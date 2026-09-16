"""SAC for encoder v5 (spec §3.3-§4): one learner at a time, the rest frozen in the loop.

Actors see only what they are deployed with: the encoder its eye stack, the decoder the
DN traces. The critic exists only during training and additionally sees DN traces and the
geometry of objects the eyes can currently see (honest-labels rule).
"""

import json
import math
import shutil
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, MultiInputPolicy
from torch import nn

from .brain import V5_CHANNELS, BrainRuntime
from .encoder import FEATURES, EyesExtractor, eyes_space
from .env import ConnectomeEnv
from .replay import FlyDictReplayBuffer

GEOMETRY = 8
LAMBDA_LOOM = 0.01
LAMBDA_LIGHT = 0.002
LEARNERS = ("encoder", "decoder", "bypass")
ACTOR_KEYS = {"encoder": "eyes", "decoder": "dn", "bypass": "currents"}
LIGHT = [V5_CHANNELS.index(c) for c in ("mi1_l", "mi1_r", "tm3_l", "tm3_r")]
LOOM = [V5_CHANNELS.index(c) for c in ("lc4_l", "lc4_r", "lplc2_l", "lplc2_r")]
# Flags stay 0/1; body-frame metres are scaled into roughly [-2, 2] for the critic.
GEOMETRY_SCALE = (1.0, 0.25, 0.25, 0.25, 1.0, 0.25, 0.25, 0.25)
# Training only (spec §7): vary the wall bands' grey so the encoder cannot key on one
# texture. Evaluation (ConnectomeEnv) keeps the arena's 0.03 band.
BAND_GREY = (0.0, 0.15)
# Critic-only warm-up (spec §4, §8.5): the actor and the entropy coefficient stay frozen for
# the first frames of every round (SB3 `num_timesteps`, summed over workers), so an untrained
# critic cannot wreck the warm-started actor (the v4 clone or the round-0 decoder).
ACTOR_WARMUP_FRAMES = 50_000

# Post-warm-up exploration regime (design §8.6, added after the round-1 collapse). The DAgger
# warm start sets log_std = WARM_START_LOG_STD on every action dimension; exploration is pinned to
# that scale. SB3's auto-alpha default (init 1.0, target -dim A) let the entropy bonus collapse the
# encoder's deterministic mean to the action-space centre while alpha was still ~1 — a blind
# encoder with constant currents (docs/results/encoder-v5/PHASE0-DIAGNOSTICS-2026-09-16.md).
WARM_START_LOG_STD = -2.5
# Sigma in [e^-4, e^-1]: the warm start sits inside, SB3's own cap of +2 (sigma ~ 7.4) does not.
LOG_STD_MIN = -4.0
LOG_STD_MAX = -1.0
# Target (differential) entropy per dimension at the warm-start sigma: 0.5·ln(2πe) + log_std.
TARGET_ENTROPY_PER_DIM = 0.5 * math.log(2.0 * math.pi * math.e) + WARM_START_LOG_STD
# Auto-alpha initial value; SB3's 1.0 dwarfs the ~0.05/step reward. Alpha stays adaptive.
ENT_COEF_INIT = 0.01


def visible_geometry(env):
    """Threat and beacon in the body frame, only while visible; zeros otherwise."""
    from .teacher import visible

    plant = env.plant
    pos = plant.pos[0]
    yaw = float(plant.rpy[0, 2])
    c, s = np.cos(yaw), np.sin(yaw)

    def body(point):
        d = np.asarray(point, dtype=float) - pos
        return [c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]]

    out = np.zeros(GEOMETRY, np.float32)
    threat = env.roam["threat"] if env.roam is not None else None
    if threat is not None and visible(env, plant.obstacle, "obstacle"):
        out[0] = 1.0
        out[1:4] = body(plant.obstacle)
    if env.beacon_visible():
        out[4] = 1.0
        out[5:8] = body(plant.target)
    return out


def learner_spaces(learner, n_features=2022, spatial=False):
    """Observation (actor key + critic-only keys) and action space for one learner."""
    keys = {"geometry": spaces.Box(-np.inf, np.inf, (GEOMETRY,), np.float32)}
    action = spaces.Box(-1, 1, (4,), np.float32)
    if learner == "encoder":
        keys["eyes"] = eyes_space()
        keys["dn"] = spaces.Box(0, 1, (n_features,), np.float32)
        n_currents = _spatial_dim() if spatial else len(V5_CHANNELS)
        action = spaces.Box(-1, 1, (n_currents,), np.float32)
    elif learner == "decoder":
        keys["dn"] = spaces.Box(0, 1, (n_features,), np.float32)
    else:
        keys["currents"] = spaces.Box(0, 2, (len(V5_CHANNELS),), np.float32)
    return spaces.Dict(keys), action


def _spatial_dim():
    from .spatial_encoder import flat_dim

    return flat_dim()


class SacRoamEnv(gym.Env):
    """Free roam (L3, respawn) seen by one learner: encoder, decoder or brain bypass."""

    metadata = {}

    def __init__(
        self,
        learner,
        decoder=None,
        encoder=None,
        level=3,
        lambda_loom=LAMBDA_LOOM,
        lambda_light=LAMBDA_LIGHT,
        spatial=False,
    ):
        if learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}")
        if spatial and learner == "bypass":
            raise ValueError("the spatial v6 path has no brain-bypass control")
        self.spatial = bool(spatial)
        if learner == "encoder":
            if decoder is None:
                raise ValueError("encoder learning needs a frozen decoder")
            from .brain import LEARNED_EXTERNAL_V6

            brain = BrainRuntime(
                encoder=LEARNED_EXTERNAL_V6 if self.spatial else "external"
            )
            # The frozen decoder is scenery here; validate() pairs it with the learned
            # encoder under the strict version check.
            brain.load_policy(decoder, check_encoder=False)
        else:
            if encoder is None:
                raise ValueError(f"{learner} learning needs a frozen encoder")
            brain = BrainRuntime(encoder=encoder)
        self.learner = learner
        self.lambda_loom = lambda_loom
        self.lambda_light = lambda_light
        self.env = ConnectomeEnv(
            task="free_roam", level=level, respawn=True, brain=brain
        )
        n = len(brain.feature_ids)
        self.observation_space, self.action_space = learner_spaces(
            learner, n, spatial=self.spatial
        )
        self.features = np.zeros(n, np.float32)
        if self.spatial:
            from .spatial_encoder import group_slices

            self._groups = group_slices()

    def _obs(self):
        brain, keys = self.env.brain, self.observation_space.spaces
        obs = {"geometry": visible_geometry(self.env)}
        if "eyes" in keys:
            obs["eyes"] = brain.stack.array()
        if "dn" in keys:
            obs["dn"] = self.features
        if "currents" in keys:
            obs["currents"] = brain.encoder.currents(brain.stack.array())
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.features, info = self.env.reset(seed=seed)
        self._randomise_bands()
        return self._obs(), info

    def _randomise_bands(self):
        plant = self.env.plant
        grey = float(self.np_random.uniform(*BAND_GREY))
        for g in range(plant.model.ngeom):
            name = mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.endswith("_band"):
                plant.model.geom_rgba[g, :3] = grey  # alpha stays: ghost mode owns it
        # The first stacked frame was rendered before the recolour; start the stack again.
        self.env.brain.stack.clear()
        self.env.brain.push_frame(self.env._frame())

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        brain = self.env.brain
        cost = loom_cost = light_cost = 0.0
        if self.learner == "encoder":
            currents = brain.set_currents(action + 1.0)
            velocity = brain.infer(self.features) / self.env.plant.limits
            if self.spatial:
                loom_cost = self.lambda_loom * float(
                    currents[self._groups["loom"]].mean()
                )
                light_cost = self.lambda_light * float(
                    currents[self._groups["light"]].mean()
                )
            else:
                loom_cost = self.lambda_loom * float(currents[LOOM].mean())
                light_cost = self.lambda_light * float(currents[LIGHT].mean())
            cost = loom_cost + light_cost
        else:
            velocity = action
        self.features, reward, terminated, truncated, info = self.env.step(
            np.clip(velocity, -1, 1)
        )
        info["metabolic_cost"] = cost
        info["loom_cost"] = loom_cost
        info["light_cost"] = light_cost
        return self._obs(), float(reward - cost), terminated, truncated, info

    def close(self):
        self.env.close()


class KeyNormalizer(BaseFeaturesExtractor):
    """One observation key, standardised with mean/scale buffers (like NeuralNormalizer)."""

    def __init__(self, observation_space, key):
        n = observation_space[key].shape[0]
        super().__init__(observation_space, n)
        self.key = key
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))

    def forward(self, observations):
        return (observations[self.key] - self.mean) / self.scale


class CriticExtractor(BaseFeaturesExtractor):
    """Training-only view: every key present (eyes, DN traces, currents, visible geometry)."""

    def __init__(self, observation_space):
        keys = observation_space.spaces
        n_dn = keys["dn"].shape[0] if "dn" in keys else 0
        dim = GEOMETRY + (2 * FEATURES if "eyes" in keys else 0) + (64 if n_dn else 0)
        dim += len(V5_CHANNELS) if "currents" in keys else 0
        super().__init__(observation_space, dim)
        self.register_buffer("geometry_scale", torch.tensor(GEOMETRY_SCALE))
        self.eyes = EyesExtractor(observation_space) if "eyes" in keys else None
        self.dn_norm = KeyNormalizer(observation_space, "dn") if n_dn else None
        self.dn = nn.Sequential(nn.Linear(n_dn, 64), nn.ReLU()) if n_dn else None
        self.has_currents = "currents" in keys

    def forward(self, observations):
        parts = [observations["geometry"] * self.geometry_scale]
        if self.eyes is not None:
            parts.append(self.eyes(observations))
        if self.dn is not None:
            parts.append(self.dn(self.dn_norm(observations)))
        if self.has_currents:
            parts.append(observations["currents"])
        return torch.cat(parts, 1)


class ClampedActor(Actor):
    """SAC actor whose `log_std` is clamped to [LOG_STD_MIN, LOG_STD_MAX].

    SB3's own cap allows `log_std = +2` (sigma ~ 7.4); a tanh-squashed actor at that spread
    saturates and loses control of its mean. Keeping sigma near the warm start leaves the mean in
    control, which is where the deployed encoder's currents come from (`encoder.py:138`).
    """

    def get_action_dist_params(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = torch.clamp(self.log_std(latent_pi), LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}


class AsymmetricSACPolicy(MultiInputPolicy):
    """SAC policy whose actor reads one deployable key while the critic reads every key."""

    def __init__(self, *args, actor_key="eyes", **kwargs):
        self.actor_key = actor_key
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        if self.actor_key == "eyes":
            extractor = EyesExtractor(self.observation_space)
        else:
            extractor = KeyNormalizer(self.observation_space, self.actor_key)
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, extractor)
        return ClampedActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None):
        return super().make_critic(CriticExtractor(self.observation_space))

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["actor_key"] = self.actor_key
        return data


class WarmupSAC(SAC):
    """SAC that trains only the critic while `num_timesteps < actor_warmup`.

    During warm-up the actor's and the entropy coefficient's optimizer steps are skipped, so
    their parameters and Adam state stay bitwise untouched; the critic and its Polyak target
    train as usual. Freezing alpha too matters: with the actor frozen, auto-alpha would chase
    the target entropy for the whole warm-up and hand the unfrozen actor a runaway entropy
    bonus. `actor_warmup` is saved in the `.zip`; `WarmupSAC.load(..., actor_warmup=n)`
    overrides it (a `.zip` saved by plain SAC otherwise gets the default).
    """

    def __init__(self, *args, actor_warmup=ACTOR_WARMUP_FRAMES, **kwargs):
        self.actor_warmup = int(actor_warmup)
        super().__init__(*args, **kwargs)

    @property
    def actor_frozen(self):
        return self.num_timesteps < self.actor_warmup

    def train(self, gradient_steps, batch_size=64):
        self.logger.record("train/actor_frozen", int(self.actor_frozen))
        if not self.actor_frozen:
            return super().train(gradient_steps, batch_size)
        self._train_critic_only(gradient_steps, batch_size)

    def _train_critic_only(self, gradient_steps, batch_size):
        """Critic-only gradient steps: SAC's loop with the actor and entropy losses removed.

        The actor's and alpha's parameters and Adam state stay bitwise untouched, and the
        actor forward-backward that only the actor loss needs is not run at all (the warm-up
        spends its compute on the critic, not on optimising an actor that cannot move). Mirrors
        stable_baselines3 2.9.0 ``SAC.train``; the parameters it moves and the losses it logs
        are pinned by ``test_warmup_trains_only_the_critic_then_normal_sac``.
        """
        from stable_baselines3.common.utils import polyak_update
        from torch.nn import functional as F

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        # Frozen alpha: read once, never stepped.
        ent_coef = (
            torch.exp(self.log_ent_coef.detach())
            if self.log_ent_coef is not None
            else self.ent_coef_tensor
        )
        critic_losses = []
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", ent_coef.item())
        self.logger.record("train/critic_loss", np.mean(critic_losses))


def build_sac(
    learner,
    env,
    buffer_size=100_000,
    seed=42,
    device="auto",
    learning_starts=5_000,
    actor_warmup=ACTOR_WARMUP_FRAMES,
    n_step=1,
    optimize_memory=False,
    spatial=False,
):
    action_dim = int(np.prod(env.action_space.shape))
    policy_class = AsymmetricSACPolicy
    pi_arch = [] if learner == "encoder" else [64, 64]
    if spatial:
        from .spatial_policy import SpatialSACPolicy

        policy_class = SpatialSACPolicy
        pi_arch = []
    return WarmupSAC(
        policy_class,
        env,
        actor_warmup=actor_warmup,
        buffer_size=buffer_size,
        replay_buffer_class=FlyDictReplayBuffer,
        replay_buffer_kwargs={"n_steps": int(n_step), "gamma": 0.99},
        optimize_memory_usage=bool(optimize_memory),
        batch_size=256,
        learning_starts=learning_starts,
        train_freq=1,
        gradient_steps=2,
        gamma=0.99,
        learning_rate=3e-4,
        # Post-warm-up exploration: small initial alpha pinned to the warm-start sigma (design §8.6).
        ent_coef=f"auto_{ENT_COEF_INIT}",
        target_entropy=action_dim * TARGET_ENTROPY_PER_DIM,
        seed=seed,
        device=device,
        verbose=1,
        policy_kwargs={
            "actor_key": ACTOR_KEYS[learner],
            # The encoder's head is linear on the eye features (LearnedEncoder format);
            # the decoder's hidden layers are tanh (Rust actor format). The v6 spatial
            # actor's latent must stay empty so its mean head is the identity.
            "net_arch": {
                "pi": pi_arch,
                "qf": [256, 256],
            },
            "activation_fn": nn.Tanh,
        },
    )


def apply_entropy_regime(model, action_dim):
    """Pin a loaded model to the post-warm-up entropy regime (design §8.6).

    `WarmupSAC.load` restores the saved `ent_coef`, `target_entropy` and alpha, so a model saved
    before the regime landed (the round-0 decoder, or any round chained off it) keeps the old
    automatic schedule — alpha 1.0 at unfreeze, target entropy ``-dim``. A fresh round always
    starts pinned to the warm-start sigma, so loaded rounds are reset to match; `--resume` keeps
    its own evolved alpha and does not call this.
    """
    model.ent_coef = f"auto_{ENT_COEF_INIT}"
    model.target_entropy = float(action_dim) * TARGET_ENTROPY_PER_DIM
    if model.log_ent_coef is not None:
        with torch.no_grad():
            model.log_ent_coef.fill_(math.log(ENT_COEF_INIT))
    return model


def set_dn_stats(model, mean, scale):
    """Standardise DN traces the same way in the actor, the critic and the target critic."""
    mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
    scale = torch.as_tensor(np.asarray(scale), dtype=torch.float32)
    for net in (model.actor, model.critic, model.critic_target):
        for module in net.modules():
            if isinstance(module, KeyNormalizer) and module.key == "dn":
                module.mean.copy_(mean)
                module.scale.copy_(scale)


class SpacesOnlyEnv(gym.Env):
    """Just the spaces SAC needs to build a model; no brain or renderer."""

    metadata = {}

    def __init__(self, learner, n_features=2022, spatial=False):
        self.observation_space, self.action_space = learner_spaces(
            learner, n_features, spatial=spatial
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {
            k: np.zeros(s.shape, s.dtype)
            for k, s in self.observation_space.spaces.items()
        }
        return obs, {}

    def step(self, action):
        return self.reset()[0], 0.0, True, False, {}


def export_decoder(model, brain, path, limits):
    """SAC decoder actor -> Rust actor JSON (tanh hidden, tanh output) with a parity check."""
    actor = model.actor
    layers = []
    for layer in list(actor.latent_pi) + [actor.mu]:
        if isinstance(layer, nn.Linear):
            layers.append(
                {
                    "weights": layer.weight.detach().cpu().tolist(),
                    "bias": layer.bias.detach().cpu().tolist(),
                }
            )
        elif not isinstance(layer, nn.Tanh):
            raise ValueError("only tanh MLP export supported")
    norm = actor.features_extractor
    limits = np.asarray(limits, dtype=float)
    payload = {
        "version": 1,
        "encoder_version": brain.encoder_version,
        "dataset_hash": brain.dataset_hash,
        "feature_ids": brain.feature_ids,
        "mean": norm.mean.cpu().tolist(),
        "scale": norm.scale.cpu().tolist(),
        "layers": layers,
        "action_limits": limits.tolist(),
        "output": "tanh",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    brain.load_policy(path)
    rng = np.random.default_rng(123)
    error = 0.0
    for x in rng.uniform(0, 1, (32, len(brain.feature_ids))).astype(np.float32):
        obs = {"dn": x, "geometry": np.zeros(GEOMETRY, np.float32)}
        expected = model.predict(obs, deterministic=True)[0] * limits
        error = max(error, float(np.max(np.abs(expected - brain.infer(x)))))
    if error > 1e-4:
        raise RuntimeError(f"Rust policy parity failed: {error}")
    return error


def warm_start_decoder(
    model, paths, dataset_hash, encoder_version, steps=4000, holdout=0.1, seed=72
):
    """Behaviour-clone teacher labels onto the SAC decoder's tanh head.

    The loss is on the pre-tanh mean against atanh of the clipped labels, so saturated
    labels keep a gradient; the reported per-drive MSE stays in tanh (action) space.
    `paths` must have been collected with `encoder_version`.
    """
    from .distill import _load, class_weights
    from .teacher import DRIVES

    x, y, drive, flight = _load(paths, dataset_hash, encoder_version)
    rng = np.random.default_rng(seed)
    unique = np.unique(flight)
    held = rng.choice(unique, max(1, int(len(unique) * holdout)), replace=False)
    test = np.isin(flight, held)
    train = np.flatnonzero(~test)
    set_dn_stats(model, x[train].mean(0), np.maximum(x[train].std(0), 0.003))
    actor, device = model.actor, model.device
    weight = class_weights(drive, ~test)[drive].astype(np.float32)
    # tanh never reaches +-1: stop labels just short of saturation.
    target = np.clip(y, -0.97, 0.97).astype(np.float32)
    pre_tanh_target = np.arctanh(target)
    params = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    torch.manual_seed(seed)

    def pre_tanh(ids):
        obs = {"dn": torch.as_tensor(x[ids], device=device)}
        return actor.mu(actor.latent_pi(actor.features_extractor(obs)))

    losses = []
    for _ in range(steps):
        ids = rng.choice(train, 256)
        w = torch.as_tensor(weight[ids, None], device=device)
        z_target = torch.as_tensor(pre_tanh_target[ids], device=device)
        err = (pre_tanh(ids) - z_target).square()
        loss = (w * err).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    report = {
        "samples": int(len(x)),
        "held_out_flights": int(len(held)),
        "loss_first": losses[0],
        "loss_last": float(np.mean(losses[-20:])),
        "drives": {name: {} for name in DRIVES},
    }
    with torch.no_grad():
        # Start SAC exploration narrow around the cloned behaviour.
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(WARM_START_LOG_STD)
        for split, mask in (("train", ~test), ("held_out", test)):
            ids = np.flatnonzero(mask)
            errs = [
                (
                    torch.tanh(pre_tanh(ids[i : i + 4096]))
                    - torch.as_tensor(target[ids[i : i + 4096]], device=device)
                )
                .square()
                .cpu()
                .numpy()
                for i in range(0, len(ids), 4096)
            ]
            err = np.concatenate(errs) if errs else np.zeros((0, target.shape[1]))
            for k, name in enumerate(DRIVES):
                sel = drive[mask] == k
                if not sel.any():
                    report["drives"][name][f"{split}_mse"] = None
                    continue
                # Per-action-axis error (vy-vz spec M1): vx, vy, vz, yaw.
                per_axis = err[sel].mean(0)
                report["drives"][name][f"{split}_mse"] = float(per_axis.mean())
                for axis, value in zip(
                    ("vx", "vy", "vz", "yaw"), per_axis, strict=True
                ):
                    report["drives"][name][f"{split}_{axis}_mse"] = float(value)
    return report


def init_decoder(paths, encoder, output, steps=4000, device="auto"):
    """Round-0 decoder: DAgger labels on a tanh head, exported for the given encoder."""
    from .arena import ArenaSpec

    brain = BrainRuntime(encoder=encoder)
    # buffer_size=1: this model is only cloned and saved; rounds reload it with a real buffer.
    model = build_sac(
        "decoder",
        SpacesOnlyEnv("decoder", len(brain.feature_ids)),
        buffer_size=1,
        device=device,
    )
    report = warm_start_decoder(
        model, paths, brain.dataset_hash, brain.encoder_version, steps
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "decoder")
    report["export_max_error"] = export_decoder(
        model, brain, out / "decoder.json", ArenaSpec().limits
    )
    report["encoder_version"] = brain.encoder_version
    report["paths"] = [str(p) for p in paths]
    (out / "warm-start.json").write_text(json.dumps(report, indent=2))
    return report


def export_checkpoint(checkpoint, learner, output, encoder=None, device="auto"):
    """Deployable artifact from any SAC `.zip` (a round output or a CheckpointCallback
    checkpoint): `encoder.pt` for `learner="encoder"`, or a parity-checked `decoder.json`
    (pinned to `encoder`) for `learner="decoder"`. Reuses the same helpers a round uses to
    export its own result, so a mid-round checkpoint can be validated early.
    """
    from stable_baselines3 import SAC

    from .arena import ArenaSpec
    from .encoder import LearnedEncoder

    # buffer_size=1: exporting only reads the actor's weights, no replay buffer needed.
    model = SAC.load(checkpoint, device=device, buffer_size=1)
    out = Path(output)
    if learner == "encoder":
        version = LearnedEncoder.from_actor(model.actor).save(out)
        return {"learner": "encoder", "encoder_version": version, "output": str(out)}
    if learner == "decoder":
        if encoder is None:
            raise ValueError("a decoder export needs --encoder")
        brain = BrainRuntime(encoder=encoder)
        error = export_decoder(model, brain, out, ArenaSpec().limits)
        return {
            "learner": "decoder",
            "export_max_error": error,
            "encoder_version": brain.encoder_version,
            "output": str(out),
        }
    raise ValueError("learner must be 'encoder' or 'decoder'")


_DECODER_FIELDS = {
    "version",
    "encoder_version",
    "dataset_hash",
    "feature_ids",
    "mean",
    "scale",
    "layers",
    "action_limits",
    "output",
}


def repin_decoder(source, encoder, output):
    """Copy a frozen decoder JSON unchanged onto a new encoder, so `sac-validate`/
    `encoder-checks` can measure the new encoder (`check_encoder=False`, same pairing the
    encoder-learning env uses) before a decoder round is spent re-training against it.

    Refuses any source JSON whose `output` or `layers` it cannot copy verbatim.
    """
    from .encoder import LearnedEncoder

    data = json.loads(Path(source).read_text())
    missing = _DECODER_FIELDS - data.keys()
    if missing:
        raise ValueError(f"{source}: missing fields {sorted(missing)}")
    if data["output"] != "tanh":
        raise ValueError(
            f"{source}: only tanh-output decoders are supported, got {data['output']!r}"
        )
    layers = data["layers"]
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"{source}: no layers to copy verbatim")
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict) or "weights" not in layer or "bias" not in layer:
            raise ValueError(f"{source}: layer {i} missing weights/bias")
        if len(layer["weights"]) != len(layer["bias"]):
            raise ValueError(f"{source}: layer {i} weights/bias shape mismatch")

    new_encoder = LearnedEncoder.load(encoder)
    pinned = dict(data)
    pinned["repinned_from"] = data["encoder_version"]
    pinned["encoder_version"] = new_encoder.version
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pinned, indent=2))
    return {
        "output": str(out),
        "encoder_version": new_encoder.version,
        "repinned_from": data["encoder_version"],
    }


class MetabolicLogger(BaseCallback):
    def _on_step(self):
        infos = self.locals["infos"]
        for key in ("metabolic_cost", "loom_cost", "light_cost"):
            values = [info.get(key, 0.0) for info in infos]
            self.logger.record_mean(f"rollout/{key}", float(np.mean(values)))
        return True


class ProbeLogger(BaseCallback):
    """Records actor exploration and probe statistics the standard SAC logs omit.

    On a fixed observation batch (sampled once, seeded, from the training observation space) it
    records the mean clamped `log_std`, the deterministic action's mean level, and its variation
    across states. The round-1 encoder's deployed mean collapsed to a constant while `log_std`
    stayed in range, so `log_std` alone would not have caught it; `probe/action_state_std -> 0`
    does (docs/results/encoder-v5/PHASE0-DIAGNOSTICS-2026-09-16.md).
    """

    def __init__(self, every=1000, batch=64, seed=0):
        super().__init__()
        self.every = int(every)
        self.batch = int(batch)
        self.seed = int(seed)
        self.probe = None

    def _make_probe(self):
        space = self.training_env.observation_space
        rng = np.random.default_rng(self.seed)
        probe = {}
        for key, sub in space.spaces.items():
            low = np.where(np.isfinite(sub.low), sub.low, -1.0)
            high = np.where(np.isfinite(sub.high), sub.high, 1.0)
            probe[key] = rng.uniform(low, high, (self.batch, *sub.shape)).astype(
                sub.dtype
            )
        return probe

    def _on_step(self):
        if self.model.num_timesteps % self.every:
            return True
        if self.probe is None:
            self.probe = self._make_probe()
        probe = {
            k: torch.as_tensor(v, device=self.model.device)
            for k, v in self.probe.items()
        }
        actor = self.model.actor
        with torch.no_grad():
            mean, log_std, _ = actor.get_action_dist_params(probe)
            tanh_mean = torch.tanh(mean)
        self.logger.record("probe/log_std_mean", float(log_std.mean()))
        self.logger.record("probe/action_mean", float(tanh_mean.mean()))
        self.logger.record("probe/action_state_std", float(tanh_mean.std(0).mean()))
        return True


class ActionLogger(BaseCallback):
    """Records the mean |vy| and |vz| the decoder commanded (vy-vz spec M3).

    Only the 4-axis velocity decoder is logged; the encoder/bypass actors write 8 currents, where
    indices 1 and 2 are not velocity axes.
    """

    def _on_step(self):
        actions = self.locals.get("actions")
        if actions is None:
            return True
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != 4:
            return True
        self.logger.record_mean(
            "rollout/action_vy_absmean", float(np.abs(actions[:, 1]).mean())
        )
        self.logger.record_mean(
            "rollout/action_vz_absmean", float(np.abs(actions[:, 2]).mean())
        )
        return True


class ResumeCheckpoint(BaseCallback):
    """Keeps one model+replay-buffer snapshot so a crashed round can resume (Pack 2 S2).

    Model and buffer are written back to back on the same callback step, so they describe the
    same ``num_timesteps``; each snapshot overwrites the previous one, so at most one copy sits
    on disk. Resume trusts the model's timestep count, not a separate metadata file.
    """

    def __init__(self, out, every):
        super().__init__()
        self.dir = Path(out) / "resume"
        self.every = max(1, int(every))
        self._next = self.every

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self.dir / "model")
        self.model.save_replay_buffer(self.dir / "buffer.pkl")
        self._next += self.every
        return True


def _factory(learner, rank, seed, decoder, encoder, level):
    def make():
        from stable_baselines3.common.monitor import Monitor

        torch.set_num_threads(1)
        env = Monitor(
            SacRoamEnv(learner, decoder=decoder, encoder=encoder, level=level)
        )
        env.reset(seed=seed + rank)
        return env

    return make


def train_round(
    learner,
    output,
    frames,
    decoder=None,
    encoder=None,
    init=None,
    workers=6,
    seed=42,
    buffer_size=100_000,
    device="auto",
    level=3,
    learning_starts=5_000,
    actor_warmup=ACTOR_WARMUP_FRAMES,
    n_step=1,
    optimize_memory=False,
    resume=False,
    resume_every=50_000,
    keep_resume=False,
):
    """One SAC round for one learner; the other half is frozen inside the environment.

    The first `actor_warmup` frames train only the critic (`WarmupSAC`); 0 disables it.
    `learn(reset_num_timesteps=True)` counts every round from frame 0, so a round resumed
    from a checkpoint repeats the warm-up unless `actor_warmup=0` is passed.

    `n_step>1` uses n-step returns and `optimize_memory=True` halves the buffer's RAM (both
    via `FlyDictReplayBuffer`); the defaults reproduce the pre-Pack-2 round exactly. With
    `resume=True` a `resume/` snapshot left by the same round continues it instead of starting
    over; `frames` is then the round's total, and the warm-up is not repeated because
    `num_timesteps` is restored.
    """
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .arena import ArenaSpec
    from .encoder import LearnedEncoder

    # Fail before spawning workers: a missing frozen partner would only surface inside one.
    if learner == "encoder" and decoder is None:
        raise ValueError("an encoder round needs a frozen decoder")
    if learner in ("decoder", "bypass") and encoder is None:
        raise ValueError(f"a {learner} round needs a frozen encoder")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), 6))
    snapshot = out / "resume"
    resumed_from = None
    decoder = str(Path(decoder).resolve()) if decoder else None
    encoder = str(Path(encoder).resolve()) if encoder else None
    factories = [
        _factory(learner, r, seed, decoder, encoder, level) for r in range(workers)
    ]
    # Each env owns a full brain and renderer; spawn keeps EGL state per process.
    vec = (
        SubprocVecEnv(factories, start_method="spawn")
        if workers > 1
        else DummyVecEnv(factories)
    )
    try:
        buffer_kwargs = {
            "replay_buffer_class": FlyDictReplayBuffer,
            "replay_buffer_kwargs": {"n_steps": int(n_step), "gamma": 0.99},
            "optimize_memory_usage": bool(optimize_memory),
            "n_steps": int(n_step),
        }
        if resume and (snapshot / "model.zip").exists():
            # Continue the interrupted attempt. The snapshot's model and buffer are a matched
            # pair from the same step, and its num_timesteps means the warm-up is already over.
            model = WarmupSAC.load(
                snapshot / "model.zip",
                env=vec,
                device=device,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                seed=seed,
                actor_warmup=int(actor_warmup),
                **buffer_kwargs,
            )
            model.load_replay_buffer(snapshot / "buffer.pkl")
            resumed_from = int(model.num_timesteps)
        elif init is not None and str(init).endswith(".zip"):
            # Fresh replay buffer each round: the frozen partner changed, old transitions are stale.
            # WarmupSAC.load (not SAC.load) keeps train() warm-up-aware; the argument, not
            # the value saved in the .zip, sets this round's warm-up.
            model = WarmupSAC.load(
                init,
                env=vec,
                device=device,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                seed=seed,
                actor_warmup=int(actor_warmup),
                **buffer_kwargs,
            )
            # The .zip carries the regime it was trained under; every fresh round uses the pinned
            # one (design §8.6), so an older round-0/DAgger init cannot silently restore `auto`.
            apply_entropy_regime(model, int(np.prod(vec.action_space.shape)))
        else:
            model = build_sac(
                learner,
                vec,
                buffer_size,
                seed,
                device,
                learning_starts,
                actor_warmup,
                n_step,
                optimize_memory,
            )
            if init is not None:
                if learner != "encoder":
                    raise ValueError(
                        "a .pt init is the v4 clone for the first encoder round"
                    )
                state = torch.load(init, map_location="cpu", weights_only=True)
                model.actor.features_extractor.load_state_dict(state["extractor"])
                model.actor.mu.load_state_dict(state["mu"])
                with torch.no_grad():
                    model.actor.log_std.weight.zero_()
                    model.actor.log_std.bias.fill_(WARM_START_LOG_STD)
        if learner == "encoder":
            stats = json.loads(Path(decoder).read_text())
            set_dn_stats(model, stats["mean"], stats["scale"])
        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=max(1, 50_000 // workers),
                    save_path=str(out / "checkpoints"),
                    name_prefix=learner,
                ),
                MetabolicLogger(),
                ProbeLogger(),
                ActionLogger(),
                ResumeCheckpoint(out, resume_every),
            ]
        )
        # A resumed round runs the remainder of its target; with reset_num_timesteps=False,
        # `learn` adds `total_timesteps` to the restored count.
        remaining = (
            frames if resumed_from is None else max(1, int(frames) - resumed_from)
        )
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=resumed_from is None,
        )
        model.save(out / learner)
        if snapshot.exists() and not keep_resume:
            shutil.rmtree(
                snapshot
            )  # the round finished; the one-shot buffer is not needed
        report = {
            "learner": learner,
            "frames": int(model.num_timesteps),
            "decoder": decoder,
            "encoder": encoder,
            "init": str(init) if init is not None else None,
            "workers": workers,
            "seed": seed,
            "buffer_size": buffer_size,
            "actor_warmup": int(model.actor_warmup),
            "n_step": int(n_step),
            "optimize_memory": bool(optimize_memory),
            "resumed_from": resumed_from,
            "snapshot": str(snapshot) if snapshot.exists() else None,
        }
        if learner == "encoder":
            report["encoder_version"] = LearnedEncoder.from_actor(model.actor).save(
                out / "encoder.pt"
            )
    finally:
        vec.close()
    # Built after the workers are gone: parity/export only needs the actor's weights.
    if learner == "decoder":
        brain = BrainRuntime(encoder=encoder)
        report["export_max_error"] = export_decoder(
            model, brain, out / "decoder.json", ArenaSpec().limits
        )
        report["encoder_version"] = brain.encoder_version
    (out / "round.json").write_text(json.dumps(report, indent=2))
    return report


def validate(
    decoder,
    encoder,
    output,
    seeds=10,
    seed_base=9000,
    seconds=60,
    workers=6,
    level=3,
    policy_checks=True,
):
    """End-of-round check on validation seeds: near-dodge (intact, ghost), foraging, E1-E2.

    E1/E2 come from the fixed probe (teacher-flown, so labels are policy-independent); the
    policy-driven numbers are kept alongside as `E1_policy`/`E2_policy` (spec R1).
    """
    from .distill import screen
    from .roam_eval import encoder_checks

    output = Path(output)
    decoder = str(Path(decoder).resolve())
    encoder = str(Path(encoder).resolve())
    key = f"policy:{decoder}"
    report = screen(
        None,
        output.with_suffix(".screen.json"),
        seeds,
        seconds,
        level,
        workers,
        seed_base=seed_base,
        combos=[(key, "none"), (key, "ghost")],
        encoder=encoder,
    )
    checks = encoder_checks(
        decoder,
        output.with_suffix(".checks.json"),
        encoder,
        seeds,
        seconds,
        workers,
        seed_base,
        level,
        controller="teacher",
    )
    policy_checks = (
        encoder_checks(
            decoder,
            output.with_suffix(".checks-policy.json"),
            encoder,
            seeds,
            seconds,
            workers,
            seed_base,
            level,
            controller="policy",
        )
        if policy_checks
        else None
    )
    none, ghost = report["results"][f"{key}|none"], report["results"][f"{key}|ghost"]
    summary = {
        "decoder": decoder,
        "encoder": encoder,
        "seeds": [seed_base, seed_base + seeds - 1],
        "near_dodge_rate": none["near_dodge_rate"],
        "balanced_dodge_rate": none["balanced_dodge_rate"],
        "ghost_near_dodge_rate": ghost["near_dodge_rate"],
        "beacons_per_min": none["beacons_per_min"],
        "collisions_per_min": none["collisions_per_min"],
        "E1": checks["E1"],
        "E2": checks["E2"],
        "E1_policy": policy_checks["E1"] if policy_checks else None,
        "E2_policy": policy_checks["E2"] if policy_checks else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    return summary
