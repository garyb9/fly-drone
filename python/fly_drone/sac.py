"""SAC for encoder v5 (spec §3.3-§4): one learner at a time, the rest frozen in the loop.

Actors see only what they are deployed with: the encoder its eye stack, the decoder the
DN traces. The critic exists only during training and additionally sees DN traces and the
geometry of objects the eyes can currently see (honest-labels rule).
"""

import json
import math
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


def learner_spaces(learner, n_features=2022):
    """Observation (actor key + critic-only keys) and action space for one learner."""
    keys = {"geometry": spaces.Box(-np.inf, np.inf, (GEOMETRY,), np.float32)}
    action = spaces.Box(-1, 1, (4,), np.float32)
    if learner == "encoder":
        keys["eyes"] = eyes_space()
        keys["dn"] = spaces.Box(0, 1, (n_features,), np.float32)
        action = spaces.Box(-1, 1, (len(V5_CHANNELS),), np.float32)
    elif learner == "decoder":
        keys["dn"] = spaces.Box(0, 1, (n_features,), np.float32)
    else:
        keys["currents"] = spaces.Box(0, 2, (len(V5_CHANNELS),), np.float32)
    return spaces.Dict(keys), action


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
    ):
        if learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}")
        if learner == "encoder":
            if decoder is None:
                raise ValueError("encoder learning needs a frozen decoder")
            brain = BrainRuntime(encoder="external")
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
        self.observation_space, self.action_space = learner_spaces(learner, n)
        self.features = np.zeros(n, np.float32)

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
        self.env.brain.push_frame(plant.camera())

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        brain = self.env.brain
        cost = loom_cost = light_cost = 0.0
        if self.learner == "encoder":
            currents = brain.set_currents(action + 1.0)
            velocity = brain.infer(self.features) / self.env.plant.limits
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
        frozen = self.actor_frozen
        self.logger.record("train/actor_frozen", int(frozen))
        if not frozen:
            return super().train(gradient_steps, batch_size)
        optimizers = [self.actor.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        for optimizer in optimizers:
            optimizer.step = _skip_step  # instance attribute shadows the class method
        try:
            return super().train(gradient_steps, batch_size)
        finally:
            for optimizer in optimizers:
                del optimizer.step


def _skip_step(closure=None):
    return None


def build_sac(
    learner,
    env,
    buffer_size=100_000,
    seed=42,
    device="auto",
    learning_starts=5_000,
    actor_warmup=ACTOR_WARMUP_FRAMES,
):
    action_dim = int(np.prod(env.action_space.shape))
    return WarmupSAC(
        AsymmetricSACPolicy,
        env,
        actor_warmup=actor_warmup,
        buffer_size=buffer_size,
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
            # the decoder's hidden layers are tanh (Rust actor format).
            "net_arch": {
                "pi": [] if learner == "encoder" else [64, 64],
                "qf": [256, 256],
            },
            "activation_fn": nn.Tanh,
        },
    )


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

    def __init__(self, learner, n_features=2022):
        self.observation_space, self.action_space = learner_spaces(learner, n_features)

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
                .mean(1)
                .cpu()
                .numpy()
                for i in range(0, len(ids), 4096)
            ]
            err = np.concatenate(errs) if errs else np.zeros(0)
            for k, name in enumerate(DRIVES):
                sel = drive[mask] == k
                report["drives"][name][f"{split}_mse"] = (
                    float(err[sel].mean()) if sel.any() else None
                )
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
):
    """One SAC round for one learner; the other half is frozen inside the environment.

    The first `actor_warmup` frames train only the critic (`WarmupSAC`); 0 disables it.
    `learn(reset_num_timesteps=True)` counts every round from frame 0, so a round resumed
    from a checkpoint repeats the warm-up unless `actor_warmup=0` is passed.
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
        if init is not None and str(init).endswith(".zip"):
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
            )
        else:
            model = build_sac(
                learner, vec, buffer_size, seed, device, learning_starts, actor_warmup
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
            ]
        )
        model.learn(
            total_timesteps=frames, callback=callbacks, reset_num_timesteps=True
        )
        model.save(out / learner)
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
    decoder, encoder, output, seeds=10, seed_base=9000, seconds=60, workers=6, level=3
):
    """End-of-round check on validation seeds: near-dodge (intact, ghost), foraging, E1-E2."""
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
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    return summary
