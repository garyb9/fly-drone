"""SAC for encoder v5 (spec §3.3-§4): one learner at a time, the rest frozen in the loop.

Actors see only what they are deployed with: the encoder its eye stack, the decoder the
DN traces. The critic exists only during training and additionally sees DN traces and the
geometry of objects the eyes can currently see (honest-labels rule).
"""

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import MultiInputPolicy
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
        cost = 0.0
        if self.learner == "encoder":
            currents = brain.set_currents(action + 1.0)
            velocity = brain.infer(self.features) / self.env.plant.limits
            cost = self.lambda_loom * float(
                currents[LOOM].mean()
            ) + self.lambda_light * float(currents[LIGHT].mean())
        else:
            velocity = action
        self.features, reward, terminated, truncated, info = self.env.step(
            np.clip(velocity, -1, 1)
        )
        info["metabolic_cost"] = cost
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
        return super().make_actor(extractor)

    def make_critic(self, features_extractor=None):
        return super().make_critic(CriticExtractor(self.observation_space))

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["actor_key"] = self.actor_key
        return data


def build_sac(
    learner, env, buffer_size=100_000, seed=42, device="auto", learning_starts=5_000
):
    from stable_baselines3 import SAC

    return SAC(
        AsymmetricSACPolicy,
        env,
        buffer_size=buffer_size,
        batch_size=256,
        learning_starts=learning_starts,
        train_freq=1,
        gradient_steps=2,
        gamma=0.99,
        learning_rate=3e-4,
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
    import json
    from pathlib import Path

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


def warm_start_decoder(model, paths, dataset_hash, steps=4000, holdout=0.1, seed=72):
    """Behaviour-clone stage-1 DAgger labels onto the SAC decoder's tanh head."""
    from .distill import _load, class_weights
    from .teacher import DRIVES

    x, y, drive, flight = _load(paths, dataset_hash)
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
    params = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    torch.manual_seed(seed)

    def predict(ids):
        obs = {"dn": torch.as_tensor(x[ids], device=device)}
        return torch.tanh(actor.mu(actor.latent_pi(actor.features_extractor(obs))))

    losses = []
    for _ in range(steps):
        ids = rng.choice(train, 256)
        w = torch.as_tensor(weight[ids, None], device=device)
        err = (predict(ids) - torch.as_tensor(target[ids], device=device)).square()
        loss = (w * err).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
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
        actor.log_std.bias.fill_(-2.5)
        for split, mask in (("train", ~test), ("held_out", test)):
            ids = np.flatnonzero(mask)
            errs = [
                (
                    predict(ids[i : i + 4096])
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
    import json
    from pathlib import Path

    from .arena import ArenaSpec

    brain = BrainRuntime(encoder=encoder)
    # buffer_size=1: this model is only cloned and saved; rounds reload it with a real buffer.
    model = build_sac(
        "decoder",
        SpacesOnlyEnv("decoder", len(brain.feature_ids)),
        buffer_size=1,
        device=device,
    )
    report = warm_start_decoder(model, paths, brain.dataset_hash, steps)
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
