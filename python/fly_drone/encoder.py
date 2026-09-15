"""Learned eye encoder (v5): two cameras' 3-frame luma stacks -> 8 input currents.

The encoder sees only its own camera frames. It speaks to the frozen connectome by one
uniform, bounded current per anatomical input population (brain.V5_CHANNELS).
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .brain import (
    ENCODER_VERSION,
    EYE_SHAPE,
    LEARNED_PREFIX,
    STACK_FRAMES,
    V4_TO_V5,
    V5_CHANNELS,
    FrameStack,
)

FEATURES = 64
# Clone loss weight on the left-right difference of each pathway (the steering signal).
DIFF_WEIGHT = 1.0
# A loom target, or the left-right light difference, above this marks a rare frame to oversample.
ACTIVE_THRESHOLD = 0.05
# (pathway, left index, right index) in V5_CHANNELS order.
PAIRS = tuple(
    (name[:-2], i, V5_CHANNELS.index(name[:-2] + "_r"))
    for i, name in enumerate(V5_CHANNELS)
    if name.endswith("_l")
)


def eyes_space(frames=STACK_FRAMES):
    return spaces.Box(0, 255, (2 * frames, *EYE_SHAPE), np.uint8)


def v4_targets(cues):
    return np.asarray(cues)[..., list(V4_TO_V5)]


class EyeNet(nn.Module):
    """One eye: frames + a side channel, 48x64 -> 24x32 -> 12x16 -> 6x8 -> 64 features."""

    def __init__(self, frames=STACK_FRAMES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(frames + 1, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.dense = nn.Sequential(nn.Linear(32 * 6 * 8, FEATURES), nn.ReLU())

    def forward(self, x):
        return self.dense(self.conv(x))


class EyesExtractor(BaseFeaturesExtractor):
    """Both eyes through one EyeNet; the right eye is mirrored and flagged -1."""

    def __init__(self, observation_space, frames=STACK_FRAMES):
        super().__init__(observation_space, 2 * FEATURES)
        self.frames = frames
        self.eye = EyeNet(frames)

    def forward(self, observations):
        eyes = observations["eyes"]
        f = self.frames
        side = eyes.new_ones(eyes.shape[0], 1, *EYE_SHAPE)
        left = torch.cat([eyes[:, :f], side], 1)
        right = torch.cat([torch.flip(eyes[:, f:], dims=[3]), -side], 1)
        return torch.cat([self.eye(left), self.eye(right)], 1)


def weights_hash(state):
    digest = hashlib.sha256()
    for part in sorted(state):
        for key in sorted(state[part]):
            digest.update(f"{part}.{key}".encode())
            digest.update(
                state[part][key].detach().cpu().numpy().astype(np.float32).tobytes()
            )
    return digest.hexdigest()[:16]


class LearnedEncoder:
    """Deployable encoder: SAC actor's eye extractor + mean head, tanh-squashed to [0, 2]."""

    def __init__(self, extractor, mu):
        self.extractor = extractor.cpu().eval()
        self.mu = mu.cpu().eval()

    @property
    def version(self):
        return LEARNED_PREFIX + weights_hash(self.state())

    def state(self):
        return {"extractor": self.extractor.state_dict(), "mu": self.mu.state_dict()}

    @classmethod
    def fresh(cls, seed=0):
        torch.manual_seed(seed)
        space = spaces.Dict({"eyes": eyes_space()})
        return cls(EyesExtractor(space), nn.Linear(2 * FEATURES, len(V5_CHANNELS)))

    @classmethod
    def from_actor(cls, actor):
        return cls(copy.deepcopy(actor.features_extractor), copy.deepcopy(actor.mu))

    @classmethod
    def load(cls, path):
        # One brain + renderer per worker process: keep torch to one thread each.
        torch.set_num_threads(1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        enc = cls.fresh()
        enc.extractor.load_state_dict(state["extractor"])
        enc.mu.load_state_dict(state["mu"])
        return enc

    def currents(self, stack):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(stack)[None], dtype=torch.float32) / 255.0
            out = torch.tanh(self.mu(self.extractor({"eyes": x}))) + 1.0
        return out[0].numpy().astype(np.float32)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), path)
        return self.version


def _clone_job(job):
    """Fly the teacher on v4; pair each frame's stack with the v4 cues it produced."""
    from .brain import BrainRuntime
    from .distill import NOISE_AXES, _roam_env
    from .teacher import teacher_action

    seeds, seconds, level, noise = job
    env = _roam_env(level, BrainRuntime())
    stack = FrameStack()
    stacks, cues, flights = [], [], []
    try:
        for seed in seeds:
            rng = np.random.default_rng(int(seed) + 7919)
            env.reset(seed=int(seed))
            stack.clear()
            for _ in range(int(seconds / 0.04)):
                action = teacher_action(env)[0].copy()
                action[list(NOISE_AXES)] += rng.normal(0, noise, len(NOISE_AXES))
                *_, info = env.step(np.clip(action, -1, 1))
                # plant.images holds the frame the v4 encoder read during this step.
                stack.push(env.plant.images)
                stacks.append(stack.array())
                cues.append(env.brain.cues.copy())
                flights.append(int(seed))
                if any(e["type"] == "collision" for e in info["events"]):
                    stack.clear()  # respawn cleared the v4 encoder's history too
    finally:
        env.close()
    return stacks, cues, flights


def collect_clone(
    path, flights=32, seconds=60, workers=6, seed_base=600, level=3, noise=0.2
):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    seeds = np.arange(seed_base, seed_base + flights)
    jobs = [
        (c.tolist(), seconds, level, noise)
        for c in np.array_split(seeds, workers)
        if len(c)
    ]
    stacks, cues, flight_ids = [], [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for s, c, f in pool.map(_clone_job, jobs):
            stacks += s
            cues += c
            flight_ids += f
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        stacks=np.asarray(stacks, dtype=np.uint8),
        cues=np.asarray(cues, dtype=np.float32),
        flight=np.asarray(flight_ids, dtype=np.int32),
        encoder_version=ENCODER_VERSION,
    )
    return {"frames": len(stacks), "flights": int(flights)}


def _channels(prefix):
    return [i for i, name in enumerate(V5_CHANNELS) if name.startswith(prefix)]


def clone_pools(train_ids, targets):
    """Loom-active frames (any loom target > 0.05) and light-side frames (|mi1_l - mi1_r| > 0.05).

    mi1 and tm3 targets are identical by construction, so the mi1 pair stands for light.
    """
    loom_channels = _channels("lc4") + _channels("lplc2")
    loom = train_ids[targets[train_ids][:, loom_channels].max(1) > ACTIVE_THRESHOLD]
    _, left, right = next(p for p in PAIRS if p[0] == "mi1")
    light_diff = targets[train_ids, left] - targets[train_ids, right]
    side = train_ids[np.abs(light_diff) > ACTIVE_THRESHOLD]
    return loom, side


def clone_batch_ids(rng, train_ids, pools, batch):
    """A third from each pool, the rest uniform; an empty pool's share is drawn uniformly."""
    third = batch // 3
    ids = [rng.choice(pool, third) for pool in pools if len(pool)]
    ids.append(rng.choice(train_ids, batch - third * len(ids)))
    return np.concatenate(ids)


def clone_loss(pred, target):
    """Per-channel MSE plus DIFF_WEIGHT x MSE of each pathway's left-right difference."""
    left = [p[1] for p in PAIRS]
    right = [p[2] for p in PAIRS]
    diff = (pred[:, left] - pred[:, right]) - (target[:, left] - target[:, right])
    return (pred - target).square().mean() + DIFF_WEIGHT * diff.square().mean()


def _r(p, t):
    return float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 and t.std() > 1e-9 else None


def fit_clone(paths, output, steps=20000, batch=256, holdout=0.1, device=None, seed=0):
    """Supervised copy of v4 onto the learned encoder, so SAC starts from a working system."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    stacks, cues, flights = [], [], []
    for i, p in enumerate(paths):
        d = np.load(p)
        if str(d["encoder_version"]) != ENCODER_VERSION:
            raise ValueError(f"{p}: not collected on encoder v4")
        stacks.append(d["stacks"])
        cues.append(d["cues"])
        flights.append(d["flight"].astype(np.int64) + i * 1_000_000)
    stacks, flights = np.concatenate(stacks), np.concatenate(flights)
    targets = v4_targets(np.concatenate(cues)).astype(np.float32)
    rng = np.random.default_rng(72)
    unique = np.unique(flights)
    held = rng.choice(unique, max(1, int(len(unique) * holdout)), replace=False)
    test = np.isin(flights, held)
    train_ids, test_ids = np.flatnonzero(~test), np.flatnonzero(test)
    # Loom and light-side frames are rare: each fills a third of every batch.
    pools = clone_pools(train_ids, targets)

    enc = LearnedEncoder.fresh(seed)
    net = nn.ModuleDict({"extractor": enc.extractor, "mu": enc.mu}).to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    def predict(ids):
        x = torch.as_tensor(stacks[ids], dtype=torch.float32, device=device) / 255.0
        return torch.tanh(net["mu"](net["extractor"]({"eyes": x}))) + 1.0

    for _ in range(steps):
        ids = clone_batch_ids(rng, train_ids, pools, batch)
        loss = clone_loss(predict(ids), torch.as_tensor(targets[ids], device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        pred = np.concatenate(
            [
                predict(test_ids[i : i + 1024]).cpu().numpy()
                for i in range(0, len(test_ids), 1024)
            ]
        )
    truth = targets[test_ids]
    report = {
        "frames": int(len(stacks)),
        "held_out_flights": int(len(held)),
        "held_out": {},
    }
    for k, name in enumerate(V5_CHANNELS):
        p, t = pred[:, k], truth[:, k]
        report["held_out"][name] = {"mse": float(np.mean((p - t) ** 2)), "r": _r(p, t)}
    report["held_out_differences"] = {}
    for pathway, left, right in PAIRS:
        dp, dt = pred[:, left] - pred[:, right], truth[:, left] - truth[:, right]
        report["held_out_differences"][pathway] = {
            "r": _r(dp, dt),
            "rmse": float(np.sqrt(np.mean((dp - dt) ** 2))),
            "gain": float(dp.std() / dt.std())
            if dp.std() > 1e-9 and dt.std() > 1e-9
            else None,
        }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    report["version"] = LearnedEncoder(net["extractor"], net["mu"]).save(
        out / "encoder.pt"
    )
    report["paths"] = [str(p) for p in paths]
    (out / "clone.json").write_text(json.dumps(report, indent=2))
    return report
