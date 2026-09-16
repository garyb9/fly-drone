"""Distil the composite teacher into ONE decoder by closed-loop collection (DAgger).

The decoder input is only the 2,022 descending/motor traces. Simulator state makes
labels (teacher.py, visibility-gated) and never reaches the decoder.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from .brain import ENCODER_VERSION
from .teacher import DRIVES

NOISE_AXES = (0, 1, 3)


def _roam_env(level, brain=None):
    from .env import ConnectomeEnv

    return ConnectomeEnv(task="free_roam", level=level, respawn=True, brain=brain)


def _collect_job(job):
    from .brain import BrainRuntime
    from .teacher import teacher_action

    seeds, seconds, level, student, beta, noise, stride, *rest = job
    brain = BrainRuntime(encoder=rest[0] if rest else None)
    if student:
        brain.load_policy(student)
    env = _roam_env(level, brain)
    xs, ys, drives, flights = [], [], [], []
    try:
        for seed in seeds:
            rng = np.random.default_rng(int(seed) + 7919)
            obs, _ = env.reset(seed=int(seed))
            for frame in range(int(seconds / 0.04)):
                label, drive = teacher_action(env)
                if frame % stride == 0:
                    xs.append(obs.astype(np.float16))
                    ys.append(label.astype(np.float32))
                    drives.append(DRIVES.index(drive))
                    flights.append(int(seed))
                if student and rng.random() >= beta:
                    action = brain.infer(obs) / env.plant.limits
                else:
                    action = label.copy()
                    action[list(NOISE_AXES)] += rng.normal(0, noise, len(NOISE_AXES))
                obs, *_ = env.step(np.clip(action, -1, 1))
    finally:
        env.close()
    return xs, ys, drives, flights


def collect(
    path,
    flights=128,
    seconds=60,
    student=None,
    beta=1.0,
    noise=0.2,
    stride=2,
    workers=6,
    seed_base=200,
    levels=None,
    encoder=None,
):
    """Teacher-labelled features; beta < 1 flies the student that often (DAgger).

    `encoder` is a learned encoder `.pt` driving the brain (None: v4). A student must be
    pinned to the same encoder.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from .arena import LEVELS

    if encoder is None:
        encoder_version = ENCODER_VERSION
    elif str(encoder) == "external":
        raise ValueError(
            'encoder="external" cannot collect: pass a saved learned encoder .pt path'
        )
    else:
        from .brain import _is_spatial

        encoder = str(Path(encoder).resolve())
        if _is_spatial(encoder):
            from .spatial_encoder import SpatialEncoder

            encoder_version = SpatialEncoder.load(encoder).version
        else:
            from .encoder import LearnedEncoder

            encoder_version = LearnedEncoder.load(encoder).version
    seeds = np.arange(seed_base, seed_base + flights)
    levels = sorted(levels if levels is not None else LEVELS)
    if not levels or any(level not in LEVELS for level in levels):
        raise ValueError(f"levels must be drawn from {sorted(LEVELS)}")
    jobs = []
    for i, level in enumerate(levels):
        level_seeds = seeds[i :: len(levels)]
        for chunk in np.array_split(level_seeds, max(1, workers // len(levels))):
            if len(chunk):
                jobs.append(
                    (
                        chunk.tolist(),
                        seconds,
                        level,
                        str(Path(student).resolve()) if student else None,
                        beta,
                        noise,
                        stride,
                        encoder,
                    )
                )
    xs, ys, drives, flight_ids = [], [], [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for x, y, d, f in pool.map(_collect_job, jobs):
            xs += x
            ys += y
            drives += d
            flight_ids += f
    from .brain import BrainRuntime

    dataset_hash = BrainRuntime().dataset_hash
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=np.asarray(xs, dtype=np.float16),
        y=np.asarray(ys, dtype=np.float32),
        drive=np.asarray(drives, dtype=np.int8),
        flight=np.asarray(flight_ids, dtype=np.int32),
        dataset_hash=dataset_hash,
        encoder_version=encoder_version,
        student=str(student),
        beta=beta,
    )
    counts = np.bincount(np.asarray(drives, dtype=int), minlength=len(DRIVES))
    return {
        "samples": len(xs),
        "drives": dict(zip(DRIVES, counts.tolist(), strict=True)),
    }


def _load(paths, dataset_hash, encoder_version=ENCODER_VERSION):
    xs, ys, drives, flights = [], [], [], []
    for i, path in enumerate(paths):
        d = np.load(path)
        if str(d["dataset_hash"]) != dataset_hash:
            raise ValueError(f"{path}: collected on a different connectome")
        if str(d["encoder_version"]) != encoder_version:
            raise ValueError(
                f"{path}: collected with encoder {d['encoder_version']}, "
                f"expected {encoder_version}"
            )
        xs.append(d["x"].astype(np.float32))
        ys.append(d["y"])
        drives.append(d["drive"].astype(int))
        # Flights are unique per file; offset so held-out splits stay per flight.
        flights.append(d["flight"].astype(np.int64) + i * 1_000_000)
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(drives),
        np.concatenate(flights),
    )


class _SpacesOnly:
    """Just the spaces PPO needs to build a policy; no brain or renderer."""

    def __init__(self, n):
        import gymnasium as gym
        from gymnasium import spaces

        class Env(gym.Env):
            observation_space = spaces.Box(0, 1, (n,), np.float32)
            action_space = spaces.Box(-1, 1, (4,), np.float32)

            def reset(self, seed=None, options=None):
                return np.zeros(n, np.float32), {}

            def step(self, action):
                return np.zeros(n, np.float32), 0.0, True, False, {}

        self.env = Env()


def build_model(n_features, net_arch=(64, 64), env=None):
    import torch
    from stable_baselines3 import PPO

    from .normalizer import NeuralNormalizer

    return PPO(
        "MlpPolicy",
        env if env is not None else _SpacesOnly(n_features).env,
        seed=42,
        device="cpu",
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        learning_rate=1e-4,
        policy_kwargs={
            "net_arch": {"pi": list(net_arch), "vf": list(net_arch)},
            "activation_fn": torch.nn.Tanh,
            "features_extractor_class": NeuralNormalizer,
        },
        verbose=0,
    )


def class_weights(drive, mask):
    """Threat frames are rare: weight every drive present in `mask` equally (mean 1)."""
    counts = np.bincount(drive[mask], minlength=len(DRIVES)).astype(float)
    weight = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    return weight / weight[counts > 0].mean()


def fit(paths, output, net_arch=(64, 64), steps=4000, holdout=0.1):
    """Class-balanced behaviour cloning of the teacher onto the decoder head."""
    import torch

    from .arena import ArenaSpec
    from .brain import BrainRuntime
    from .training import export_actor

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    brain = BrainRuntime()
    x, y, drive, flight = _load(paths, brain.dataset_hash)
    rng = np.random.default_rng(72)
    unique = np.unique(flight)
    held = set(rng.choice(unique, max(1, int(len(unique) * holdout)), replace=False))
    test = np.array([f in held for f in flight])
    class_weight = class_weights(drive, ~test)

    model = build_model(x.shape[1], net_arch)
    norm = model.policy.features_extractor
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    wt = torch.tensor(class_weight[drive], dtype=torch.float32)
    train_ids = torch.tensor(np.flatnonzero(~test))
    norm.mean.copy_(xt[train_ids].mean(0))
    norm.scale.copy_(xt[train_ids].std(0).clamp(min=0.003))
    params = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    opt = torch.optim.Adam(params, lr=1e-3)
    torch.manual_seed(72)

    def predict(ids):
        return model.policy.action_net(
            model.policy.mlp_extractor.forward_actor(norm(xt[ids]))
        )

    for _ in range(steps):
        ids = train_ids[torch.randint(len(train_ids), (256,))]
        loss = (wt[ids, None] * (predict(ids) - yt[ids]).square()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    report = {"samples": int(len(x)), "held_out_flights": len(held), "drives": {}}
    with torch.no_grad():
        model.policy.log_std.fill_(-2.5)
        for split, mask in (("train", ~test), ("held_out", test)):
            ids = torch.tensor(np.flatnonzero(mask))
            if not len(ids):
                continue
            err = (predict(ids) - yt[ids]).square().mean(1).numpy()
            for k, name in enumerate(DRIVES):
                sel = drive[mask] == k
                entry = report["drives"].setdefault(name, {})
                entry[f"{split}_samples"] = int(sel.sum())
                entry[f"{split}_mse"] = float(err[sel].mean()) if sel.any() else None
                if sel.any():
                    target = y[mask][sel]
                    var = float(target.var(0).sum())
                    entry[f"{split}_r2"] = (
                        1.0 - float(err[sel].mean()) * 4 / var if var > 1e-9 else None
                    )
    model.save(out / "warm-ppo")
    report["export_max_error"] = export_actor(
        model, brain, out / "warm-actor.json", limits=ArenaSpec().limits
    )
    report["net_arch"] = list(net_arch)
    report["paths"] = [str(p) for p in paths]
    (out / "warm-start.json").write_text(json.dumps(report, indent=2))
    return report


def _screen_job(job):
    from .brain import BrainRuntime
    from .feasibility import CueController, RandomController
    from .teacher import teacher_action

    controller, seeds, seconds, level, ablation, *rest = job
    encoder = rest[0] if rest else None
    brain = BrainRuntime(encoder=encoder)
    bypass = None
    if controller.startswith("policy:"):
        brain.load_policy(controller.split(":", 1)[1])
    elif controller.startswith("bypass:"):
        from stable_baselines3 import SAC

        from .encoder import LearnedEncoder

        if not isinstance(brain.encoder, LearnedEncoder):
            raise ValueError(
                "bypass: controller requires a LearnedEncoder "
                f"(got encoder={encoder!r}); pass a saved encoder path"
            )
        bypass = SAC.load(controller.split(":", 1)[1], device="cpu")
    env = _roam_env(level, brain)
    env.ablation = ablation
    runs = []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=int(seed))
            script = (
                CueController()
                if controller == "cue_script"
                else RandomController(seed)
                if controller == "random"
                else None
            )
            frames = int(seconds / 0.04)
            yaw_commands = []
            stuck = 0
            start = time.perf_counter()
            for _ in range(frames):
                if controller == "teacher":
                    action = teacher_action(env)[0]
                elif script is not None:
                    action = script.act(env.brain.cues)
                elif bypass is not None:
                    # E3 control: a decoder that reads the encoder's currents, not the brain.
                    bypass_obs = {
                        "currents": brain.encoder.currents(brain.stack.array()),
                        "geometry": np.zeros(8, np.float32),
                    }
                    action = bypass.predict(bypass_obs, deterministic=True)[0]
                else:
                    action = brain.infer(obs) / env.plant.limits
                yaw_commands.append(float(action[3]))
                obs, _, _, _, info = env.step(action)
                stuck += info["speed"] < 0.05
            threats = info["threats"]
            minutes = seconds / 60
            runs.append(
                {
                    "seed": int(seed),
                    "beacons_per_min": info["beacons_collected"] / minutes,
                    "collisions_per_min": info["collisions"] / minutes,
                    "collision_kinds": info["collision_kinds"],
                    "threats": len(threats),
                    "threats_dodged": int(sum(t["dodged"] for t in threats)),
                    "threats_hit": int(sum(t["hit"] for t in threats)),
                    "threat_log": threats,
                    "visited_cells": info["visited_cells"],
                    "mean_yaw_command": float(np.mean(yaw_commands)),
                    "slow_fraction": stuck / frames,
                    "wall_seconds": time.perf_counter() - start,
                }
            )
    finally:
        env.close()
    return controller, ablation, runs


def near_dodge_rates(runs):
    """Pre-registered A3 scoring: only throws that hit or came within 2 m count.

    Mirrors roam_eval.dodge_rates so a screen reads the same number as acceptance.
    """
    near = [
        t for r in runs for t in r["threat_log"] if t["hit"] or t["min_distance"] < 2.0
    ]
    sides = {
        name: [t["dodged"] for t in near if (t["side"] > 0) == left]
        for name, left in (("left", True), ("right", False))
    }
    by_side = {k: float(np.mean(v)) if v else None for k, v in sides.items()}
    return {
        "near_threats": len(near),
        "near_dodge_rate": float(np.mean([t["dodged"] for t in near]))
        if near
        else None,
        "near_dodge_by_side": by_side,
        "balanced_dodge_rate": min(by_side.values())
        if all(v is not None for v in by_side.values())
        else None,
    }


def summarise(runs):
    threats = sum(r["threats"] for r in runs)
    return {
        **near_dodge_rates(runs),
        "beacons_per_min": float(np.mean([r["beacons_per_min"] for r in runs])),
        "collisions_per_min": float(np.mean([r["collisions_per_min"] for r in runs])),
        "threat_dodge_rate": sum(r["threats_dodged"] for r in runs) / threats
        if threats
        else None,
        "threats": threats,
        "mean_visited_cells": float(np.mean([r["visited_cells"] for r in runs])),
        "mean_abs_yaw_bias": float(np.mean([abs(r["mean_yaw_command"]) for r in runs])),
        "slow_fraction": float(np.mean([r["slow_fraction"] for r in runs])),
        "runs": sorted(runs, key=lambda r: r["seed"]),
    }


def screen(
    controllers,
    output,
    seeds=10,
    seconds=60,
    level=3,
    workers=6,
    seed_base=5000,
    ablations=("none",),
    combos=None,
    encoder=None,
):
    """Closed-loop screen; combos lists explicit (controller, ablation) pairs.

    `encoder` is only used for `policy:`/`bypass:` controllers; every baseline flies v4.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    all_seeds = np.arange(seed_base, seed_base + seeds)
    combos = combos or [(c, a) for c in controllers for a in ablations]
    per = max(1, workers // len(combos))
    learned = ("policy:", "bypass:")
    if encoder and not any(c.startswith(learned) for c, _ in combos):
        print(
            "warning: --encoder is ignored (no policy:/bypass: controller flies it; "
            "every baseline flies v4)",
            file=sys.stderr,
        )
    jobs = [
        (
            c,
            chunk.tolist(),
            seconds,
            level,
            a,
            encoder if c.startswith(learned) else None,
        )
        for c, a in combos
        for chunk in np.array_split(all_seeds, min(per, seeds))
        if len(chunk)
    ]
    collected = {}
    start = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for controller, ablation, runs in pool.map(_screen_job, jobs):
            collected.setdefault(f"{controller}|{ablation}", []).extend(runs)
    report = {
        "seeds": [int(all_seeds[0]), int(all_seeds[-1])],
        "seconds": seconds,
        "level": level,
        "wall_seconds": time.perf_counter() - start,
        "results": {k: summarise(v) for k, v in collected.items()},
        "encoder": str(encoder) if encoder else None,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report
