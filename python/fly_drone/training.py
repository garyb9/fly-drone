import json
import time
from pathlib import Path

import numpy as np

from .plant import LIMITS


def export_actor(model, brain, path, limits=None):
    import torch

    limits = LIMITS if limits is None else np.asarray(limits, dtype=float)
    layers = []
    for layer in list(model.policy.mlp_extractor.policy_net) + [
        model.policy.action_net
    ]:
        if isinstance(layer, torch.nn.Linear):
            layers.append(
                {
                    "weights": layer.weight.detach().cpu().tolist(),
                    "bias": layer.bias.detach().cpu().tolist(),
                }
            )
        elif not isinstance(layer, torch.nn.Tanh):
            raise ValueError("only tanh MLP export supported")
    payload = {
        "version": 1,
        # The runtime's encoder, not the v4 constant: a learned-brain clone must carry the
        # `learned-v5:`/`learned-v6:` identity or `load_policy` rejects it. A v4 brain keeps
        # `ENCODER_VERSION`, so legacy exports are unchanged.
        "encoder_version": brain.encoder_version,
        "dataset_hash": brain.dataset_hash,
        "feature_ids": brain.feature_ids,
        "mean": model.policy.features_extractor.mean.cpu().tolist()
        if hasattr(model.policy.features_extractor, "mean")
        else [0.0] * len(brain.feature_ids),
        "scale": model.policy.features_extractor.scale.cpu().tolist()
        if hasattr(model.policy.features_extractor, "scale")
        else [1.0] * len(brain.feature_ids),
        "layers": layers,
        "action_limits": limits.tolist(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    brain.load_policy(path)
    rng = np.random.default_rng(123)
    error = 0.0
    for x in rng.uniform(0, 1, (32, len(brain.feature_ids))).astype(np.float32):
        expected = model.predict(x, deterministic=True)[0] * limits
        error = max(error, float(np.max(np.abs(expected - brain.infer(x)))))
    if error > 1e-4:
        raise RuntimeError(f"Rust policy parity failed: {error}")
    return error


def _env_kwargs(task):
    """Free roam trains in the full L3 arena and respawns after a crash, as evaluated."""
    if task == "free_roam":
        return {"task": task, "level": 3, "respawn": True}
    return {"task": task}


def action_limits(task):
    """Physical scale of a normalised action; free roam flies faster than the room."""
    if task == "free_roam":
        from .arena import ArenaSpec

        return np.asarray(ArenaSpec().limits, dtype=float)
    return LIMITS


def _env_factory(task, rank, seed):
    def make():
        from stable_baselines3.common.monitor import Monitor

        from .env import ConnectomeEnv

        env = Monitor(ConnectomeEnv(**_env_kwargs(task)))
        env.reset(seed=seed + rank)
        return env

    return make


def train(
    steps=20000,
    output="runs/visual",
    task="visual",
    resume=None,
    calibration=None,
    envs=4,
    teacher_scale=0.4,
    seed=42,
    learning_rate=3e-4,
    log_std=None,
):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .assay import sensory_assay
    from .brain import BrainRuntime
    from .normalizer import NeuralNormalizer

    torch.set_num_threads(1)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    assay = sensory_assay(out / "sensory-assay.json")
    if not assay["passed"]:
        raise RuntimeError(
            "Sensory causal gate failed; training was not started. "
            "Inspect sensory-assay.json."
        )
    envs = max(1, int(envs))
    factories = [_env_factory(task, rank, seed) for rank in range(envs)]
    # Each env owns a full brain and renderer; spawn keeps EGL state per process.
    vec = (
        SubprocVecEnv(factories, start_method="spawn")
        if envs > 1
        else DummyVecEnv(factories)
    )
    # Export parity runs against a separate Rust runtime in this process.
    brain = BrainRuntime()
    try:
        if resume:
            model = PPO.load(resume, env=vec, device="cpu")
            # A warm-started checkpoint carries lr 1e-5 and log_std -2.5; keeping them
            # froze a resumed looming run (log_std unchanged after 30k steps).
            model.learning_rate = learning_rate
            model.lr_schedule = lambda _: learning_rate
            if log_std is not None:
                with torch.no_grad():
                    model.policy.log_std.fill_(log_std)
        else:
            model = PPO(
                "MlpPolicy",
                vec,
                seed=seed,
                device="cpu",
                n_steps=max(64, 1024 // envs),
                batch_size=64,
                n_epochs=5,
                learning_rate=3e-4,
                policy_kwargs={
                    "net_arch": {"pi": [32, 32], "vf": [32, 32]},
                    "activation_fn": torch.nn.Tanh,
                    "features_extractor_class": NeuralNormalizer,
                },
                verbose=1,
            )
        if calibration and not resume:
            from .calibration import warm_start

            result = warm_start(
                model, calibration, brain.dataset_hash, yaw_scale=teacher_scale
            )
            (out / "warm-start.json").write_text(json.dumps(result, indent=2))
            export_actor(
                model, brain, out / "warm-actor.json", limits=action_limits(task)
            )
            model.save(out / "warm-ppo")
            model.learning_rate = 1e-5
            model.lr_schedule = lambda _: 1e-5
        model.learn(
            total_timesteps=steps,
            callback=CheckpointCallback(
                save_freq=max(1, 5000 // envs), save_path=str(out / "checkpoints")
            ),
            reset_num_timesteps=not bool(resume),
        )
        model.save(out / "ppo")
        error = export_actor(
            model, brain, out / "actor.json", limits=action_limits(task)
        )
        (out / "training.json").write_text(
            json.dumps(
                {
                    "steps_requested": steps,
                    "steps_total": model.num_timesteps,
                    "task": task,
                    "seed": seed,
                    "envs": envs,
                    "teacher_scale": teacher_scale if calibration else None,
                    "resume": resume,
                    "learning_rate": learning_rate if resume else None,
                    "log_std": log_std,
                    "export_max_error": error,
                },
                indent=2,
            )
        )
    finally:
        vec.close()


def export_checkpoint(checkpoint, output):
    """Export a saved PPO zip to a Rust actor JSON (with parity check)."""
    from stable_baselines3 import PPO

    from .brain import BrainRuntime

    model = PPO.load(checkpoint, device="cpu")
    return export_actor(model, BrainRuntime(), output)


ABLATIONS = ("zero", "sensory", "shuffle")
TASK_NOTES = {
    "visual": "Halve the initial target bearing within 10 s; balanced by target side. "
    "Hover: trained policy, 30 s, settled RMS after 2 s.",
    "looming": "Survive an obstacle flown at the drone and be within 0.25 m of the "
    "start when it first comes within 2 m; balanced by obstacle side.",
    "approach": "End within 0.6 m (horizontal) of a target 2.2-3.0 m away after 15 s; "
    "balanced by initial target side.",
    "track": "Mean |bearing| < 0.3 rad from 5 s to 30 s while the target orbits and "
    "reverses; balanced by initial orbit direction.",
    "steer_dodge": "Halve the target bearing AND dodge a mid-episode obstacle "
    "threat-specifically; balanced by obstacle side.",
    "escape": "Survive an obstacle aimed just below the drone, threat-specifically, "
    "with a climb of at least 0.1 m; balanced by obstacle side.",
}
FRAME_HZ = 25


def _rollout_chunk(job):
    """Roll out one policy over seeds in a single process (own brain + renderer)."""
    from .brain import BrainRuntime
    from .env import HORIZON_FRAMES, ConnectomeEnv, EpisodeTracker

    policy, mode, seeds, seconds, task = job
    brain = BrainRuntime()
    brain.load_policy(policy)
    env = ConnectomeEnv(
        task=task, brain=brain, ablation="none" if mode == "hover" else mode
    )
    frames = min(int(seconds * FRAME_HZ), HORIZON_FRAMES[task])
    runs = []
    sim_seconds = 0.0
    start = time.perf_counter()
    try:
        for seed in seeds:
            obs, info = env.reset(seed=int(seed))
            tracker = EpisodeTracker(task, info)
            terminated = False
            for _ in range(frames):
                obs, _, done, truncated, info = env.step(
                    brain.infer(obs) / env.plant.limits
                )
                tracker.update(info)
                if done:
                    terminated = True
                    break
                if truncated:
                    break
            sim_seconds += env.plant.data.time
            runs.append({"seed": int(seed), **tracker.finish(terminated)})
    finally:
        env.close()
    return mode, runs, sim_seconds, time.perf_counter() - start


def _summary(runs, sim_seconds, wall_seconds):
    by_side = {
        s: [r["success"] for r in runs if r.get("side") == s] for s in ("left", "right")
    }
    side_rates = {s: float(np.mean(v)) if v else None for s, v in by_side.items()}
    # A blind policy that always turns one way scores ~50% success but ~0% balanced.
    balanced = min((v for v in side_rates.values() if v is not None), default=0.0)
    return {
        "success_rate": float(np.mean([r["success"] for r in runs])),
        "success_by_side": side_rates,
        "balanced_success": balanced,
        "collision_rate": float(np.mean([r["collision"] for r in runs])),
        "termination_rate": float(np.mean([r["terminated"] for r in runs])),
        "mean_pre_launch_displacement": float(
            np.mean([r["pre_launch_displacement"] for r in runs])
        ),
        "mean_displacement_at_threat": float(
            np.mean([r["displacement_at_threat"] for r in runs])
        ),
        "survival_rate": float(np.mean([r["survived"] for r in runs])),
        "mean_final_bearing": float(np.mean([r["final_bearing"] for r in runs])),
        "real_time_factor": sim_seconds / wall_seconds if wall_seconds else None,
        "runs": sorted(runs, key=lambda r: r["seed"]),
    }


def evaluate(
    policy,
    episodes=50,
    output="runs/evaluation.json",
    seconds=None,
    workers=8,
    hover_episodes=5,
    hover_seconds=30,
    task="visual",
):
    """Held-out trials, ablation controls and (visual) policy hover, in parallel."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from .env import EVAL_SECONDS, SETTLE_SECONDS

    if task not in EVAL_SECONDS:
        raise ValueError(f"evaluate task must be one of {sorted(EVAL_SECONDS)}")
    seconds = EVAL_SECONDS[task] if seconds is None else seconds
    policy = str(Path(policy).resolve())
    workers = max(1, int(workers))
    seeds = np.arange(1000, 1000 + episodes)
    jobs = [
        (policy, mode, chunk.tolist(), seconds, task)
        for mode in ("none", *ABLATIONS)
        for chunk in np.array_split(seeds, min(workers, episodes))
    ]
    if task != "visual":
        hover_episodes = 0
    hover_seeds = np.arange(2000, 2000 + hover_episodes)
    jobs += [
        (policy, "hover", chunk.tolist(), hover_seconds, "visual")
        for chunk in np.array_split(hover_seeds, min(workers, max(1, hover_episodes)))
        if len(chunk)
    ]
    collected = {}
    start = time.perf_counter()
    # Spawn: MuJoCo/EGL renderer state must not be inherited through fork.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for mode, runs, sim, wall in pool.map(_rollout_chunk, jobs):
            entry = collected.setdefault(mode, [[], 0.0, 0.0])
            entry[0] += runs
            entry[1] += sim
            entry[2] += wall
    modes = {m: _summary(*collected[m]) for m in ("none", *ABLATIONS)}
    report = {
        "policy": policy,
        "task": task,
        "episodes": episodes,
        "seconds": seconds,
        "workers": workers,
        "wall_seconds": time.perf_counter() - start,
        "modes": modes,
    }
    none = modes["none"]
    ablations = {
        m: {
            "success_rate": modes[m]["success_rate"],
            "balanced_success": modes[m]["balanced_success"],
        }
        for m in ABLATIONS
    }
    acceptance = {
        "success_rate": none["success_rate"],
        "balanced_success": none["balanced_success"],
        "passed": none["success_rate"] >= 0.8 and none["balanced_success"] >= 0.8,
        "ablations": ablations,
        "ablation_passed": all(
            none["success_rate"] > modes[m]["success_rate"]
            and none["balanced_success"] > modes[m]["balanced_success"]
            for m in ABLATIONS
        ),
        "note": TASK_NOTES[task],
    }
    if task == "visual":
        hover_runs = collected.get("hover", [[], 0.0, 0.0])[0]
        hover_rms = [r["settled_altitude_rms"] for r in hover_runs]
        report["hover"] = {
            "seconds": hover_seconds,
            "settle_seconds": SETTLE_SECONDS,
            "max_settled_altitude_rms": max(
                (v for v in hover_rms if v is not None), default=None
            ),
            "runs": sorted(hover_runs, key=lambda r: r["seed"]),
        }
        acceptance["hover_passed"] = bool(
            hover_runs
            and all(not r["terminated"] for r in hover_runs)
            and all(r["frames"] >= int(hover_seconds * FRAME_HZ) for r in hover_runs)
            and report["hover"]["max_settled_altitude_rms"] is not None
            and report["hover"]["max_settled_altitude_rms"] < 0.15
        )
    report["acceptance"] = acceptance
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report
