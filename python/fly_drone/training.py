import json
import time
from pathlib import Path

import numpy as np

from .brain import ENCODER_VERSION
from .plant import LIMITS


def export_actor(model, brain, path):
    import torch

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
        "encoder_version": ENCODER_VERSION,
        "dataset_hash": brain.dataset_hash,
        "feature_ids": brain.feature_ids,
        "mean": model.policy.features_extractor.mean.cpu().tolist()
        if hasattr(model.policy.features_extractor, "mean")
        else [0.0] * len(brain.feature_ids),
        "scale": model.policy.features_extractor.scale.cpu().tolist()
        if hasattr(model.policy.features_extractor, "scale")
        else [1.0] * len(brain.feature_ids),
        "layers": layers,
        "action_limits": LIMITS.tolist(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    brain.load_policy(path)
    rng = np.random.default_rng(123)
    error = 0.0
    for x in rng.uniform(0, 1, (32, len(brain.feature_ids))).astype(np.float32):
        expected = model.predict(x, deterministic=True)[0] * LIMITS
        error = max(error, float(np.max(np.abs(expected - brain.infer(x)))))
    if error > 1e-4:
        raise RuntimeError(f"Rust policy parity failed: {error}")
    return error


def _env_factory(task, rank, seed):
    def make():
        from stable_baselines3.common.monitor import Monitor

        from .env import ConnectomeEnv

        env = Monitor(ConnectomeEnv(task=task))
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
            export_actor(model, brain, out / "warm-actor.json")
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
        error = export_actor(model, brain, out / "actor.json")
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
FRAME_HZ = 25
SETTLE_SECONDS = 2.0


def _rollout_chunk(job):
    """Roll out one policy over seeds in a single process (own brain + renderer)."""
    from .brain import BrainRuntime
    from .env import HORIZON_FRAMES, ConnectomeEnv

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
            if task == "looming":
                side = info["obstacle_side"]
            else:
                side = 1 if info["bearing"] > 0 else -1
            initial = abs(info["bearing"])
            zs = []
            terminated = False
            collision = False
            min_distance = info["obstacle_distance"]
            pre_launch_displacement = 0.0
            for _ in range(frames):
                obs, _, done, truncated, info = env.step(brain.infer(obs) / LIMITS)
                zs.append(float(env.plant.pos[0, 2]))
                min_distance = min(min_distance, info["obstacle_distance"])
                if not info["launched"]:
                    pre_launch_displacement = max(
                        pre_launch_displacement, info["displacement"]
                    )
                if done:
                    terminated = True
                    collision = info["collision"]
                    break
                if truncated:
                    break
            sim_seconds += env.plant.data.time
            zs = np.asarray(zs)
            settled = zs[int(SETTLE_SECONDS * FRAME_HZ) :]
            final = abs(info["bearing"])
            if task == "looming":
                success = not terminated
            else:
                success = not terminated and final < initial * 0.5
            runs.append(
                {
                    "seed": int(seed),
                    "side": "left" if side > 0 else "right",
                    "target_side": "left" if side > 0 else "right",
                    "success": bool(success),
                    "terminated": terminated,
                    "collision": collision,
                    "initial_bearing": initial,
                    "final_bearing": final,
                    "min_obstacle_distance": float(min_distance),
                    "pre_launch_displacement": float(pre_launch_displacement),
                    "final_displacement": float(info["displacement"]),
                    "altitude_rms": float(np.sqrt(np.mean((zs - 1) ** 2))),
                    "settled_altitude_rms": float(np.sqrt(np.mean((settled - 1) ** 2)))
                    if len(settled)
                    else None,
                    "frames": len(zs),
                }
            )
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
        "mean_final_bearing": float(np.mean([r["final_bearing"] for r in runs])),
        "real_time_factor": sim_seconds / wall_seconds if wall_seconds else None,
        "runs": sorted(runs, key=lambda r: r["seed"]),
    }


def evaluate(
    policy,
    episodes=50,
    output="runs/evaluation.json",
    seconds=10,
    workers=8,
    hover_episodes=5,
    hover_seconds=30,
    task="visual",
):
    """Held-out trials, ablation controls and (visual) policy hover, in parallel."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    if task not in ("visual", "looming"):
        raise ValueError("evaluate task must be visual or looming")
    policy = str(Path(policy).resolve())
    workers = max(1, int(workers))
    seeds = np.arange(1000, 1000 + episodes)
    jobs = [
        (policy, mode, chunk.tolist(), seconds, task)
        for mode in ("none", *ABLATIONS)
        for chunk in np.array_split(seeds, min(workers, episodes))
    ]
    if task == "looming":
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
    beats_ablations = all(
        none["success_rate"] > modes[m]["success_rate"]
        and none["balanced_success"] > modes[m]["balanced_success"]
        for m in ABLATIONS
    )
    ablations = {
        m: {
            "success_rate": modes[m]["success_rate"],
            "balanced_success": modes[m]["balanced_success"],
        }
        for m in ABLATIONS
    }
    if task == "looming":
        report["acceptance"] = {
            "avoidance_rate": none["success_rate"],
            "avoidance_balanced": none["balanced_success"],
            "avoidance_passed": none["success_rate"] >= 0.8
            and none["balanced_success"] >= 0.8,
            "ablations": ablations,
            "ablation_passed": beats_ablations,
            "note": "Success = no contact or crash while an obstacle flies at the "
            "drone; balanced by obstacle side. Pre-launch displacement exposes "
            "blind dodging.",
        }
    else:
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
        report["acceptance"] = {
            "steering_success_rate": none["success_rate"],
            "steering_balanced_success": none["balanced_success"],
            "steering_passed": none["success_rate"] >= 0.8
            and none["balanced_success"] >= 0.8,
            "ablations": ablations,
            "ablation_passed": beats_ablations,
            "hover_passed": bool(
                hover_runs
                and all(not r["terminated"] for r in hover_runs)
                and all(
                    r["frames"] >= int(hover_seconds * FRAME_HZ) for r in hover_runs
                )
                and report["hover"]["max_settled_altitude_rms"] is not None
                and report["hover"]["max_settled_altitude_rms"] < 0.15
            ),
            "note": "Hover uses the trained policy with a lateral target for 30 s; "
            "settled RMS excludes the first 2 s.",
        }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report
