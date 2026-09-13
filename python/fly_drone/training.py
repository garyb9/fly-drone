import json
import time
from pathlib import Path

import numpy as np

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
        "encoder_version": "bright-contrast-400-v1",
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


def train(
    steps=20000, output="runs/visual", task="visual", resume=None, calibration=None
):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from .assay import sensory_assay
    from .env import ConnectomeEnv
    from .normalizer import NeuralNormalizer

    torch.set_num_threads(1)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    assay = sensory_assay(out / "sensory-assay.json")
    if not assay["passed"]:
        raise RuntimeError(
            "Sensory causal gate failed; training was not started. Inspect sensory-assay.json."
        )
    env = ConnectomeEnv(task=task)
    try:
        if resume:
            model = PPO.load(resume, env=env, device="cpu")
        else:
            model = PPO(
                "MlpPolicy",
                env,
                seed=42,
                device="cpu",
                n_steps=256,
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

            result = warm_start(model, calibration)
            (out / "warm-start.json").write_text(json.dumps(result, indent=2))
            export_actor(model, env.brain, out / "warm-actor.json")
            model.save(out / "warm-ppo")
            model.learning_rate = 1e-5
            model.lr_schedule = lambda _: 1e-5
        model.learn(
            total_timesteps=steps,
            callback=CheckpointCallback(
                save_freq=5000, save_path=str(out / "checkpoints")
            ),
            reset_num_timesteps=not bool(resume),
        )
        model.save(out / "ppo")
        error = export_actor(model, env.brain, out / "actor.json")
        (out / "training.json").write_text(
            json.dumps(
                {
                    "steps_requested": steps,
                    "steps_total": model.num_timesteps,
                    "task": task,
                    "seed": 42,
                    "export_max_error": error,
                },
                indent=2,
            )
        )
    finally:
        env.close()


ABLATIONS = ("zero", "sensory", "shuffle")
FRAME_HZ = 25
SETTLE_SECONDS = 2.0


def _rollout_chunk(job):
    """Roll out one policy over seeds in a single process (own brain + renderer)."""
    from .brain import BrainRuntime
    from .env import ConnectomeEnv

    policy, mode, seeds, seconds = job
    brain = BrainRuntime()
    brain.load_policy(policy)
    env = ConnectomeEnv(brain=brain, ablation=mode)
    runs = []
    sim_seconds = 0.0
    start = time.perf_counter()
    try:
        for seed in seeds:
            obs, info = env.reset(seed=int(seed))
            initial = abs(info["bearing"])
            zs = []
            collision = False
            for _ in range(int(seconds * FRAME_HZ)):
                obs, _, done, truncated, info = env.step(brain.infer(obs) / LIMITS)
                zs.append(float(env.plant.pos[0, 2]))
                if done:
                    collision = True
                    break
                if truncated:
                    break
            sim_seconds += env.plant.data.time
            zs = np.asarray(zs)
            settled = zs[int(SETTLE_SECONDS * FRAME_HZ) :]
            final = abs(info["bearing"])
            runs.append(
                {
                    "seed": int(seed),
                    "success": bool(not collision and final < initial * 0.5),
                    "collision": collision,
                    "initial_bearing": initial,
                    "final_bearing": final,
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
    return {
        "success_rate": float(np.mean([r["success"] for r in runs])),
        "collision_rate": float(np.mean([r["collision"] for r in runs])),
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
):
    """Held-out steering trials, ablation controls and policy hover, in parallel."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    policy = str(Path(policy).resolve())
    workers = max(1, int(workers))
    seeds = np.arange(1000, 1000 + episodes)
    jobs = [
        (policy, mode, chunk.tolist(), seconds)
        for mode in ("none", *ABLATIONS)
        for chunk in np.array_split(seeds, min(workers, episodes))
    ]
    hover_seeds = np.arange(2000, 2000 + hover_episodes)
    jobs += [
        (policy, "hover", chunk.tolist(), hover_seconds)
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
    hover_runs = collected.get("hover", [[], 0.0, 0.0])[0]
    hover_rms = [r["settled_altitude_rms"] for r in hover_runs]
    hover = {
        "seconds": hover_seconds,
        "settle_seconds": SETTLE_SECONDS,
        "max_settled_altitude_rms": max(
            (v for v in hover_rms if v is not None), default=None
        ),
        "runs": sorted(hover_runs, key=lambda r: r["seed"]),
    }
    report = {
        "policy": policy,
        "episodes": episodes,
        "seconds": seconds,
        "workers": workers,
        "wall_seconds": time.perf_counter() - start,
        "modes": modes,
        "hover": hover,
    }
    report["acceptance"] = {
        "steering_success_rate": modes["none"]["success_rate"],
        "steering_passed": modes["none"]["success_rate"] >= 0.8,
        "ablation_success_rates": {m: modes[m]["success_rate"] for m in ABLATIONS},
        "ablation_passed": all(
            modes["none"]["success_rate"] > modes[m]["success_rate"] for m in ABLATIONS
        ),
        "hover_passed": bool(
            hover_runs
            and all(not r["collision"] for r in hover_runs)
            and all(r["frames"] >= int(hover_seconds * FRAME_HZ) for r in hover_runs)
            and hover["max_settled_altitude_rms"] is not None
            and hover["max_settled_altitude_rms"] < 0.15
        ),
        "note": "Hover uses the trained policy with a lateral target for 30 s; "
        "settled RMS excludes the first 2 s.",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report
