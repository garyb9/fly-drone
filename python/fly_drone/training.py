import json
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


def evaluate(policy, episodes=50, output="runs/evaluation.json", seconds=10):
    from .brain import BrainRuntime
    from .env import ConnectomeEnv

    brain = BrainRuntime()
    brain.load_policy(policy)
    report = {}
    for mode in ("none", "zero", "sensory", "shuffle"):
        env = ConnectomeEnv(brain=brain, ablation=mode)
        runs = []
        try:
            for seed in range(1000, 1000 + episodes):
                obs, _ = env.reset(seed=seed)
                initial = env.info()["bearing"]
                zs = []
                collision = False
                for _ in range(int(seconds * 25)):
                    obs, _, done, truncated, info = env.step(brain.infer(obs) / LIMITS)
                    zs.append(env.plant.pos[0, 2])
                    if done:
                        collision = True
                        break
                    if truncated:
                        break
                final = abs(
                    np.arctan2(np.sin(info["bearing"]), np.cos(info["bearing"]))
                )
                success = not collision and final < abs(initial) * 0.5
                runs.append(
                    {
                        "seed": seed,
                        "success": success,
                        "collision": collision,
                        "initial_bearing": initial,
                        "final_bearing": final,
                        "altitude_rms": float(
                            np.sqrt(np.mean((np.asarray(zs) - 1) ** 2))
                        ),
                    }
                )
            report[mode] = {
                "success_rate": float(np.mean([r["success"] for r in runs])),
                "runs": runs,
            }
        finally:
            env.close()
    report["acceptance"] = {
        "steering_passed": report["none"]["success_rate"] >= 0.8,
        "ablation_passed": all(
            report["none"]["success_rate"] > report[m]["success_rate"]
            for m in ("zero", "sensory", "shuffle")
        ),
        "note": "This evaluation does not replace the separate 30-second hover test.",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report
