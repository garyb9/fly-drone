"""Optional supervised warm start of the decoder, followed by PPO.
Teacher uses simulator bearing only while collecting training labels. The actor
sees ONLY measured neural features, both here and at inference.
"""

from pathlib import Path

import numpy as np

from .brain import ENCODER_VERSION, BrainRuntime
from .plant import DronePlant


def collect(path="runs/calibration.npz", trials=64):
    b = BrainRuntime()
    p = DronePlant()
    rng = np.random.default_rng(72)
    xs = []
    ys = []
    angles = []
    try:
        for trial in range(trials):
            b.reset(trial + 200)
            angle = rng.uniform(-0.75, 0.75)
            p.set_objects(
                target=[2, 2 * np.tan(angle), 1],
                obstacle=[3, -3 if angle >= 0 else 3, 1],
            )
            images = p.camera().copy()
            b.sense(images)
            for t in range(35):
                # Sustain observed frame; adaptation matches real camera updates.
                if t:
                    b.sense(images)
                b.step(8)
                xs.append(b.features())
                ys.append([0, 0, 0, float(np.clip(angle * 1.5 / 0.8, -1, 1))])
                angles.append(angle)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            x=np.array(xs),
            y=np.array(ys, dtype=np.float32),
            angles=np.array(angles),
            dataset_hash=b.dataset_hash,
            encoder_version=ENCODER_VERSION,
        )
        print(
            f"Collected {len(xs)} neural observations from {trials} rendered target positions",
            flush=True,
        )
    finally:
        p.close()


DODGE_RANGE = 2.0


def dodge_label(info):
    """Teacher lateral command: flee the obstacle's side once it is launched and near.

    2.0 m matches where gain-150 loom input starts driving LC4/LPLC2, so the decoder
    is not asked to react before the threat is visible.
    """
    if info["launched"] and info["obstacle_distance"] < DODGE_RANGE:
        return -float(info["obstacle_side"])
    return 0.0


def collect_closed_loop(
    path="runs/calibration.npz",
    trials=64,
    frames=100,
    noise=0.3,
    teacher_scale=0.4,
    task="visual",
):
    """Record features while a noisy proportional yaw teacher flies the drone.

    Static frames never contain rotation-induced loom input; features seen in
    closed-loop flight do, so a decoder fitted only on static frames fails there.
    Labels still come from simulator bearing, used offline only.
    """
    from .env import ConnectomeEnv

    env = ConnectomeEnv(task=task)
    rng = np.random.default_rng(72)
    xs = []
    ys = []
    angles = []
    try:
        for trial in range(trials):
            obs, info = env.reset(seed=trial + 200)
            for _ in range(frames):
                xs.append(obs)
                angles.append(info["bearing"])
                if task == "looming":
                    label = dodge_label(info)
                    ys.append([0, label, 0, 0])
                    vy = np.clip(label + rng.normal(0, noise), -1, 1)
                    action = np.array([0, vy, 0, 0])
                else:
                    label = float(np.clip(info["bearing"] * 1.5 / 0.8, -1, 1))
                    ys.append([0, 0, 0, label])
                    yaw = np.clip(teacher_scale * label + rng.normal(0, noise), -1, 1)
                    action = np.array([0, 0, 0, yaw])
                obs, _, done, _, info = env.step(action)
                if done:
                    break
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            x=np.array(xs),
            y=np.array(ys, dtype=np.float32),
            angles=np.array(angles),
            dataset_hash=env.brain.dataset_hash,
            encoder_version=ENCODER_VERSION,
            mode=f"closed-loop-{task}",
        )
        print(
            f"Collected {len(xs)} closed-loop {task} observations from {trials} flights",
            flush=True,
        )
    finally:
        env.close()


def warm_start(model, path, dataset_hash, steps=1500, yaw_scale=0.4):
    import torch

    d = np.load(path)
    if str(d["dataset_hash"]) != dataset_hash:
        raise ValueError("calibration graph mismatch")
    if str(d["encoder_version"]) != ENCODER_VERSION:
        raise ValueError("calibration was collected with a different camera/encoder")
    x = torch.tensor(d["x"])
    y = torch.tensor(d["y"])
    y[:, 3] *= yaw_scale
    torch.manual_seed(72)
    opt = torch.optim.Adam(
        list(model.policy.mlp_extractor.policy_net.parameters())
        + list(model.policy.action_net.parameters()),
        lr=1e-3,
    )
    # Normalize the small visually responsive population while keeping frozen
    # anatomical membership. Constant motor features cannot dominate the input.
    norm = model.policy.features_extractor
    norm.mean.copy_(x.mean(0))
    norm.scale.copy_(x.std(0).clamp(min=0.003))
    for _ in range(steps):
        ids = torch.randint(len(x), (128,))
        features = norm(x[ids])
        out = model.policy.action_net(
            model.policy.mlp_extractor.forward_actor(features)
        )
        loss = (out - y[ids]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.policy.log_std.fill_(-2.5)
        out = model.policy.action_net(model.policy.mlp_extractor.forward_actor(norm(x)))
        result = {
            "training_mse": float((out - y).square().mean()),
            "samples": len(x),
            "steps": steps,
            "yaw_teacher_scale": yaw_scale,
            "note": "Supervised decoder warm start, not held-out flight evaluation.",
        }
    return result
