"""Optional supervised warm start of the decoder, followed by PPO.
Teacher uses simulator bearing only while collecting training labels. The actor
sees ONLY measured neural features, both here and at inference.
"""

from pathlib import Path

import numpy as np

from .brain import BrainRuntime
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
            encoder_version="bright-contrast-400-v1",
        )
        print(
            f"Collected {len(xs)} neural observations from {trials} rendered target positions",
            flush=True,
        )
    finally:
        p.close()


def warm_start(model, path, steps=1500):
    import torch

    d = np.load(path)
    if str(d["dataset_hash"]) != model.get_env().envs[0].unwrapped.brain.dataset_hash:
        raise ValueError("calibration graph mismatch")
    x = torch.tensor(d["x"])
    y = torch.tensor(d["y"])
    y[:, 3] *= 0.4
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
            "yaw_teacher_scale": 0.4,
            "note": "Supervised decoder warm start, not held-out flight evaluation.",
        }
    return result
