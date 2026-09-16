"""Check a learned encoder is not degenerate on real held-out frames.

The v6 mid-round gate: a collapsed encoder pins a group to a near-constant value (the round-1
encoder drove light to a constant 2.0), which a random-noise in-run probe misses because a healthy
v6 net also barely varies on noise. This measures how much each scored group's per-frame mean
varies over real flight frames. Exit 1 if any group is too flat.

    env -u PYTHONPATH .venv/bin/python scripts/encoder_liveness.py runs/v6/round1/encoder/encoder.pt
"""

import argparse
import sys

import numpy as np

# Per-frame mean std floors, set between the healthy clone (~0.83 light / ~0.22 loom) and the
# collapsed round-1 encoder (~0.006 / ~0.049).
FLOORS = {"light": 0.10, "loom": 0.08}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("encoder", help="learned-v6 .pt")
    ap.add_argument("--data", default="runs/v5/clone/data.npz")
    ap.add_argument("--frames", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from fly_drone.spatial_encoder import CHANNEL_ORDER, SpatialEncoder, group_slices

    stacks = np.load(args.data)["stacks"][: args.frames]
    enc = SpatialEncoder.load(args.encoder)
    with torch.no_grad():
        x = torch.as_tensor(stacks, dtype=torch.float32) / 255.0
        out = enc.net(x)
        currents = np.concatenate(
            [
                (torch.tanh(out[name]) + 1.0).reshape(len(stacks), -1).numpy()
                for name in CHANNEL_ORDER
            ],
            axis=1,
        )
    groups = group_slices()
    ok = True
    for group, floor in FLOORS.items():
        std = float(currents[:, groups[group]].mean(1).std())
        print(f"{group}: frame-mean std {std:.4f} (floor {floor})")
        ok = ok and std >= floor
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
