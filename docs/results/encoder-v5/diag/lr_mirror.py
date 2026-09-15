# Held-out L-R fidelity and mirror consistency of the L-R difference: d(mirror(x)) should equal -d(x).
# Usage: lr_mirror.py ENCODER DATA [DATA ...] — reproduces fit_clone's held-out split over the given files
# (flight ids offset by file index * 1_000_000, default_rng(72), 10% of flights).
import sys

import numpy as np
import torch

from fly_drone.encoder import LearnedEncoder, v4_targets

paths = sys.argv[2:]
flights = np.concatenate(
    [np.load(p)["flight"].astype(np.int64) + i * 1_000_000 for i, p in enumerate(paths)]
)
u = np.unique(flights)
rng = np.random.default_rng(72)
held = rng.choice(u, max(1, int(len(u) * 0.1)), replace=False)
stacks, cues, start = [], [], 0
for p in paths:
    d = np.load(p)
    n = len(d["flight"])
    local = np.flatnonzero(np.isin(flights[start : start + n], held))
    if len(local):
        stacks.append(d["stacks"][local])
        cues.append(d["cues"][local])
    start += n
stacks = np.concatenate(stacks)
t = v4_targets(np.concatenate(cues))
print(f"held-out flights {len(held)}, frames {len(stacks)}")
f = stacks.shape[1] // 2
mirror = np.concatenate([stacks[:, f:, :, ::-1], stacks[:, :f, :, ::-1]], axis=1).copy()
enc = LearnedEncoder.load(sys.argv[1])


def run(x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            b = torch.as_tensor(x[i : i + 512], dtype=torch.float32) / 255.0
            out.append((torch.tanh(enc.mu(enc.extractor({"eyes": b}))) + 1).numpy())
    return np.concatenate(out)


p, pm = run(stacks), run(mirror)
for a, name in ((0, "mi1"), (2, "tm3"), (4, "lc4"), (6, "lplc2")):
    dt, dp, dm = t[:, a] - t[:, a + 1], p[:, a] - p[:, a + 1], pm[:, a] - pm[:, a + 1]
    print(
        f"{name}: L-R r {np.corrcoef(dt, dp)[0, 1]:.3f} rmse {np.sqrt(np.mean((dt - dp) ** 2)):.3f} "
        f"| mirror diff r(dp, -dm) {np.corrcoef(dp, -dm)[0, 1]:.3f} rmse {np.sqrt(np.mean((dp + dm) ** 2)):.3f} "
        f"| mean bias (dp+dm)/2 {np.mean((dp + dm) / 2):+.4f}"
    )
