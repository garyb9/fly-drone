# Brain sensitivity: DN/motor features for v4 cues vs v4 cues + small Gaussian noise (clipped to [0, 2]).
import numpy as np
from fly_drone.brain import BrainRuntime
from fly_drone.encoder import v4_targets
d = np.load("runs/v5/clone/data.npz"); ids = np.flatnonzero(d["flight"] == 600)[:500]
cues = v4_targets(d["cues"][ids]).astype(np.float32)
def run(seq):
    rt = BrainRuntime(encoder="external"); rt.reset(0); out = []
    for c in seq:
        rt.set_currents(c); out.append(rt.step(8))
    return np.array(out)
A = run(cues)
for sigma in (0.01, 0.03):
    noisy = np.clip(cues + np.random.default_rng(1).normal(0, sigma, cues.shape).astype(np.float32), 0, 2)
    C = run(noisy)
    both = (A.std(0) > 1e-6) & (C.std(0) > 1e-6)
    r = np.array([np.corrcoef(A[:, i], C[:, i])[0, 1] for i in np.flatnonzero(both)])
    cr = [round(float(np.corrcoef(cues[:, k], noisy[:, k])[0, 1]), 3) for k in range(8)]
    print(f"sigma {sigma}: currents r {cr}  features median r {np.median(r):.3f} frac r<0.9 {np.mean(r < 0.9):.3f} mean|diff| {np.abs(A - C).mean():.5f}")
