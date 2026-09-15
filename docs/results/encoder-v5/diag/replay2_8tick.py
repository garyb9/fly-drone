# Open-loop replay of one held-out flight of the mirror clone's split (data.npz + data2.npz):
# DN/motor features under v4 cues vs clone currents, both through the v5 runtime (bit-identical to v4 for equal cues).
import sys
import numpy as np, torch
from fly_drone.brain import BrainRuntime
from fly_drone.encoder import LearnedEncoder, v4_targets
enc_path, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500
paths = ["runs/v5/clone/data.npz", "runs/v5/clone/data2.npz"]
flights = np.concatenate([np.load(p)["flight"].astype(np.int64) + i * 1_000_000 for i, p in enumerate(paths)])
u = np.unique(flights); held = np.random.default_rng(72).choice(u, max(1, int(len(u) * 0.1)), replace=False)
flight = int(sorted(held)[0]); file_i, local_flight = divmod(flight, 1_000_000)
d = np.load(paths[file_i]); ids = np.flatnonzero(d["flight"] == local_flight)[:N]
print("held-out flights", sorted(int(h) for h in held), "replaying", flight, "frames", len(ids))
cues = v4_targets(d["cues"][ids]); stacks = d["stacks"][ids]
enc = LearnedEncoder.load(enc_path)
with torch.no_grad():
    clone = (torch.tanh(enc.mu(enc.extractor({"eyes": torch.as_tensor(stacks, dtype=torch.float32) / 255.0}))) + 1).numpy()
def run(seq):
    rt = BrainRuntime(encoder="external"); rt.reset(0); out = []
    for c in seq:
        rt.set_currents(c); out.append(rt.step(8))
    return np.array(out)
A, C = run(cues), run(clone)
both = (A.std(0) > 1e-6) & (C.std(0) > 1e-6)
r = np.array([np.corrcoef(A[:, i], C[:, i])[0, 1] for i in np.flatnonzero(both)])
print(f"currents r per channel {[round(float(np.corrcoef(cues[:, k], clone[:, k])[0, 1]), 3) for k in range(8)]}")
print(f"features active {int(((A.std(0) > 1e-6) | (C.std(0) > 1e-6)).sum())}  median r {np.median(r):.3f}  frac r<0.9 {np.mean(r < 0.9):.3f}  mean|diff| {np.abs(A - C).mean():.5f}  max|diff| {np.abs(A - C).max():.3f}")
