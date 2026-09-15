# Open-loop replay of one held-out clone flight: DN/motor features under
# (A) v4 runtime + v4 cues, (B) v5 runtime + v4 cues copied to 8 channels, (C) v5 runtime + clone currents.
import numpy as np, torch, json
from fly_drone.brain import BrainRuntime
from fly_drone.encoder import LearnedEncoder, v4_targets
N = 500
d = np.load('runs/v5/clone/data.npz'); fl = d['flight']
ids = np.flatnonzero(fl == 625)[:N]
cues4 = d['cues'][ids]
enc = LearnedEncoder.load('runs/v5/clone/encoder.pt')
with torch.no_grad():
    x = torch.as_tensor(d['stacks'][ids], dtype=torch.float32) / 255.
    clone = (torch.tanh(enc.mu(enc.extractor({'eyes': x}))) + 1).numpy()
def run(rt, seq):
    rt.reset(0); out = []
    for c in seq:
        rt.set_currents(c); out.append(rt.step(40))
    return np.array(out)
A = run(BrainRuntime(), cues4)
v5 = BrainRuntime(encoder='external')
B = run(v5, v4_targets(cues4))
C = run(v5, clone)
def cmp(name, P, Q):
    act = (P.std(0) > 1e-6) | (Q.std(0) > 1e-6)
    r = [np.corrcoef(P[:, i], Q[:, i])[0, 1] for i in np.flatnonzero((P.std(0) > 1e-6) & (Q.std(0) > 1e-6))]
    print(f'{name}: features {P.shape[1]} active {act.sum()} max|diff| {np.abs(P-Q).max():.4f} '
          f'mean|diff| {np.abs(P-Q).mean():.5f} median r {np.median(r) if r else float("nan"):.3f} '
          f'frac r<0.9 {np.mean(np.array(r)<0.9) if r else float("nan"):.3f}')
cmp('A v4 vs B v5(v4 cues)', A, B)
cmp('B v5(v4 cues) vs C v5(clone)', B, C)
cmp('A v4 vs C v5(clone)', A, C)
np.savez_compressed('runs/v5/diag/replay.npz', A=A, B=B, C=C, cues4=cues4, clone=clone)
