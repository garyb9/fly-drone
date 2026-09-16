"""D4: offline drift of exported encoder currents during round 1.

Deployed currents are the deterministic mean: currents = tanh(mu(extractor(eyes))) + 1 in [0, 2]
(encoder.py:138-142). Entropy-driven drift moves channels toward the mid-range 1.0 (tanh 0);
saturation pushes them to 0/2. Same held-out flights and seed as lr_fidelity.py.

Run with no args for the round-1 checkpoint timeline, or pass encoder .pt paths.
"""

import sys

import numpy as np
import torch

from fly_drone.encoder import LearnedEncoder

LABELS = ("mi1_L", "mi1_R", "tm3_L", "tm3_R", "lc4_L", "lc4_R", "lplc2_L", "lplc2_R")
DEFAULT = [
    ("clone  (round0 init)", "runs/v5/clone/encoder.pt"),
    ("r1 ckpt  50k", "runs/v5/diag/d2/encoder_49998.pt"),
    ("r1 ckpt 100k", "runs/v5/diag/d2/encoder_99996.pt"),
    ("r1 ckpt 150k", "runs/v5/diag/d2/encoder_149994.pt"),
    ("r1 ckpt 200k", "runs/v5/diag/d2/encoder_199992.pt"),
    ("r1 final 350k", "runs/v5/round1/encoder/encoder.pt"),
]


def held_out_ids():
    d = np.load("runs/v5/clone/data.npz")
    fl = d["flight"]
    u = np.unique(fl)
    rng = np.random.default_rng(72)
    held = rng.choice(u, max(1, int(len(u) * 0.1)), replace=False)
    ids = np.flatnonzero(np.isin(fl, held))
    return d["stacks"][ids], len(held)


def currents(path, stacks):
    enc = LearnedEncoder.load(path)
    out = []
    with torch.no_grad():
        for i in range(0, len(stacks), 512):
            x = torch.as_tensor(stacks[i : i + 512], dtype=torch.float32) / 255.0
            out.append((torch.tanh(enc.mu(enc.extractor({"eyes": x}))) + 1).numpy())
    return np.concatenate(out)


def main(argv):
    stacks, n_flights = held_out_ids()
    pairs = [(a, "runs/v5/clone/encoder.pt")] if argv else DEFAULT
    if argv:
        pairs = [(p, p) for p in argv]
    clone = currents("runs/v5/clone/encoder.pt", stacks)
    print(f"held-out flights {n_flights}, frames {len(stacks)}; reference = clone")
    print(
        f"{'encoder':22s} {'mean':>6s} {'std':>6s} {'sat>0.9':>8s} {'mean|d|':>8s} "
        f"   {'mi1_L':>6s} {'tm3_L':>6s} {'lc4_L':>6s} {'lplc2_L':>7s}   (channel means)"
    )
    for name, path in pairs:
        cur = clone if path == "runs/v5/clone/encoder.pt" else currents(path, stacks)
        sat = float(np.mean(np.abs(cur - 1.0) > 0.9))
        lc = [cur[:, c].mean() for c in (0, 2, 4, 6)]
        print(
            f"{name:22s} {cur.mean():6.3f} {cur.std():6.3f} {sat:8.3f} "
            f"{np.abs(cur - clone).mean():8.3f}   {lc[0]:6.3f} {lc[1]:6.3f} {lc[2]:6.3f} {lc[3]:7.3f}"
        )
    print()
    print(f"{'channel':10s} {'clone mean':>10s} {'clone std':>9s} {'r1 final mean':>13s} {'r1 final std':>12s}")
    final = currents("runs/v5/round1/encoder/encoder.pt", stacks)
    for c, label in enumerate(LABELS):
        print(
            f"{label:10s} {clone[:, c].mean():10.3f} {clone[:, c].std():9.3f} "
            f"{final[:, c].mean():13.3f} {final[:, c].std():12.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
