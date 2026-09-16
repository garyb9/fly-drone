"""Trace where a uniform Mi1/Tm3 injection propagates in the frozen graph.

Diagnostic for the M1b gate: if spatial Mi1/Tm3 input does not reach LC4/LPLC2,
this shows the first stage where the signal dies.
"""

import collections

import numpy as np
from fly_drone.brain import BrainRuntime

WATCH = [
    "Mi1",
    "Tm3",
    "L1",
    "L2",
    "L3",
    "T4a",
    "T4b",
    "T5a",
    "T5b",
    "LC4",
    "LPLC2",
    "T2",
    "T3",
    "LPTC",
]


def watch(br, ids):
    return float(np.mean(br.core.activity(ids))) if ids else float("nan")


def main():
    br = BrainRuntime()
    by_type = collections.defaultdict(list)
    for i, c in enumerate(br.cells):
        by_type[c["type"]].append(i)

    mi = [i for i, c in enumerate(br.cells) if c["type"] in ("Mi1", "Tm3")]
    role = br.core.input_role("all_on", mi)
    print("Mi1+Tm3 cells:", len(mi))

    br.reset(7)
    header = ["Mi1", "Tm3"] + [w for w in WATCH if w not in ("Mi1", "Tm3")]
    print(f"{'tick':>5s} " + " ".join(f"{t:>7s}" for t in header))
    for t in range(0, 241, 30):
        for _ in range(30):
            br.core.inject(role, 1.6)
            br.core.step(1)
        vals = {w: watch(br, by_type.get(w, [])) for w in header}
        print(f"{t + 30:5d} " + " ".join(f"{vals[w]:7.3f}" for w in header))

    # also report raw spike-ish count for a few types
    print("\nfeature (DN+VNC) mean:", float(np.mean(br.features())))


if __name__ == "__main__":
    main()
