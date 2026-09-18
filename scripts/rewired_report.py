"""Lateralisation probe on the canonical bundle vs a rewired null model.

Decoder-free: inject one side's visual population and measure whether the effect
stays ipsilateral in the loom/escape circuit. The real wiring lateralises; a
degree-matched random rewiring is expected to smear or lose it. This is the causal
control for "the connectome, not an anonymous graph, does the work" — it needs no
training and touches no acceptance threshold.

See ``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream B.

Usage:
    python scripts/make_rewired_bundle.py --mode swap --seed 0
    python scripts/rewired_report.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fly_drone.brain import BrainRuntime
from fly_drone.identity import load_identity

ROOT = Path(__file__).resolve().parents[1]

# Left-side populations to inject. LC4/LPLC2 is v4's guaranteed loom route; T2/Tm4
# are the v6 retinotopic drivers (spec 2026-09-16 M1b).
PROBES = {
    "loom_direct": ("LC4", "LPLC2"),
    "tm4": ("Tm4",),
    "t2": ("T2",),
}
# Injecting the same population we measure makes laterality tautological; for these
# probes only the downstream escape readout is informative.
DIRECT_PROBES = frozenset({"loom_direct"})


def measure(br, lc4, lplc2, escape):
    return {
        "lc4_l": float(np.mean(br.core.activity(lc4["l"]))),
        "lc4_r": float(np.mean(br.core.activity(lc4["r"]))),
        "lplc2_l": float(np.mean(br.core.activity(lplc2["l"]))),
        "lplc2_r": float(np.mean(br.core.activity(lplc2["r"]))),
        "escape": float(br.core.readout(escape)),
    }


def probe(br, types, side, ticks, current, seed):
    ids = [
        i
        for i, c in enumerate(br.cells)
        if c["side"].lower() == side and c["type"] in types
    ]
    if not ids:
        raise ValueError(f"no cells for {types} side {side}")
    br.reset(seed)
    role = br.core.input_role(f"rewired_{'_'.join(types)}_{side}", ids)
    for _ in range(ticks):
        br.core.inject(role, current)
        br.core.step(1)
    return len(ids)


def run_bundle(data, ticks, current, seed):
    br = BrainRuntime(data=data)
    lc4 = {s: br._cells(s, ("LC4",)) for s in ("l", "r")}
    lplc2 = {s: br._cells(s, ("LPLC2",)) for s in ("l", "r")}
    escape = br.readouts["escape"]
    probes = {}
    for name, types in PROBES.items():
        n = probe(br, types, "l", ticks, current, seed)
        m = measure(br, lc4, lplc2, escape)
        ipsi = m["lc4_l"] + m["lplc2_l"]
        contra = m["lc4_r"] + m["lplc2_r"]
        m["n_cells"] = n
        if name in DIRECT_PROBES:
            m["laterality"] = None
            m["note"] = "input cells measured directly; use escape"
        else:
            m["laterality"] = (ipsi - contra) / (ipsi + contra + 1e-9)
        probes[name] = m
    dataset_hash, bundle_hash, alternate = load_identity(data)
    meta = {
        "data": str(data),
        "dataset_hash": dataset_hash,
        "bundle_hash": bundle_hash,
        "alternate": alternate,
        "n_neurons": br.core.neuron_count(),
        "probes": probes,
    }
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intact", default=str(ROOT / "data/malecns"))
    ap.add_argument("--rewired", default=str(ROOT / "data/malecns-rewired"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--current", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    intact = run_bundle(args.intact, args.ticks, args.current, args.seed)
    rewired = run_bundle(args.rewired, args.ticks, args.current, args.seed)
    report = {
        "note": (
            "Lateralisation of the loom pathway under a degree-matched rewiring. "
            "Diagnostic only; not an acceptance run."
        ),
        "ticks": args.ticks,
        "current": args.current,
        "seed": args.seed,
        "intact": intact,
        "rewired": rewired,
    }
    for name in PROBES:
        i = intact["probes"][name]["laterality"]
        r = rewired["probes"][name]["laterality"]
        report.setdefault("laterality", {})[name] = {"intact": i, "rewired": r}
        if i is None:
            print(f"{name:12s} laterality n/a (input cells measured directly)")
        else:
            print(f"{name:12s} laterality intact {i:+.3f}  rewired {r:+.3f}")

    out = args.out or str(
        ROOT / "runs/diagnostics" / f"rewired-{rewired['bundle_hash'][:12]}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
