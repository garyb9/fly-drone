"""Compare loom-pathway lateralisation across transmitter-sign conventions.

Decoder-free probe over S1 (canonical), S2 (fly.ai: histamine inhibitory) and S3
(FlyGM: glutamate excitatory). Inject one side's Tm4/T2 and measure whether the
effect stays ipsilateral in the loom/escape circuit.

Diagnostic only; not an acceptance run. See
``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream E.

Usage:
    python scripts/make_sign_bundle.py --convention s2 --feather ...
    python scripts/make_sign_bundle.py --convention s3 --feather ...
    python scripts/sign_variant_report.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fly_drone.brain import BrainRuntime
from fly_drone.identity import load_identity

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = {
    "s1": ROOT / "data/malecns",
    "s2": ROOT / "data/malecns-sign-s2",
    "s3": ROOT / "data/malecns-sign-s3",
}
PROBES = {"tm4": ("Tm4",), "t2": ("T2",)}


def probe_bundle(data, ticks, current, seed):
    brain = BrainRuntime(data=data)
    lc4 = {s: brain._cells(s, ("LC4",)) for s in ("l", "r")}
    lplc2 = {s: brain._cells(s, ("LPLC2",)) for s in ("l", "r")}
    escape = brain.readouts["escape"]
    results = {}
    for name, types in PROBES.items():
        ids = [
            i
            for i, c in enumerate(brain.cells)
            if c["side"].lower() == "l" and c["type"] in types
        ]
        brain.reset(seed)
        role = brain.core.input_role(f"sign_{name}", ids)
        for _ in range(ticks):
            brain.core.inject(role, current)
            brain.core.step(1)
        ipsi = float(np.mean(brain.core.activity(lc4["l"]))) + float(
            np.mean(brain.core.activity(lplc2["l"]))
        )
        contra = float(np.mean(brain.core.activity(lc4["r"]))) + float(
            np.mean(brain.core.activity(lplc2["r"]))
        )
        results[name] = {
            "laterality": (ipsi - contra) / (ipsi + contra + 1e-9),
            "escape": float(brain.core.readout(escape)),
        }
    dataset_hash, bundle_hash, _ = load_identity(data)
    return {
        "data": str(data),
        "dataset_hash": dataset_hash,
        "bundle_hash": bundle_hash,
        "probes": results,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--current", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "runs/diagnostics/sign-variant.json"))
    args = ap.parse_args()

    report = {"note": "Sign-convention probe; diagnostic, not an acceptance run."}
    for name, data in BUNDLES.items():
        if not Path(data).exists():
            print(f"{name}: missing {data} (run scripts/make_sign_bundle.py)")
            continue
        info = probe_bundle(data, args.ticks, args.current, args.seed)
        report[name] = info
        row = "  ".join(
            f"{p} lat {info['probes'][p]['laterality']:+.3f} esc "
            f"{info['probes'][p]['escape']:.3f}"
            for p in PROBES
        )
        print(f"{name} [{info['bundle_hash'][:12]}]: {row}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
