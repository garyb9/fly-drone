"""Build a degree-matched rewired MaleCNS bundle (diagnostic null model).

Only the wiring changes: every neuron, edge weight and source out-degree is kept,
and edge destinations are shuffled (``shuffle``) or swapped between edge pairs
preserving both in- and out-degree (``swap``). The bundle is generated on demand,
git-ignored, and never flown.

See ``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream B.

Usage:
    python scripts/make_rewired_bundle.py --source data/malecns \\
        --out data/malecns-rewired --mode swap --seed 0
"""

import argparse
from pathlib import Path

from fly_drone.rewire import rewire_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(ROOT / "data/malecns"))
    ap.add_argument("--out", default=str(ROOT / "data/malecns-rewired"))
    ap.add_argument("--mode", choices=("shuffle", "swap"), default="swap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=20, help="swap passes")
    ap.add_argument("--batch", type=int, default=1_000_000, help="candidate pairs/pass")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    info = rewire_bundle(
        args.source, args.out, args.mode, args.seed, args.rounds, args.batch, args.force
    )
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
