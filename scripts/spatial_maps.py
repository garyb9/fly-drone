"""Inspect the v6 retinotopic maps and, optionally, an encoder's current maps.

    env -u PYTHONPATH .venv/bin/python scripts/spatial_maps.py
    env -u PYTHONPATH .venv/bin/python scripts/spatial_maps.py --encoder runs/v6/encoder.pt

Thin wrapper around ``fly_drone.maps``; the CLI command is ``fly-drone spatial-maps``.
"""

import argparse
from pathlib import Path

from fly_drone.maps import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "runs" / "probe" / "maps"))
    ap.add_argument("--encoder", default=None, help="learned-v6 .pt to visualise")
    args = ap.parse_args()
    result = run(args.output, args.encoder)
    for r in result["rows"]:
        print(
            f"{r['channel']:10s} {r['population']:6s} {r['side']:4s} "
            f"{r['grid'][0]:>2d}x{r['grid'][1]:<2d}  {r['occupancy']:5.2f} {r['cells']:6d}"
        )
    print(f"\nflat_dim = {result['flat_dim']} currents/frame")


if __name__ == "__main__":
    main()
