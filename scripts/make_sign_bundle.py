"""Build alternate transmitter-sign bundles from the raw neurotransmitter feather.

The canonical bundle's flag bit is exactly ``inhibitory = {glutamate, gaba}``; this
writes S2 (fly.ai: also histamine) and S3 (FlyGM: glutamate excitatory, GABA
inhibitory) as additive bundles. Edges, positions and group ids are untouched.

Requires the optional Arrow reader: ``pip install -e ".[data]"``.

See ``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream E.

Usage:
    python scripts/make_sign_bundle.py --convention s2 \\
        --feather /path/to/body-neurotransmitters-male-cns-v1.0.feather
"""

import argparse
from pathlib import Path

from fly_drone.identity import load_identity
from fly_drone.signs import CONVENTIONS, build_sign_bundle

ROOT = Path(__file__).resolve().parents[1]


def load_labels(feather_path):
    try:
        import pyarrow.feather as feather
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "reading the neurotransmitter feather needs pyarrow; "
            'install with `pip install -e ".[data]"`'
        ) from exc
    table = feather.read_table(feather_path, columns=["body", "consensus_nt"])
    labels = {}
    for body, label in zip(
        table.column("body").to_pylist(),
        table.column("consensus_nt").to_pylist(),
        strict=True,
    ):
        labels.setdefault(int(body), label)
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(ROOT / "data/malecns"))
    ap.add_argument("--feather", required=True)
    ap.add_argument("--convention", choices=sorted(CONVENTIONS), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    labels = load_labels(args.feather)
    out = args.out or str(ROOT / f"data/malecns-sign-{args.convention}")
    info = build_sign_bundle(args.source, out, args.convention, labels, args.force)
    dataset_hash, bundle_hash, alternate = load_identity(out)
    print(f"convention: {info['convention']}  out: {info['out']}")
    print(f"dataset_hash: {dataset_hash}")
    print(f"bundle_hash:  {bundle_hash}  alternate: {alternate}")


if __name__ == "__main__":
    main()
