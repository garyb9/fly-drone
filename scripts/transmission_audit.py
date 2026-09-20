"""Which neurons are wired into the graph but transmit nothing?

The canonical bundle's declared model says "unresolved/modulatory source current zero"
(``manifest.json``, ``docs/neuron-model.md`` §1). That hypothesis is implemented upstream by zeroing
those neurons' outgoing edge weights, so the cells integrate input and spike but never propagate.
This audit quantifies it per transmitter, directly from ``graph.bin``'s CSR.

Cheap and offline: no arena, no brain runtime, no long run.

Run: ``env -u PYTHONPATH .venv/bin/python scripts/transmission_audit.py``
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data" / "malecns"
HEADER = 32  # graph.bin: magic, version, n_nodes, n_edges (u64) -> CSR offsets


def load_out_weights(bundle=CANONICAL):
    """``(out_degree, summed_abs_out_weight)`` per neuron, from the CSR directly."""
    raw = np.fromfile(Path(bundle) / "graph.bin", dtype=np.uint8)
    n_nodes = int(np.frombuffer(raw[8:12].tobytes(), "<u4")[0])
    offsets = np.frombuffer(
        raw[HEADER : HEADER + (n_nodes + 1) * 4].tobytes(), "<u4"
    ).astype(np.int64)
    n_edges = int(offsets[-1])
    weight_base = HEADER + (n_nodes + 1) * 4 + n_edges * 4
    weights = np.abs(
        np.frombuffer(
            raw[weight_base : weight_base + n_edges * 2].tobytes(), "<i2"
        ).astype(np.int64)
    )
    cumulative = np.concatenate([[0], np.cumsum(weights)])
    out_weight = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
    return np.diff(offsets), out_weight, n_edges


def run(bundle=CANONICAL, output="docs/results/liveness/transmission-audit.json"):
    bundle = Path(bundle)
    out_degree, out_weight, n_edges = load_out_weights(bundle)
    cells = json.loads((bundle / "cells.json").read_text())
    transmitter = np.array([c.get("nt") for c in cells])
    wired = out_degree > 0
    silent = wired & (out_weight == 0)

    per_transmitter = {}
    for name in sorted(set(transmitter.tolist())):
        mask = transmitter == name
        has_edges = mask & wired
        per_transmitter[str(name)] = {
            "cells": int(mask.sum()),
            "with_out_edges": int(has_edges.sum()),
            "silent_out": int((mask & silent).sum()),
            "mean_abs_out_weight": round(
                float(out_weight[has_edges].mean()) if has_edges.any() else 0.0, 3
            ),
        }

    report = {
        "bundle": bundle.name,
        "n_neurons": len(cells),
        "n_edges": n_edges,
        "silent_out_neurons": int(silent.sum()),
        "silent_out_edges": int(out_degree[silent].sum()),
        "per_transmitter": per_transmitter,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print_table(report):
    print(
        f"{'transmitter':16}{'cells':>8}{'w/ edges':>10}{'silent':>9}{'mean |w|':>11}"
    )
    for name, row in sorted(
        report["per_transmitter"].items(), key=lambda kv: -kv[1]["cells"]
    ):
        print(
            f"{name:16}{row['cells']:8d}{row['with_out_edges']:10d}"
            f"{row['silent_out']:9d}{row['mean_abs_out_weight']:11.1f}"
        )
    neurons, edges = report["silent_out_neurons"], report["silent_out_edges"]
    print(
        f"\nwired but transmitting nothing: {neurons} neurons "
        f"({100 * neurons / report['n_neurons']:.1f}%), "
        f"{edges} edges ({100 * edges / report['n_edges']:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(CANONICAL))
    parser.add_argument(
        "--output", default="docs/results/liveness/transmission-audit.json"
    )
    args = parser.parse_args()
    report = run(args.bundle, args.output)
    _print_table(report)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
