"""Build the P2 additive dynamics bundle from the canonical MaleCNS bundle.

The graph and neurons are unchanged (the wiring stays frozen); only a new
``dynamics.bin`` (per-neuron leak/threshold) plus a manifest marker are added, so
the bundle gets its own ``bundle_hash`` and is alternate. The large immutable
files are symlinked. Generated directories are git-ignored, like the sign and
feedback bundles.
"""

import argparse
import json
from pathlib import Path

from fly_drone.dynamics import (
    EXCITABLE_PARAMS,
    SHIU_PARAMS,
    encode,
    excitable_arrays,
    globals_dict,
    sha256,
    uniform_arrays,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data" / "malecns"
DEFAULT_OUTPUT = ROOT / "data" / "malecns-dynamics"
PRIORS = ("shiu", "excitable")
SHARED = (
    "graph.bin",
    "neurons.bin",
    "cells.json",
    "groups.json",
    "sensory-mappings.json",
    "ATTRIBUTION.md",
)


def build(canonical=CANONICAL, output=DEFAULT_OUTPUT, prior="shiu", scale=None):
    canonical, output = Path(canonical), Path(output)
    if prior not in PRIORS:
        raise ValueError(f"unknown prior {prior!r}; choose from {PRIORS}")
    manifest = json.loads((canonical / "manifest.json").read_text())
    if "dynamics" in manifest:
        raise ValueError("canonical manifest already carries a dynamics marker")
    n = int(manifest["n_neurons"])
    if prior == "excitable":
        scale = float(EXCITABLE_PARAMS["threshold_scale"] if scale is None else scale)
        leak, threshold = excitable_arrays(n, scale)
        marker = {
            **EXCITABLE_PARAMS,
            "threshold_scale": scale,
            "n_neurons": n,
            "sha256": sha256(encode(leak, threshold)),
            "note": "Declared excitability prior (P3): per-neuron threshold scaled by "
            f"{scale} so the threshold-bound model can fire. Declared, not fitted to "
            "activity. Canonical bundle untouched.",
        }
    else:
        leak, threshold = uniform_arrays(n)
        marker = {
            **SHIU_PARAMS,
            **globals_dict(),
            "n_neurons": n,
            "sha256": sha256(encode(leak, threshold)),
            "note": "Declared Shiu-2024 LIF prior; per-neuron arrays are uniform and "
            "replaceable by a recording-fitted array. Not fitted to activity.",
        }
    payload = encode(leak, threshold)
    manifest["dynamics"] = marker
    output.mkdir(parents=True, exist_ok=True)
    for name in SHARED:
        target = output / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to((canonical / name).resolve())
    (output / "dynamics.bin").write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output, manifest["dynamics"]


def main():
    parser = argparse.ArgumentParser(description="Build the P2/P3 dynamics bundle")
    parser.add_argument("--canonical", default=str(CANONICAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prior", choices=PRIORS, default="shiu")
    parser.add_argument(
        "--scale", type=float, default=None, help="excitable threshold scale"
    )
    args = parser.parse_args()
    output, marker = build(args.canonical, args.output, args.prior, args.scale)
    print(
        json.dumps(
            {
                "output": str(output),
                "version": marker["version"],
                "sha256": marker["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
