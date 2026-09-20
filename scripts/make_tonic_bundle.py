"""Build the P4 (C3a) additive tonic bundle from the canonical MaleCNS bundle.

The graph and neurons are unchanged — the tonic drive is declared state on identified motor
neurons, not a new connectome — so the immutable files are symlinked and only the manifest
marker (which changes ``bundle_hash`` and marks the bundle alternate) is written. Like the
feedback and dynamics bundles, the output directory is generated and git-ignored.

Anchor: the fly's wing steering motoneurons fire about once per wingbeat continuously in flight
and steering modulates that ongoing train rather than recruiting it from silence. The project
already adopted this for the DLMn/DVMn power muscles (``b = 0.85``); this bundle applies the
same arithmetic (``docs/neuron-model.md`` §2) to the steering motoneurons, so their resting
activity trace sits at ``a = 0.5`` — mid-rate, where input can push it both ways. This is
declared state, not a fit, and it is silenceable (``silence_tonic``).
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data" / "malecns"
DEFAULT_OUTPUT = ROOT / "data" / "malecns-tonic"
TONIC_VERSION = "declared-tonic-v1"
# Byte-identical to canonical: symlinked so a regenerated bundle cannot drift.
SHARED = (
    "graph.bin",
    "neurons.bin",
    "cells.json",
    "groups.json",
    "sensory-mappings.json",
    "ATTRIBUTION.md",
)
# The declared flight-state tonic drive, per role. b = 0.85 free-runs a cell (b > 1 - lambda
# ~ 0.221) and, from reset, reaches threshold on the second tick -> mean activity 0.5
# (docs/neuron-model.md §2). The power pair is the existing hard-coded bias, now declared.
ROLES = {"power_l": 0.85, "power_r": 0.85, "steer_l": 0.85, "steer_r": 0.85}


def build(canonical=CANONICAL, output=DEFAULT_OUTPUT, roles=None):
    canonical, output = Path(canonical), Path(output)
    manifest = json.loads((canonical / "manifest.json").read_text())
    if "tonic" in manifest:
        raise ValueError("canonical manifest already carries a tonic marker")
    manifest["tonic"] = {
        "version": TONIC_VERSION,
        "source": "declared; flight-state tonic drive on identified motoneurons; no activity fit",
        "roles": dict(ROLES if roles is None else roles),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name in SHARED:
        target = output / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to((canonical / name).resolve())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description="Build the P4 (C3a) tonic bundle")
    parser.add_argument("--canonical", default=str(CANONICAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = build(args.canonical, args.output)
    print(json.dumps({"output": str(output), "tonic": TONIC_VERSION, "roles": ROLES}))


if __name__ == "__main__":
    main()
