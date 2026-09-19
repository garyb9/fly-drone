"""Build the P1 additive feedback bundle from the canonical MaleCNS bundle.

The graph and neurons are unchanged --- the feedback roles are new annotated
inputs, not a new connectome --- so the large immutable files are symlinked and
only a manifest marker (which changes ``bundle_hash`` and marks the bundle
alternate) plus ``feedback-mappings.json`` are written. Like the sign and rewired
bundles, the output directory is generated and git-ignored.
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data" / "malecns"
DEFAULT_OUTPUT = ROOT / "data" / "malecns-feedback"
DEFAULT_MAPPINGS = ROOT / "docs" / "results" / "feedback" / "feedback-mappings.json"
FEEDBACK_VERSION = "c1-ascending-v1"
# Byte-identical to canonical: symlinked so a regenerated bundle cannot drift.
SHARED = (
    "graph.bin",
    "neurons.bin",
    "cells.json",
    "groups.json",
    "sensory-mappings.json",
    "ATTRIBUTION.md",
)


def build(canonical=CANONICAL, output=DEFAULT_OUTPUT, mappings=DEFAULT_MAPPINGS):
    canonical, output = Path(canonical), Path(output)
    manifest = json.loads((canonical / "manifest.json").read_text())
    if "feedback" in manifest:
        raise ValueError("canonical manifest already carries a feedback marker")
    manifest["feedback"] = FEEDBACK_VERSION
    output.mkdir(parents=True, exist_ok=True)
    for name in SHARED:
        target = output / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to((canonical / name).resolve())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(mappings, output / "feedback-mappings.json")
    return output


def main():
    parser = argparse.ArgumentParser(description="Build the P1 feedback bundle")
    parser.add_argument("--canonical", default=str(CANONICAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--mappings", default=str(DEFAULT_MAPPINGS))
    args = parser.parse_args()
    output = build(args.canonical, args.output, args.mappings)
    print(json.dumps({"output": str(output), "feedback": FEEDBACK_VERSION}))


if __name__ == "__main__":
    main()
