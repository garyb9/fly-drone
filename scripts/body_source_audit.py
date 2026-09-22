"""Read-only inventory of sibling body sources; not a runtime causal assay.

Fingerprints dirty sources without printing their contents or changing the sibling.
The ledger is a reviewed classification of the current implementation, not a parser
that can certify arbitrary future code. Missing anchors invalidate the inventory.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

LEDGER = [
    ("src/body/integrate.ts", "const gravity", "passive", "gravity and damping"),
    ("src/body/body.ts", "// soft bounds", "assistance", "boundary return force"),
    (
        "src/body/body.ts",
        "this.movement.wrench",
        "assistance",
        "manual/landing controller",
    ),
    (
        "src/body/body.ts",
        "m.firedImpulse",
        "assistance",
        "legacy threshold escape impulse",
    ),
    (
        "src/body/body.ts",
        "resolveSphere(",
        "passive_proxy",
        "sphere contact resolution",
    ),
    (
        "src/body/articulation.ts",
        "s.energyL =",
        "neural_mechanics",
        "wing activation decay",
    ),
    (
        "src/body/articulation.ts",
        "const flight =",
        "assistance",
        "support/escape flight gate",
    ),
    ("src/body/articulation.ts", "s.headYaw =", "neural_proxy", "neck motor joint map"),
    (
        "src/body/articulation.ts",
        "s.antennaL =",
        "neural_proxy",
        "antenna motor joint map",
    ),
    (
        "src/body/articulation.ts",
        'read(r, "proboscis_r"), s.feed',
        "assistance",
        "DN feeding override",
    ),
    (
        "src/body/articulation.ts",
        "Math.sin(s.time * 2 * Math.PI * 4)",
        "assistance",
        "authored abdominal pumping",
    ),
    (
        "src/body/articulation.ts",
        "l.phase =",
        "assistance",
        "authored tripod oscillator",
    ),
    (
        "src/body/articulation.ts",
        "const sweep =",
        "assistance",
        "authored grooming waveform",
    ),
    (
        "src/body/articulation.ts",
        "if (reach &&",
        "assistance",
        "surface-targeted foot IK",
    ),
    (
        "src/body/articulation.ts",
        "l.contact = !!hit",
        "assistance",
        "phase/groom/escape contact gating",
    ),
    (
        "src/body/articulation.ts",
        "scale(error, c.contactSpring",
        "passive_proxy",
        "contact spring/damping conditional on motor drive",
    ),
    (
        "src/body/articulation.ts",
        "if (s.supported) torque",
        "assistance",
        "support alignment torque",
    ),
    (
        "src/body/articulation.ts",
        "const leading =",
        "assistance",
        "adjacent-surface lookahead",
    ),
    (
        "src/body/articulation.ts",
        "const lift =",
        "neural_proxy",
        "calibrated wing-power lift",
    ),
    (
        "src/body/articulation.ts",
        "const target = add(landing.point",
        "assistance",
        "landing destination controller",
    ),
    (
        "src/body/wrench.ts",
        "P.ESCAPE_IMPULSE",
        "assistance",
        "escape impulse independent of actuator mechanics",
    ),
    (
        "src/body/wrench.ts",
        "P.CRUISE_THRUST",
        "assistance",
        "legacy non-neural cruise term (conditional)",
    ),
]


def inventory(root):
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args]).decode()

    changed = git("diff", "--name-only", "HEAD", "-z").split("\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    relevant = sorted(
        {
            name
            for name in changed + untracked
            if name
            and name.startswith(
                (
                    "src/body/",
                    "src/sensing/",
                    "src/sim/",
                    "src/bridge/",
                    "src/viz/",
                    "pipeline/",
                    "public/models/fly/",
                    "docs/body-",
                    "examples/body-",
                )
            )
        }
        | {row[0] for row in LEDGER}
    )
    sources = {}
    for name in relevant:
        path = root / name
        sources[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    ledger = []
    for name, anchor, category, meaning in LEDGER:
        lines = (root / name).read_text().splitlines()
        matches = [i + 1 for i, line in enumerate(lines) if anchor in line]
        ledger.append(
            {
                "source": name,
                "lines": matches,
                "category": category,
                "meaning": meaning,
                "anchor_found": bool(matches),
            }
        )
    return {
        "schema": "body-source-audit-v1",
        "sibling_revision": git("rev-parse", "HEAD").strip(),
        "dirty_tracked_count": len([p for p in changed if p]),
        "untracked_count": len([p for p in untracked if p]),
        "source_sha256": sources,
        "ledger": ledger,
        "anchors_complete": all(row["anchor_found"] for row in ledger),
        "runtime_validated": False,
        "canonical_brain_parity_validated": False,
        "neural_only_certified": False,
        "verdict": "proxy mechanics and behavioral assistance coexist; isolation required",
        "next": "separate neural-only body path; preserve dirty originals; validate forces before biomechanics",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sibling", default="../fly-playground")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = inventory(args.sibling)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(path),
                "anchors_complete": report["anchors_complete"],
                "neural_only_certified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
