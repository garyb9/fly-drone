"""Cross-check this project against the independent fly.ai / flybrain build.

Two parts:

1. **Edge-count reconciliation** (always runs). This project keeps edges with >=3
   synaptic contacts (``data/malecns/manifest.json``); the surveyed projects report
   a much larger directed-connection count. The difference is recorded, not hidden,
   and needs the raw flat-connectome file to resolve.
2. **Laterality probe** (runs only if ``flybrain`` is installed). Fly.ai publishes
   that stimulating left LC4+LPLC2 raises the left giant fibre DNp01 far more than
   the right. We reproduce the direction on the same connectome with an independent
   implementation. Optional dependency: ``pip install flybrain`` (~260 MB data on
   first use). Never part of the default test suite.

See ``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream C
and ``docs/external-prior-art.md`` section 5.

Usage:
    python scripts/check_fly_ai_oracle.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIS_EDGES = json.loads((ROOT / "data/malecns/manifest.json").read_text())["n_edges"]
FLY_AI_EDGES = 25_582_938  # reported by fly.ai / FlyDrones / DOOMFLY for MaleCNS v1.0
STIM_TICKS = 60


def edge_report():
    print("edge-count reconciliation")
    print(f"  this bundle, >=3 contacts : {THIS_EDGES:,}")
    print(f"  fly.ai / FlyDrones / DOOMFLY reported: {FLY_AI_EDGES:,}")
    print(
        "  ratio %.2fx. Same source file; the surveyed builds likely keep below-3\n"
        "  contacts or do not deduplicate ordered pairs. Resolving needs the raw\n"
        "  flat-connectome feather, which is not vendored here. Recorded as open."
        % (FLY_AI_EDGES / THIS_EDGES)
    )


def _cells(brain, types, side):
    for candidate in (side, side.lower(), side.upper()):
        try:
            ids = brain.cells(types, side=candidate)
        except Exception:  # noqa: BLE001 - optional dependency, API may differ
            continue
        if ids:
            return set(int(i) for i in ids)
    return set()


def laterality():
    try:
        from flybrain import FlyBrain
    except ImportError:
        print("laterality probe skipped: install `flybrain` to run it (optional).")
        return None
    brain = FlyBrain(device="auto")
    stim = _cells(brain, ["LC4", "LPLC2"], "L")
    dn_l = _cells(brain, ["DNp01"], "L")
    dn_r = _cells(brain, ["DNp01"], "R")
    if not (stim and dn_l and dn_r):
        print("laterality probe skipped: cell lookup returned nothing.")
        return None
    brain.stimulate(sorted(stim), 0.8)
    fired_l = fired_r = 0
    for _ in range(STIM_TICKS):
        fired = set(int(i) for i in brain.step())
        fired_l += len(fired & dn_l)
        fired_r += len(fired & dn_r)
    result = {"dnp01_l_spikes": fired_l, "dnp01_r_spikes": fired_r}
    print(
        f"laterality: left DNp01 {fired_l} spikes, right {fired_r} (left >= right expected)"
    )
    return result


def main():
    edge_report()
    result = laterality()
    if result is not None:
        out = ROOT / "runs/diagnostics/fly-ai-oracle.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
