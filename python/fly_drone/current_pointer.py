"""Best-available (not necessarily accepted) free-roam encoder/decoder pointer.

`docs/results/accepted-policies.json` only ever names checkpoints that passed
`roam_eval.ACCEPTANCE` — that bar is never relaxed (AGENTS.md). While the free-roam
decoder is still in training, this module maintains a separate, explicitly-labelled
"current" pointer so the live viewer can default to *something* moving, without ever
being mistaken for an accepted result.

Regenerate with `fly-drone current-pointer --task free_roam`. This never modifies
`accepted-policies.json`, and never reads the mid-flight files another training run is
actively writing (`runs/v5/round0*`, `runs/v5/*-data*`) unless they carry a full,
passing `evaluate_free_roam` report.
"""

import json
from pathlib import Path

from .brain import ENCODER_VERSION, ROOT

MANIFEST = ROOT / "docs" / "results" / "current-policies.json"
FALLBACK_DECODER = "runs/v5/dagger/it0/warm-actor.json"
REQUIRED_ACTOR_KEYS = ("encoder_version", "layers", "action_limits", "feature_ids")


def _combined_pass(acceptance):
    """True only if every criterion in an evaluate_free_roam acceptance block passed."""
    flags = [
        v["passed"]
        for v in acceptance.values()
        if isinstance(v, dict) and "passed" in v
    ]
    return bool(flags) and all(flags)


def _gated_candidate(root):
    """A committed or local evaluation*.json for free_roam with every A-criterion passing."""
    best = None
    for path in sorted((root / "runs" / "v5").rglob("evaluation*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if data.get("task") != "free_roam":
            continue
        if not _combined_pass(data.get("acceptance", {})):
            continue
        actor = data.get("policy")
        if not actor:
            continue
        if best is None or path.stat().st_mtime > best[1]:
            best = (data, path.stat().st_mtime, actor)
    if best is None:
        return None
    data, _, actor = best
    return {
        "status": "gated",
        "encoder": data.get("encoder"),
        "encoder_version": None,
        "decoder": actor,
        "source_run": str(Path(actor).parent),
        "gate_report": str(best[0].get("path", "")) or None,
    }


def _fallback_candidate(root):
    path = root / FALLBACK_DECODER
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    if any(k not in data for k in REQUIRED_ACTOR_KEYS):
        return None
    return {
        "status": "interim",
        "encoder": None,
        "encoder_version": ENCODER_VERSION,
        "decoder": FALLBACK_DECODER,
        "source_run": str(Path(FALLBACK_DECODER).parent),
        "gate_report": None,
    }


def scan(task="free_roam", root=ROOT):
    """Find the best available free-roam pair without touching a live training run.

    Preference order: a fully-passing evaluate_free_roam report ("gated"), else the
    Stage-1 DAgger `it0` actor under the frozen v4 encoder ("interim", the only
    finished, non-diagnostic checkpoint that isn't a failed/mid-flight sanity-gate
    attempt), else "none" if even that is missing locally.
    """
    if task != "free_roam":
        raise ValueError("current-pointer only supports the free_roam task today")
    root = Path(root)
    candidate = _gated_candidate(root) or _fallback_candidate(root)
    if candidate is None:
        return {
            "status": "none",
            "encoder": None,
            "encoder_version": None,
            "decoder": None,
            "source_run": None,
            "gate_report": None,
        }
    return candidate


def update(task="free_roam", root=ROOT, manifest=MANIFEST, dry_run=False):
    import datetime

    entry = scan(task, root)
    entry["generated_at"] = (
        datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    )
    manifest = Path(manifest)
    payload = {
        "note": (
            "Best available v5 free-roam pair, NOT vetted against roam_eval.ACCEPTANCE. "
            "Never read by --accepted; regenerate with `fly-drone current-pointer "
            "--task free_roam`."
        ),
        "policies": {task: entry},
    }
    if manifest.is_file():
        try:
            existing = json.loads(manifest.read_text())
            payload["policies"] = {**existing.get("policies", {}), task: entry}
        except (ValueError, OSError):
            pass
    if not dry_run:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def current_policies(manifest=None, root=ROOT):
    """Best-available (interim or gated) free-roam pair: (present, missing) decoder paths.

    Mirrors server.accepted_policies()'s (present, missing) contract, but reads
    current-policies.json instead, and skips any task whose status is "none".
    """
    manifest = Path(manifest or MANIFEST)
    if not manifest.is_file():
        return {}, {}
    present, missing = {}, {}
    for task, entry in json.loads(manifest.read_text())["policies"].items():
        if entry.get("status") == "none" or not entry.get("decoder"):
            continue
        path = Path(root) / entry["decoder"]
        (present if path.is_file() else missing)[task] = str(path)
    return present, missing


def current_entry(task, manifest=None, root=ROOT):
    """The full current-pointer entry for one task (decoder + encoder paths), or None."""
    manifest = Path(manifest or MANIFEST)
    if not manifest.is_file():
        return None
    entry = json.loads(manifest.read_text())["policies"].get(task)
    if not entry or entry.get("status") == "none" or not entry.get("decoder"):
        return None
    decoder = Path(root) / entry["decoder"]
    if not decoder.is_file():
        return None
    encoder = Path(root) / entry["encoder"] if entry.get("encoder") else None
    if encoder is not None and not encoder.is_file():
        return None
    return {"decoder": str(decoder), "encoder": str(encoder) if encoder else None}
