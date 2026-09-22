"""Additive evidence gate; never changes pre-registered behavioral arithmetic.

Legacy ``acceptance.passed`` is a behavioral score, not permission to promote an
artifact. Consumers of new reports must use ``completeness.eligible_for_promotion``.
"""

import hashlib
import math
import subprocess
from pathlib import Path


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_provenance(policy, bundle=None, encoder=None, relay=False):
    """Fingerprint runtime inputs without instantiating another brain.

    Missing identities are recorded as errors, never inferred from a filename.
    The source fingerprint includes tracked changes and untracked source files.
    Call before and after an evaluation to detect concurrent source/input edits.
    """
    from . import _brain
    from .adapter import DEFAULT_BRIDGE_PATH
    from .brain import DATA, ENCODER_VERSION, ROOT

    errors = []
    files = {}
    data = Path(bundle) if bundle else DATA
    required = (
        "manifest.json",
        "graph.bin",
        "neurons.bin",
        "cells.json",
        "groups.json",
        "sensory-mappings.json",
    )
    paths = {f"bundle/{name}": data / name for name in required}
    paths["brain_runtime"] = Path(_brain.__file__)
    for name in ("dynamics.bin", "feedback-mappings.json"):
        if (data / name).exists():
            paths[f"bundle/{name}"] = data / name
    if policy == "adapter":
        paths["controller"] = DEFAULT_BRIDGE_PATH
    elif policy.startswith(("adapter:", "policy:")):
        paths["controller"] = Path(policy.split(":", 1)[1])
    else:
        errors.append("controller identity unavailable")
    if encoder is not None:
        paths["encoder"] = Path(encoder)
    for name, path in paths.items():
        try:
            files[name] = _digest(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    source = {}
    try:

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE
            )

        source["revision"] = git("rev-parse", "HEAD").decode().strip()
        names = (
            git(
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "python",
                "crates",
                "web",
                "Cargo.toml",
                "Cargo.lock",
                "pyproject.toml",
                "uv.lock",
            )
            .decode()
            .split("\0")
        )
        fingerprints = []
        for name in sorted(set(filter(None, names))):
            path = ROOT / name
            fingerprints.append((name, _digest(path) if path.is_file() else "absent"))
        source["fingerprint"] = hashlib.sha256(repr(fingerprints).encode()).hexdigest()
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"source provenance: {exc}")
    return {
        "source": source,
        "files": files,
        "sensory": "artifact" if encoder else ENCODER_VERSION,
        "relay": bool(relay),
        "errors": errors,
    }


def report_completeness(
    report, *, expected_conditions, expected_seeds, probe_names, realtime=None
):
    """Check evidence coverage independently of score values.

    ``realtime`` must be a live end-to-end measurement with positive
    ``simulated_seconds``, ``wall_seconds``, and a nonempty ``evidence`` reference.
    Offline worker throughput is deliberately not accepted as live A7 evidence.
    Provenance consists of matching ``before``/``after`` captures in the report.
    """
    seeds = list(expected_seeds)
    issues = []
    if not seeds or len(set(seeds)) != len(seeds):
        issues.append("expected seed list must be nonempty and unique")

    def finite(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    def check_runs(label, runs):
        observed = [r.get("seed") for r in runs]
        if len(observed) != len(seeds) or set(observed) != set(seeds):
            issues.append(f"{label}: missing, duplicate, or unexpected seeds")
        if any(r.get("status", "completed") != "completed" for r in runs):
            issues.append(f"{label}: unfinished trials")

    results = report.get("results") or {}
    for condition in expected_conditions:
        if condition not in results:
            issues.append(f"missing condition: {condition}")
        summary = results.get(condition, {})
        runs = summary.get("runs", [])
        check_runs(condition, runs)
        if not all(
            finite(summary.get(k))
            for k in (
                "beacons_per_min",
                "collisions_per_min",
                "mean_visited_cells",
                "slow_fraction",
                "mean_abs_yaw_bias",
            )
        ):
            issues.append(f"{condition}: summary metrics unavailable or nonfinite")
        if any(
            not all(
                finite(r.get(k))
                for k in (
                    "beacons_per_min",
                    "collisions_per_min",
                )
            )
            or not isinstance(r.get("collision_kinds"), dict)
            or not isinstance(r.get("threat_log"), list)
            for r in runs
        ):
            issues.append(f"{condition}: trial metrics unavailable or nonfinite")
    probes = report.get("probes") or {}
    for name in probe_names:
        probe = probes.get(name, {})
        runs = probe.get("runs", [])
        check_runs(f"probe {name}", runs)
        if {r.get("side") for r in runs} != {"left", "right"}:
            issues.append(f"probe {name}: both sides required")
        if not runs or any(not isinstance(r.get("success"), bool) for r in runs):
            issues.append(f"probe {name}: success observations missing")
        if not finite(probe.get("balanced")):
            issues.append(f"probe {name}: balanced score unavailable")
    for condition in expected_conditions:
        if condition.endswith("|ghost") or condition == f"{report.get('policy')}|none":
            sides = set()
            for run in results.get(condition, {}).get("runs", []):
                for threat in run.get("threat_log", []):
                    if threat.get("hit") or threat.get("min_distance", math.inf) < 2:
                        side = threat.get("side", 0)
                        if side:
                            sides.add("left" if side > 0 else "right")
            if sides != {"left", "right"}:
                issues.append(f"{condition}: near threats on both sides required")

    provenance = report.get("provenance") or {}
    before, after = provenance.get("before"), provenance.get("after")
    provenance_complete = bool(
        before
        and after
        and before == after
        and not before.get("errors")
        and before.get("files", {}).get("controller")
        and all(
            before.get("files", {}).get(f"bundle/{name}")
            for name in (
                "manifest.json",
                "graph.bin",
                "neurons.bin",
                "cells.json",
                "groups.json",
                "sensory-mappings.json",
            )
        )
        and before.get("sensory")
        and before.get("source", {}).get("revision")
        and before.get("source", {}).get("fingerprint")
    )
    timing = realtime or {}
    numbers = [timing.get(k) for k in ("simulated_seconds", "wall_seconds")]
    realtime_complete = bool(
        timing.get("mode") == "live_end_to_end"
        and timing.get("evidence")
        and all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            and v > 0
            for v in numbers
        )
    )
    factor = numbers[0] / numbers[1] if realtime_complete else None
    a7 = {
        "available": realtime_complete,
        "real_time_factor": factor,
        "passed": factor >= 1 if realtime_complete else None,
        "evidence": timing if realtime_complete else None,
        "note": "live end-to-end measurement"
        if realtime_complete
        else "unavailable: offline throughput is not live A7 evidence",
    }
    behavior_complete = not issues
    if not provenance_complete:
        issues.append("provenance missing, invalid, or changed during evaluation")
    if not realtime_complete:
        issues.append("live A7 evidence unavailable")
    return {
        "version": 1,
        "behavioral_complete": behavior_complete,
        "provenance_complete": provenance_complete,
        "realtime_complete": realtime_complete,
        "complete": behavior_complete and provenance_complete and realtime_complete,
        "A7": a7,
        "eligible_for_promotion": bool(
            behavior_complete
            and provenance_complete
            and realtime_complete
            and a7["passed"]
            and report.get("acceptance", {}).get("passed")
        ),
        "issues": issues,
    }
