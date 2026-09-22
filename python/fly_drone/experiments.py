"""Bounded, foreground experiment supervision; subprocess success is not science."""

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def _write(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read(path):
    return json.loads(path.read_text())


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError("identifiers must contain only letters, numbers, _, - or .")
    return value


def _fingerprint(cwd, identities):
    """Hash tracked source and declared artifacts, never arbitrary ignored files."""
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=cwd).decode().split("\0")
    )
    paths = {
        cwd / name
        for name in names
        if name and not name.startswith(("docs/results/", "runs/"))
    }
    untracked = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "python",
                "tests",
                "scripts",
                "web",
                "src",
            ],
            cwd=cwd,
        )
        .decode()
        .split("\0")
    )
    source_extensions = {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".c",
        ".cpp",
        ".h",
        ".rs",
        ".sh",
        ".html",
        ".css",
    }
    paths.update(
        cwd / name
        for name in untracked
        if name and Path(name).suffix in source_extensions
    )
    for name in identities:
        path = (cwd / name).resolve()
        if not path.exists():
            raise ValueError(f"missing identity: {path}")
        paths.update(path.rglob("*") if path.is_dir() else [path])
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_dir():
            continue
        digest.update(str(path).encode())
        digest.update(path.read_bytes() if path.exists() else b"<deleted>")
    return {
        "sha256": digest.hexdigest(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode()
        .strip(),
    }


def _manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def validate(manifest):
    for key in ("run_id", "stage"):
        _name(manifest[key])
    for key in ("hypothesis", "decision_rules"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"{key} must be preregistered text")
    workers = manifest.get("workers", 1)
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 6
    ):
        raise ValueError("workers must be 1..6")
    limit = manifest.get("max_seconds", 7200)
    if (
        not isinstance(limit, (float, int))
        or not math.isfinite(limit)
        or not 0 < limit <= 7200
    ):
        raise ValueError("max_seconds must be >0 and <=7200")
    if not isinstance(manifest.get("identities"), list) or not all(
        isinstance(path, str) for path in manifest["identities"]
    ):
        raise ValueError("identities must list explicit source/artifact paths")
    trials = manifest["trials"]
    if (
        not isinstance(trials, list)
        or not trials
        or len({_name(t["id"]) for t in trials}) != len(trials)
    ):
        raise ValueError("trials require unique IDs")
    for trial in trials:
        if (
            not isinstance(trial.get("argv"), list)
            or not trial["argv"]
            or not all(isinstance(x, str) for x in trial["argv"])
        ):
            raise ValueError("argv must be a nonempty string list")
        if not isinstance(trial.get("output"), str):
            raise ValueError("trial output JSON path required")
        criterion = trial.get("criterion")
        if criterion is not None:
            if criterion.get("op") not in ("eq", "gte", "lte") or not criterion.get(
                "field"
            ):
                raise ValueError("criterion needs dotted field and eq/gte/lte op")
            if "value" not in criterion:
                raise ValueError("criterion value required")
    outputs = [
        str((Path(manifest.get("cwd", Path.cwd())) / t["output"]).resolve())
        for t in trials
    ]
    if len(set(outputs)) != len(outputs):
        raise ValueError("trial outputs must be unique")


def status(run_id, root="runs/experiments"):
    return _read(Path(root) / _name(run_id) / "status.json")


def stop(run_id, root="runs/experiments"):
    directory = Path(root) / _name(run_id)
    state = _read(directory / "status.json")
    # Never signal a stored PID: the live supervisor owns and verifies its child.
    (directory / "stop.request").touch()
    return {"run_id": run_id, "stop_requested": True, "state": state["state"]}


def _criterion(data, criterion):
    value = data
    for part in criterion["field"].split("."):
        value = value[part]
    target = criterion["value"]
    if criterion["op"] == "eq":
        return value == target
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("criterion requires finite numeric result")
    return value >= target if criterion["op"] == "gte" else value <= target


def run(manifest_path=None, *, run_id=None, root="runs/experiments"):
    """Start/resume synchronously. A global advisory lock serializes heavy runs."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    initial = (
        _read(Path(manifest_path))
        if manifest_path is not None
        else _read(root / _name(run_id) / "manifest.json")
    )
    workdir = Path(initial.get("cwd", Path.cwd())).resolve()
    repository = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=workdir)
        .decode()
        .strip()
    )
    coordinator = repository / "runs/experiments"
    coordinator.mkdir(parents=True, exist_ok=True)
    with (coordinator / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "another experiment is running in this repository"
            ) from error
        if manifest_path is not None:
            manifest = _read(Path(manifest_path))
            validate(manifest)
            cwd = Path(manifest.get("cwd", Path.cwd())).resolve()
            manifest["cwd"] = str(cwd)
            directory = root / manifest["run_id"]
            identity = _fingerprint(cwd, manifest["identities"])
            directory.mkdir()  # Never overwrite a prior run.
            _write(directory / "manifest.json", manifest)
            state = {
                "run_id": manifest["run_id"],
                "stage": manifest["stage"],
                "state": "created",
                "elapsed_seconds": 0.0,
                "identity": identity,
                "manifest_sha256": _manifest_hash(manifest),
                "completed": {},
                "current_trial": None,
            }
            _write(directory / "status.json", state)
        else:
            directory = root / _name(run_id)
            manifest = _read(directory / "manifest.json")
            validate(manifest)
            cwd = Path(manifest["cwd"])
            state = _read(directory / "status.json")
            if state["manifest_sha256"] != _manifest_hash(manifest):
                raise ValueError("manifest changed; cannot resume")
            if state["identity"] != _fingerprint(cwd, manifest["identities"]):
                raise ValueError("source/artifact identity mismatch; cannot resume")
            for trial_id, result in state["completed"].items():
                output = Path(result["output"])
                if (
                    not output.exists()
                    or hashlib.sha256(output.read_bytes()).hexdigest()
                    != result["sha256"]
                ):
                    raise ValueError(f"completed output changed: {trial_id}")
            if state["state"] == "running":
                raise ValueError(
                    "unclean supervisor exit: inspect owned workers before recovery"
                )
            (directory / "stop.request").unlink(missing_ok=True)
        registry_path = coordinator / ".registry.json"
        registered = set(_read(registry_path) if registry_path.exists() else [])
        registered.update(str(path.parent) for path in root.glob("*/status.json"))
        registered.add(str(directory))
        _write(registry_path, sorted(registered))
        for name in registered:
            path = Path(name) / "status.json"
            if (
                path.exists()
                and path.parent != directory
                and _read(path)["state"] == "running"
            ):
                raise ValueError(
                    "unclean run registered; inspect owned workers before starting another"
                )
        other_seconds = sum(
            item["elapsed_seconds"]
            for name in registered
            if (path := Path(name) / "status.json").exists()
            if path.parent != directory
            and (item := _read(path))["stage"] == manifest["stage"]
        )
        return _supervise(directory, manifest, state, other_seconds)


def _supervise(directory, manifest, state, other_seconds):
    started = time.monotonic()
    previous = state["elapsed_seconds"]
    cancelled = False
    child = None
    old_handlers = {}

    def cancel(*_):
        nonlocal cancelled
        cancelled = True

    def publish(event=None):
        state["elapsed_seconds"] = previous + time.monotonic() - started
        state["heartbeat"] = time.time()
        state["completed_count"] = len(state["completed"])
        state["total_trials"] = len(manifest["trials"])
        _write(directory / "status.json", state)
        if event:
            with (directory / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "time": state["heartbeat"],
                            "event": event,
                            "trial": state["current_trial"],
                        }
                    )
                    + "\n"
                )

    def interruption():
        elapsed = previous + time.monotonic() - started
        if cancelled or (directory / "stop.request").exists():
            return "cancelled"
        if (
            elapsed >= manifest.get("max_seconds", 7200)
            or elapsed + other_seconds >= 21600
        ):
            return "budget_exhausted"
        return None

    def terminate():
        if child is not None and child.returncode is None:
            # Child remains unreaped while waiting, so its process-group ID cannot
            # be recycled. Never use PIDs from a previous supervisor invocation.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            # Retain the leader as a zombie if it exits first, until descendants
            # receive the final signal. This prevents process-group ID reuse.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and _group_live(child.pid):
                time.sleep(0.05)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, cancel)
    state["state"] = "running"
    state.pop("reason", None)
    publish("started")
    try:
        for trial in manifest["trials"]:
            if trial["id"] in state["completed"]:
                continue
            reason = interruption()
            if reason:
                state["state"] = reason
                break
            output = (Path(manifest["cwd"]) / trial["output"]).resolve()
            if output.exists():
                raise ValueError(f"unregistered existing output (preserved): {output}")
            state["current_trial"] = trial["id"]
            publish("trial_started")
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["FLY_DRONE_EXPERIMENT_WORKERS"] = str(
                manifest.get("workers", 1)
            )
            with (directory / f"{trial['id']}.log").open("a") as log:
                child = subprocess.Popen(
                    trial["argv"],
                    cwd=manifest["cwd"],
                    env=environment,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                last_heartbeat = time.monotonic()
                # WNOWAIT observes termination without reaping the group leader.
                # Its PID therefore remains reserved until group cleanup completes.
                while (
                    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    is None
                ):
                    reason = interruption()
                    if reason:
                        terminate()
                        state["state"] = reason
                        break
                    if time.monotonic() - last_heartbeat >= 1:
                        publish()
                        last_heartbeat = time.monotonic()
                    time.sleep(0.05)
                if state["state"] != "running":
                    if output.exists():
                        partial = output.with_name(
                            f"{output.name}.partial.{time.time_ns()}"
                        )
                        output.rename(partial)
                        state.setdefault("partial_outputs", []).append(str(partial))
                    break
                terminate()
                if child.returncode != 0:
                    raise ValueError(f"trial {trial['id']} exited {child.returncode}")
            data = _read(output)
            scientific = (
                _criterion(data, trial["criterion"]) if trial.get("criterion") else None
            )
            state["completed"][trial["id"]] = {
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "criterion_passed": scientific,
            }
            publish("trial_completed")
        else:
            if state["identity"] != _fingerprint(
                Path(manifest["cwd"]), manifest["identities"]
            ):
                raise ValueError("source/artifact identity changed during execution")
            results = [item["criterion_passed"] for item in state["completed"].values()]
            state["state"] = (
                "failed"
                if False in results
                else "incomplete"
                if None in results
                else "passed"
            )
            if None in results:
                state["reason"] = (
                    "execution complete; scientific decision requires review"
                )
    except Exception as error:
        terminate()
        state["state"] = "incomplete"
        state["reason"] = str(error)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        state["current_trial"] = None
        publish("finished")
        _write(directory / "report.json", state)
    return state


def _group_live(group):
    """Linux /proc inspection only; no process IDs read here are signalled."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False
