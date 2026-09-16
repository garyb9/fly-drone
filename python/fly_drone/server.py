import asyncio
import base64
import io
import json
import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .arena import LEVELS, clearance
from .attribution import for_brain
from .brain import ROOT
from .env import PATHWAYS, TASKS, ConnectomeEnv, EpisodeTracker
from .fly import FlyMirror

ABLATION_MODES = ("none", "zero", "sensory", "shuffle", "light", "loom", "ghost")
POLICY_ROOTS = (ROOT / "runs", ROOT / "docs" / "results")


def safe_policy_path(path):
    """Only JSON actors inside runs/ or docs/results/ may be loaded over the socket."""
    resolved = (
        Path(path).resolve() if Path(path).is_absolute() else (ROOT / path).resolve()
    )
    if resolved.suffix != ".json" or not any(
        resolved.is_relative_to(root.resolve()) for root in POLICY_ROOTS
    ):
        raise ValueError("policy must be a .json file under runs/ or docs/results/")
    if not resolved.is_file():
        raise ValueError("policy file not found")
    return resolved


def accepted_policies(manifest=None, root=ROOT):
    """Accepted actor per task from the committed manifest: (present, missing) paths."""
    manifest = Path(manifest or root / "docs" / "results" / "accepted-policies.json")
    present, missing = {}, {}
    for task, entry in json.loads(manifest.read_text())["policies"].items():
        path = Path(root) / entry["actor"]
        (present if path.is_file() else missing)[task] = str(path)
    return present, missing


def list_reports():
    """Evaluation reports with per-seed outcomes, for the viewer's replay panel."""
    reports = []
    for root in POLICY_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("evaluation*.json")):
            try:
                data = json.loads(path.read_text())
                modes = data["modes"]
            except (ValueError, KeyError, OSError):
                continue
            policy = Path(data.get("policy", ""))
            try:
                policy = policy.resolve().relative_to(ROOT)
            except ValueError:
                pass
            reports.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "policy": str(policy),
                    "task": data.get("task", "visual"),
                    "seconds": data.get("seconds"),
                    "acceptance": {
                        k: v
                        for k, v in data.get("acceptance", {}).items()
                        if isinstance(v, bool | int | float)
                    },
                    "modes": {
                        mode: [
                            {
                                "seed": r["seed"],
                                "success": r["success"],
                                "side": r.get("side", r.get("target_side")),
                            }
                            for r in summary["runs"]
                        ]
                        for mode, summary in modes.items()
                    },
                }
            )
    return reports


class Session:
    def __init__(self, policy=None, looming_policy=None, task_policies=None):
        self.policy = policy
        self.task_policies = {task: None for task in TASKS}
        self.task_policies.update(task_policies or {})
        self.task_policies["visual"] = policy
        if looming_policy:
            self.task_policies["looming"] = looming_policy
        unknown = set(self.task_policies) - set(TASKS)
        if unknown:
            raise ValueError(f"unknown task policies: {sorted(unknown)}")
        self.commands = queue.Queue(maxsize=64)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.metadata = None
        self.room = None
        self.room_seq = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, command):
        self.commands.put_nowait(command)

    def publish_room(self, room):
        """Room geometry changes only on reset (legacy room <-> free-roam arena)."""
        with self.lock:
            self.room = room
            self.room_seq += 1

    def run(self):
        env = None
        try:
            env = ConnectomeEnv()
            fly = FlyMirror()
            task_policies = self.task_policies
            active_policy = None
            attributor = None
            policy_limits = None
            silenced = set()
            events = deque(maxlen=20)

            def use_policy(path):
                nonlocal active_policy, attributor, policy_limits
                if path and path != active_policy:
                    env.brain.load_policy(path)
                    attributor = for_brain(path, env.brain)
                    policy_limits = json.loads(Path(path).read_text())["action_limits"]
                active_policy = path

            def apply_silencing():
                # restore() clears every silence; re-apply the condition, then probes.
                env.brain.core.restore()
                if env.ablation == "sensory":
                    env.brain.silence_sensors()
                elif env.ablation in PATHWAYS:
                    env.brain.silence_inputs(PATHWAYS[env.ablation])
                for name in sorted(silenced):
                    env.brain.silence_inputs(PATHWAYS[name])

            use_policy(self.policy)
            seed = 42
            _, info = env.reset(seed=seed)
            tracker = EpisodeTracker(env.task, info)
            result = None
            cells = env.brain.cells
            # Render measured somata only. Keep feature cells and selected visual cells.
            ids = sorted(set(env.brain.feature_ids + list(range(0, len(cells), 100))))
            ids = [i for i in ids if cells[i]["measured"]]
            idx = {i: j for j, i in enumerate(ids)}
            raw = (env.brain.data / "graph.bin").read_bytes()
            n = len(cells)
            offsets = np.frombuffer(raw, dtype="<u4", count=n + 1, offset=32)
            edges = int.from_bytes(raw[16:24], "little")
            targets = np.frombuffer(
                raw, dtype="<u4", count=edges, offset=32 + 4 * (n + 1)
            )
            # Round-robin one edge per source neuron per pass, instead of draining each
            # source's edges before moving to the next: iterating `ids` in ascending order
            # and stopping at the first 1000 links meant only the first few dozen
            # (lowest-index) neurons ever contributed an edge, leaving every other rendered
            # neuron looking synapse-less regardless of where it actually sits in the brain.
            LINK_BUDGET = 1000
            edge_iters = {
                src: iter(
                    int(dst)
                    for dst in targets[offsets[src] : offsets[src + 1]]
                    if int(dst) in idx
                )
                for src in ids
            }
            links = []
            active = list(edge_iters)
            while active and len(links) < LINK_BUDGET:
                still_active = []
                for src in active:
                    dst = next(edge_iters[src], None)
                    if dst is None:
                        continue
                    links.append([idx[src], idx[dst]])
                    still_active.append(src)
                    if len(links) >= LINK_BUDGET:
                        break
                active = still_active
            self.metadata = {
                "ids": ids,
                "cells": [cells[i] for i in ids],
                "links": links,
                "groups": env.brain.groups,
                "neurons": len(cells),
                "features": len(env.brain.feature_ids),
                "policy": "trained" if self.policy else "PID baseline",
                "tasks": list(TASKS),
                "ablations": list(ABLATION_MODES),
                "levels": sorted(LEVELS),
                "dataset_hash": env.brain.dataset_hash,
                "room": env.plant.room(),
                "task_policy_status": {
                    task: "loaded" if path else "none"
                    for task, path in self.task_policies.items()
                },
            }
            self.publish_room(self.metadata["room"])
            episode = 0
            paused = False
            seq = 0
            deadline = time.perf_counter()
            start = deadline
            missed = 0
            last_error = None
            policy_status = "loaded" if active_policy else "none"
            explained = None
            while not self.stop.is_set():
                while True:
                    try:
                        c = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        op = c.get("op")
                        if op == "pause":
                            paused = bool(c["value"])
                        elif op == "reset":
                            task = c.get("task", env.task)
                            ablation = c.get("ablation", "none")
                            if task not in TASKS or ablation not in ABLATION_MODES:
                                raise ValueError("unknown task or ablation")
                            policy = c.get("policy")
                            if policy:
                                use_policy(str(safe_policy_path(policy)))
                            else:
                                use_policy(task_policies[task] or active_policy)
                            seed = int(c.get("seed", 42))
                            level = int(c.get("level", env.level))
                            if level not in LEVELS:
                                raise ValueError("unknown arena level")
                            env.task = task
                            env.ablation = ablation
                            env.level = level
                            # Free roam is continuous: crashes respawn, nothing ends it.
                            env.respawn = task == "free_roam"
                            _, info = env.reset(seed=seed)
                            tracker = EpisodeTracker(env.task, info)
                            result = None
                            apply_silencing()
                            events.clear()
                            self.publish_room(env.plant.room())
                            fly.reset()
                            episode += 1
                            paused = False
                            start = time.perf_counter()
                        elif op == "objects":
                            env.plant.set_objects(c.get("target"), c.get("obstacle"))
                        elif op in ("silence", "pulse", "hold"):
                            cell = int(c["index"])
                            if not 0 <= cell < len(cells):
                                raise ValueError("invalid neuron index")
                            if op == "silence":
                                env.brain.core.silence([cell], True)
                            elif op == "pulse":
                                env.brain.core.stimulate([cell], 1.5)
                            else:
                                env.interventions[cell] = 1.5
                        elif op in (
                            "place_beacon",
                            "launch_threat",
                            "pathway",
                            "ghost",
                        ):
                            if env.roam is None:
                                raise ValueError(
                                    "free-roam probes need the free_roam task"
                                )
                            if op == "place_beacon":
                                xy = np.array([float(c["x"]), float(c["y"])])
                                spec = env.spec
                                if (
                                    not np.isfinite(xy).all()
                                    or np.any(np.abs(xy) > spec.inner)
                                    or clearance(spec, env.plant.pillars, xy)
                                    < spec.beacon_radius + 0.3
                                ):
                                    raise ValueError(
                                        "beacon must be on free floor inside the arena"
                                    )
                                env.plant.set_objects(target=[xy[0], xy[1], 1.0])
                                env.roam["beacon_hidden"] = False
                            elif op == "launch_threat":
                                if not env.launch_threat():
                                    raise ValueError(
                                        "a threat is already flying or its path is blocked"
                                    )
                            elif op == "pathway":
                                name = c["name"]
                                if name not in PATHWAYS:
                                    raise ValueError("pathway must be light or loom")
                                if c.get("silenced", True):
                                    silenced.add(name)
                                else:
                                    silenced.discard(name)
                                apply_silencing()
                            else:
                                env.plant.set_ghost(bool(c["value"]))
                        elif op == "restore":
                            env.brain.core.restore()
                            env.interventions.clear()
                            silenced.clear()
                            apply_silencing()
                        else:
                            raise ValueError("unknown command")
                        last_error = None
                    except (ValueError, KeyError, TypeError) as exc:
                        last_error = str(exc)
                if not paused:
                    # observe() applies the zero/shuffle ablations exactly as evaluation.
                    observed = env.observe()
                    if not active_policy:
                        policy_status = "none"
                    elif not np.allclose(policy_limits, env.plant.limits):
                        # A legacy-room actor would be mis-scaled in the arena: hold.
                        policy_status = "limits mismatch"
                    else:
                        policy_status = "loaded"
                    if policy_status == "loaded":
                        command = env.brain.infer(observed) / env.plant.limits
                        explained = attributor.explain(observed)
                    else:
                        command = np.zeros(4)
                        explained = None
                    _, _, done, truncated, info = env.step(command)
                    tracker.update(info)
                    for r in env.trace:
                        fly.step(r)
                    if env.roam is not None:
                        events.extend(
                            {**e, "time": round(float(info["time"]), 2)}
                            for e in info["events"]
                        )
                    if done or (truncated and env.task != "free_roam"):
                        paused = True
                        finished = tracker.finish(done)
                        result = {
                            k: finished[k]
                            for k in ("success", "terminated", "collision")
                        }
                camera = []
                for frame in env.plant.images:
                    b = io.BytesIO()
                    Image.fromarray(frame).save(b, format="JPEG", quality=75)
                    camera.append(base64.b64encode(b.getvalue()).decode())
                seq += 1
                now = time.perf_counter()
                frame = {
                    "version": 1,
                    "seq": seq,
                    "episode": episode,
                    "tick": env.brain.tick,
                    "physics_tick": env.plant.step_counter,
                    "time": env.plant.data.time,
                    "paused": paused,
                    "state": env.plant.state(),
                    "fly": fly.state(),
                    "activity": env.brain.core.activity(ids),
                    "readouts": env.brain.read(),
                    "cues": env.brain.cues.tolist(),
                    "command": env.command.tolist(),
                    "cameras": camera,
                    "error": last_error,
                    "task": env.task,
                    "ablation": env.ablation,
                    "seed": seed,
                    "active_policy": str(Path(active_policy).name)
                    if active_policy
                    else None,
                    "outcome": {
                        "bearing": info["bearing"],
                        "target_distance": info["target_distance"],
                        "climb": info["climb"],
                        "obstacle_distance": info["obstacle_distance"],
                        "launched": info["launched"],
                        "displacement": info["displacement"],
                        "result": result,
                    },
                    "real_time_factor": env.plant.data.time / max(0.001, now - start),
                    "missed_deadlines": missed,
                    "policy_status": policy_status,
                    "attribution": explained,
                    "free_roam": None
                    if env.roam is None
                    else {
                        "level": env.level,
                        "beacons": info["beacons_collected"],
                        "collisions": info["collisions"],
                        "collision_kinds": info["collision_kinds"],
                        "threats_finished": len(info["threats"]),
                        "threats_dodged": sum(t["dodged"] for t in info["threats"]),
                        "threats_hit": sum(t["hit"] for t in info["threats"]),
                        "visited_cells": info["visited_cells"],
                        "beacon_visible": info["beacon_visible"],
                        "clearance": info["clearance"],
                        "ghost": env.plant.ghost,
                        "silenced": sorted(silenced),
                        "events": list(events),
                    },
                }
                with self.lock:
                    self.latest = frame
                deadline += 0.04
                if now > deadline:
                    missed += 1
                    deadline = now
                self.stop.wait(max(0, deadline - time.perf_counter()))
        except Exception as exc:
            import traceback

            self.error = str(exc)
            traceback.print_exc()
        finally:
            if env:
                env.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


def make_app(policy=None, looming_policy=None, task_policies=None, port=8000):
    session = Session(policy, looming_policy, task_policies)

    @asynccontextmanager
    async def lifespan(app):
        session.start()
        yield
        session.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": session.latest is not None, "error": session.error}

    @app.get("/api/reports")
    def reports():
        return list_reports()

    @app.websocket("/ws")
    async def socket(ws: WebSocket):
        # Local-only service; reject unrelated websites attempting local control.
        origin = ws.headers.get("origin", "")
        if origin and origin not in {
            f"http://{host}:{p}"
            for host in ("127.0.0.1", "localhost")
            for p in (port, 5173)
        }:
            await ws.close(code=1008)
            return
        await ws.accept()

        async def receive():
            while True:
                message = await ws.receive_json()
                if not isinstance(message, dict):
                    continue
                try:
                    session.submit(message)
                except queue.Full:
                    pass

        reader = asyncio.create_task(receive())
        try:
            while session.metadata is None:
                if session.error:
                    await ws.send_json({"error": session.error})
                    return
                await asyncio.sleep(0.1)
            await ws.send_json({"type": "metadata", **session.metadata})
            last = -1
            # metadata already carries the room; a newer one is sent when resets change it.
            last_room = 1
            while not reader.done():
                with session.lock:
                    frame = session.latest
                if session.error:
                    await ws.send_json({"error": session.error})
                    return
                with session.lock:
                    room, room_seq = session.room, session.room_seq
                if room_seq != last_room:
                    await ws.send_json({"type": "room", **room})
                    last_room = room_seq
                if frame and frame["seq"] != last:
                    await asyncio.wait_for(
                        ws.send_json({"type": "frame", **frame}), timeout=5
                    )
                    last = frame["seq"]
                await asyncio.sleep(0.02)
        except (WebSocketDisconnect, TimeoutError, RuntimeError):
            pass
        finally:
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                pass

    dist = ROOT / "web/dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="viewer")
    return app
