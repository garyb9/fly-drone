import asyncio
import base64
import io
import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .brain import ROOT
from .env import ConnectomeEnv
from .fly import FlyMirror
from .plant import LIMITS
from .training import MAX_DISPLACEMENT_AT_THREAT, THREAT_RANGE

TASKS = ("visual", "looming")
ABLATION_MODES = ("none", "zero", "sensory", "shuffle")
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
    def __init__(self, policy=None, looming_policy=None):
        self.policy = policy
        self.looming_policy = looming_policy
        self.commands = queue.Queue(maxsize=64)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.metadata = None
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, command):
        self.commands.put_nowait(command)

    def run(self):
        env = None
        try:
            env = ConnectomeEnv()
            fly = FlyMirror()
            task_policies = {"visual": self.policy, "looming": self.looming_policy}
            active_policy = None

            def use_policy(path):
                nonlocal active_policy
                if path and path != active_policy:
                    env.brain.load_policy(path)
                active_policy = path

            use_policy(self.policy)
            seed = 42
            _, info = env.reset(seed=seed)
            initial_bearing = abs(info["bearing"])
            threat_displacement = None
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
            links = []
            for src in ids:
                for dst in targets[offsets[src] : offsets[src + 1]]:
                    if int(dst) in idx:
                        links.append([idx[src], idx[int(dst)]])
                    if len(links) >= 1000:
                        break
                if len(links) >= 1000:
                    break
            self.metadata = {
                "ids": ids,
                "cells": [cells[i] for i in ids],
                "links": links,
                "neurons": len(cells),
                "features": len(env.brain.feature_ids),
                "policy": "trained" if self.policy else "PID baseline",
                "tasks": list(TASKS),
                "ablations": list(ABLATION_MODES),
                "dataset_hash": env.brain.dataset_hash,
            }
            episode = 0
            paused = False
            seq = 0
            deadline = time.perf_counter()
            start = deadline
            missed = 0
            last_error = None
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
                            env.task = task
                            env.ablation = ablation
                            _, info = env.reset(seed=seed)
                            initial_bearing = abs(info["bearing"])
                            threat_displacement = None
                            result = None
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
                        elif op == "restore":
                            env.brain.core.restore()
                            env.interventions.clear()
                        else:
                            raise ValueError("unknown command")
                        last_error = None
                    except (ValueError, KeyError, TypeError) as exc:
                        last_error = str(exc)
                if not paused:
                    # observe() applies the zero/shuffle ablations exactly as evaluation.
                    if active_policy:
                        command = env.brain.infer(env.observe()) / LIMITS
                    else:
                        command = np.zeros(4)
                    _, _, done, truncated, info = env.step(command)
                    for r in env.trace:
                        fly.step(r)
                    if (
                        info["launched"]
                        and threat_displacement is None
                        and info["obstacle_distance"] < THREAT_RANGE
                    ):
                        threat_displacement = info["displacement"]
                    if done or truncated:
                        paused = True
                        if env.task == "looming":
                            moved = (
                                info["displacement"]
                                if threat_displacement is None
                                else threat_displacement
                            )
                            success = not done and moved < MAX_DISPLACEMENT_AT_THREAT
                        else:
                            success = not done and abs(info["bearing"]) < (
                                0.5 * initial_bearing
                            )
                        result = {
                            "success": bool(success),
                            "terminated": bool(done),
                            "collision": bool(info["collision"]),
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
                        "obstacle_distance": info["obstacle_distance"],
                        "launched": info["launched"],
                        "displacement": info["displacement"],
                        "result": result,
                    },
                    "real_time_factor": env.plant.data.time / max(0.001, now - start),
                    "missed_deadlines": missed,
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


def make_app(policy=None, looming_policy=None):
    session = Session(policy, looming_policy)

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
        if origin and origin not in (
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ):
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
            while not reader.done():
                with session.lock:
                    frame = session.latest
                if session.error:
                    await ws.send_json({"error": session.error})
                    return
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
