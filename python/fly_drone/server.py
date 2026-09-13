import asyncio
import base64
import io
import queue
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .brain import ROOT
from .env import ConnectomeEnv
from .fly import FlyMirror
from .plant import LIMITS


class Session:
    def __init__(self, policy=None):
        self.policy = policy
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
            env.reset(seed=42)
            fly = FlyMirror()
            if self.policy:
                env.brain.load_policy(self.policy)
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
                            env.reset(seed=int(c.get("seed", 42)))
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
                    command = env.brain.infer() / LIMITS if self.policy else np.zeros(4)
                    _, _, done, truncated, _ = env.step(command)
                    for r in env.trace:
                        fly.step(r)
                    if done or truncated:
                        paused = True
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


def make_app(policy=None):
    session = Session(policy)

    @asynccontextmanager
    async def lifespan(app):
        session.start()
        yield
        session.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": session.latest is not None, "error": session.error}

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
