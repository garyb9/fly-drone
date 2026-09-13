import time

import pytest
from fastapi.testclient import TestClient
from fly_drone.server import make_app


def test_service_pause_reset_and_disconnect():
    with TestClient(make_app()) as client:
        deadline = time.monotonic() + 30
        while not client.get("/health").json()["ready"]:
            assert client.get("/health").json()["error"] is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        with client.websocket_connect("/ws") as ws:
            metadata = ws.receive_json()
            assert metadata["type"] == "metadata" and metadata["neurons"] == 166700
            frame = ws.receive_json()
            assert frame["type"] == "frame"
            ws.send_json({"op": "pause", "value": True})
            while not (frame := ws.receive_json())["paused"]:
                pass
            tick = frame["tick"]
            fly = frame["fly"]
            next_frame = ws.receive_json()
            assert next_frame["tick"] == tick and next_frame["fly"] == fly
            ws.send_json({"op": "reset", "seed": 99})
            while (frame := ws.receive_json())["episode"] == 0:
                pass
            assert frame["episode"] == 1 and frame["fly"]["ticks"] <= 16
            last = frame["physics_tick"]
        time.sleep(0.2)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            assert ws.receive_json()["physics_tick"] > last


def test_service_rejects_unrelated_origin():
    from starlette.websockets import WebSocketDisconnect

    with TestClient(make_app()) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws", headers={"origin": "https://unrelated.example"}
            ):
                pass


def test_replay_reset_task_ablation_and_policy_guard():
    with TestClient(make_app()) as client:
        deadline = time.monotonic() + 30
        while not client.get("/health").json()["ready"]:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert isinstance(client.get("/api/reports").json(), list)
        with client.websocket_connect("/ws") as ws:
            from fly_drone.env import TASKS

            metadata = ws.receive_json()
            assert metadata["tasks"] == list(TASKS)
            assert metadata["room"]["kind"] == "legacy"
            assert len(metadata["room"]["walls"]) == 3
            ws.send_json(
                {"op": "reset", "seed": 1003, "task": "looming", "ablation": "sensory"}
            )
            while (frame := ws.receive_json())["episode"] == 0:
                pass
            assert frame["task"] == "looming" and frame["ablation"] == "sensory"
            assert frame["seed"] == 1003 and "obstacle_distance" in frame["outcome"]
            ws.send_json({"op": "reset", "seed": 5, "task": "free_roam"})
            while (message := ws.receive_json()).get("type") != "room" or message[
                "kind"
            ] != "arena":
                pass
            assert len(message["walls"]) == 4 and len(message["bands"]) == 4
            assert len(message["pillars"]) == 16 and message["half_size"] == 8.0
            ws.send_json({"op": "reset", "policy": "/etc/passwd.json"})
            while not (frame := ws.receive_json())["error"]:
                pass
            assert "runs/" in frame["error"]
