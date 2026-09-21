import time

import pytest
from fastapi.testclient import TestClient
from fly_drone.server import make_app


def next_frame(ws):
    """Telemetry frames interleave with room messages sent after resets."""
    while (message := ws.receive_json()).get("type") != "frame":
        pass
    return message


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
            assert metadata["codecs"] == ["v1", "v2", "v3"]
            next_frame(ws)
            ws.send_json({"op": "pause", "value": True})
            while not (frame := next_frame(ws))["paused"]:
                pass
            tick = frame["tick"]
            fly = frame["fly"]
            following = next_frame(ws)
            assert following["tick"] == tick and following["fly"] == fly
            ws.send_json({"op": "reset", "seed": 99})
            while (frame := next_frame(ws))["episode"] == 0:
                pass
            assert frame["episode"] == 1 and frame["fly"]["ticks"] <= 16
            # The shipped bridge is v3 (C4a); the codec op switches to v1 and resets the sim.
            assert frame["codec"] == "v3"
            ws.send_json({"op": "codec", "value": "v1"})
            while (frame := next_frame(ws))["codec"] != "v1":
                pass
            assert frame["episode"] == 2
            last = frame["physics_tick"]
        time.sleep(0.2)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            assert next_frame(ws)["physics_tick"] > last


def test_accepted_policy_manifest_splits_present_and_missing(tmp_path):
    import json

    from fly_drone.env import TASKS
    from fly_drone.server import accepted_policies

    (tmp_path / "runs" / "a").mkdir(parents=True)
    (tmp_path / "runs" / "a" / "actor.json").write_text("{}")
    manifest = tmp_path / "accepted.json"
    manifest.write_text(
        json.dumps(
            {
                "policies": {
                    "visual": {"actor": "runs/a/actor.json"},
                    "looming": {"actor": "runs/b/actor.json"},
                }
            }
        )
    )
    present, missing = accepted_policies(manifest, tmp_path)
    assert list(present) == ["visual"] and list(missing) == ["looming"]
    committed = accepted_policies()
    assert set(committed[0]) | set(committed[1]) <= set(TASKS)


def test_metadata_reports_task_policy_status(tmp_path):
    import json

    actor = tmp_path / "actor.json"
    actor.write_text(
        json.dumps({"encoder_version": "x", "action_limits": [1, 1, 1, 1]})
    )
    with TestClient(make_app(task_policies={"free_roam": str(actor)})) as client:
        deadline = time.monotonic() + 30
        while not client.get("/health").json()["ready"]:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        with client.websocket_connect("/ws") as ws:
            metadata = ws.receive_json()
            assert metadata["task_policy_status"]["free_roam"] == "loaded"
            assert metadata["task_policy_status"]["visual"] == "none"


def test_session_bridge_selection_is_explicit():
    from fly_drone.server import Session

    assert Session().bridge == "declared"
    assert Session(task_policies={"free_roam": "x"}).bridge == "learned"
    assert (
        Session(bridge="declared", task_policies={"free_roam": "x"}).bridge
        == "declared"
    )
    assert Session(bridge="learned").bridge == "learned"
    with pytest.raises(ValueError, match="bridge"):
        Session(bridge="bogus")


def test_session_defaults_to_the_shipped_v3_bridge():
    from fly_drone.adapter import DEFAULT_V3_PATH
    from fly_drone.server import Session, codec_name

    assert Session().adapter == str(DEFAULT_V3_PATH)
    assert codec_name(Session().adapter) == "v3"
    assert codec_name(str(DEFAULT_V3_PATH)) == "v3"
    assert codec_name(None) is None


def test_service_rejects_unrelated_origin():
    from starlette.websockets import WebSocketDisconnect

    with TestClient(make_app()) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws", headers={"origin": "https://unrelated.example"}
            ):
                pass


def test_replay_reset_task_ablation_and_policy_guard():
    from fly_drone.env import TASKS

    with TestClient(make_app()) as client:
        deadline = time.monotonic() + 30
        while not client.get("/health").json()["ready"]:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert isinstance(client.get("/api/reports").json(), list)
        with client.websocket_connect("/ws") as ws:
            metadata = ws.receive_json()
            assert metadata["tasks"] == list(TASKS)
            assert metadata["room"]["kind"] == "legacy"
            assert len(metadata["room"]["walls"]) == 3
            ws.send_json(
                {"op": "reset", "seed": 1003, "task": "looming", "ablation": "sensory"}
            )
            while (frame := next_frame(ws))["episode"] == 0:
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
            frame = next_frame(ws)
            # Default bridge is the declared adapter: loaded, but no learned attribution.
            assert frame["policy_status"] == "loaded" and frame["attribution"] is None
            assert frame["active_policy"] == "declared-adapter"
            assert (
                frame["free_roam"]["level"] == 3 and frame["free_roam"]["beacons"] == 0
            )
            # Derived live-scoreboard fields share roam_eval's units.
            assert frame["attribution_seq"] == 0
            roam = frame["free_roam"]
            assert roam["elapsed"] >= 0.0
            assert roam["beacons_per_min"] == 0.0 and roam["collisions_per_min"] == 0.0
            assert 0.0 <= roam["coverage"] <= 1.0
            ws.send_json({"op": "launch_threat"})
            ws.send_json({"op": "pathway", "name": "loom", "silenced": True})
            ws.send_json({"op": "pathway", "name": "sensory", "silenced": True})
            ws.send_json({"op": "ghost", "value": True})
            ws.send_json({"op": "place_beacon", "x": 99.0, "y": 0.0})
            while not (frame := next_frame(ws))["error"]:
                pass
            assert "beacon" in frame["error"]
            roam = frame["free_roam"]
            assert roam["silenced"] == ["loom", "sensory"] and roam["ghost"] is True
            assert any(e["type"] == "threat_launched" for e in roam["events"])
            assert frame["outcome"]["launched"]
            ws.send_json({"op": "reset", "policy": "/etc/passwd.json"})
            # The earlier beacon error persists until a command replaces it.
            while "runs/" not in (next_frame(ws)["error"] or ""):
                pass
