from pathlib import Path

import pytest
from fly_drone.cli import _bridge_codec_adapter, _resolve_adapter, _roam_controller


def test_resolve_adapter_aliases_the_committed_codecs():
    from fly_drone.adapter import DEFAULT_PATH, DEFAULT_V2_PATH

    assert _resolve_adapter("v1") == str(DEFAULT_PATH)
    assert _resolve_adapter("v2") == str(DEFAULT_V2_PATH)
    assert _resolve_adapter("/tmp/x.json") == "/tmp/x.json"
    assert _resolve_adapter(None) is None


def test_bridge_codec_adapter_defaults_to_v2():
    from fly_drone.adapter import DEFAULT_V2_PATH

    assert _bridge_codec_adapter("v2") == str(DEFAULT_V2_PATH)
    with pytest.raises(ValueError, match="unknown codec"):
        _bridge_codec_adapter("v9")


def test_roam_controller_expands_the_adapter_codec():
    from fly_drone.adapter import DEFAULT_PATH, DEFAULT_V2_PATH

    assert _roam_controller("teacher", "v2") == "teacher"
    assert _roam_controller("random", "v2") == "random"
    # Bare ``adapter`` flies the --codec default; an explicit pin wins.
    assert _roam_controller("adapter", "v2") == f"adapter:{DEFAULT_V2_PATH}"
    assert _roam_controller("adapter", "v1") == f"adapter:{DEFAULT_PATH}"
    assert _roam_controller("adapter:v1", "v2") == f"adapter:{DEFAULT_PATH}"
    assert _roam_controller("adapter:v2", "v1") == f"adapter:{DEFAULT_V2_PATH}"
    pinned = _roam_controller("adapter:/tmp/a.json", "v2")
    assert pinned == "adapter:/tmp/a.json"


def test_roam_controller_routes_bypass_and_policy_paths(tmp_path):
    actor = tmp_path / "actor.zip"
    actor.write_bytes(b"")
    assert _roam_controller(str(actor), "v2") == f"bypass:{Path(actor).resolve()}"
    policy = tmp_path / "actor.json"
    policy.write_text("{}")
    assert _roam_controller(str(policy), "v2") == f"policy:{Path(policy).resolve()}"
