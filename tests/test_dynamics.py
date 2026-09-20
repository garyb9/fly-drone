import importlib.util
import struct

import numpy as np
import pytest
from fly_drone.brain import ROOT, BrainRuntime
from fly_drone.dynamics import (
    DT_MS,
    EXCITABLE_PARAMS,
    MAGIC,
    SHIU_PARAMS,
    VERSION,
    encode,
    excitable_arrays,
    sha256,
    shiu_leak,
    shiu_threshold,
    uniform_arrays,
)

SCRIPT = ROOT / "scripts" / "make_dynamics_bundle.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("make_dynamics_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dynamics_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("bundle") / "malecns-dynamics"
    _, marker = _load_builder().build(ROOT / "data" / "malecns", output)
    return output, marker


def test_shiu_prior_is_declared_and_stable():
    assert SHIU_PARAMS["v_threshold_mv"] - SHIU_PARAMS["v_resting_mv"] == 7.0
    assert shiu_threshold() == pytest.approx(1.0)
    assert shiu_leak() == pytest.approx(np.exp(-DT_MS / 20.0))
    assert SHIU_PARAMS["refrac_ms"] == 2.2


def test_encode_layout():
    leak, threshold = uniform_arrays(5)
    payload = encode(leak, threshold)
    magic, version, n = struct.unpack("<III", payload[:12])
    assert (magic, version, n) == (MAGIC, VERSION, 5)
    assert len(payload) == 12 + 5 * 8
    assert np.allclose(np.frombuffer(payload[12:], dtype="<f4")[0::2], leak)


def test_dynamics_bundle_is_additive_and_alternate(dynamics_bundle):
    output, marker = dynamics_bundle
    canonical = BrainRuntime()
    bundle = BrainRuntime(data=output)
    assert bundle.dataset_hash == canonical.dataset_hash
    assert bundle.bundle_hash != canonical.bundle_hash
    assert bundle.alternate and not canonical.alternate
    assert bundle.feedback_ids == {}
    raw = (output / "dynamics.bin").read_bytes()
    assert marker["sha256"] == sha256(raw)
    assert marker["n_neurons"] == len(bundle.cells)
    assert marker["noise_sigma"] == 0.0


def test_excitable_prior_scales_only_the_threshold():
    leak, threshold = excitable_arrays(7, 0.85)
    assert np.allclose(leak, shiu_leak())
    assert np.allclose(threshold, 0.85 * shiu_threshold())


def test_excitable_bundle_is_declared_additive_and_calibratable(tmp_path):
    builder = _load_builder()
    output = tmp_path / "malecns-excitable"
    _, marker = builder.build(ROOT / "data" / "malecns", output, "excitable", 0.85)
    assert marker["version"] == EXCITABLE_PARAMS["version"]
    assert marker["threshold_scale"] == 0.85
    assert marker["noise_sigma"] == 0.02
    bundle = BrainRuntime(data=output)
    assert bundle.alternate and bundle.bundle_hash != BrainRuntime().bundle_hash


def test_builder_rejects_an_unknown_prior(tmp_path):
    with pytest.raises(ValueError, match="unknown prior"):
        _load_builder().build(ROOT / "data" / "malecns", tmp_path / "x", "invented")
