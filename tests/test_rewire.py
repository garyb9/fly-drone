import shutil
from pathlib import Path

import numpy as np
import pytest
from fly_drone import rewire
from fly_drone.identity import load_identity

FIXTURE = Path(__file__).resolve().parents[1] / "pipeline/out/fixture"


def _source(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(FIXTURE, src)
    return src


def _graph(path):
    raw = (path / "graph.bin").read_bytes()
    return raw, rewire.read_graph(raw)


def test_shuffle_preserves_out_degree_weights_and_header_but_changes_bundles(tmp_path):
    src = _source(tmp_path)
    rewire.rewire_bundle(src, tmp_path / "out", mode="shuffle", seed=1)
    raw0, (n0, e0, off0, tgt0, wt0, _) = _graph(src)
    raw1, (n1, e1, off1, tgt1, wt1, _) = _graph(tmp_path / "out")

    assert (n1, e1) == (n0, e0)
    np.testing.assert_array_equal(off1, off0)
    np.testing.assert_array_equal(wt1, wt0)
    assert raw1[:32] == raw0[:32]
    # Out-degree per source is exact; the in-degree histogram is a permutation.
    np.testing.assert_array_equal(
        np.bincount(tgt0, minlength=n0), np.bincount(tgt1, minlength=n1)
    )
    assert not np.array_equal(tgt1, tgt0)

    d0, b0, alt0 = load_identity(src)
    d1, b1, alt1 = load_identity(tmp_path / "out")
    assert alt0 is False and alt1 is True
    assert d1 != d0 and b1 != b0


def test_swap_preserves_in_and_out_degree(tmp_path):
    src = _source(tmp_path)
    rewire.rewire_bundle(src, tmp_path / "out", mode="swap", seed=2, rounds=30)
    _, (n0, e0, off0, tgt0, wt0, _) = _graph(src)
    _, (n1, e1, off1, tgt1, wt1, _) = _graph(tmp_path / "out")

    assert (n1, e1) == (n0, e0)
    np.testing.assert_array_equal(off1, off0)
    np.testing.assert_array_equal(wt1, wt0)
    np.testing.assert_array_equal(np.diff(off0), np.diff(off1))
    # Both degree sequences are preserved exactly, unlike a global shuffle.
    np.testing.assert_array_equal(
        np.bincount(tgt0, minlength=n0), np.bincount(tgt1, minlength=n1)
    )
    assert not np.array_equal(tgt1, tgt0)


def test_existing_bundle_requires_force(tmp_path):
    src = _source(tmp_path)
    out = tmp_path / "out"
    rewire.rewire_bundle(src, out, mode="shuffle", seed=0)
    with pytest.raises(FileExistsError):
        rewire.rewire_bundle(src, out, mode="shuffle", seed=0)
    rewire.rewire_bundle(src, out, mode="shuffle", seed=0, force=True)


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        rewire.rewire_bundle(_source(tmp_path), tmp_path / "out", mode="nope")
