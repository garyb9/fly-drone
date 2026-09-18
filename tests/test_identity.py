import json

import pytest
from fly_drone.identity import (
    canonical_model,
    check_bundle_pin,
    hashes,
    is_alternate,
    load_identity,
)


def _write_bundle(tmp_path, manifest, graph=b"\x01\x02\x03", neurons=b"\x04\x05\x06"):
    (tmp_path / "graph.bin").write_bytes(graph)
    (tmp_path / "neurons.bin").write_bytes(neurons)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_dataset_hash_covers_graph_only_but_bundle_hash_covers_neurons(tmp_path):
    manifest = {"n_neurons": 3, "w_norm": 0.01, "model": "LIF"}
    _write_bundle(tmp_path, manifest, graph=b"graph", neurons=b"neurons")
    dataset_hash, bundle_hash = hashes(b"graph", b"neurons", manifest)

    _write_bundle(tmp_path, manifest, graph=b"graph", neurons=b"mutated")
    dataset_hash2, bundle_hash2 = hashes(b"graph", b"mutated", manifest)

    assert dataset_hash2 == dataset_hash
    assert bundle_hash2 != bundle_hash


def test_bundle_hash_covers_graph_and_model_fields(tmp_path):
    manifest = {"n_neurons": 3, "w_norm": 0.01, "model": "LIF"}
    _, base = hashes(b"graph", b"neurons", manifest)

    _, graph_changed = hashes(b"graph!", b"neurons", manifest)
    _, model_changed = hashes(b"graph", b"neurons", {**manifest, "model": "other"})

    assert graph_changed != base
    assert model_changed != base


def test_bundle_hash_ignores_volatile_manifest_fields():
    manifest = {"n_neurons": 3, "model": "LIF", "sources": ["a"]}
    _, base = hashes(b"graph", b"neurons", manifest)
    _, with_sources = hashes(
        b"graph", b"neurons", {**manifest, "sources": ["b"], "credit": "x"}
    )
    assert with_sources == base


def test_canonical_model_is_key_order_independent():
    a = {"model": "LIF", "n_neurons": 3, "w_norm": 0.01}
    b = {"w_norm": 0.01, "n_neurons": 3, "model": "LIF"}
    assert canonical_model(a) == canonical_model(b)


def test_alternate_detection():
    assert not is_alternate({"n_neurons": 3, "model": "LIF"})
    assert is_alternate({"sign_convention": "s2"})
    assert is_alternate({"wiring": "rewired"})


def test_load_identity_reads_a_bundle_directory(tmp_path):
    manifest = {"n_neurons": 3, "w_norm": 0.01, "model": "LIF"}
    _write_bundle(tmp_path, manifest)
    dataset_hash, bundle_hash, alternate = load_identity(tmp_path)
    assert (dataset_hash, bundle_hash) == hashes(
        b"\x01\x02\x03", b"\x04\x05\x06", manifest
    )
    assert alternate is False


def test_check_bundle_pin_ignores_canonical_bundles():
    check_bundle_pin({"encoder_version": "x"}, "bundle", alternate=False)


def test_check_bundle_pin_requires_a_matching_hash_on_alternates():
    check_bundle_pin({"bundle_hash": "bundle"}, "bundle", alternate=True)
    with pytest.raises(ValueError, match="alternate bundle"):
        check_bundle_pin({"bundle_hash": "other"}, "bundle", alternate=True)
    with pytest.raises(ValueError, match="alternate bundle"):
        check_bundle_pin({}, "bundle", alternate=True)
