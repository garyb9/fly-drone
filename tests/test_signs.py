import json

import numpy as np
import pytest
from fly_drone import signs
from fly_drone.identity import load_identity

REC = np.dtype(
    [
        ("id", "<u8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("group", "<u2"),
        ("flags", "u1"),
        ("pad", "u1"),
    ]
)


def _neurons(flags_by_body):
    raw = bytearray(16)
    np.frombuffer(raw, np.uint32, 1, 0)[0] = 0x4E594C46
    count = len(flags_by_body)
    np.frombuffer(raw, np.uint32, 1, 8)[0] = count
    for body, flags in flags_by_body.items():
        rec = np.zeros(1, REC)[0]
        rec["id"], rec["flags"] = body, flags
        raw += rec.tobytes()
    return bytes(raw)


def _flags(neurons):
    count = int(np.frombuffer(neurons, np.uint32, 1, 8)[0])
    arr = np.frombuffer(neurons, REC, count, 16)
    return {int(b): int(f) for b, f in zip(arr["id"], arr["flags"], strict=True)}


# ACh exc, Glu inh, GABA inh, histamine exc, an unlabelled inhibitory neuron.
BASE = {1: 0b0000, 2: 0b0010, 3: 0b0010, 4: 0b0000, 5: 0b0011}
LABELS = {1: "acetylcholine", 2: "glutamate", 3: "gaba", 4: "histamine"}


def test_conventions_are_the_documented_sets():
    assert signs.convention_inhibits("glutamate", "s1")
    assert not signs.convention_inhibits("histamine", "s1")
    assert signs.convention_inhibits("histamine", "s2")
    assert not signs.convention_inhibits("glutamate", "s3")
    assert signs.convention_inhibits("gaba", "s3")


def test_s1_is_the_identity_and_preserves_other_flag_bits():
    out = signs.retag_flags(_neurons(BASE), LABELS, "s1")
    assert _flags(out) == BASE


def test_s2_makes_histamine_inhibitory():
    out = _flags(signs.retag_flags(_neurons(BASE), LABELS, "s2"))
    assert out[4] & signs.INHIBITORY_BIT
    assert out[1] == BASE[1] and out[5] == BASE[5]


def test_s3_makes_glutamate_excitatory():
    out = _flags(signs.retag_flags(_neurons(BASE), LABELS, "s3"))
    assert not out[2] & signs.INHIBITORY_BIT
    assert out[3] & signs.INHIBITORY_BIT  # GABA stays inhibitory
    assert out[5] & signs.INHIBITORY_BIT  # unlabelled kept


def test_unknown_convention_rejected():
    with pytest.raises(ValueError, match="unknown convention"):
        signs.retag_flags(_neurons(BASE), LABELS, "s9")


def test_build_sign_bundle_marks_an_alternate_identity(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "neurons.bin").write_bytes(_neurons(BASE))
    (src / "graph.bin").write_bytes(b"graph")
    (src / "manifest.json").write_text(json.dumps({"n_neurons": 5, "model": "LIF"}))
    signs.build_sign_bundle(src, out, "s2", LABELS)
    _, bundle_a, alt_a = load_identity(src)
    _, bundle_b, alt_b = load_identity(out)
    assert alt_a is False and alt_b is True
    assert bundle_a != bundle_b  # neurons.bin changed and sign_convention is modelled
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sign_convention"] == "s2"
