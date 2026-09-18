"""Transmitter-sign conventions as additive bundle variants.

The canonical MaleCNS bundle encodes sign as one flag bit per neuron
(`neurons.bin`, ``is_inhibitory``), and the current mapping is exactly
``inhibitory = {glutamate, gaba}`` (verified against the
``body-neurotransmitters-male-cns-v1.0`` consensus). Surveyed projects disagree on
glutamate and histamine, so this module can retag the flags under alternate
conventions without touching edges, positions or group ids.

Building a variant needs the per-body transmitter labels from the raw feather; see
``scripts/make_sign_bundle.py``. See
``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream E.
"""

import json
import shutil
from pathlib import Path

import numpy as np

# Neurotransmitters that make a neuron inhibitory, per convention.
CONVENTIONS = {
    "s1": frozenset({"gaba", "glutamate"}),  # canonical (this project)
    "s2": frozenset({"gaba", "glutamate", "histamine"}),  # fly.ai
    "s3": frozenset({"gaba"}),  # FlyGM: ACh/Glu/ASP/His excitatory, GABA/Gly inhibitory
}
INHIBITORY_BIT = 0b0010
RECORD = 24
HEADER = 16


def convention_inhibits(label, convention):
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}")
    return label in CONVENTIONS[convention]


def retag_flags(neurons, labels, convention):
    """Return ``neurons.bin`` bytes with the inhibitory bit set per convention.

    Neurons whose body id is absent from ``labels`` keep their canonical flag.
    """
    out = bytearray(neurons)
    count = int(np.frombuffer(neurons, np.dtype("<u4"), 1, 8)[0])
    for k in range(count):
        offset = HEADER + k * RECORD
        body = int(np.frombuffer(neurons[offset : offset + 8], np.dtype("<u8"), 1)[0])
        label = labels.get(body)
        if label is None:
            continue
        flag = out[offset + 22]
        if convention_inhibits(label, convention):
            out[offset + 22] = flag | INHIBITORY_BIT
        else:
            out[offset + 22] = flag & ~INHIBITORY_BIT
    return bytes(out)


def build_sign_bundle(source, out, convention, labels, force=False):
    source, out = Path(source), Path(out)
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}")
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} exists; pass force=True to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.name != "neurons.bin":
            shutil.copy2(item, out / item.name)
    neurons = (source / "neurons.bin").read_bytes()
    (out / "neurons.bin").write_bytes(retag_flags(neurons, labels, convention))

    manifest = json.loads((source / "manifest.json").read_text())
    manifest["sign_convention"] = convention
    manifest["sign"] = {"inhibitory": sorted(CONVENTIONS[convention])}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"convention": convention, "out": str(out)}
