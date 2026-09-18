"""Bundle identity: which model a decoder is allowed to run against.

``dataset_hash`` is the legacy graph-only hash, kept so every accepted actor keeps
loading unchanged. ``bundle_hash`` additionally covers ``neurons.bin`` (where each
neuron's inhibitory flag lives) and the manifest fields that define the model, so a
sign or wiring variant cannot silently load an actor trained on another convention.

See ``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` Step 0.
"""

import hashlib
import json
from pathlib import Path

# Manifest fields that define the model. Volatile fields (source URLs and their
# hashes, measured-somata counts, credit text) are excluded so a re-download that
# yields identical model bytes yields an identical identity.
MODEL_KEYS = (
    "version",
    "n_neurons",
    "core_count",
    "n_edges",
    "w_norm",
    "model",
    "selection",
    "wiring",
    "sign_convention",
)

# A bundle carrying any of these keys is an alternate, non-canonical model.
ALTERNATE_KEYS = ("wiring", "sign_convention")


def canonical_model(manifest):
    """Canonical bytes for the model-defining manifest fields present."""
    payload = {k: manifest[k] for k in MODEL_KEYS if k in manifest}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def is_alternate(manifest):
    return any(k in manifest for k in ALTERNATE_KEYS)


def hashes(graph, neurons, manifest):
    """(dataset_hash, bundle_hash) for raw bundle bytes and a parsed manifest."""
    dataset_hash = hashlib.sha256(graph).hexdigest()
    h = hashlib.sha256()
    h.update(graph)
    h.update(neurons)
    h.update(canonical_model(manifest))
    return dataset_hash, h.hexdigest()


def load_identity(data):
    """(dataset_hash, bundle_hash, alternate) for a bundle directory."""
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    dataset_hash, bundle_hash = hashes(
        (data / "graph.bin").read_bytes(),
        (data / "neurons.bin").read_bytes(),
        manifest,
    )
    return dataset_hash, bundle_hash, is_alternate(manifest)


def check_bundle_pin(policy, bundle_hash, alternate):
    """Refuse to run a policy on an alternate bundle it does not pin."""
    if not alternate:
        return
    pinned = policy.get("bundle_hash")
    if pinned != bundle_hash:
        raise ValueError(
            f"actor bundle_hash {pinned!r} does not match alternate bundle "
            f"{bundle_hash!r}"
        )
