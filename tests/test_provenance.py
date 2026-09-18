"""Guard the vendored fly-playground references against accidental edits.

``docs/source-provenance.json`` records the SHA-256 of every copied original. Some
copies were later adapted (and are tracked as such), but the TypeScript body/scene
reference under ``web/src/fly/`` and ``web/src/app/config.ts`` is reference-only:
the running viewer renders the server's fly body (``python/fly_drone/fly.py``), so
those files must stay byte-for-byte identical to the recorded original. This test
fails loudly if a future cleanup edits or deletes them.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "docs" / "source-provenance.json"

# Manifest sources whose copies are reference-only rather than subsequently adapted.
VENDORED_REFERENCE_PREFIXES = ("fly-playground/src/body/",)
VENDORED_REFERENCE_FILES = {"fly-playground/src/app/config.ts"}


def _recorded_files():
    return json.loads(PROVENANCE.read_text())["files"]


def test_every_recorded_destination_exists():
    missing = [
        entry["destination"]
        for entry in _recorded_files()
        if not (ROOT / entry["destination"]).exists()
    ]
    assert missing == []


def test_destinations_are_unique():
    destinations = [entry["destination"] for entry in _recorded_files()]
    assert len(destinations) == len(set(destinations))


def test_vendored_references_are_unmodified():
    for entry in _recorded_files():
        source = entry["source"]
        reference_only = source.startswith(VENDORED_REFERENCE_PREFIXES) or (
            source in VENDORED_REFERENCE_FILES
        )
        if not reference_only:
            continue
        path = ROOT / entry["destination"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], (
            f"{entry['destination']} differs from its vendored original {source}; "
            "do not edit it. The runtime fly body is python/fly_drone/fly.py; update "
            "docs/source-provenance.json only if the copy is intentionally re-synced."
        )
