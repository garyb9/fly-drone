"""Rewire a project graph bundle while preserving its degree sequence.

Reads and writes the committed CSR ``graph.bin`` format
(``crates/brain-core/src/core/format.rs``). Only edge destinations change; every
neuron, edge weight and source out-degree is kept. Used by
``scripts/make_rewired_bundle.py`` as a diagnostic null model for the causal claim
that the connectome's specific wiring does the work.

See ``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream B.
"""

import json
import shutil
from pathlib import Path

import numpy as np

U32 = np.dtype("<u4")
U64 = np.dtype("<u8")
I16 = np.dtype("<i2")


def read_graph(raw):
    """(n_nodes, edge_count, offsets, targets, weights, targets_byte_offset)."""
    n_nodes = int(np.frombuffer(raw, U32, 1, 8)[0])
    edge_count = int(np.frombuffer(raw, U64, 1, 16)[0])
    offsets = np.frombuffer(raw, U32, n_nodes + 1, 32)
    tbase = 32 + 4 * (n_nodes + 1)
    targets = np.frombuffer(raw, U32, edge_count, tbase)
    weights = np.frombuffer(raw, I16, edge_count, tbase + 4 * edge_count)
    return n_nodes, edge_count, offsets, targets, weights, tbase


def edge_sources(offsets):
    return np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))


def target_shuffle(targets, rng):
    """Permute destinations globally: exact out-degree, changed in-degree."""
    return rng.permutation(targets).astype(U32)


def edge_swap(targets, src, n_nodes, rng, rounds=20, batch=1_000_000):
    """Swap destinations between edge pairs, preserving in- and out-degree.

    A batch-internal collision can introduce a parallel edge; the format allows it
    and it does not affect the diagnostic. Returns ``(targets, swaps_applied)``.
    """
    targets = targets.copy()
    edges = targets.size
    n = np.int64(n_nodes)
    accepted = 0
    for _ in range(rounds):
        e1 = np.unique(rng.integers(0, edges, size=min(batch, edges), dtype=np.int64))
        e2 = rng.integers(0, edges, size=e1.size, dtype=np.int64)
        e2 = e2[~np.isin(e2, e1)]
        _, uniq = np.unique(e2, return_index=True)
        e1, e2 = e1[uniq], e2[uniq]
        a, b = src[e1], targets[e1]
        c, d = src[e2], targets[e2]
        keep = (a != d) & (c != b) & (b != d) & (a != c)
        e1, e2, a, b, c, d = (x[keep] for x in (e1, e2, a, b, c, d))
        if e1.size == 0:
            continue
        n1, n2 = a * n + d, c * n + b
        existing = np.sort(src * n + targets)
        p1 = np.clip(np.searchsorted(existing, n1), 0, existing.size - 1)
        p2 = np.clip(np.searchsorted(existing, n2), 0, existing.size - 1)
        ok = ~((existing[p1] == n1) | (existing[p2] == n2))
        e1, e2, b, d = e1[ok], e2[ok], b[ok], d[ok]
        targets[e1] = d.astype(U32)
        targets[e2] = b.astype(U32)
        accepted += int(e1.size)
    return targets, accepted


def rewire_bundle(
    source, out, mode="swap", seed=0, rounds=20, batch=1_000_000, force=False
):
    source, out = Path(source), Path(out)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} exists; pass force=True to overwrite")
    raw = (source / "graph.bin").read_bytes()
    n_nodes, edge_count, offsets, targets, weights, tbase = read_graph(raw)
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        new_targets, accepted = target_shuffle(targets, rng), edge_count
    elif mode == "swap":
        new_targets, accepted = edge_swap(
            targets, edge_sources(offsets), n_nodes, rng, rounds, batch
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    out.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.name != "graph.bin":
            shutil.copy2(item, out / item.name)
    patched = bytearray(raw)
    patched[tbase : tbase + 4 * edge_count] = new_targets.astype(U32).tobytes()
    (out / "graph.bin").write_bytes(bytes(patched))

    manifest = json.loads((source / "manifest.json").read_text())
    manifest["wiring"] = "rewired"
    manifest["rewire"] = {"mode": mode, "seed": seed, "swaps": accepted}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "n_nodes": n_nodes,
        "n_edges": edge_count,
        "mode": mode,
        "swaps": accepted,
        "out": str(out),
    }
