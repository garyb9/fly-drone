"""P2 (C2): declared per-neuron LIF dynamics, as an additive bundle.

The spec's P2 is to fit per-neuron leak/threshold to recorded activity with the
wiring frozen. No recordings are in this repository yet, so this module lands the
*infrastructure* and a **declared** prior: the published LIF parameters of
Shiu et al., *Nature* 2024 ("A Drosophila computational brain model reveals
sensorimotor processing"), mapped into this simulator's units. The mapping is a
declared interpretation, not a fit:

| Shiu-2024            | value   | here                                             |
| -------------------- | ------- | ------------------------------------------------ |
| ``V_resting``        | -52 mV  | resting = 0                                      |
| ``V_threshold``      | -45 mV  | ``v_threshold = 1.0`` (7 mV above rest, unit)    |
| ``T_mbr=C_mbr*R_mbr``| 20 ms   | ``leak = exp(-5/20)``                            |
| ``T_refractory``     | 2.2 ms  | ``refrac_ms = 2.2``                              |
| noise                | none    | ``noise_sigma = 0.0`` (deterministic)            |
| ``W_syn``            | 0.275 mV| not mapped: our graph carries raw synapse counts |

The per-neuron arrays are emitted even though this prior is uniform, so that
replacing them with a recording-fitted array is the only change a future P2 fit
needs. The bundle is alternate (its manifest carries ``dynamics``), so the
canonical bundle and every accepted actor are untouched.
"""

import hashlib
import struct

import numpy as np

MAGIC = 0x4459_4C46  # "FLYD" LE
VERSION = 1
DT_MS = 5.0

SHIU_PARAMS = {
    "version": "shiu-2024-lif-v1",
    "source": "Shiu et al., Nature 634:210-219 (2024), doi:10.1038/s41586-024-07763-9",
    "v_resting_mv": -52.0,
    "v_reset_mv": -52.0,
    "v_threshold_mv": -45.0,
    "tau_m_ms": 20.0,
    "refrac_ms": 2.2,
    "noise_sigma": 0.0,
    "w_syn_mv": 0.275,
    "tau_syn_ms": 5.0,
}


def shiu_threshold():
    """Threshold above rest, normalised so the canonical unit threshold is 1.0."""
    above = SHIU_PARAMS["v_threshold_mv"] - SHIU_PARAMS["v_resting_mv"]
    return above / 7.0


def shiu_leak():
    return float(np.exp(-DT_MS / SHIU_PARAMS["tau_m_ms"]))


def uniform_arrays(n):
    """The declared prior as per-neuron arrays (uniform, ready to be replaced)."""
    leak = np.full(n, shiu_leak(), dtype=np.float32)
    threshold = np.full(n, shiu_threshold(), dtype=np.float32)
    return leak, threshold


# Declared excitability prior (P3 follow-up). The model is threshold-bound: with raw contact
# weights and a unit threshold, the network sits near spike threshold and weak sensory drive
# does not propagate (scripts/readout_audit.py; external-prior-art.md fly.ai finding #2). This
# declared prior scales the per-neuron threshold down by a fixed factor so typical input can
# fire. It is a declared interpretation, not fitted to activity, and carries its own version.
EXCITABLE_PARAMS = {
    "version": "declared-excitable-v1",
    "source": "declared threshold-scale prior; no activity fit",
    "threshold_scale": 0.5,
    "v_reset": 0.0,
    "refrac_ms": 2.0,
    "noise_sigma": 0.02,
}


def excitable_arrays(n, scale=EXCITABLE_PARAMS["threshold_scale"]):
    """Leak unchanged; per-neuron threshold scaled by a declared excitability factor."""
    leak = np.full(n, shiu_leak(), dtype=np.float32)
    threshold = np.full(n, shiu_threshold() * float(scale), dtype=np.float32)
    return leak, threshold


def encode(leak, v_threshold):
    """``dynamics.bin`` bytes: 12-byte header + n * (leak f32, v_threshold f32)."""
    leak = np.asarray(leak, dtype="<f4")
    v_threshold = np.asarray(v_threshold, dtype="<f4")
    if leak.shape != v_threshold.shape:
        raise ValueError("leak and v_threshold arrays must have equal length")
    header = struct.pack("<III", MAGIC, VERSION, leak.size)
    body = np.empty(leak.size * 2, dtype="<f4")
    body[0::2] = leak
    body[1::2] = v_threshold
    return header + body.tobytes()


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def globals_dict():
    return {
        "v_reset": 0.0,
        "refrac_ms": SHIU_PARAMS["refrac_ms"],
        "noise_sigma": SHIU_PARAMS["noise_sigma"],
    }
