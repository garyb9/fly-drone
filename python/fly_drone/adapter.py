"""Declared body adapter: a fixed neural readout to ``[vx, vy, vz, yaw_rate]``.

P0 of ``docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md``.
The codec is a declared function of neural activity only. There is no teacher, no
learned policy and no pose: the constants are fixed by calibration on a declared
stimulus battery and carried with an identity, exactly like an actor's
``encoder_version``/``dataset_hash``.

The two readouts the codec uses:

- **laterality**: mean left-side minus mean right-side activity over the 2,022
  descending + VNC motor traces. Vision drives one side, so the sign says which.
- **escape**: the ``escape`` readout (giant-fibre motor cells), which fires on
  looming and is the brain's own climb signal.

The command is one fixed formula. When calm it steers toward the more active side
(a light); when ``escape`` fires it flips to steer away, sidesteps and climbs. This
is a declared reflex-level interpretation of the connectome's activity, not a hidden
mode switch: the same constants run for every frame of a run.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .brain import ROOT, BrainRuntime
from .identity import check_bundle_pin

DEFAULT_PATH = ROOT / "docs" / "results" / "adapter" / "adapter.json"
# Escape readout level above rest that counts as a full loom response (rest ~0, loom
# ~0.74-0.87 in the battery). Declared, not fitted.
ESCAPE_SCALE = 0.5
# Sidestep gain as a fraction of the steering command. Declared, not fitted.
SIDE_FRACTION = 0.5

# Declared stimulus battery: (name, injected v4 cues [light_l, light_r, loom_l, loom_r],
# target body command [vx, vy, vz, yaw_rate]). Targets are fly ethology, not a teacher:
# light drives forward and a turn toward it; loom drives a turn away and a climb; rest
# holds. ``vy`` is left at zero (sidestep is a declared fraction of steering).
STIMULI = [
    ("rest", [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]),
    ("light_l", [2, 0, 0, 0], [0.5, 0.0, 0.0, 1.0]),
    ("light_r", [0, 2, 0, 0], [0.5, 0.0, 0.0, -1.0]),
    ("loom_l", [0, 0, 2, 0], [0.0, 0.0, 1.0, -1.0]),
    ("loom_r", [0, 0, 0, 2], [0.0, 0.0, 1.0, 1.0]),
    ("light_both", [2, 2, 0, 0], [0.5, 0.0, 0.0, 0.0]),
    ("loom_both", [0, 0, 2, 2], [0.0, 0.0, 1.0, 0.0]),
]
# Battery settling ticks: matches scripts/stimulus_battery.py so the two diagnostics
# read the same brain state.
SETTLE_TICKS = 80


def side_masks(brain):
    """Indices into ``brain.features()`` for left- and right-side cells (cached)."""
    masks = getattr(brain, "_adapter_masks", None)
    if masks is None:
        left, right = [], []
        for j, i in enumerate(brain.feature_ids):
            side = brain.cells[i]["side"].lower()
            if side == "l":
                left.append(j)
            elif side == "r":
                right.append(j)
        if not left or not right:
            raise ValueError("connectome has no sided descending/motor cells")
        masks = (np.array(left), np.array(right))
        brain._adapter_masks = masks
    return masks


def feature_positions(brain):
    """Map each readout cell id to its position in ``brain.features()`` (cached)."""
    positions = getattr(brain, "_adapter_positions", None)
    if positions is None:
        positions = {i: j for j, i in enumerate(brain.feature_ids)}
        brain._adapter_positions = positions
    return positions


def _readouts(brain, features):
    """``(escape, power)`` means read from the same neural vector the codec steers by.

    Reading both signals from one vector is what makes the zero/shuffle ablations real:
    the environment ablates the vector it hands the bridge, and the declared adapter has
    no second, un-ablated channel into the connectome.
    """
    positions = feature_positions(brain)
    means = {
        key: float(np.mean([features[positions[i]] for i in ids]))
        for key, ids in brain.readout_ids.items()
        if key in ("escape", "power_l", "power_r")
    }
    return means["escape"], 0.5 * (means["power_l"] + means["power_r"])


def _laterality(brain, features):
    left, right = side_masks(brain)
    return float(features[left].mean() - features[right].mean())


def _latch(escape, rest_escape):
    """Loom level in [0, 1] from the escape readout above its rest baseline."""
    return float(np.clip((escape - rest_escape) / ESCAPE_SCALE, 0.0, 1.0))


def _params(brain, settle=SETTLE_TICKS):
    """Calibrate the declared gains on the stimulus battery.

    The structure is fixed (laterality steers, escape gates and climbs, activity drives
    forward); only the scalar gains and the rest baselines come from this fit.
    """
    names, lat, escape, power, targets = [], [], [], [], []
    for _ in range(3):
        for name, cues, target in STIMULI:
            brain.reset(11)
            brain.set_currents(np.asarray(cues, dtype=np.float32))
            brain.step(settle)
            features = brain.features()
            names.append(name)
            lat.append(_laterality(brain, features))
            esc, pw = _readouts(brain, features)
            escape.append(esc)
            power.append(pw)
            targets.append(target)
    rest = [i for i, n in enumerate(names) if n == "rest"]
    rest_escape = float(np.mean([escape[i] for i in rest]))
    rest_power = float(np.mean([power[i] for i in rest]))
    rest_lat = float(np.mean([lat[i] for i in rest]))
    loom = np.array([_latch(e, rest_escape) for e in escape])
    activity = np.maximum(0.0, np.asarray(power) - rest_power)
    target = np.asarray(targets, dtype=float)
    steer = (np.asarray(lat) - rest_lat) * (1.0 - 2.0 * loom)

    def gain(x, y):
        denominator = float(x @ x)
        if denominator < 1e-12:
            raise ValueError("declared battery cannot calibrate this gain")
        return float(x @ y / denominator)

    params = {
        "escape_scale": ESCAPE_SCALE,
        "side_fraction": SIDE_FRACTION,
        "rest_escape": round(rest_escape, 6),
        "rest_power": round(rest_power, 6),
        "rest_lat": round(rest_lat, 8),
        "g_yaw": round(gain(steer, target[:, 3]), 6),
        "g_climb": round(gain(loom, target[:, 2]), 6),
        "g_fwd": round(gain(activity, target[:, 0]), 6),
    }
    if not (abs(params["g_yaw"]) > 0 and params["g_climb"] > 0 and params["g_fwd"] > 0):
        raise ValueError(f"declared battery gained no usable drive: {params}")
    return params


def _version(params, dataset_hash):
    payload = json.dumps(
        {"params": params, "dataset_hash": dataset_hash},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "declared-v1:" + hashlib.sha256(payload).hexdigest()[:16]


class Adapter:
    """A calibrated declared codec, bound to the connectome it was calibrated on."""

    def __init__(self, params, dataset_hash, bundle_hash, alternate=False):
        self.params = params
        self.dataset_hash = dataset_hash
        self.bundle_hash = bundle_hash
        self.alternate = alternate
        self.version = _version(params, dataset_hash)

    @classmethod
    def calibrate(cls, brain=None, settle=SETTLE_TICKS):
        brain = brain or BrainRuntime()
        params = _params(brain, settle)
        return cls(params, brain.dataset_hash, brain.bundle_hash, brain.alternate)

    @classmethod
    def from_payload(cls, payload):
        return cls(
            dict(payload["params"]),
            payload["dataset_hash"],
            payload["bundle_hash"],
            bool(payload.get("alternate", False)),
        )

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        adapter = cls.from_payload(payload)
        if payload.get("adapter_version") != adapter.version:
            raise ValueError("adapter identity does not match its constants")
        return adapter

    @property
    def payload(self):
        return {
            "version": 1,
            "adapter_version": self.version,
            "dataset_hash": self.dataset_hash,
            "bundle_hash": self.bundle_hash,
            "alternate": self.alternate,
            "note": "Declared body adapter (P0); no teacher, no learned policy.",
            "params": self.params,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload, indent=2) + "\n")
        return path

    def check(self, brain):
        """Refuse to run against a connectome other than the calibrated one."""
        if self.dataset_hash != brain.dataset_hash:
            raise ValueError(
                f"adapter dataset_hash {self.dataset_hash!r} does not match runtime "
                f"{brain.dataset_hash!r}"
            )
        check_bundle_pin(
            {"bundle_hash": self.bundle_hash}, brain.bundle_hash, brain.alternate
        )

    def command(self, brain, features=None):
        """The body command ``[vx, vy, vz, yaw_rate]`` in ``[-1, 1]`` from neural activity.

        ``features`` is the (possibly ablated) 2,022-vector the environment hands the body
        bridge; when omitted the live ``brain.features()`` is read. Pass the environment's
        ``observe()`` output so zero/shuffle/silencing conditions act on the codec.
        """
        p = self.params
        vector = (
            brain.features() if features is None else np.asarray(features, dtype=float)
        )
        escape, power = _readouts(brain, vector)
        loom = _latch(escape, p["rest_escape"])
        steer = (_laterality(brain, vector) - p["rest_lat"]) * (1.0 - 2.0 * loom)
        activity = max(0.0, power - p["rest_power"])
        yaw = p["g_yaw"] * steer
        return np.clip(
            [p["g_fwd"] * activity, p["side_fraction"] * yaw, p["g_climb"] * loom, yaw],
            -1.0,
            1.0,
        )


_DEFAULT = None


def load_default(path=DEFAULT_PATH):
    """The calibrated adapter committed at P0, loaded once per process."""
    global _DEFAULT
    if _DEFAULT is None:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"declared adapter not calibrated: {path} "
                "(run `fly-drone adapter-calibrate`)"
            )
        _DEFAULT = Adapter.load(path)
    return _DEFAULT


def declared_command(brain, features=None, path=None):
    """Module entry point used by the environment and server: command from neural activity.

    Pass the environment's (possibly ablated) feature vector as ``features`` so the causal
    conditions apply to the declared bridge exactly as they do to a learned decoder.
    """
    adapter = Adapter.load(path) if path else load_default()
    adapter.check(brain)
    return adapter.command(brain, features=features)
