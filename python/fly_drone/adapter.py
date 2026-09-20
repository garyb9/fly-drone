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
# The same codec re-pinned to the P1 feedback bundle (an alternate identity).
DEFAULT_FEEDBACK_PATH = ROOT / "docs" / "results" / "adapter" / "adapter-feedback.json"
# The same codec re-pinned to the P2 dynamics bundle (an alternate identity).
DEFAULT_DYNAMICS_PATH = ROOT / "docs" / "results" / "adapter" / "adapter-dynamics.json"
# The same codec on the P3 declared optic-flow relay front-end (an additive bridge version).
DEFAULT_RELAY_PATH = ROOT / "docs" / "results" / "adapter" / "adapter-relay.json"
# P4 codec v2: steering from the wing steering motoneurons, two-sided forward drive.
DEFAULT_V2_PATH = ROOT / "docs" / "results" / "adapter" / "adapter-v2.json"
# The codec re-pinned to the P4 (C3a) tonic bundle (an alternate identity).
DEFAULT_TONIC_PATH = ROOT / "docs" / "results" / "adapter" / "adapter-tonic.json"
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


def _steer_diff(brain, features):
    """Left-minus-right mean of the fly's wing steering motoneurons (P4 codec v2).

    Read from the same possibly-ablated ``features`` vector as everything else, so the
    zero/shuffle/silencing conditions act on the steering command exactly as they do on the
    forward and escape channels.
    """
    positions = feature_positions(brain)
    means = {
        key: float(np.mean([features[positions[i]] for i in ids]))
        for key, ids in brain.readout_ids.items()
        if key in ("steer_l", "steer_r")
    }
    return means["steer_l"] - means["steer_r"]


def _gain(x, y):
    """Least-squares scalar gain ``y ~ g * x`` (the declared battery's only fitted scalars)."""
    denominator = float(x @ x)
    if denominator < 1e-12:
        raise ValueError("declared battery cannot calibrate this gain")
    return float(x @ y / denominator)


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

    params = {
        "escape_scale": ESCAPE_SCALE,
        "side_fraction": SIDE_FRACTION,
        "rest_escape": round(rest_escape, 6),
        "rest_power": round(rest_power, 6),
        "rest_lat": round(rest_lat, 8),
        "g_yaw": round(_gain(steer, target[:, 3]), 6),
        "g_climb": round(_gain(loom, target[:, 2]), 6),
        "g_fwd": round(_gain(activity, target[:, 0]), 6),
    }
    if not (abs(params["g_yaw"]) > 0 and params["g_climb"] > 0 and params["g_fwd"] > 0):
        raise ValueError(f"declared battery gained no usable drive: {params}")
    return params


def _params_v2(brain, settle=SETTLE_TICKS):
    """Calibrate codec v2 on the same battery with a faithful readout (P4 workstream A3).

    Three changes from v1, each separately ablatable:
    - steering reads the wing steering motoneurons (``steer_l``/``steer_r``) and is normalised
      by their maximum battery excursion (``steer_span``), not the 2,022-cell population mean;
    - forward drive is two-sided (``power - rest_power`` may be negative) and normalised by
      ``power_span``;
    - ``g_fwd`` is fit only on battery rows whose declared ``vx`` target is non-zero, so the
      loom rows (target ``vx = 0`` but the largest ``power`` excursion) no longer drag the gain
      down. This is the recorded choice from the spec's A3.4 (row mask, not a re-declared loom
      target).

    A calibration-time falsifier rejects a codec that cannot command at least 0.05 m/s on the
    brightest non-loom stimulus (A3.5).
    """
    names, escape, power, steer_d, targets = [], [], [], [], []
    for _ in range(3):
        for name, cues, target in STIMULI:
            brain.reset(11)
            brain.set_currents(np.asarray(cues, dtype=np.float32))
            brain.step(settle)
            features = brain.features()
            names.append(name)
            esc, pw = _readouts(brain, features)
            escape.append(esc)
            power.append(pw)
            steer_d.append(_steer_diff(brain, features))
            targets.append(target)
    rest = [i for i, n in enumerate(names) if n == "rest"]
    rest_escape = float(np.mean([escape[i] for i in rest]))
    rest_power = float(np.mean([power[i] for i in rest]))
    rest_steer = float(np.mean([steer_d[i] for i in rest]))
    loom = np.array([_latch(e, rest_escape) for e in escape])
    target = np.asarray(targets, dtype=float)
    power_exc = np.asarray(power) - rest_power
    steer_exc = np.asarray(steer_d) - rest_steer
    power_span = float(np.max(np.abs(power_exc)))
    steer_span = float(np.max(np.abs(steer_exc)))
    if power_span < 1e-12 or steer_span < 1e-12:
        raise ValueError("declared battery cannot calibrate codec v2 spans")
    drive = np.clip(power_exc / power_span, -1.0, 1.0)
    mask = target[:, 0] != 0.0
    if not mask.any():
        raise ValueError("declared battery has no non-zero vx target to fit g_fwd")
    params = {
        "escape_scale": ESCAPE_SCALE,
        "side_fraction": SIDE_FRACTION,
        "rest_escape": round(rest_escape, 6),
        "rest_power": round(rest_power, 6),
        "rest_steer": round(rest_steer, 8),
        "power_span": round(power_span, 8),
        "steer_span": round(steer_span, 8),
        "g_climb": round(_gain(loom, target[:, 2]), 6),
        "g_fwd": round(_gain(drive[mask], target[mask, 0]), 6),
    }
    from .plant import LIMITS

    bright = names.index("light_both")
    commanded = abs(params["g_fwd"] * float(drive[bright])) * float(LIMITS[0])
    if commanded < 0.05:
        raise ValueError(
            f"codec v2 cannot command motion: brightest non-loom stimulus commands "
            f"{commanded:.4f} m/s (< 0.05 m/s)"
        )
    return params


def _version(params, dataset_hash, visual=None, codec=None):
    payload = {"params": params, "dataset_hash": dataset_hash}
    if visual:
        # A declared sensory front-end changes the bridge, so it belongs in its identity.
        payload["visual"] = visual
    if codec:
        # A codec revision changes the command formula, so it belongs in the identity too.
        payload["codec"] = codec
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"declared-{codec or 'v1'}:" + hashlib.sha256(encoded).hexdigest()[:16]


class Adapter:
    """A calibrated declared codec, bound to the connectome it was calibrated on."""

    def __init__(
        self,
        params,
        dataset_hash,
        bundle_hash,
        alternate=False,
        visual=None,
        codec=None,
    ):
        self.params = params
        self.dataset_hash = dataset_hash
        self.bundle_hash = bundle_hash
        self.alternate = alternate
        self.visual = visual
        self.codec = codec
        self.version = _version(params, dataset_hash, visual, codec)

    @classmethod
    def calibrate(cls, brain=None, settle=SETTLE_TICKS, visual=None, codec=None):
        brain = brain or BrainRuntime(relay=visual == "relay")
        params = _params_v2(brain, settle) if codec == "v2" else _params(brain, settle)
        return cls(
            params,
            brain.dataset_hash,
            brain.bundle_hash,
            brain.alternate,
            visual,
            codec,
        )

    @classmethod
    def from_payload(cls, payload):
        return cls(
            dict(payload["params"]),
            payload["dataset_hash"],
            payload["bundle_hash"],
            bool(payload.get("alternate", False)),
            payload.get("visual"),
            payload.get("codec"),
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
        payload = {
            "version": 1,
            "adapter_version": self.version,
            "dataset_hash": self.dataset_hash,
            "bundle_hash": self.bundle_hash,
            "alternate": self.alternate,
            "note": "Declared body adapter (P0); no teacher, no learned policy.",
            "params": self.params,
        }
        if self.visual:
            payload["visual"] = self.visual
        if self.codec:
            payload["codec"] = self.codec
        return payload

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
        if self.codec == "v2":
            # A3.1/A3.3: steering from the wing steering motoneurons, normalised by their
            # battery excursion. A3.2: two-sided forward drive (slowing is commandable).
            raw = (_steer_diff(brain, vector) - p["rest_steer"]) / p["steer_span"]
            steer = float(np.clip(raw, -1.0, 1.0)) * (1.0 - 2.0 * loom)
            drive = float(
                np.clip((power - p["rest_power"]) / p["power_span"], -1.0, 1.0)
            )
            yaw = steer
            vx = p["g_fwd"] * drive
        else:
            steer = (_laterality(brain, vector) - p["rest_lat"]) * (1.0 - 2.0 * loom)
            yaw = p["g_yaw"] * steer
            vx = p["g_fwd"] * max(0.0, power - p["rest_power"])
        return np.clip(
            [vx, p["side_fraction"] * yaw, p["g_climb"] * loom, yaw],
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
