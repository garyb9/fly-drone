import hashlib
import json
from pathlib import Path

import numpy as np

from ._brain import Brain

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/malecns"
# Must match brain-core policy::ENCODER_VERSION (camera geometry + cue encoder).
ENCODER_VERSION = "bright-contrast-400-splay075-noaa-loom150-v4"
LEARNED_PREFIX = "learned-v5:"
LEARNED_EXTERNAL = LEARNED_PREFIX + "external"
STACK_FRAMES = 3
EYE_SHAPE = (48, 64)
# Encoder v5 drives each anatomical input population with its own uniform current.
V5_CHANNELS = (
    "mi1_l",
    "mi1_r",
    "tm3_l",
    "tm3_r",
    "lc4_l",
    "lc4_r",
    "lplc2_l",
    "lplc2_r",
)
V5_TYPES = {"mi1": "Mi1", "tm3": "Tm3", "lc4": "LC4", "lplc2": "LPLC2"}
# v4 cues [light_l, light_r, loom_l, loom_r] copied onto the 8 v5 channels.
V4_TO_V5 = (0, 1, 0, 1, 2, 3, 2, 3)
PATHWAY_TYPES = {"light": ("mi1", "tm3"), "looming": ("lc4", "lplc2")}


def luma_u8(images):
    """Rec. 709 luma of packed RGB eyes, the same weights as the Rust v4 encoder."""
    rgb = np.asarray(images, dtype=np.float32)
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.round(y).astype(np.uint8)


class FrameStack:
    """The learned encoder's own recent frames; an empty stack repeats the first frame."""

    def __init__(self, frames=STACK_FRAMES):
        self.frames = frames
        self.buffer = None

    def clear(self):
        self.buffer = None

    def push(self, images):
        y = luma_u8(images)[:, None]
        if self.buffer is None:
            self.buffer = np.repeat(y, self.frames, axis=1)
        else:
            self.buffer = np.concatenate([self.buffer[:, 1:], y], axis=1)

    def array(self):
        if self.buffer is None:
            return np.zeros((2 * self.frames, *EYE_SHAPE), np.uint8)
        return self.buffer.reshape(2 * self.frames, *EYE_SHAPE).copy()


class BrainRuntime:
    def __init__(self, seed=42, data=DATA, encoder=None):
        self.data = Path(data)
        manifest = json.loads((self.data / "manifest.json").read_text())
        raw = (self.data / "graph.bin").read_bytes()
        self.dataset_hash = hashlib.sha256(raw).hexdigest()
        self.core = Brain((self.data / "neurons.bin").read_bytes(), raw, seed)
        if self.core.neuron_count() != manifest["n_neurons"]:
            raise ValueError("manifest neuron count mismatch")
        self.cells = json.loads((self.data / "cells.json").read_text())
        groups = json.loads((self.data / "groups.json").read_text())
        sensory = json.loads((self.data / "sensory-mappings.json").read_text())[
            "inputs"
        ]
        self.encoder = encoder
        self.pathway_ids = {}
        if encoder is None:
            self.encoder_version = ENCODER_VERSION
            self.input_ids = {k: sensory[k] for k in ("light_l", "light_r")}
            for side in ("l", "r"):
                self.input_ids["looming_" + side] = self._cells(side, ("LC4", "LPLC2"))
        else:
            if isinstance(encoder, str) and encoder == "external":
                self.encoder_version = LEARNED_EXTERNAL
            else:
                from .encoder import LearnedEncoder

                if not isinstance(encoder, LearnedEncoder):
                    self.encoder = LearnedEncoder.load(encoder)
                self.encoder_version = self.encoder.version
            self.input_ids = {
                ch: self._cells(ch[-1], (V5_TYPES[ch[:-2]],)) for ch in V5_CHANNELS
            }
            for pathway, types in PATHWAY_TYPES.items():
                for side in ("l", "r"):
                    self.pathway_ids[f"{pathway}_{side}"] = sum(
                        (self.input_ids[f"{t}_{side}"] for t in types), []
                    )
        if any(not ids for ids in self.input_ids.values()):
            raise ValueError("required sensory role is empty")
        self.inputs = {k: self.core.input_role(k, v) for k, v in self.input_ids.items()}
        self.readout_ids = {k: v for k, v in groups["roles"]["readout"].items() if v}
        for side in ("l", "r"):
            self.readout_ids["power_" + side] = [
                i
                for i, c in enumerate(self.cells)
                if c["side"].lower() == side
                and c["type"].startswith(("DLMn ", "DVMn "))
            ]
            self.readout_ids["steer_" + side] = [
                i
                for i, c in enumerate(self.cells)
                if c["side"].lower() == side
                and c["type"]
                in (
                    "b1 MN",
                    "b2 MN",
                    "b3 MN",
                    "i1 MN",
                    "i2 MN",
                    "iii1 MN",
                    "iii3 MN",
                    "hg1 MN",
                    "hg2 MN",
                    "hg3 MN",
                    "hg4 MN",
                )
            ]
        self.readouts = {
            k: self.core.readout_role(k, v) for k, v in self.readout_ids.items()
        }
        self.tonic = self.readout_ids["power_l"] + self.readout_ids["power_r"]
        self.core.bias(self.tonic, 0.85)
        names = groups["groups"]
        self.feature_ids = [
            i
            for i, c in enumerate(self.cells)
            if names[c["group"]] in ("descending_neuron", "vnc_motor")
        ]
        self.tick = 0
        self.cues = np.zeros(len(self.input_ids), dtype=np.float32)
        self.stack = FrameStack()

    def _cells(self, side, types):
        return [
            i
            for i, c in enumerate(self.cells)
            if c["side"].lower() == side and c["type"] in types
        ]

    @property
    def learned(self):
        return self.encoder is not None

    def reset(self, seed):
        self.core.reset(seed)
        self.core.bias(self.tonic, 0.85)
        self.tick = 0
        self.cues[:] = 0
        self.stack.clear()

    def sense(self, images):
        if self.learned:
            raise RuntimeError("learned encoders sense via push_frame + encode_stack")
        self.cues = np.array(
            self.core.encode(
                images[0].tobytes(),
                images[1].tobytes(),
                images.shape[2],
                images.shape[1],
            ),
            dtype=np.float32,
        )
        return self.cues

    def set_currents(self, currents):
        currents = np.clip(np.asarray(currents, dtype=np.float32), 0.0, 2.0)
        if currents.shape != self.cues.shape:
            raise ValueError(f"expected {self.cues.shape[0]} currents")
        self.cues = currents
        return self.cues

    def push_frame(self, images):
        self.stack.push(images)

    def encode_stack(self):
        """Learned encoder: currents from the stack. External: the learner already set them."""
        if self.learned and not isinstance(self.encoder, str):
            self.set_currents(self.encoder.currents(self.stack.array()))
        return self.cues

    def step(self, ticks=1):
        for _ in range(ticks):
            for role, value in zip(self.inputs.values(), self.cues, strict=True):
                self.core.inject(role, float(value))
            self.core.step(1)
            self.tick += 1
        return self.features()

    def features(self):
        return np.asarray(self.core.activity(self.feature_ids), dtype=np.float32)

    def read(self):
        return {k: self.core.readout(v) for k, v in self.readouts.items()}

    def load_policy(self, path, check_encoder=True):
        text = Path(path).read_text()
        version = json.loads(text)["encoder_version"]
        # A decoder reads activity shaped by one encoder; check_encoder=False only for a
        # frozen decoder inside an encoder-learning environment.
        if check_encoder and version != self.encoder_version:
            raise ValueError(
                f"actor encoder {version!r} does not match runtime {self.encoder_version!r}"
            )
        self.core.load_policy(text, self.dataset_hash, self.feature_ids)

    def infer(self, features=None):
        return np.asarray(
            self.core.infer(
                (self.features() if features is None else features).tolist()
            ),
            dtype=np.float64,
        )

    def silence_sensors(self):
        self.core.silence(sorted(set(sum(self.input_ids.values(), []))), True)

    def silence_inputs(self, roles):
        """Silence one sensory pathway, e.g. ("looming_l", "looming_r")."""
        ids = {**self.input_ids, **self.pathway_ids}
        self.core.silence(sorted(set(sum((ids[r] for r in roles), []))), True)

    def clear_vision_history(self):
        """Forget previous frames so a respawn does not read as dark-area growth."""
        self.core.clear_vision_history()
        self.stack.clear()
