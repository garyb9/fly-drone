import hashlib
import json
from pathlib import Path

import numpy as np

from ._brain import Brain

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/malecns"
# Must match brain-core policy::ENCODER_VERSION (camera geometry + cue encoder).
ENCODER_VERSION = "bright-contrast-400-splay075-noaa-v3"


class BrainRuntime:
    def __init__(self, seed=42, data=DATA):
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
        self.input_ids = {k: sensory[k] for k in ("light_l", "light_r")}
        for side in ("l", "r"):
            self.input_ids["looming_" + side] = [
                i
                for i, c in enumerate(self.cells)
                if c["side"].lower() == side and c["type"] in ("LC4", "LPLC2")
            ]
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
        self.cues = np.zeros(4, dtype=np.float32)

    def reset(self, seed):
        self.core.reset(seed)
        self.core.bias(self.tonic, 0.85)
        self.tick = 0
        self.cues[:] = 0

    def sense(self, images):
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

    def load_policy(self, path):
        self.core.load_policy(
            Path(path).read_text(), self.dataset_hash, self.feature_ids
        )

    def infer(self, features=None):
        return np.asarray(
            self.core.infer(
                (self.features() if features is None else features).tolist()
            ),
            dtype=np.float64,
        )

    def silence_sensors(self):
        self.core.silence(sorted(set(sum(self.input_ids.values(), []))), True)
