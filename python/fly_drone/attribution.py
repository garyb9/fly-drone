"""Which neurons drive each motion command: exact gradient x input on the decoder.

The decoder is a small tanh MLP (policy.rs), so the Jacobian of every command with
respect to every descending/motor feature is exact and cheap. A contribution is
d(command)/d(feature) * (feature - normaliser mean): how far this cell's current
activity pushes the command away from what the average activity would produce.
"""

import json
from pathlib import Path

import numpy as np

CHANNELS = ("forward", "lateral", "climb", "yaw")


class Attributor:
    def __init__(self, actor, feature_types, feature_ids=None):
        if isinstance(actor, str | Path):
            actor = json.loads(Path(actor).read_text())
        self.mean = np.asarray(actor["mean"], dtype=np.float64)
        self.scale = np.asarray(actor["scale"], dtype=np.float64)
        self.layers = [
            (
                np.asarray(layer["weights"], dtype=np.float64),
                np.asarray(layer["bias"], dtype=np.float64),
            )
            for layer in actor["layers"]
        ]
        self.limits = np.asarray(actor["action_limits"], dtype=np.float64)
        if len(feature_types) != len(self.mean):
            raise ValueError("one cell type per decoder feature required")
        self.types = np.asarray(feature_types)
        self.ids = np.asarray(
            feature_ids if feature_ids is not None else range(len(self.mean))
        )
        self.unique_types, self.type_index = np.unique(self.types, return_inverse=True)

    def forward(self, features):
        """Commands in physical units and the Jacobian d(command)/d(feature)."""
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        jac = np.diag(1.0 / self.scale)
        for i, (w, b) in enumerate(self.layers):
            z = w @ x + b
            if i + 1 < len(self.layers):
                x = np.tanh(z)
                jac = (1.0 - x**2)[:, None] * (w @ jac)
            else:
                x = z
                jac = w @ jac
        # clamp(-1, 1) has zero gradient once saturated.
        inside = np.abs(x) < 1.0
        command = np.clip(x, -1.0, 1.0) * self.limits
        jac = jac * (inside * self.limits)[:, None]
        return command, jac

    def contributions(self, features):
        command, jac = self.forward(features)
        return command, jac * (np.asarray(features, dtype=np.float64) - self.mean)[
            None, :
        ]

    def explain(self, features, top=5):
        """Top cell types per channel (signed, physical units) and top cell ids."""
        command, contrib = self.contributions(features)
        by_type = np.zeros((len(CHANNELS), len(self.unique_types)))
        for c in range(len(CHANNELS)):
            np.add.at(by_type[c], self.type_index, contrib[c])
        out = {"command": command.tolist(), "channels": {}}
        for c, name in enumerate(CHANNELS):
            order = np.argsort(-np.abs(by_type[c]))[:top]
            cells = np.argsort(-np.abs(contrib[c]))[:top]
            out["channels"][name] = {
                "types": [
                    {"type": str(self.unique_types[k]), "value": float(by_type[c, k])}
                    for k in order
                ],
                "cells": [int(self.ids[k]) for k in cells],
            }
        return out


def for_brain(actor, brain):
    """Attributor using the runtime's feature ids and their anatomical cell types."""
    types = [brain.cells[i]["type"] or "unannotated" for i in brain.feature_ids]
    return Attributor(actor, types, brain.feature_ids)
