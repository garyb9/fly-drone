import json
import time
from pathlib import Path

import numpy as np
import psutil

from .brain import BrainRuntime


def sensory_assay(output=None):
    brain = BrainRuntime()
    results = {}
    traces = {}
    latency = []
    for condition in ["dark", "left", "right", "left_silenced", "right_silenced"]:
        brain.reset(42)
        if condition.endswith("silenced"):
            brain.silence_sensors()
        images = np.zeros((2, 48, 64, 3), dtype=np.uint8)
        if condition.startswith("left"):
            images[0] = 255
        if condition.startswith("right"):
            images[1] = 255
        brain.sense(images)
        features = []
        for _ in range(200):
            start = time.perf_counter()
            brain.step()
            latency.append(time.perf_counter() - start)
            features.append(brain.features())
        traces[condition] = np.asarray(features)
        results[condition] = {
            "final_readouts": brain.read(),
            "mean_feature_activity": float(np.mean(features)),
        }
    separation = float(np.max(np.abs(traces["left"] - traces["right"])))
    effect = float(np.max(np.abs(traces["left"] - traces["left_silenced"])))
    silenced_error = float(np.max(np.abs(traces["left_silenced"] - traces["dark"])))
    from .plant import DronePlant

    plant = DronePlant()
    rendered = {}
    try:
        for name, side, silence in [
            ("left", 1, False),
            ("right", -1, False),
            ("left_silenced", 1, True),
            ("right_silenced", -1, True),
        ]:
            brain.reset(42)
            if silence:
                brain.silence_sensors()
            plant.set_objects(target=[2, side * 1.2, 1], obstacle=[3, -3, 1])
            images = plant.camera().copy()
            samples = []
            for _ in range(25):
                brain.sense(images)
                brain.step(8)
                samples.append(brain.features())
            rendered[name] = np.array(samples)
    finally:
        plant.close()
    rendered_separation = float(np.max(np.abs(rendered["left"] - rendered["right"])))
    rendered_silenced = float(
        np.max(np.abs(rendered["left_silenced"] - rendered["right_silenced"]))
    )
    report = {
        "dataset_hash": brain.dataset_hash,
        "neurons": brain.core.neuron_count(),
        "features": len(brain.feature_ids),
        "conditions": results,
        "left_right_max_difference": separation,
        "silencing_max_effect": effect,
        "silenced_dark_max_difference": silenced_error,
        "rendered_left_right_max_difference": rendered_separation,
        "rendered_silenced_difference": rendered_silenced,
        "passed": separation > 1e-4
        and effect > 1e-4
        and silenced_error < 1e-6
        and rendered_separation > 0.01
        and rendered_silenced < 1e-6,
        "tick_ms_p50": float(np.percentile(latency, 50) * 1000),
        "tick_ms_p95": float(np.percentile(latency, 95) * 1000),
        "rss_mb": psutil.Process().memory_info().rss / 1e6,
        "meaning": "Software causal assay; not behavioral or biological validation. Synthetic RGB and rendered target stimuli.",
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2))
    return report
