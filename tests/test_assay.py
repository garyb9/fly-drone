from fly_drone.assay import sensory_assay


def test_rendered_camera_stimulus_reaches_frozen_graph(tmp_path):
    report = sensory_assay(tmp_path / "assay.json")
    assert report["passed"]
    assert report["rendered_left_right_max_difference"] > 0.01
    assert report["rendered_silenced_difference"] < 1e-6


def test_light_cue_sign_follows_target_side():
    import numpy as np
    from fly_drone.brain import BrainRuntime
    from fly_drone.plant import DronePlant

    brain = BrainRuntime()
    plant = DronePlant()
    try:
        for bearing in (-0.6, -0.3, -0.1, 0.1, 0.3, 0.6):
            plant.set_objects(
                target=[2 * np.cos(bearing), 2 * np.sin(bearing), 1],
                obstacle=[3.5, -3.5, 1],
            )
            image = plant.camera().copy()
            brain.reset(1)
            brain.sense(image)
            cues = brain.sense(image)
            difference = cues[0] - cues[1]
            assert np.sign(difference) == np.sign(bearing), (bearing, cues)
            assert abs(difference) > 0.5, (bearing, cues)
    finally:
        plant.close()
