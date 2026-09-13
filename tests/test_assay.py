from fly_drone.assay import sensory_assay


def test_rendered_camera_stimulus_reaches_frozen_graph(tmp_path):
    report = sensory_assay(tmp_path / "assay.json")
    assert report["passed"]
    assert report["rendered_left_right_max_difference"] > 0.01
    assert report["rendered_silenced_difference"] < 1e-6
