from fly_drone.stabiliser import measure_step_response


def test_vertical_step_response_faster_than_lateral():
    lateral = measure_step_response(1, warmup_seconds=1.0, step_seconds=2.0)
    vertical = measure_step_response(2, warmup_seconds=1.0, step_seconds=2.0)
    assert lateral["reached_80pct"]
    assert vertical["reached_80pct"]
    # Vertical acceleration is direct-thrust, not tilt-dependent, so it should reach a
    # commanded step markedly faster than lateral (measured ~0.69s vs ~1.48s).
    assert vertical["time_to_80pct_s"] < lateral["time_to_80pct_s"]


def test_rejects_bad_axis():
    import pytest

    with pytest.raises(ValueError):
        measure_step_response(0)
