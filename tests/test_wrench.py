import numpy as np
from fly_drone import wrench
from fly_drone.wrench import (
    HOVER_RPM,
    HOVER_THRUST,
    MAX_RPM,
    MAX_THRUST,
    RateController,
    intent_to_wrench,
    mix,
    rotor_wrench,
    thrust_for_vertical,
)


def test_hover_is_four_equal_rotors():
    rpm = mix(HOVER_THRUST, [0.0, 0.0, 0.0])
    assert np.allclose(rpm, HOVER_RPM, rtol=1e-6)
    assert np.allclose(wrench.hover_rpm(), HOVER_RPM)
    assert 14400 < HOVER_RPM < 14600  # matches control-and-physics.md


def test_rotor_wrench_round_trips_through_the_mixer():
    forces = np.array([0.070, 0.060, 0.065, 0.070])
    rpm = np.sqrt(forces / wrench.KF)
    thrust, torques = rotor_wrench(rpm)
    recovered = mix(thrust, torques)
    assert np.allclose(recovered, rpm, rtol=1e-6)


def test_hover_wrench_round_trips():
    thrust, torques = rotor_wrench(wrench.hover_rpm())
    assert abs(thrust - HOVER_THRUST) < 1e-9
    assert np.allclose(torques, 0.0, atol=1e-12)


def test_more_thrust_means_more_rpm():
    rpm = mix(1.2 * HOVER_THRUST, [0.0, 0.0, 0.0])
    assert np.all(rpm > HOVER_RPM)


def test_positive_roll_torque_speeds_the_front_rotors():
    rpm = mix(HOVER_THRUST, [1e-4, 0.0, 0.0])
    assert rpm[0] > rpm[2] and rpm[1] > rpm[3]


def test_infeasible_wrench_saturates_within_the_envelope():
    rpm = mix(10.0 * MAX_THRUST, [1.0, -1.0, 1.0])
    assert np.isfinite(rpm).all()
    assert (rpm >= 0).all() and (rpm <= MAX_RPM + 1e-9).all()


def test_rate_controller_maps_zero_error_to_zero_and_tracks_sign():
    ctrl = RateController()
    assert np.allclose(ctrl.torques([0.0, 0.0, 0.0]), 0.0)
    assert np.all(ctrl.torques([1.0, 1.0, 1.0]) > 0)
    ctrl.reset()
    assert np.all(ctrl.torques([-1.0, -1.0, -1.0]) < 0)


def test_vertical_thrust_tracks_error_and_clips():
    assert thrust_for_vertical(0.5, 0.0) > HOVER_THRUST
    assert thrust_for_vertical(-0.5, 0.0) < HOVER_THRUST
    assert thrust_for_vertical(100.0, 0.0) == MAX_THRUST


def test_intent_to_wrench_is_zero_at_zero_and_antisymmetric():
    thrust, torques = intent_to_wrench([0.0, 0.0, 0.0, 0.0])
    assert thrust == HOVER_THRUST and np.allclose(torques, 0.0)
    _, forward = intent_to_wrench([0.5, 0.5, 0.0, 0.5])
    _, backward = intent_to_wrench([-0.5, -0.5, 0.0, -0.5])
    assert np.allclose(forward, -backward)
