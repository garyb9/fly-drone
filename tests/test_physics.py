import mujoco
import numpy as np
import pytest
from fly_drone.plant import DronePlant


@pytest.fixture
def plant():
    p = DronePlant(vision=False, drag=False)
    yield p
    p.close()


def test_equal_rpm_supports_weight(plant):
    plant._physics(np.full(4, plant.HOVER_RPM), 0)
    f = plant.data.xfrc_applied[plant.model.body("drone0").id]
    np.testing.assert_allclose(f[:3], [0, 0, 0.027 * 9.81], atol=1e-10)
    np.testing.assert_allclose(f[3:], 0, atol=1e-10)


@pytest.mark.parametrize(
    "motor,signs",
    [(0, [1, -1, -1]), (1, [1, 1, 1]), (2, [-1, 1, -1]), (3, [-1, -1, 1])],
)
def test_each_motor_torque_sign(plant, motor, signs):
    rpm = np.full(4, plant.HOVER_RPM)
    rpm[motor] *= 1.1
    plant._physics(rpm, 0)
    np.testing.assert_array_equal(
        np.sign(plant.data.xfrc_applied[plant.model.body("drone0").id, 3:]), signs
    )


def test_motor_lag_saturation_and_bad_inputs(plant):
    initial = plant.actual.copy()
    plant.advance_rpm(np.full(4, plant.MAX_RPM * 2))
    assert np.all(plant.actual > initial) and np.all(plant.actual < plant.MAX_RPM)
    np.testing.assert_array_equal(plant.commanded, np.full(4, plant.MAX_RPM))
    with pytest.raises(ValueError):
        plant.advance([np.nan] * 4)


def test_hover_recovers_vertical_perturbation(plant):
    plant.data.qpos[2] = 1.12
    mujoco.mj_forward(plant.model, plant.data)
    plant._updateAndStoreKinematicInformation()
    z = []
    for i in range(2000):
        plant.advance(np.zeros(4))
        z.append(plant.pos[0, 2])
    assert np.sqrt(np.mean((np.array(z[400:]) - 1) ** 2)) < 0.15
    assert abs(z[-1] - 1) < 0.03


def test_commands_move_in_expected_direction(plant):
    for _ in range(400):
        plant.advance([0.1, 0, 0, 0])
    assert plant.pos[0, 0] > 0.04
    assert abs(plant.pos[0, 1]) < 0.02
    assert abs(plant.pos[0, 2] - 1) < 0.15


def test_camera_shape_and_visual_response():
    p = DronePlant()
    try:
        p.set_objects(target=[1, 1, 1])
        a = p.camera().copy()
        p.set_objects(target=[1, -1, 1])
        b = p.camera().copy()
        assert a.shape == (2, 48, 64, 3) and a.dtype == np.uint8
        assert np.max(np.abs(a.astype(float) - b)) > 50
    finally:
        p.close()
