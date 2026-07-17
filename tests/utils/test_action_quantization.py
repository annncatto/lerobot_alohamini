import pytest

from lerobot.utils.action_quantization import snap_planar_velocity


def test_snap_planar_velocity_uses_deadband_and_preserves_other_actions():
    action = {"x.vel": 0.02, "y.vel": -0.08, "theta.vel": 3.0, "arm.pos": 12.0}

    snapped = snap_planar_velocity(action, speed=0.15, deadband=0.05)

    assert snapped == {"x.vel": 0.0, "y.vel": -0.15, "theta.vel": 3.0, "arm.pos": 12.0}
    assert action["y.vel"] == -0.08


@pytest.mark.parametrize("speed,deadband", [(-0.15, 0.05), (0.15, -0.05)])
def test_snap_planar_velocity_rejects_negative_parameters(speed, deadband):
    with pytest.raises(ValueError):
        snap_planar_velocity({"x.vel": 0.1}, speed=speed, deadband=deadband)
