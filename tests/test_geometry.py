import numpy as np
from miners.geometry import global_to_ego


def test_identity_pose():
    points = np.array([
        [10.0, 2.0, 0.0],
        [5.0, -3.0, 1.0],
    ])

    result = global_to_ego(
        points,
        ego_translation=[0.0, 0.0, 0.0],
        ego_quaternion=[1.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(result, points)


def test_translation():
    points = np.array([
        [15.0, 5.0, 2.0],
    ])

    result = global_to_ego(
        points,
        ego_translation=[10.0, 2.0, 1.0],
        ego_quaternion=[1.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(result, [[5.0, 3.0, 1.0]])


def test_yaw_90_degrees():
    # 90 degree yaw quaternion.
    angle = np.pi / 2
    qw = np.cos(angle / 2)
    qz = np.sin(angle / 2)

    # A global point one meter along global +x.
    points = np.array([[1.0, 0.0, 0.0]])

    result = global_to_ego(
        points,
        ego_translation=[0.0, 0.0, 0.0],
        ego_quaternion=[qw, 0.0, 0.0, qz],
    )

    # Ego +x points toward global +y after a 90-degree yaw,
    # so global +x appears as ego -y.
    np.testing.assert_allclose(result, [[0.0, -1.0, 0.0]], atol=1e-7)
