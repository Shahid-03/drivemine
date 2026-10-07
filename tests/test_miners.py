import numpy as np

from miners.miners import (
    hard_braking,
    pedestrian_near_path,
    close_following,
)


def test_hard_braking():
    # 1-second intervals.
    # Position differences:
    # 10, 9, 6, 2 meters
    # Velocities:
    # 10, 9, 6, 2 m/s
    # Accelerations:
    # -1, -3, -4 m/s^2
    positions = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [19.0, 0.0, 0.0],
        [25.0, 0.0, 0.0],
        [27.0, 0.0, 0.0],
    ])

    timestamps = np.array([
        0,
        1_000_000,
        2_000_000,
        3_000_000,
        4_000_000,
    ])

    result = hard_braking(positions, timestamps, threshold=3.0)

    np.testing.assert_array_equal(result, [4])


def test_pedestrian_near_path():
    pedestrians = np.array([
        [10.0, 1.0, 0.0],   # should match
        [15.0, 2.5, 0.0],   # too far laterally
        [25.0, 0.5, 0.0],   # too far ahead
        [-5.0, 0.0, 0.0],   # behind
    ])

    result = pedestrian_near_path(pedestrians)

    np.testing.assert_array_equal(result, [0])


def test_close_following():
    vehicles = np.array([
        [8.0, 1.0, 0.0],    # should match
        [12.0, 0.0, 0.0],   # too far
        [5.0, 3.0, 0.0],    # too far laterally
        [-3.0, 0.0, 0.0],   # behind
    ])

    result = close_following(vehicles)

    np.testing.assert_array_equal(result, [0])
