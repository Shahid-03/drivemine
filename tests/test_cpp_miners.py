import numpy as np
import cpp_miners

from miners.miners import hard_braking


def test_cpp_matches_numpy():
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
    ], dtype=np.float64)

    x = np.ascontiguousarray(positions[:, 0])
    y = np.ascontiguousarray(positions[:, 1])

    numpy_result = hard_braking(positions, timestamps, 3.0)
    cpp_result = cpp_miners.hard_braking(x, y, timestamps, 3.0)

    np.testing.assert_array_equal(cpp_result, numpy_result)
