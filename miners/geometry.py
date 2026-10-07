import numpy as np


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ],
    ], dtype=np.float64)


def global_to_ego(points, ego_translation, ego_quaternion):
    """
    Transform Nx3 points from the global frame into the ego frame.
    """
    points = np.asarray(points, dtype=np.float64)
    translation = np.asarray(ego_translation, dtype=np.float64)

    qw, qx, qy, qz = ego_quaternion
    rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)

    return (rotation.T @ (points - translation).T).T