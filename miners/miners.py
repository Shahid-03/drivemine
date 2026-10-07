import numpy as np


def hard_braking(ego_positions, timestamps, threshold=3.0):
    positions = np.asarray(ego_positions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    if len(positions) < 3:
        return np.array([], dtype=int)

    dt = np.diff(timestamps) / 1e6

    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")

    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])

    velocity = np.hypot(dx, dy) / dt
    acceleration = np.diff(velocity) / dt[1:]

    return np.where(acceleration < -threshold)[0] + 2
    """
    Detect hard braking from ego-frame longitudinal position.

    Parameters
    ----------
    ego_positions : (N, 3) array
        Ego positions in ego/world-consistent coordinates.
        x is longitudinal/forward.
    timestamps : (N,) array
        Timestamps in microseconds.
    threshold : float
        Braking threshold in m/s^2.

    Returns
    -------
    indices : ndarray
        Sample indices where longitudinal acceleration
        falls below -threshold.
    """
    positions = np.asarray(ego_positions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    if len(positions) < 3:
        return np.array([], dtype=int)

    dt = np.diff(timestamps) / 1e6

    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")

    velocity = np.diff(positions[:, 0]) / dt
    acceleration = np.diff(velocity) / dt[1:]

    return np.where(acceleration < -threshold)[0] + 2


def pedestrian_near_path(
    pedestrian_positions,
    lateral_threshold=2.0,
    ahead_threshold=20.0,
):
    """
    Find pedestrians within a simple ego-path region.

    x: forward
    y: lateral
    """
    positions = np.asarray(pedestrian_positions, dtype=np.float64)

    mask = (
        (positions[:, 0] > 0)
        & (positions[:, 0] <= ahead_threshold)
        & (np.abs(positions[:, 1]) <= lateral_threshold)
    )

    return np.flatnonzero(mask)


def close_following(
    vehicle_positions,
    gap_threshold=10.0,
    lateral_threshold=2.5,
):
    """
    Find vehicles close ahead of ego.
    """
    positions = np.asarray(vehicle_positions, dtype=np.float64)

    mask = (
        (positions[:, 0] > 0)
        & (positions[:, 0] <= gap_threshold)
        & (np.abs(positions[:, 1]) <= lateral_threshold)
    )

    return np.flatnonzero(mask)
