"""Continuous 6D rotation encoding for ForceVLA end-effector poses.

Dataset fields are still xyz + roll/pitch/yaw + gripper. The policy transform
rewrites the three Euler angles as the first two columns of the rotation matrix
(Zhou et al.) so `DeltaActions` never sees a ±π branch cut. Inverse conversion
goes back to extrinsic XYZ Euler for the robot.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

XYZ_DIMS = 3
ROTATION_6D_DIMS = 6
GRIPPER_DIMS = 1
POSE_DIMS = XYZ_DIMS + ROTATION_6D_DIMS  # 9
ROBOT_DIMS = POSE_DIMS + GRIPPER_DIMS  # 10
RAW_RPY_SLICE = slice(3, 6)
RAW_GRIPPER = 6
RAW_FORCE_SLICE = slice(7, 13)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    rpy = np.asarray(rpy, dtype=np.float64)
    return Rotation.from_euler("xyz", rpy.reshape(-1, 3)).as_matrix().reshape(*rpy.shape[:-1], 3, 3)


def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_euler("xyz").reshape(*matrix.shape[:-2], 3)


def matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def sixd_to_matrix(sixd: np.ndarray) -> np.ndarray:
    sixd = np.asarray(sixd, dtype=np.float64)
    a1 = sixd[..., :3]
    a2 = sixd[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def geodesic_angle(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    relative = np.matmul(np.swapaxes(r_a, -1, -2), r_b)
    traces = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(traces)


def rpy_to_6d(rpy: np.ndarray) -> np.ndarray:
    return matrix_to_6d(rpy_to_matrix(rpy)).astype(np.float32)


def sixd_to_rpy(sixd: np.ndarray) -> np.ndarray:
    return matrix_to_rpy(sixd_to_matrix(sixd)).astype(np.float32)


def convert_state(state: np.ndarray) -> np.ndarray:
    """Rewrite a raw ForceVLA state vector: xyz + 6D + gripper [+ force]."""
    state = np.asarray(state, dtype=np.float64)
    leading, last = state.shape[:-1], state.shape[-1]
    if last < 7:
        raise ValueError(f"Expected state [..., >=7], got {state.shape}")
    flat = state.reshape(-1, last)
    rotation = rpy_to_6d(flat[:, RAW_RPY_SLICE])
    parts = [flat[:, :XYZ_DIMS], rotation, flat[:, RAW_GRIPPER : RAW_GRIPPER + 1]]
    if last >= 13:
        parts.append(flat[:, RAW_FORCE_SLICE])
    elif last > 7:
        parts.append(flat[:, 7:])
    return np.concatenate(parts, axis=-1).reshape(*leading, -1).astype(np.float32)


def convert_action(action: np.ndarray) -> np.ndarray:
    """Rewrite a raw ForceVLA action: xyz + 6D + gripper."""
    action = np.asarray(action, dtype=np.float64)
    leading, last = action.shape[:-1], action.shape[-1]
    if last < 7:
        raise ValueError(f"Expected action [..., >=7], got {action.shape}")
    flat = action.reshape(-1, last)
    rotation = rpy_to_6d(flat[:, RAW_RPY_SLICE])
    tail = flat[:, 7:] if last > 7 else np.empty((flat.shape[0], 0), dtype=np.float64)
    converted = np.concatenate([flat[:, :XYZ_DIMS], rotation, flat[:, RAW_GRIPPER : RAW_GRIPPER + 1], tail], axis=-1)
    return converted.reshape(*leading, -1).astype(np.float32)


def actions_6d_to_rpy(action: np.ndarray) -> np.ndarray:
    """Map a 10D xyz+6D+gripper action back to the robot's xyz+rpy+gripper."""
    action = np.asarray(action, dtype=np.float64)
    leading, last = action.shape[:-1], action.shape[-1]
    if last < ROBOT_DIMS:
        raise ValueError(f"Expected 6D action [..., >=10], got {action.shape}")
    flat = action.reshape(-1, last)
    rpy = sixd_to_rpy(flat[:, XYZ_DIMS : POSE_DIMS])
    restored = np.concatenate([flat[:, :XYZ_DIMS], rpy, flat[:, POSE_DIMS : ROBOT_DIMS], flat[:, ROBOT_DIMS:]], axis=-1)
    return restored.reshape(*leading, -1).astype(np.float32)
