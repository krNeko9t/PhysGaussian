"""
Coordinate transformation utilities.

.. note::

    Coordinate-system alignment (source -> internal Y-up) is now handled
    by :mod:`physics_sim.coord`.  The functions below handle the
    **MPM domain mapping** ([0, 2]^3) and camera/observant helpers which
    are still used by backends and the camera stage.
"""

import torch
import numpy as np


# ── MPM domain helpers ────────────────────────────────────────────────

def transform2origin(position_tensor, scale=1.0):
    """Centre the point cloud at the origin and scale it."""
    min_pos = torch.min(position_tensor, 0)[0]
    max_pos = torch.max(position_tensor, 0)[0]
    max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    scale = scale / max_diff
    original_mean_pos = original_mean_pos.to(device="cuda")
    scale = scale.to(device="cuda")
    new_position_tensor = (position_tensor - original_mean_pos) * scale
    return new_position_tensor, scale, original_mean_pos


def shift2center111(position_tensor):
    """Shift positions so the centre sits at (1,1,1) inside the [0,2]^3 domain."""
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor + tensor111


def transform_with_reference(position_tensor, scale, original_mean_pos):
    """Apply a precomputed transform2origin() mapping."""
    return (position_tensor - original_mean_pos) * scale


def world_to_mpm_positions(
    world_positions: torch.Tensor,
    rotation_matrices,
    scale_origin: torch.Tensor,
    original_mean_pos: torch.Tensor,
):
    """World-space positions -> MPM domain positions.

    .. deprecated:: Use ``physics_sim.coord`` for alignment;
        domain mapping is internal to each backend.
    """
    rotated = apply_rotations(world_positions, rotation_matrices)
    transformed = transform_with_reference(rotated, scale_origin, original_mean_pos)
    return shift2center111(transformed)


def world_to_mpm_directions(
    world_dirs: torch.Tensor,
    rotation_matrices,
):
    """World-space direction vectors -> MPM-space directions.

    .. deprecated:: Use ``physics_sim.coord.align_directions`` instead.
    """
    return apply_rotations(world_dirs, rotation_matrices)


# ── Rotation helpers (deprecated – use physics_sim.coord) ────────────

def generate_rotation_matrix(degree, axis):
    cos_theta = torch.cos(degree / 180.0 * 3.1415926)
    sin_theta = torch.sin(degree / 180.0 * 3.1415926)
    if axis == 0:
        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:
        rotation_matrix = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    elif axis == 2:
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError("Invalid axis selection")
    return rotation_matrix.cuda()


def generate_rotation_matrices(degrees, axises):
    assert len(degrees) == len(axises)
    matrices = []
    for i in range(len(degrees)):
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))
    return matrices


def apply_rotation(position_tensor, rotation_matrix):
    return torch.mm(position_tensor, rotation_matrix.T)


def apply_rotations(position_tensor, rotation_matrices):
    for R in rotation_matrices:
        position_tensor = apply_rotation(position_tensor, R)
    return position_tensor


def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated


def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device="cuda")
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]
    return mat.view(-1, 3, 3)


def get_upper_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device="cuda")
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]
    return upper_mat


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for R in rotation_matrices:
        cov_tensor = apply_cov_rotation(cov_tensor, R)
    return get_upper_from_mat(cov_tensor)


# ── Inverse transforms (deprecated) ──────────────────────────────────

def undotransform2origin(position_tensor, scale, original_mean_pos):
    return original_mean_pos + position_tensor / scale


def undoshift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor - tensor111


def apply_inverse_rotation(position_tensor, rotation_matrix):
    return torch.mm(position_tensor, rotation_matrix)


def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor


def apply_inverse_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        cov_tensor = apply_cov_rotation(cov_tensor, R.T)
    return get_upper_from_mat(cov_tensor)


def undo_all_transforms(input, rotation_matrices, scale_origin, original_mean_pos):
    """Convenience: undo shift -> undo scale/translate -> undo rotation."""
    return apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )


def mpm_to_world_positions(
    mpm_positions: torch.Tensor,
    rotation_matrices,
    scale_origin: torch.Tensor,
    original_mean_pos: torch.Tensor,
):
    """MPM domain positions -> world-space positions."""
    return undo_all_transforms(
        mpm_positions, rotation_matrices, scale_origin, original_mean_pos
    )


# ── Camera / observant coordinate helpers ─────────────────────────────

def generate_local_coord(vertical_vector):
    """Build a local coordinate frame from a vertical direction vector."""
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:
        horizontal_1 = np.array([0.72, 0.37, -0.67])
    horizontal_1 = horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)
    return vertical_vector, horizontal_1, horizontal_2


def get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
):
    """Convert MPM-space camera center and up to world space.

    .. deprecated:: Camera setup now uses the scene center directly
        in internal Y-up coordinates.
    """
    viewpoint_center_worldspace = undo_all_transforms(
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos
    )
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center
    worldspace_up = undo_all_transforms(
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace
    viewpoint_center_worldspace = np.squeeze(
        viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0
    )
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )
    observant_coordinates = np.column_stack((h1, h2, vertical))
    return viewpoint_center_worldspace, observant_coordinates
