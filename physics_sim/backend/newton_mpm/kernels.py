"""
Warp kernels for Newton MPM backend.

Computes covariance and rotation from the deformation gradient (particle_transform)
tracked by Newton's implicit MPM solver.
"""

import warp as wp


# ── Helper: rotation matrix → quaternion (wxyz) ──────────────────────────

@wp.func
def _rotmat_to_quat_wxyz(R: wp.mat33) -> wp.vec4:
    """Shepperd's method: 3×3 rotation matrix → quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = wp.sqrt(trace + 1.0) * 2.0  # s = 4w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = wp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # s = 4x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = wp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # s = 4y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = wp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # s = 4z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    norm = wp.sqrt(w * w + x * x + y * y + z * z)
    inv_n = 1.0 / wp.max(norm, 1e-12)
    return wp.vec4(w * inv_n, x * inv_n, y * inv_n, z * inv_n)


@wp.func
def _quat_mul_wxyz(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    """Hamilton product of two quaternions in (w, x, y, z) layout."""
    aw = a[0]; ax = a[1]; ay = a[2]; az = a[3]
    bw = b[0]; bx = b[1]; by = b[2]; bz = b[3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return wp.vec4(ow, ox, oy, oz)


# ── Covariance kernel ─────────────────────────────────────────────────────

@wp.kernel
def compute_cov_from_F(
    F: wp.array(dtype=wp.mat33),
    init_cov: wp.array(dtype=float),       # (N*6,) upper-triangle, flat
    out_cov: wp.array(dtype=float),         # (N*6,) upper-triangle, flat
):
    """cov = F @ init_cov @ F^T  (symmetric → 6 upper-triangle components)."""
    p = wp.tid()

    Fp = F[p]

    # Reconstruct symmetric 3×3 from packed upper-triangle
    c = wp.mat33(0.0)
    c[0, 0] = init_cov[p * 6 + 0]
    c[0, 1] = init_cov[p * 6 + 1]
    c[0, 2] = init_cov[p * 6 + 2]
    c[1, 0] = init_cov[p * 6 + 1]
    c[1, 1] = init_cov[p * 6 + 3]
    c[1, 2] = init_cov[p * 6 + 4]
    c[2, 0] = init_cov[p * 6 + 2]
    c[2, 1] = init_cov[p * 6 + 4]
    c[2, 2] = init_cov[p * 6 + 5]

    cov = Fp * c * wp.transpose(Fp)

    out_cov[p * 6 + 0] = cov[0, 0]
    out_cov[p * 6 + 1] = cov[0, 1]
    out_cov[p * 6 + 2] = cov[0, 2]
    out_cov[p * 6 + 3] = cov[1, 1]
    out_cov[p * 6 + 4] = cov[1, 2]
    out_cov[p * 6 + 5] = cov[2, 2]


# ── Rotation-only kernel (3DGS path) ─────────────────────────────────────

@wp.kernel
def compute_R_from_F(
    F: wp.array(dtype=wp.mat33),
    out_R: wp.array(dtype=wp.mat33),
):
    """Extract rotation via polar SVD:  F = U Σ V^T  →  R = U V^T."""
    p = wp.tid()

    Fp = F[p]

    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(Fp, U, sig, V)

    # Ensure proper rotation (det > 0)
    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # R = U V^T, stored transposed to match WarpMPM convention
    R = U * wp.transpose(V)
    out_R[p] = wp.transpose(R)


# ── Combined rotation + quats + scales kernel (2DGS path) ────────────────

@wp.kernel
def compute_R_quats_scales_from_F(
    F: wp.array(dtype=wp.mat33),
    init_quats: wp.array(dtype=wp.vec4),    # (N,) wxyz
    init_scales: wp.array(dtype=float),     # (N * num_scales,) flat
    num_scales: int,
    out_R: wp.array(dtype=wp.mat33),
    out_quats: wp.array(dtype=wp.vec4),     # (N,) wxyz
    out_scales: wp.array(dtype=float),      # (N * num_scales,) flat
):
    """SVD of F → rotation matrix (for SH), quaternion, and scales (for 2DGS).

    Reuses the same SVD as ``compute_R_from_F``, additionally composing
    ``R_deform @ R_init`` as a quaternion and scaling singular values by
    initial scales.
    """
    p = wp.tid()

    Fp = F[p]

    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(Fp, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    R = U * wp.transpose(V)
    out_R[p] = wp.transpose(R)

    # R_deform → quaternion, then compose with init_quat
    q_deform = _rotmat_to_quat_wxyz(R)
    q_init = init_quats[p]
    out_quats[p] = _quat_mul_wxyz(q_deform, q_init)

    # Scales: sig_i * init_scale_i (approximation for small deformation)
    for s in range(num_scales):
        out_scales[p * num_scales + s] = sig[s] * init_scales[p * num_scales + s]
