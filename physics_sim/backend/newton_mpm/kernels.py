"""
Warp kernels for Newton MPM backend.

Computes covariance and rotation from the deformation gradient (particle_transform)
tracked by Newton's implicit MPM solver.
"""

import warp as wp


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
