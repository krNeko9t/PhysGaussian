"""
Newton VBD backend — unified rigid + soft body simulation.

Uses a **single** ``ModelBuilder`` and ``SolverVBD`` to handle both
rigid bodies (``add_body`` + ``add_shape_*``) and FEM soft bodies
(``add_soft_grid``) in one scene, with automatic contact handling.

Each object in the config declares ``"physics": "rigid"`` or
``"physics": "soft"``.  Rigid bodies reuse the same collision-geometry
strategies as ``NewtonRigidBackend``.  Soft bodies use a **regular
tetrahedral grid** (Newton's ``add_soft_grid``) covering the GS
particle bounding box — this produces uniform, well-conditioned
elements that Newton's VBD solver is optimized for.

GS particles are **embedded** inside the tet grid via barycentric
coordinates, so the coarse FEM grid drives the dense GS point cloud.

Lifecycle (called by the pipeline):
    initialize  → store particles, create ModelBuilder
    set_material → create rigid/soft bodies per object
    set_boundary_conditions → add ground planes / walls
    finalize    → builder.color(), finalize, create VBD solver
    step        → clear forces, collide, solver step, swap states
    get_state   → rigid: broadcast R,t;  soft: bary-interp from tets
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverVBD

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.backend.newton_vbd.barycentric import compute_barycentric
from physics_sim.backend.newton_vbd.rigid_mesh import create_rigid_body
from physics_sim.backend.newton_vbd.state_export import export_state
from physics_sim.coord import (
    E_GRAVITY_MISSING,
    gravity_contract_error,
    normalize_internal_gravity,
)
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


# ── Per-object data structures ───────────────────────────────────────

@dataclass
class _RigidInfo:
    """Back-mapping data for a rigid body."""
    body_idx: int
    particle_indices: list[int]
    init_local_pos: torch.Tensor    # (N, 3)  body-local coords
    init_cov_3x3: torch.Tensor      # (N, 3, 3)
    init_quats: Optional[torch.Tensor] = None   # (N, 4) wxyz — for 2DGS
    init_scales: Optional[torch.Tensor] = None   # (N, 2|3) — for 2DGS


@dataclass
class _SoftInfo:
    """Back-mapping data for an FEM soft body."""
    particle_indices: list[int]
    tet_ids: np.ndarray              # (N,)  which tet each GS particle is in
    bary_coords: np.ndarray          # (N, 4) barycentric weights
    vert_offset: int                 # offset in global particle_q
    vert_count: int                  # number of tet mesh vertices
    tet_cells: np.ndarray            # (T, 4)  for computing deformation gradient
    rest_verts: np.ndarray           # (V, 3)  rest-pose tet vertices
    init_cov_6: torch.Tensor         # (N, 6)


# ═════════════════════════════════════════════════════════════════════
# Backend
# ═════════════════════════════════════════════════════════════════════

class NewtonVBDBackend(PhysicsBackend):
    """Unified rigid + soft body backend using Newton VBD solver."""

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self._builder: Optional[newton.ModelBuilder] = None
        self._model: Optional[newton.Model] = None
        self._solver: Optional[SolverVBD] = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._collision_pipeline = None
        self._contacts = None

        # Per-object info
        self._rigid_bodies: list[_RigidInfo] = []
        self._soft_bodies: list[_SoftInfo] = []
        self._n_particles: int = 0

        # Initial data (kept for get_state)
        self._init_positions: Optional[torch.Tensor] = None
        self._init_covariances: Optional[torch.Tensor] = None

        # Config
        self._gravity: tuple[float, float, float] | None = None
        self._solver_iterations = 10
        self._collision_geo = "convex_hull"  # for rigid bodies
        self._alpha = None
        self._max_triangles = 500
        self._grid_lim = 2.0

        # Track particle offset for soft bodies
        self._particle_offset = 0

        # Soft body deformation options
        self._debug_soft_no_deformation = False  # skip F update, only interp pos
        self._sv_clamp_min = 0.1   # min allowed singular value of F
        self._sv_clamp_max = 5.0   # max allowed singular value of F
        self._frame_counter = 0    # for periodic diagnostics

        # Self-contact parameters (default: OFF — enabling with wrong
        # radius/margin relative to mesh resolution completely blocks
        # gravity because the solver spends all iterations resolving
        # self-contact instead of falling).
        self._particle_self_contact = False
        self._particle_self_contact_radius = 0.001
        self._particle_self_contact_margin = 0.002

        # Soft contact stiffness (particle↔rigid shape contact).
        # Newton official examples use 1e2 for ground contact.
        # Too high → explosion on impact; too low → penetration.
        self._soft_contact_ke = 1e2
        self._soft_contact_kd = 1e-5
        self._soft_contact_mu = 0.5

        # Max contact buffer size (Newton internal)
        self._rigid_contact_max = 100_000

    # ── PhysicsBackend interface ─────────────────────────────────────

    def initialize(
        self,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        covariances: torch.Tensor,
        **kwargs,
    ) -> None:
        self._n_particles = positions.shape[0]
        self._init_positions = positions.clone().to(self._device)
        self._init_covariances = covariances.clone().to(self._device)
        self._grid_lim = float(kwargs.get("grid_lim", 2.0))

        iq = kwargs.get("init_quats")
        self._init_quats = iq.clone().to(self._device) if iq is not None else None
        isc = kwargs.get("init_scales")
        self._init_scales = isc.clone().to(self._device) if isc is not None else None

        # Collision geometry for rigid bodies
        self._collision_geo = kwargs.get("collision_geometry", "convex_hull")
        self._alpha = kwargs.get("alpha", None)
        self._max_triangles = int(kwargs.get("max_triangles", 500))

        # Soft body deformation gradient options
        self._debug_soft_no_deformation = bool(
            kwargs.get("debug_soft_no_deformation", False)
        )
        self._sv_clamp_min = float(kwargs.get("sv_clamp_min", 0.1))
        self._sv_clamp_max = float(kwargs.get("sv_clamp_max", 5.0))
        if self._debug_soft_no_deformation:
            LOGGER.info(
                "[NewtonVBD] DEBUG: soft body deformation gradient DISABLED "
                "(positions only, no cov/rot update)"
            )
        else:
            LOGGER.info(
                f"[NewtonVBD] F singular value clamp: "
                f"[{self._sv_clamp_min}, {self._sv_clamp_max}]"
            )

        # Self-contact parameters
        self._particle_self_contact = bool(
            kwargs.get("particle_self_contact", False)
        )
        self._particle_self_contact_radius = float(
            kwargs.get("particle_self_contact_radius", 0.001)
        )
        self._particle_self_contact_margin = float(
            kwargs.get("particle_self_contact_margin", 0.002)
        )
        self._soft_contact_ke = float(
            kwargs.get("soft_contact_ke", 1e2)
        )
        self._soft_contact_kd = float(
            kwargs.get("soft_contact_kd", 1e-5)
        )
        self._soft_contact_mu = float(
            kwargs.get("soft_contact_mu", 0.5)
        )

        # Max contact buffer
        self._rigid_contact_max = int(
            kwargs.get("rigid_contact_max", 100_000)
        )

        # Contact margin
        contact_margin = kwargs.get("contact_margin", 0.01)

        # Create builder
        self._builder = newton.ModelBuilder()
        self._builder.default_shape_cfg.contact_margin = float(contact_margin)

        LOGGER.info(
            f"[NewtonVBD] collision_geometry={self._collision_geo}, "
            f"contact_margin={contact_margin}"
        )

    def set_material(self, material_params: dict) -> None:
        builder = self._builder
        assert builder is not None

        # Gravity
        if "g" not in material_params:
            raise gravity_contract_error(
                E_GRAVITY_MISSING,
                backend="newton_vbd",
                config_path="material.g",
                detail="set_material() missing required gravity vector",
                suggestion="pass g as [0, -|g|, 0], usually from backend_init._resolve_gravity",
            )
        self._gravity = normalize_internal_gravity(
            material_params.get("g"),
            backend="newton_vbd",
            config_path="material.g",
            allow_scalar=False,
        )

        # Solver options
        solver_opts = material_params.get("newton_solver_opts", {})
        self._solver_iterations = solver_opts.get("iterations", 10)

        # Default shape config for rigid bodies
        default_mu = float(material_params.get("mu", 0.5))
        default_density = float(material_params.get("density", 500.0))
        base_cfg = newton.ModelBuilder.ShapeConfig(
            density=default_density,
            mu=default_mu,
        )

        # Create bodies from per-object definitions
        per_object = material_params.get("per_object")
        if per_object is not None:
            for obj in per_object:
                mat = obj.get("material", {})
                physics_type = mat.get("physics", "rigid")

                if physics_type == "rigid":
                    cfg = copy(base_cfg)
                    cfg.density = float(mat.get("density", cfg.density))
                    if "mu" in mat:
                        cfg.mu = float(mat["mu"])
                    self._create_rigid_body(
                        particle_indices=obj["particle_indices"],
                        shape_cfg=cfg,
                        name=obj.get("name", "?"),
                        collision_geo=mat.get("collision_geometry", None),
                    )
                elif physics_type == "soft":
                    self._create_soft_body(
                        particle_indices=obj["particle_indices"],
                        material=mat,
                        name=obj.get("name", "?"),
                    )
                else:
                    raise ValueError(
                        f"Unknown physics type '{physics_type}' for "
                        f"object '{obj.get('name', '?')}'. "
                        "Use 'rigid' or 'soft'."
                    )
        else:
            # Single body → default to rigid
            self._create_rigid_body(
                particle_indices=list(range(self._n_particles)),
                shape_cfg=base_cfg,
                name="single_body",
            )

    def set_boundary_conditions(
        self, bc_params: list, time_params: dict
    ) -> None:
        builder = self._builder
        assert builder is not None

        if not isinstance(bc_params, list):
            return

        for bc in bc_params:
            bc_type = bc.get("type", "")
            if bc_type == "surface_collider":
                normal = bc["normal"]
                point = bc["point"]
                d = -(normal[0] * point[0] + normal[1] * point[1] +
                      normal[2] * point[2])
                surface = bc.get("surface", "slip")
                mu = 0.5
                if surface == "sticky":
                    mu = 1.0
                elif surface == "slip":
                    mu = 0.0
                if "friction" in bc:
                    mu = float(bc["friction"])

                plane_cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
                builder.add_shape_plane(
                    plane=(float(normal[0]), float(normal[1]),
                           float(normal[2]), float(d)),
                    cfg=plane_cfg,
                )
                LOGGER.info(
                    f"[NewtonVBD] Plane: normal={normal}, "
                    f"point={point}, mu={mu}"
                )

    def finalize(self) -> None:
        builder = self._builder
        assert builder is not None

        # VBD requires coloring
        builder.color()

        self._model = builder.finalize(device=self._device)
        self._model.set_gravity(self._gravity)
        LOGGER.info("[NewtonVBD] gravity=%s", self._gravity)

        # Soft contact parameters (particle-shape and self-contact)
        self._model.soft_contact_ke = self._soft_contact_ke
        self._model.soft_contact_kd = self._soft_contact_kd
        self._model.soft_contact_mu = self._soft_contact_mu

        # Limit contact buffer
        self._model.rigid_contact_max = self._rigid_contact_max

        # Create VBD solver
        self._solver = SolverVBD(
            self._model,
            iterations=self._solver_iterations,
            particle_enable_self_contact=self._particle_self_contact,
            particle_self_contact_radius=self._particle_self_contact_radius,
            particle_self_contact_margin=self._particle_self_contact_margin,
        )
        LOGGER.info(
            f"[NewtonVBD] SolverVBD: iterations={self._solver_iterations}, "
            f"self_contact={self._particle_self_contact}"
        )
        if self._particle_self_contact:
            LOGGER.info(
                f"[NewtonVBD]   self_contact_radius="
                f"{self._particle_self_contact_radius}, "
                f"margin={self._particle_self_contact_margin}, "
                f"ke={self._soft_contact_ke}"
            )

        # Double-buffered states
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()

        # Collision pipeline (Newton ≥1.1: `CollisionPipeline`, broad_phase as string)
        self._collision_pipeline = newton.CollisionPipeline(
            self._model,
            broad_phase="sap",
        )
        self._contacts = self._model.collide(
            self._state_0,
            collision_pipeline=self._collision_pipeline,
        )

        self._builder = None

    def step(self, dt: float, frame: int) -> None:
        self._state_0.clear_forces()
        self._contacts = self._model.collide(
            self._state_0,
            collision_pipeline=self._collision_pipeline,
        )
        self._solver.step(
            self._state_0,
            self._state_1,
            self._control,
            self._contacts,
            dt,
        )
        self._state_0, self._state_1 = self._state_1, self._state_0

    def get_state(self) -> SimulationState:
        state, self._frame_counter = export_state(
            state_0=self._state_0,
            rigid_bodies=self._rigid_bodies,
            soft_bodies=self._soft_bodies,
            n_particles=self._n_particles,
            device=self._device,
            init_quats=self._init_quats,
            init_scales=self._init_scales,
            frame_counter=self._frame_counter,
            debug_soft_no_deformation=self._debug_soft_no_deformation,
            sv_clamp_min=self._sv_clamp_min,
            sv_clamp_max=self._sv_clamp_max,
        )
        return state

    # ── Rigid body creation ──────────────────────────────────────────

    def _create_rigid_body(
        self,
        particle_indices: list[int],
        shape_cfg: newton.ModelBuilder.ShapeConfig,
        name: str,
        collision_geo: str | None = None,
    ) -> None:
        builder = self._builder
        assert builder is not None
        result = create_rigid_body(
            builder=builder,
            init_positions=self._init_positions,
            init_covariances=self._init_covariances,
            init_quats=self._init_quats,
            init_scales=self._init_scales,
            device=self._device,
            particle_indices=particle_indices,
            shape_cfg=shape_cfg,
            name=name,
            collision_geo=collision_geo,
            default_collision_geo=self._collision_geo,
            alpha=self._alpha,
            max_triangles=self._max_triangles,
        )
        self._rigid_bodies.append(
            _RigidInfo(
                body_idx=result.body_idx,
                particle_indices=particle_indices,
                init_local_pos=result.init_local_pos,
                init_cov_3x3=result.init_cov_3x3,
                init_quats=result.init_quats,
                init_scales=result.init_scales,
            )
        )

    # ── Soft body creation ───────────────────────────────────────────

    def _create_soft_body(
        self,
        particle_indices: list[int],
        material: dict,
        name: str,
    ) -> None:
        """Create a soft body using Newton's ``add_soft_grid``.

        Instead of the fragile TetGen pipeline (alpha shape → tet mesh →
        degenerate filter), we generate a **uniform regular tet grid**
        covering the GS particle bounding box.  This matches Newton's
        own examples and produces well-conditioned elements that VBD is
        optimized for.

        Configurable per-object material keys:
            cell_size      : float — tet cell size (auto if omitted)
            grid_resolution: int   — cells per longest axis (default 8,
                                     only used when cell_size is omitted)
            grid_padding   : float — padding around bbox (default 0.05)
            density        : float — mass density   (default 1000)
            k_mu           : float — shear modulus   (default 1e5)
            k_lambda       : float — bulk modulus    (default 1e5)
            k_damp         : float — damping coeff   (default 1e-3)
        """
        builder = self._builder
        assert builder is not None

        idx_t = torch.tensor(particle_indices, dtype=torch.long)
        pos = self._init_positions[idx_t].float()
        pos_np = pos.detach().cpu().numpy()

        # ── Bounding box with padding ──────────────────────────────────
        bbox_min = pos_np.min(axis=0)
        bbox_max = pos_np.max(axis=0)
        padding = float(material.get("grid_padding", 0.05))
        bbox_min = bbox_min - padding
        bbox_max = bbox_max + padding
        extent = bbox_max - bbox_min

        # ── Grid parameters ────────────────────────────────────────────
        # cell_size: explicit value, or auto-compute from grid_resolution.
        cell_size = material.get("cell_size", None)
        if cell_size is not None:
            cell_size = float(cell_size)
        else:
            grid_res = int(material.get("grid_resolution", 8))
            cell_size = float(extent.max() / max(grid_res, 1))

        dim_x = max(1, int(np.ceil(extent[0] / cell_size)))
        dim_y = max(1, int(np.ceil(extent[1] / cell_size)))
        dim_z = max(1, int(np.ceil(extent[2] / cell_size)))

        # Recalculate per-axis cell sizes to exactly cover the bbox.
        cell_x = extent[0] / dim_x
        cell_y = extent[1] / dim_y
        cell_z = extent[2] / dim_z

        # ── Material parameters ────────────────────────────────────────
        density = float(material.get("density", 1e3))
        k_mu = float(material.get("k_mu", 1e5))
        k_lambda = float(material.get("k_lambda", 1e5))
        k_damp = float(material.get("k_damp", 1e-3))

        # ── Add regular tet grid via Newton's API ──────────────────────
        vert_offset = self._particle_offset

        builder.add_soft_grid(
            pos=wp.vec3(
                float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_x,
            cell_y=cell_y,
            cell_z=cell_z,
            density=density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )

        vert_count = (dim_x + 1) * (dim_y + 1) * (dim_z + 1)
        self._particle_offset += vert_count

        # ── Reconstruct grid vertices in numpy ─────────────────────────
        # Exactly mirrors Newton's add_soft_grid vertex generation order
        # (z-major, then y, then x) so indices match particle_q layout.
        grid_verts = np.zeros((vert_count, 3), dtype=np.float64)
        vi = 0
        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    grid_verts[vi] = [
                        x * cell_x + bbox_min[0],
                        y * cell_y + bbox_min[1],
                        z * cell_z + bbox_min[2],
                    ]
                    vi += 1

        # ── Reconstruct tet connectivity ───────────────────────────────
        # Exactly mirrors Newton's 5-tet alternating decomposition.
        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        tet_list = []
        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = grid_index(x, y, z)
                    v1 = grid_index(x + 1, y, z)
                    v2 = grid_index(x + 1, y, z + 1)
                    v3 = grid_index(x, y, z + 1)
                    v4 = grid_index(x, y + 1, z)
                    v5 = grid_index(x + 1, y + 1, z)
                    v6 = grid_index(x + 1, y + 1, z + 1)
                    v7 = grid_index(x, y + 1, z + 1)

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        tet_list.extend([
                            [v0, v1, v4, v3],
                            [v2, v3, v6, v1],
                            [v5, v4, v1, v6],
                            [v7, v6, v3, v4],
                            [v4, v1, v6, v3],
                        ])
                    else:
                        tet_list.extend([
                            [v1, v2, v5, v0],
                            [v3, v0, v7, v2],
                            [v4, v7, v0, v5],
                            [v6, v5, v2, v7],
                            [v5, v2, v7, v0],
                        ])

        tet_cells = np.array(tet_list, dtype=np.int32)

        # ── Embed GS particles in tet grid ─────────────────────────────
        LOGGER.info(
            f"[NewtonVBD] Embedding {len(particle_indices)} GS particles "
            f"in {tet_cells.shape[0]} tets..."
        )
        tet_ids, bary, outside_count = compute_barycentric(
            pos_np, grid_verts, tet_cells
        )

        if outside_count > 0:
            pct = 100.0 * outside_count / len(particle_indices)
            LOGGER.warning(
                f"[NewtonVBD] WARNING: {outside_count}/{len(particle_indices)} "
                f"({pct:.1f}%) GS particles outside grid → clamped"
            )
        else:
            LOGGER.info(
                f"[NewtonVBD] All {len(particle_indices)} particles "
                f"inside grid"
            )

        cov6 = self._init_covariances[idx_t]

        self._soft_bodies.append(_SoftInfo(
            particle_indices=particle_indices,
            tet_ids=tet_ids,
            bary_coords=bary,
            vert_offset=vert_offset,
            vert_count=vert_count,
            tet_cells=tet_cells,
            rest_verts=grid_verts.copy(),
            init_cov_6=cov6,
        ))
        LOGGER.info(
            f"[NewtonVBD] Soft '{name}': {len(particle_indices)} GS particles, "
            f"grid {dim_x}x{dim_y}x{dim_z} = {vert_count} verts, "
            f"{tet_cells.shape[0]} tets, "
            f"cell=[{cell_x:.4f},{cell_y:.4f},{cell_z:.4f}], "
            f"density={density}, k_mu={k_mu:.0e}, k_lambda={k_lambda:.0e}, "
            f"k_damp={k_damp:.0e}"
        )
