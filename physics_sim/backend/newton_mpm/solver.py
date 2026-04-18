"""
Newton implicit-MPM backend adapter that implements the PhysicsBackend interface.

This wraps Newton's SolverImplicitMPM and exposes a clean, backend-agnostic API
through PhysicsBackend / SimulationState, matching the existing WarpMPMBackend.
"""

from __future__ import annotations

import math
import numpy as np
import warp as wp
import torch

import newton
from newton.solvers import SolverImplicitMPM

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.coord import (
    E_GRAVITY_MISSING,
    gravity_contract_error,
    normalize_internal_gravity,
)
from physics_sim.backend.newton_mpm.kernels import (
    compute_cov_from_F,
    compute_R_from_F,
    compute_R_quats_scales_from_F,
)


# ── Material presets ─────────────────────────────────────────────────────
# Calibrated against Newton's own MPM examples (example_mpm_multi_material,
# example_mpm_granular).  Only yield / plasticity parameters differ between
# presets — young_modulus and poisson_ratio come from the config.
#
# Key Newton MPM parameters:
#   yield_stress      – shear yield threshold (0 = instantly flows)
#   yield_pressure    – compression yield threshold
#   friction          – Coulomb friction coefficient
#   tensile_yield_ratio – tensile strength (0 = no tensile, 1 = full)
#   hardening         – plastic strain hardening

_MATERIAL_PRESETS: dict[str, dict] = {
    # Granular: flows freely, friction-dominated (Newton default behaviour)
    "sand": dict(
        friction=None,            # from friction_angle if given, else 0.68
        yield_pressure=1.0e12,
        yield_stress=0.0,         # zero → instantly yields in shear
        tensile_yield_ratio=0.0,
        hardening=0.0,
    ),
    # Viscoelastic: soft, deforms, slight bounce, settles (jelly/gelatin)
    # Moderate yield allows some plastic dissipation → no infinite bouncing.
    # Hardening provides progressive stiffening → mimics viscoelastic damping.
    "jelly": dict(
        friction=0.0,
        yield_pressure=1.0e6,
        yield_stress=5.0e4,       # high but not infinite → mostly elastic, slight flow
        tensile_yield_ratio=0.1,
        hardening=3.0,            # deform more → get stiffer → settles
    ),
    # Snow: compressible, soft, hardens under strain (from Newton example)
    "snow": dict(
        friction=0.1,
        yield_pressure=2.0e4,
        yield_stress=1.0e3,
        tensile_yield_ratio=0.05,
        hardening=10.0,
    ),
    # Mud: viscous, cohesive, no friction (from Newton example)
    "mud": dict(
        friction=0.0,
        yield_pressure=1.0e10,
        yield_stress=3.0e2,
        tensile_yield_ratio=1.0,
        hardening=2.0,
    ),
    # Metal: stiff, high yield, minimal plasticity
    "metal": dict(
        friction=0.3,
        yield_pressure=1.0e12,
        yield_stress=1.0e8,
        tensile_yield_ratio=0.0,
        hardening=0.0,
    ),
    # Foam: soft, compressible
    "foam": dict(
        friction=0.5,
        yield_pressure=1.0e6,
        yield_stress=1.0e4,
        tensile_yield_ratio=0.1,
        hardening=5.0,
    ),
    # Plasticine: soft, deforms permanently
    "plasticine": dict(
        friction=0.5,
        yield_pressure=1.0e6,
        yield_stress=5.0e3,
        tensile_yield_ratio=0.1,
        hardening=3.0,
    ),
}


def _friction_from_angle(friction_angle_deg: float) -> float:
    """Convert friction angle (degrees) to Coulomb friction coefficient."""
    rad = friction_angle_deg / 180.0 * math.pi
    return math.tan(rad)


class NewtonMPMBackend(PhysicsBackend):
    """Physics backend powered by Newton's implicit MPM solver."""

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self._builder: newton.ModelBuilder | None = None
        self._model: newton.Model | None = None
        self._solver: SolverImplicitMPM | None = None
        self._state_0: newton.State | None = None
        self._state_1: newton.State | None = None
        self._control: newton.Control | None = None

        # Covariance tracking (Newton doesn't store per-particle cov natively)
        self._init_cov: wp.array | None = None   # (N*6,) float, flat
        self._out_cov: wp.array | None = None     # (N*6,) float, flat
        self._out_R: wp.array | None = None       # (N,) mat33

        # 2DGS support: quats + scales output from SVD
        self._init_quats_wp: wp.array | None = None    # (N,) vec4 wxyz
        self._init_scales_wp: wp.array | None = None   # (N*S,) float flat
        self._out_quats_wp: wp.array | None = None     # (N,) vec4 wxyz
        self._out_scales_wp: wp.array | None = None    # (N*S,) float flat
        self._num_scales: int = 0

        self._n_particles: int = 0
        self._grid_lim: float = 2.0
        self._time: float = 0.0
        self._substep_dt: float = 0.0  # last substep dt, for deferred frame update
        self._frames_dirty: bool = False  # whether particle frames need updating

        # Solver options — start from Newton's own defaults.
        # Overrides come from config JSON ("newton_mpm" → "solver" section)
        # and are applied in set_material().
        self._solver_opts = SolverImplicitMPM.Config()
        self._material_params: dict = {}
        self._cov_np: np.ndarray | None = None
        self._volumes: np.ndarray | None = None
        self._bc_params: list = []
        self._time_params: dict = {}

    # ------------------------------------------------------------------
    # PhysicsBackend interface
    # ------------------------------------------------------------------

    def initialize(
        self,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        covariances: torch.Tensor,
        *,
        n_grid: int = 100,
        grid_lim: float = 2.0,
        **kwargs,
    ) -> None:
        """Build a Newton Model from the preprocessed GS particle data.

        Args:
            positions:   (N, 3) particle positions in rotated world space.
            volumes:     (N,)   per-particle volumes (used for mass).
            covariances: (N, 6) upper-triangle covariance matrices.
            n_grid:      Grid resolution (default 100).
            grid_lim:    Ignored (kept for interface compat).  voxel_size
                         is computed from particle bounding box / n_grid.
        """
        self._n_particles = positions.shape[0]
        n = self._n_particles

        # ── Compute voxel_size from particle bounding box ────────────
        pos_np = positions.detach().cpu().numpy().astype(np.float32)
        lo = pos_np.min(axis=0)
        hi = pos_np.max(axis=0)
        max_extent = float((hi - lo).max())
        if max_extent < 1e-8:
            max_extent = 1.0
        voxel_size = max_extent / max(n_grid, 1)
        self._grid_lim = max_extent
        self._bbox_lo = lo.tolist()
        self._bbox_hi = hi.tolist()
        self._solver_opts.voxel_size = voxel_size
        print(f"[NewtonMPM] bbox extent={max_extent:.4f}, voxel_size={voxel_size:.6f}")

        # ── Build Newton ModelBuilder (finalized in finalize()) ──────
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        vol_np = volumes.detach().cpu().numpy().astype(np.float32)

        # Compute per-particle mass from volume (density set later in set_material)
        # Use a placeholder density of 1.0; actual mass will be recomputed.
        mass_np = vol_np.copy()  # mass = vol * density, density=1 placeholder

        # Compute radius from volume (sphere approximation)
        radius_np = np.cbrt(vol_np * 3.0 / (4.0 * math.pi)).astype(np.float32)
        radius_np = np.maximum(radius_np, 1e-6)

        # Batch add particles
        pos_list = [wp.vec3(float(pos_np[i, 0]), float(pos_np[i, 1]), float(pos_np[i, 2]))
                    for i in range(n)]
        vel_list = [wp.vec3(0.0, 0.0, 0.0)] * n
        mass_list = [float(mass_np[i]) for i in range(n)]
        radius_list = [float(radius_np[i]) for i in range(n)]
        flags_list = [int(newton.ParticleFlags.ACTIVE)] * n

        builder.add_particles(
            pos=pos_list,
            vel=vel_list,
            mass=mass_list,
            radius=radius_list,
            flags=flags_list,
        )

        # Store builder + arrays; finalize() will create model + solver.
        self._builder = builder
        self._cov_np = (
            covariances.detach().cpu().numpy().astype(np.float32).reshape(-1)
        )
        self._volumes = vol_np

        # 2DGS: store initial quats + scales for SVD-based output
        iq = kwargs.get("init_quats")
        isc = kwargs.get("init_scales")
        if iq is not None and isc is not None:
            self._init_quats_np = iq.detach().cpu().numpy().astype(np.float32)
            self._init_scales_np = isc.detach().cpu().numpy().astype(np.float32)
            self._num_scales = isc.shape[1]
        else:
            self._init_quats_np = None
            self._init_scales_np = None

    def set_material(self, material_params: dict) -> None:
        """Configure Newton MPM material from the existing config dict.

        Supports the same keys as WarpMPMBackend:
            material, E, nu, density, friction_angle, yield_stress,
            hardening, g, rpic_damping, grid_v_damping_scale, etc.
        """
        # Store material params; applied to the finalized model in finalize().
        self._material_params = material_params

        # ── Solver options from material params ──────────────────────
        # Transfer scheme: map rpic_damping → "pic" / "apic"
        rpic = material_params.get("rpic_damping", 0.0)
        if rpic < 0:
            self._solver_opts.transfer_scheme = "pic"
        else:
            self._solver_opts.transfer_scheme = "apic"

        # Apply solver option overrides from config ("newton_mpm" → "solver").
        # Any key that exists as an attribute on SolverImplicitMPM.Config
        # can be overridden here.  Unrecognised keys are warned about.
        newton_opts = material_params.get("newton_solver_opts", {})
        _SOLVER_OPT_TYPES = {
            "max_iterations": int,
            "tolerance": float,
            "solver": str,
            "grid_type": str,
            "transfer_scheme": str,
            "air_drag": float,
            "grid_padding": int,
        }
        for key, val in newton_opts.items():
            if hasattr(self._solver_opts, key):
                cast = _SOLVER_OPT_TYPES.get(key, type(val))
                setattr(self._solver_opts, key, cast(val))
                print(f"[NewtonMPM] solver.{key} = {cast(val)}")
            else:
                print(f"[NewtonMPM] Warning: unknown solver option '{key}', ignoring.")

        # NOTE: Material fields on the Newton model are not available until
        # builder.finalize() is called. We apply physical parameters in finalize().

    def _apply_material_to_model(self) -> None:
        """Apply stored material parameters to the finalized Newton model."""
        material_params = self._material_params
        model = self._model
        assert model is not None
        assert self._volumes is not None

        # ── Gravity ──────────────────────────────────────────────────
        if "g" not in material_params:
            raise gravity_contract_error(
                E_GRAVITY_MISSING,
                backend="newton_mpm",
                config_path="material.g",
                detail="_apply_material_to_model() missing required gravity vector",
                suggestion="pass g as [0, -|g|, 0], usually from backend_init._resolve_gravity",
            )
        g = normalize_internal_gravity(
            material_params.get("g"),
            backend="newton_mpm",
            config_path="material.g",
            allow_scalar=False,
        )
        model.set_gravity(g)

        # ── Material preset ──────────────────────────────────────────
        mat_name = material_params.get("material", "jelly")
        preset = _MATERIAL_PRESETS.get(mat_name, _MATERIAL_PRESETS["jelly"])

        E = material_params.get("E", 1e5)
        nu = material_params.get("nu", 0.3)

        model.mpm.young_modulus.fill_(float(E))
        model.mpm.poisson_ratio.fill_(float(nu))

        # Friction: from preset or friction_angle conversion
        friction = preset.get("friction")
        if mat_name == "sand" and "friction_angle" in material_params:
            friction = _friction_from_angle(material_params["friction_angle"])
        if friction is not None:
            model.mpm.friction.fill_(float(friction))

        # Yield parameters from preset, with config overrides
        yp = preset.get("yield_pressure", 1e12)
        model.mpm.yield_pressure.fill_(float(yp))

        ys = preset.get("yield_stress", 0.0)
        if "yield_stress" in material_params:
            ys = material_params["yield_stress"]
        if ys is not None:
            model.mpm.yield_stress.fill_(float(ys))

        tyr = preset.get("tensile_yield_ratio", 0.0)
        model.mpm.tensile_yield_ratio.fill_(float(tyr))

        h = preset.get("hardening", 0.0)
        if "hardening" in material_params:
            h = material_params["hardening"]
        model.mpm.hardening.fill_(float(h))

        # ── Density → recompute mass ─────────────────────────────────
        density = material_params.get("density", 200.0)
        mass_np = (self._volumes * density).astype(np.float32)
        mass_wp = wp.from_numpy(mass_np, dtype=float, device=self._device)
        wp.copy(mass_wp, model.particle_mass)

        inv_mass_np = np.where(mass_np > 0.0, 1.0 / mass_np, 0.0).astype(np.float32)
        inv_mass_wp = wp.from_numpy(inv_mass_np, dtype=float, device=self._device)
        wp.copy(inv_mass_wp, model.particle_inv_mass)

        # Per-object material overrides (multi-object mode)
        if "per_object" in material_params:
            self._apply_per_object_materials(material_params["per_object"])

    def _apply_per_object_materials(self, per_object: list[dict]) -> None:
        """Apply per-object material overrides using Newton's native API.

        Uses the same ``model.mpm.attr[wp_idx].fill_(val)`` pattern as
        Newton's own ``example_mpm_multi_material.py``.
        """
        model = self._model

        for obj in per_object:
            raw_idx = obj["particle_indices"]
            if len(raw_idx) == 0:
                continue
            mat = obj.get("material", {})
            name = obj.get("name", "?")

            # Build a warp index array (same approach as Newton's examples)
            idx_wp = wp.array(
                np.array(raw_idx, dtype=np.int32), dtype=int, device=self._device
            )

            # Resolve preset
            mat_name = mat.get("material", "jelly")
            preset = _MATERIAL_PRESETS.get(mat_name, _MATERIAL_PRESETS["jelly"])

            # ── Elastic params (from config, per-object) ──────────────
            if "E" in mat:
                model.mpm.young_modulus[idx_wp].fill_(float(mat["E"]))
            if "nu" in mat:
                model.mpm.poisson_ratio[idx_wp].fill_(float(mat["nu"]))

            # ── Yield / plasticity (from preset, overrideable by config) ─
            model.mpm.yield_pressure[idx_wp].fill_(
                float(preset.get("yield_pressure", 1e12)))
            model.mpm.tensile_yield_ratio[idx_wp].fill_(
                float(preset.get("tensile_yield_ratio", 0.0)))

            ys = preset.get("yield_stress", 0.0)
            if "yield_stress" in mat:
                ys = mat["yield_stress"]
            model.mpm.yield_stress[idx_wp].fill_(float(ys))

            h = preset.get("hardening", 0.0)
            if "hardening" in mat:
                h = mat["hardening"]
            model.mpm.hardening[idx_wp].fill_(float(h))

            # Friction: preset, or from friction_angle if sand
            friction = preset.get("friction")
            if friction is None:
                # Default for sand when no friction_angle given
                friction = 0.68
            if mat_name == "sand" and "friction_angle" in mat:
                friction = _friction_from_angle(mat["friction_angle"])
            model.mpm.friction[idx_wp].fill_(float(friction))

            # ── Density → mass ────────────────────────────────────────
            if "density" in mat:
                density = float(mat["density"])
                mass_np = model.particle_mass.numpy()
                inv_mass_np = model.particle_inv_mass.numpy()
                for i in raw_idx:
                    m = self._volumes[i] * density
                    mass_np[i] = m
                    inv_mass_np[i] = 1.0 / m if m > 0 else 0.0
                model.particle_mass.assign(
                    wp.from_numpy(mass_np.astype(np.float32), dtype=float,
                                  device=self._device))
                model.particle_inv_mass.assign(
                    wp.from_numpy(inv_mass_np.astype(np.float32), dtype=float,
                                  device=self._device))

            print(
                f"[NewtonMPM] Object '{name}': {len(raw_idx)} particles, "
                f"preset={mat_name}, "
                f"E={mat.get('E','(base)')}, "
                f"yield_stress={ys}, "
                f"friction={friction}, "
                f"density={mat.get('density','(base)')}"
            )

    def set_boundary_conditions(self, bc_params: list, time_params: dict) -> None:
        """Register boundary conditions.

        Newton MPM handles boundaries differently from WarpMPM:
        - Ground planes / surface colliders → Newton's collision pipeline
        - Particle impulses → applied via state.particle_f
        - Cuboid velocity BCs → Newton doesn't have direct equivalent;
          we store them and apply in step() via particle velocity overrides.

        For Phase 1, we support: ground_plane (implicit), bounding_box,
        surface_collider, and store other BC types for manual application.
        """
        self._bc_params = bc_params
        self._time_params = time_params

        # Particle-level BCs that need per-step application
        self._velocity_bcs = []
        self._impulse_bcs = []
        self._release_bcs = []
        self._velocity_rotation_bcs = []

        builder = self._builder
        if builder is None:
            raise RuntimeError("initialize() must be called before set_boundary_conditions().")

        for bc in bc_params:
            bc_type = bc["type"]
            if bc_type == "bounding_box":
                # Add 6 planes enclosing the particle bounding box with margin
                margin = self._solver_opts.voxel_size * 2.0
                wall_cfg = newton.ModelBuilder.ShapeConfig(ke=0.0, kd=0.0, mu=0.0)
                lo = self._bbox_lo
                hi = self._bbox_hi
                planes = [
                    (1.0, 0.0, 0.0, -(lo[0] - margin)),
                    (-1.0, 0.0, 0.0, (hi[0] + margin)),
                    (0.0, 1.0, 0.0, -(lo[1] - margin)),
                    (0.0, -1.0, 0.0, (hi[1] + margin)),
                    (0.0, 0.0, 1.0, -(lo[2] - margin)),
                    (0.0, 0.0, -1.0, (hi[2] + margin)),
                ]
                for p in planes:
                    builder.add_shape_plane(plane=p, cfg=wall_cfg)
            elif bc_type == "surface_collider":
                normal = bc["normal"]
                point = bc["point"]
                d = -(
                    normal[0] * point[0]
                    + normal[1] * point[1]
                    + normal[2] * point[2]
                )
                mu = float(bc.get("friction", 0.0))
                cfg = newton.ModelBuilder.ShapeConfig(ke=0.0, kd=0.0, mu=mu)
                builder.add_shape_plane(
                    plane=(float(normal[0]), float(normal[1]), float(normal[2]), float(d)),
                    cfg=cfg,
                )
            elif bc_type == "cuboid":
                self._velocity_bcs.append(bc)
            elif bc_type == "particle_impulse":
                self._impulse_bcs.append(bc)
            elif bc_type == "enforce_particle_translation":
                self._velocity_bcs.append(bc)
            elif bc_type == "release_particles_sequentially":
                self._release_bcs.append(bc)
            elif bc_type == "enforce_particle_velocity_rotation":
                self._velocity_rotation_bcs.append(bc)
            else:
                print(f"[NewtonMPM] Warning: unsupported BC type '{bc_type}', skipping.")

    def finalize(self) -> None:
        """Create the Newton solver and initial states. Must be called after
        set_material() and set_boundary_conditions()."""
        builder = self._builder
        if builder is None:
            raise RuntimeError("initialize() must be called before finalize().")
        if self._material_params is None:
            raise RuntimeError("set_material() must be called before finalize().")

        # Finalize model after all shapes are registered.
        self._model = builder.finalize(device=self._device)
        self._builder = None
        model = self._model

        # ── Covariance tracking buffers ───────────────────────────────
        n = self._n_particles
        assert self._cov_np is not None
        self._init_cov = wp.from_numpy(self._cov_np, dtype=float, device=self._device)
        self._out_cov = wp.zeros(n * 6, dtype=float, device=self._device)
        self._out_R = wp.zeros(n, dtype=wp.mat33, device=self._device)

        # ── 2DGS quats/scales buffers ─────────────────────────────────
        if self._init_quats_np is not None:
            self._init_quats_wp = wp.from_numpy(
                self._init_quats_np.reshape(-1, 4), dtype=wp.vec4, device=self._device,
            )
            self._init_scales_wp = wp.from_numpy(
                self._init_scales_np.reshape(-1), dtype=float, device=self._device,
            )
            self._out_quats_wp = wp.zeros(n, dtype=wp.vec4, device=self._device)
            self._out_scales_wp = wp.zeros(
                n * self._num_scales, dtype=float, device=self._device,
            )

        # Apply material params now that model exists.
        self._apply_material_to_model()

        # ── Create solver ────────────────────────────────────────────
        self._solver = SolverImplicitMPM(model, self._solver_opts)

        # ── Create double-buffered states + control ──────────────────
        self._state_0 = model.state()
        self._state_1 = model.state()
        self._control = model.control()
        self._time = 0.0

        # Handle release_particles_sequentially: initially deactivate particles
        if self._release_bcs:
            self._setup_sequential_release()

    def step(self, dt: float, frame: int) -> None:
        """Advance simulation by one substep."""
        # Apply per-step boundary conditions (skip numpy round-trip when empty)
        if self._velocity_bcs:
            self._apply_velocity_bcs(dt)
        if self._impulse_bcs:
            self._apply_impulse_bcs(dt)

        # Newton solver step
        self._solver.step(self._state_0, self._state_1, self._control, None, dt)

        # Swap states (defer update_particle_frames to get_state for performance)
        self._state_0, self._state_1 = self._state_1, self._state_0
        self._substep_dt = dt
        self._frames_dirty = True
        self._time += dt

    def get_state(self) -> SimulationState:
        """Export current simulation state as PyTorch tensors."""
        n = self._n_particles
        state = self._state_0

        # ── Update particle deformation frames (deferred from step) ──
        # Only done once per get_state() call, not every substep.
        if self._frames_dirty:
            # state_0 is current, state_1 is previous (after swap).
            # update_particle_frames reads velocity gradient from state_0
            # and previous transform from state_1, writes to state_0.
            self._solver.update_particle_frames(
                self._state_1, self._state_0, self._substep_dt
            )
            self._frames_dirty = False

        # ── Positions ────────────────────────────────────────────────
        pos = wp.to_torch(state.particle_q)  # (N, 3)

        # ── Velocities ───────────────────────────────────────────────
        vel = wp.to_torch(state.particle_qd)  # (N, 3)

        # ── Covariance from F ────────────────────────────────────────
        F = state.mpm.particle_transform  # (N,) mat33 warp array
        wp.launch(
            compute_cov_from_F,
            dim=n,
            inputs=[F, self._init_cov, self._out_cov],
            device=self._device,
        )
        cov = wp.to_torch(self._out_cov).view(n, 6)

        # ── Rotation (+ quats/scales for 2DGS) from F ────────────────
        out_quats_t = None
        out_scales_t = None

        if self._init_quats_wp is not None:
            wp.launch(
                compute_R_quats_scales_from_F,
                dim=n,
                inputs=[
                    F,
                    self._init_quats_wp,
                    self._init_scales_wp,
                    self._num_scales,
                    self._out_R,
                    self._out_quats_wp,
                    self._out_scales_wp,
                ],
                device=self._device,
            )
            out_quats_t = wp.to_torch(self._out_quats_wp).view(n, 4)
            out_scales_t = wp.to_torch(self._out_scales_wp).view(n, self._num_scales)
        else:
            wp.launch(
                compute_R_from_F,
                dim=n,
                inputs=[F, self._out_R],
                device=self._device,
            )
        rot = wp.to_torch(self._out_R).view(n, 3, 3)

        return SimulationState(
            positions=pos,
            covariances=cov,
            rotations=rot,
            velocities=vel,
            quats=out_quats_t,
            scales=out_scales_t,
        )

    # ------------------------------------------------------------------
    # Boundary condition helpers (per-step application)
    # ------------------------------------------------------------------

    def _apply_velocity_bcs(self, dt: float) -> None:
        """Apply cuboid / translation velocity BCs by overriding particle velocities."""
        if not self._velocity_bcs:
            return

        state = self._state_0
        pos_np = state.particle_q.numpy()
        vel_np = state.particle_qd.numpy()

        for bc in self._velocity_bcs:
            start = bc.get("start_time", 0.0)
            end = bc.get("end_time", 1e3)
            if not (start <= self._time <= end):
                continue

            point = np.array(bc["point"], dtype=np.float32)
            size = np.array(bc["size"], dtype=np.float32)
            velocity = np.array(bc["velocity"], dtype=np.float32)

            lo = point - size
            hi = point + size
            mask = np.all((pos_np >= lo) & (pos_np <= hi), axis=1)

            if np.any(mask):
                vel_np[mask] = velocity

        # Write back
        state.particle_qd.assign(wp.from_numpy(vel_np.astype(np.float32), dtype=wp.vec3, device=self._device))

    def _apply_impulse_bcs(self, dt: float) -> None:
        """Apply particle impulse BCs as force on particles."""
        if not self._impulse_bcs:
            return

        state = self._state_0
        pos_np = state.particle_q.numpy()

        for bc in self._impulse_bcs:
            start = bc.get("start_time", 0.0)
            num_dt = bc.get("num_dt", 1)
            # Impulse active for num_dt substeps starting at start_time
            if not (start <= self._time < start + num_dt * dt + 1e-10):
                continue

            force = np.array(bc["force"], dtype=np.float32)
            point = np.array(bc.get("point", [1, 1, 1]), dtype=np.float32)
            size = np.array(bc.get("size", [1, 1, 1]), dtype=np.float32)

            lo = point - size
            hi = point + size
            mask = np.all((pos_np >= lo) & (pos_np <= hi), axis=1)

            if np.any(mask):
                # Add force * dt as velocity impulse (since Newton uses F = ma)
                mass_np = self._model.particle_mass.numpy()
                vel_np = state.particle_qd.numpy()
                for i in np.where(mask)[0]:
                    m = mass_np[i]
                    if m > 0:
                        vel_np[i] += force * dt / m
                state.particle_qd.assign(wp.from_numpy(vel_np.astype(np.float32), dtype=wp.vec3, device=self._device))

    def _setup_sequential_release(self) -> None:
        """Pre-deactivate particles for sequential release."""
        # Newton doesn't have a direct "selection" mechanism like WarpMPM.
        # For now, we'll track released layers via particle flags.
        # Simplified: release all particles (skip sequential release for Phase 1).
        pass

    def _apply_release_bcs(self) -> None:
        """Gradually release particle layers over time."""
        # Phase 1 simplified: all particles active from start.
        # Full implementation would toggle ParticleFlags.ACTIVE per layer.
        pass

    # ------------------------------------------------------------------
    # Extra accessors
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> newton.Model:
        return self._model

    @property
    def raw_solver(self) -> SolverImplicitMPM:
        return self._solver
