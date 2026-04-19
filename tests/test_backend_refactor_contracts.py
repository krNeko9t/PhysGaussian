from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_newton_common_helpers_are_single_source():
    rigid_src = _read("physics_sim/backend/newton_rigid/solver.py")
    vbd_src = _read("physics_sim/backend/newton_vbd/solver.py")

    for token in (
        "def _rotmat_to_wp_quat",
        "def _quat_to_rotmat",
        "def _compose_body_quat_wxyz",
        "def _unpack_cov6_to_3x3",
        "def _pack_cov3x3_to_6",
    ):
        assert token not in rigid_src
        assert token not in vbd_src
    assert "backend.newton_common" in rigid_src


def test_solver_modules_are_split_and_keep_orchestrator_role():
    rigid_src = _read("physics_sim/backend/newton_rigid/solver.py")
    vbd_src = _read("physics_sim/backend/newton_vbd/solver.py")
    mpm_src = _read("physics_sim/backend/newton_mpm/solver.py")

    assert "from physics_sim.backend.newton_rigid.collider_builders import" in rigid_src
    assert "from physics_sim.backend.newton_rigid.state_export import" in rigid_src
    assert "def _create_body_mesh(" not in rigid_src

    assert "from physics_sim.backend.newton_vbd.barycentric import" in vbd_src
    assert "from physics_sim.backend.newton_vbd.rigid_mesh import" in vbd_src
    assert "from physics_sim.backend.newton_vbd.state_export import" in vbd_src
    assert "def _create_rigid_mesh(" not in vbd_src
    assert "def _compute_barycentric(" not in vbd_src

    assert "from physics_sim.backend.newton_mpm.materials import" in mpm_src
    assert "from physics_sim.backend.newton_mpm.boundary_conditions import" in mpm_src
    assert "from physics_sim.backend.newton_mpm.state_export import" in mpm_src
    assert "def _apply_material_to_model(" not in mpm_src
    assert "def _apply_velocity_bcs(" not in mpm_src


def test_coord_domain_split_keeps_public_facade():
    coord_src = _read("physics_sim/coord.py")
    assert "from physics_sim.coord_gravity import" in coord_src
    assert "from physics_sim.coord_axes import" in coord_src
    assert "from physics_sim.coord_ops import" in coord_src
    assert (REPO_ROOT / "physics_sim/coord_gravity.py").exists()
    assert (REPO_ROOT / "physics_sim/coord_camera.py").exists()
    assert (REPO_ROOT / "physics_sim/coord_axes.py").exists()
    assert (REPO_ROOT / "physics_sim/coord_ops.py").exists()


def test_surface_friction_resolution_logic_is_centralized():
    shared = _read("physics_sim/backend/newton_common/boundary.py")
    mpm = _read("physics_sim/backend/newton_mpm/boundary_conditions.py")
    rigid = _read("physics_sim/backend/newton_rigid/solver.py")
    vbd = _read("physics_sim/backend/newton_vbd/solver.py")

    assert "def resolve_surface_friction" in shared
    assert "def surface_plane_from_bc" in shared
    assert "surface_plane_from_bc" in mpm
    assert "surface_plane_from_bc" in rigid
    assert "surface_plane_from_bc" in vbd


def test_material_friction_resolution_has_single_path():
    src = _read("physics_sim/backend/newton_mpm/materials.py")
    assert "def resolve_friction(" in src
    assert 'if "friction" in material_cfg' in src
    assert 'if "friction_angle" in material_cfg' in src
    assert 'if mat_name == "sand":' in src


def test_registry_unknown_errors_are_unified():
    backend_src = _read("physics_sim/backend/registry.py")
    render_src = _read("physics_sim/render/registries.py")

    assert "unknown_registry_error" in backend_src
    assert "unknown_registry_error" in render_src
    assert "raise ValueError(f\"Unknown backend type" not in backend_src
    assert "raise KeyError(" not in render_src


def test_rigid_state_export_is_centralized():
    shared = _read("physics_sim/backend/newton_common/rigid_state.py")
    rigid = _read("physics_sim/backend/newton_rigid/state_export.py")
    vbd = _read("physics_sim/backend/newton_vbd/state_export.py")

    assert "def populate_rigid_particles" in shared
    assert "def sanitize_body_poses" in shared
    assert "populate_rigid_particles(" in rigid
    assert "populate_rigid_particles(" in vbd


def test_rigid_geometry_builders_are_shared():
    shared = _read("physics_sim/backend/newton_common/rigid_geometry.py")
    rigid = _read("physics_sim/backend/newton_rigid/collider_builders.py")
    vbd = _read("physics_sim/backend/newton_vbd/rigid_mesh.py")

    assert "def create_rigid_body_geometry" in shared
    assert "create_rigid_body_geometry(" in rigid
    assert "create_rigid_body_geometry(" in vbd
    assert "def _create_mesh_body(" not in rigid
    assert "def _create_rigid_mesh(" not in vbd


def test_camera_math_is_split_from_factory():
    camera_src = _read("physics_sim/render/camera.py")
    assert "from physics_sim.render.camera_math import" in camera_src
    assert (REPO_ROOT / "physics_sim/render/camera_math.py").exists()


def test_particle_filling_chunk_flow_is_split():
    src = _read("physics_sim/preprocessing/particle_filling.py")
    assert "from physics_sim.preprocessing.particle_filling_chunks import" in src
    assert "run_densify_stage(" in src
    assert "run_dense_fill_stage(" in src
    assert "run_internal_fill_stage(" in src
    assert (REPO_ROOT / "physics_sim/preprocessing/particle_filling_chunks.py").exists()


def test_video_stage_no_longer_uses_os_system():
    video_src = _read("physics_sim/stages/video.py")
    assert "os.system(" not in video_src
    assert "subprocess.run(" in video_src
    assert "returncode" in video_src
