"""Stage 1: Scene assembly and tensor concatenation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from physics_sim.config.models import SimConfig
from physics_sim.preprocessing.transform import generate_rotation_matrices
from physics_sim.renderer.gs_renderer import GaussianRenderer
from physics_sim.scene import SceneObject, assemble_scene


@dataclass
class SceneData:
    """All scene tensors and metadata produced by scene assembly."""

    sim_objects: list[SceneObject]
    static_chunks: list[SceneObject]
    collider_objects: list[SceneObject]

    gs_type: str
    gs_num: int
    per_object_info: list[dict]

    sim_init_pos: torch.Tensor
    sim_init_cov: torch.Tensor
    sim_init_vol: torch.Tensor
    sim_shs: torch.Tensor
    sim_opacity: torch.Tensor
    sim_quats: torch.Tensor
    sim_scales: torch.Tensor

    static_pos: Optional[torch.Tensor] = None
    static_cov: Optional[torch.Tensor] = None
    static_opacity: Optional[torch.Tensor] = None
    static_shs: Optional[torch.Tensor] = None
    static_quats: Optional[torch.Tensor] = None
    static_scales: Optional[torch.Tensor] = None

    axis_perm: str = "xyz"
    rotation_matrices: list[torch.Tensor] = field(default_factory=list)


def _estimate_volumes(pos: torch.Tensor, n_grid: int) -> torch.Tensor:
    lo = pos.min(dim=0)[0]
    hi = pos.max(dim=0)[0]
    extent = (hi - lo).max().item()
    if extent < 1e-8:
        extent = 1.0
    dx = extent / max(n_grid, 1)
    return torch.full((pos.shape[0],), float(dx ** 3), device=pos.device)


def _build_per_object_info(sim_objects: list[SceneObject]) -> list[dict]:
    info = []
    offset = 0
    for obj in sim_objects:
        n = obj.n_particles
        info.append(dict(
            name=obj.name,
            particle_indices=list(range(offset, offset + n)),
            material=obj.material,
        ))
        offset += n
    return info


def setup_scene(
    cfg: SimConfig,
    renderer: GaussianRenderer,
    config_dir: str = "",
    device: str = "cuda:0",
    sh_degree: int = 3,
) -> SceneData:
    """Assemble the scene and concatenate tensors for simulation."""
    print("Assembling scene...")

    objects = assemble_scene(cfg, renderer, config_dir=config_dir)

    sim_objects = [o for o in objects if o.role == "dynamic"]
    static_objects = [o for o in objects if o.role == "render_only"]
    collider_objects = [o for o in objects if o.role == "collider_only"]

    if not sim_objects and cfg.backend.type != "none":
        raise ValueError("No dynamic objects found — nothing to simulate.")

    gs_type = (sim_objects or objects)[0].gs_type

    pp = cfg.preprocess
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(pp.rotation_degree), pp.rotation_axis,
    )

    n_grid = getattr(cfg.backend, "n_grid", 200)

    if sim_objects:
        sim_init_pos = torch.cat([o.positions for o in sim_objects], dim=0).to(device)
        sim_init_cov = torch.cat([o.covariances for o in sim_objects], dim=0).to(device)
        sim_init_vol = _estimate_volumes(sim_init_pos, n_grid)
        sim_shs = torch.cat([o.shs for o in sim_objects], dim=0)
        sim_opacity = torch.cat([o.opacities for o in sim_objects], dim=0)
        sim_quats = torch.cat([o.quats for o in sim_objects], dim=0)
        sim_scales = torch.cat([o.scales for o in sim_objects], dim=0)
        gs_num = sim_init_pos.shape[0]
        per_object_info = _build_per_object_info(sim_objects)
    else:
        sh_c = (sh_degree + 1) ** 2
        sim_init_pos = torch.zeros(0, 3, device=device)
        sim_init_cov = torch.zeros(0, 6, device=device)
        sim_init_vol = torch.zeros(0, device=device)
        sim_shs = torch.zeros(0, sh_c, 3, device=device)
        sim_opacity = torch.zeros(0, 1, device=device)
        sim_quats = torch.zeros(0, 4, device=device)
        sim_scales = torch.zeros(0, 3, device=device)
        gs_num = 0
        per_object_info = []

    # Static chunks: render_only objects + collider_only objects that render
    static_chunks = list(static_objects)
    for obj in collider_objects:
        render_flag = (obj.collider or {}).get("render", True)
        if render_flag and obj.n_particles > 0:
            static_chunks.append(obj)

    static_pos = static_cov = static_opacity = static_shs = None
    static_quats_t = static_scales_t = None
    if static_chunks:
        static_pos = torch.cat([o.positions for o in static_chunks], dim=0)
        static_cov = torch.cat([o.covariances for o in static_chunks], dim=0)
        static_opacity = torch.cat([o.opacities for o in static_chunks], dim=0)
        static_shs = torch.cat([o.shs for o in static_chunks], dim=0)
        static_quats_t = torch.cat([o.quats for o in static_chunks], dim=0)
        static_scales_t = torch.cat([o.scales for o in static_chunks], dim=0)

    return SceneData(
        sim_objects=sim_objects,
        static_chunks=static_chunks,
        collider_objects=collider_objects,
        gs_type=gs_type,
        gs_num=gs_num,
        per_object_info=per_object_info,
        sim_init_pos=sim_init_pos,
        sim_init_cov=sim_init_cov,
        sim_init_vol=sim_init_vol,
        sim_shs=sim_shs,
        sim_opacity=sim_opacity,
        sim_quats=sim_quats,
        sim_scales=sim_scales,
        static_pos=static_pos,
        static_cov=static_cov,
        static_opacity=static_opacity,
        static_shs=static_shs,
        static_quats=static_quats_t,
        static_scales=static_scales_t,
        axis_perm=pp.axis_permutation,
        rotation_matrices=rotation_matrices,
    )
