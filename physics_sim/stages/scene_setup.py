"""Stage 1: Scene assembly and tensor concatenation.

After this stage, **all tensors are in the internal Y-up coordinate
system**.  The ``SceneData.coord`` field records the original convention
so that SH evaluation can transform view directions back to PLY-native
space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from physics_sim.backend.spec import PartRuntimeInfo
from physics_sim.config.models import SimConfig
from physics_sim.coord import SourceAxes
from physics_sim.render.interfaces import SceneAssetLoader
from physics_sim.scene.constraint_resolver import (
    ResolvedConstraint,
    resolve_constraints,
)
from physics_sim.sh_contract import sh_coeff_count
from physics_sim.scene import SceneObject, assemble_scene


__all__ = [
    "CoordContext",
    "DynamicSceneInit",
    "PartRuntimeInfo",
    "RenderSetup",
    "SceneData",
    "StaticRenderChunk",
    "setup_scene",
]


@dataclass
class DynamicSceneInit:
    """Initial tensors for all dynamic (simulated) particles, concatenated."""
    pos: torch.Tensor
    cov: torch.Tensor
    vol: torch.Tensor
    shs: torch.Tensor
    opacity: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor

    @property
    def n_particles(self) -> int:
        return self.pos.shape[0]


@dataclass
class StaticRenderChunk:
    """Static render-only + collider-render particles, concatenated once.

    Built at setup time and referenced read-only by the frame loop so the
    per-frame ``torch.cat`` fusion can reuse these tensors without copies.
    """
    pos: torch.Tensor
    cov: torch.Tensor
    opacity: torch.Tensor
    shs: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor

    @property
    def n_particles(self) -> int:
        return self.pos.shape[0]


@dataclass
class RenderSetup:
    """Frame-invariant render inputs pre-concatenated at setup time.

    ``opacity_all`` and ``shs_all`` are dynamic + static tensors merged
    once; the frame loop references them directly instead of calling
    ``torch.cat`` every frame.  ``static_identity`` is the eye-matrix
    block used for static particles' view-space rotations.
    """
    opacity_all: torch.Tensor           # (N_dyn + N_static, 1)
    shs_all: torch.Tensor                # (N_dyn + N_static, C, 3)
    static_identity: Optional[torch.Tensor] = None  # (N_static, 3, 3)


@dataclass
class CoordContext:
    """Coordinate-system metadata for rendering SH and aligning directions."""
    source_axes: SourceAxes = field(default_factory=SourceAxes.identity)
    alignment_inv: Optional[torch.Tensor] = None  # 3x3, internal -> source


@dataclass
class SceneData:
    """All scene tensors and metadata produced by scene assembly."""

    sim_objects: list[SceneObject]
    static_chunks: list[SceneObject]
    collider_objects: list[SceneObject]

    gs_type: str
    gs_num: int

    parts_runtime: list[PartRuntimeInfo]
    dynamic_init: DynamicSceneInit
    static_render: Optional[StaticRenderChunk]
    render_setup: RenderSetup
    coord: CoordContext
    resolved_constraints: list[ResolvedConstraint] = field(default_factory=list)


def _estimate_volumes(pos: torch.Tensor, n_grid: int) -> torch.Tensor:
    lo = pos.min(dim=0)[0]
    hi = pos.max(dim=0)[0]
    extent = (hi - lo).max().item()
    if extent < 1e-8:
        extent = 1.0
    dx = extent / max(n_grid, 1)
    return torch.full((pos.shape[0],), float(dx ** 3), device=pos.device)


def _build_parts_runtime(sim_objects: list[SceneObject]) -> list[PartRuntimeInfo]:
    info = []
    offset = 0
    for obj in sim_objects:
        if obj.material is None:
            raise ValueError(
                f"dynamic object {obj.name!r} has no material — "
                "dynamic objects require a MaterialSpec"
            )
        n = obj.n_particles
        info.append(PartRuntimeInfo(
            name=obj.name,
            particle_indices=list(range(offset, offset + n)),
            material=obj.material,
            initial_velocity=obj.initial_velocity,
        ))
        offset += n
    return info


def _empty_dynamic_init(
    device: str,
    sh_degree: int,
    sh_channels: int | None,
) -> DynamicSceneInit:
    sh_c = sh_channels if sh_channels is not None else sh_coeff_count(sh_degree)
    return DynamicSceneInit(
        pos=torch.zeros(0, 3, device=device),
        cov=torch.zeros(0, 6, device=device),
        vol=torch.zeros(0, device=device),
        shs=torch.zeros(0, sh_c, 3, device=device),
        opacity=torch.zeros(0, 1, device=device),
        quats=torch.zeros(0, 4, device=device),
        scales=torch.zeros(0, 3, device=device),
    )


def _build_dynamic_init(
    sim_objects: list[SceneObject],
    device: str,
    n_grid: int,
) -> DynamicSceneInit:
    pos = torch.cat([o.positions for o in sim_objects], dim=0).to(device)
    cov = torch.cat([o.covariances for o in sim_objects], dim=0).to(device)
    return DynamicSceneInit(
        pos=pos,
        cov=cov,
        vol=_estimate_volumes(pos, n_grid),
        shs=torch.cat([o.shs for o in sim_objects], dim=0),
        opacity=torch.cat([o.opacities for o in sim_objects], dim=0),
        quats=torch.cat([o.quats for o in sim_objects], dim=0),
        scales=torch.cat([o.scales for o in sim_objects], dim=0),
    )


def _build_static_render(static_chunks: list[SceneObject]) -> Optional[StaticRenderChunk]:
    if not static_chunks:
        return None
    return StaticRenderChunk(
        pos=torch.cat([o.positions for o in static_chunks], dim=0),
        cov=torch.cat([o.covariances for o in static_chunks], dim=0),
        opacity=torch.cat([o.opacities for o in static_chunks], dim=0),
        shs=torch.cat([o.shs for o in static_chunks], dim=0),
        quats=torch.cat([o.quats for o in static_chunks], dim=0),
        scales=torch.cat([o.scales for o in static_chunks], dim=0),
    )


def _build_render_setup(
    dynamic_init: DynamicSceneInit,
    static_render: Optional[StaticRenderChunk],
) -> RenderSetup:
    """Pre-concatenate the frame-invariant render inputs.

    ``opacity`` and ``shs`` never change during simulation — the per-frame
    loop only needs to concatenate the dynamic particles' positions /
    covariances / rotations with the static ones.  Concatenating these
    twice-per-frame was wasteful; merging once here removes that work.
    """
    if static_render is None or static_render.n_particles == 0:
        return RenderSetup(
            opacity_all=dynamic_init.opacity,
            shs_all=dynamic_init.shs,
            static_identity=None,
        )
    opacity_all = torch.cat([dynamic_init.opacity, static_render.opacity], dim=0)
    shs_all = torch.cat([dynamic_init.shs, static_render.shs], dim=0)
    static_count = static_render.n_particles
    ref = dynamic_init.pos if dynamic_init.n_particles > 0 else static_render.pos
    static_identity = (
        torch.eye(3, device=ref.device, dtype=ref.dtype)
        .unsqueeze(0)
        .expand(static_count, -1, -1)
        .contiguous()
    )
    return RenderSetup(
        opacity_all=opacity_all,
        shs_all=shs_all,
        static_identity=static_identity,
    )


def setup_scene(
    cfg: SimConfig,
    loader: SceneAssetLoader,
    config_dir: str = "",
    device: str = "cuda:0",
    sh_degree: int = 3,
    sh_channels: int | None = None,
) -> SceneData:
    """Assemble the scene and concatenate tensors for simulation.

    All returned tensors are in the internal Y-up coordinate system.
    """
    print("Assembling scene...")

    source_axes = SourceAxes.from_config(
        cfg.preprocess.source_up, cfg.preprocess.source_front,
    )

    objects = assemble_scene(cfg, loader, config_dir=config_dir)

    sim_objects = [o for o in objects if o.role == "dynamic"]
    static_objects = [o for o in objects if o.role == "render_only"]
    collider_objects = [o for o in objects if o.role == "collider_only"]

    if not sim_objects and cfg.backend.type != "none":
        raise ValueError("No dynamic objects found — nothing to simulate.")

    gs_type = (sim_objects or objects)[0].gs_type

    n_grid = getattr(cfg.backend, "n_grid", 200)

    if sim_objects:
        dynamic_init = _build_dynamic_init(sim_objects, device, n_grid)
        parts_runtime = _build_parts_runtime(sim_objects)
    else:
        dynamic_init = _empty_dynamic_init(device, sh_degree, sh_channels)
        parts_runtime = []

    # Resolve scene-graph constraints into concrete particle-index sets.
    scene = cfg.scene
    resolved_constraints = resolve_constraints(
        scene=scene,
        scene_objects=objects,
        parts_runtime=parts_runtime,
    )

    # Static chunks: render_only objects + collider_only objects that render
    static_chunks = list(static_objects)
    for obj in collider_objects:
        render_flag = obj.collider.render if obj.collider is not None else True
        if render_flag and obj.n_particles > 0:
            static_chunks.append(obj)

    static_render = _build_static_render(static_chunks)
    render_setup = _build_render_setup(dynamic_init, static_render)

    alignment_inv = (
        source_axes.A_inv.to(device) if not source_axes.is_identity else None
    )

    return SceneData(
        sim_objects=sim_objects,
        static_chunks=static_chunks,
        collider_objects=collider_objects,
        gs_type=gs_type,
        gs_num=dynamic_init.n_particles,
        parts_runtime=parts_runtime,
        resolved_constraints=resolved_constraints,
        dynamic_init=dynamic_init,
        static_render=static_render,
        render_setup=render_setup,
        coord=CoordContext(source_axes=source_axes, alignment_inv=alignment_inv),
    )
