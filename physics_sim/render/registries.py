"""Extensible registries for asset loaders, camera builders, and rasterizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from physics_sim.errors import unknown_registry_error
from physics_sim.render.backend_base import create_raster_backend
from physics_sim.render.camera import CameraFactory
from physics_sim.render.gaussian_asset_loader import GaussianAssetLoader
from physics_sim.render.interfaces import Rasterizer

AssetLoaderFactory = Callable[..., object]
RasterizerFactory = Callable[..., Rasterizer]
CameraFactoryBuilder = Callable[..., CameraFactory]
CameraModeResolver = Callable[..., Any]

_ASSET_LOADERS: dict[str, AssetLoaderFactory] = {}
_RASTERIZERS: dict[str, RasterizerFactory] = {}
_CAMERA_BUILDERS: dict[str, CameraFactoryBuilder] = {}
_CAMERA_MODES: dict[str, CameraModeResolver] = {}


def register_asset_loader(name: str, factory: AssetLoaderFactory) -> None:
    _ASSET_LOADERS[name] = factory


def create_asset_loader(name: str, **kwargs):
    if name not in _ASSET_LOADERS:
        raise unknown_registry_error(
            registry="asset_loader",
            key=name,
            available=_ASSET_LOADERS.keys(),
        )
    return _ASSET_LOADERS[name](**kwargs)


def register_rasterizer(name: str, factory: RasterizerFactory) -> None:
    _RASTERIZERS[name] = factory


def create_rasterizer(name: str, **kwargs) -> Rasterizer:
    if name not in _RASTERIZERS:
        raise unknown_registry_error(
            registry="rasterizer",
            key=name,
            available=_RASTERIZERS.keys(),
        )
    return _RASTERIZERS[name](**kwargs)


def register_camera_builder(name: str, factory: CameraFactoryBuilder) -> None:
    _CAMERA_BUILDERS[name] = factory


def create_camera_builder(name: str, **kwargs) -> CameraFactory:
    if name not in _CAMERA_BUILDERS:
        raise unknown_registry_error(
            registry="camera_builder",
            key=name,
            available=_CAMERA_BUILDERS.keys(),
        )
    return _CAMERA_BUILDERS[name](**kwargs)


def register_camera_mode(name: str, resolver: CameraModeResolver) -> None:
    _CAMERA_MODES[name] = resolver


def resolve_camera_for_mode(name: str, **kwargs):
    if name not in _CAMERA_MODES:
        raise unknown_registry_error(
            registry="camera_mode",
            key=name,
            available=_CAMERA_MODES.keys(),
        )
    return _CAMERA_MODES[name](**kwargs)


register_asset_loader("ply", lambda *, sh_degree: GaussianAssetLoader(sh_degree=sh_degree))
register_rasterizer(
    "gsplat",
    lambda: create_raster_backend("gsplat"),
)
register_rasterizer(
    "diffrast",
    lambda: create_raster_backend("diffrast"),
)
register_camera_builder("default", lambda: CameraFactory())
register_camera_mode(
    "external",
    lambda *, camera_builder, camera_params, center_view_world_space,
    observant_coordinates, current_frame, source_axes: camera_builder.build_camera_external(
        camera_params=camera_params,
        center_view_world_space=center_view_world_space,
        observant_coordinates=observant_coordinates,
        current_frame=current_frame,
        source_axes=source_axes,
    ),
)
register_camera_mode(
    "orbit",
    lambda *, camera_builder, camera_params, center_view_world_space,
    observant_coordinates, current_frame, **_: camera_builder.build_camera_orbit(
        camera_params=camera_params,
        center_view_world_space=center_view_world_space,
        observant_coordinates=observant_coordinates,
        current_frame=current_frame,
    ),
)
register_camera_mode(
    "fixed",
    lambda *, camera_builder, camera_params, **_: camera_builder.build_camera_fixed(
        camera_params=camera_params,
    ),
)
