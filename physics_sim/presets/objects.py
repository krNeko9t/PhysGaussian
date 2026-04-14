"""Object preset factories.

Each function returns a fully-formed ``ObjectConfig``.  The caller can
override any field via keyword arguments.
"""

from __future__ import annotations

from physics_sim.config.models import (
    ObjectConfig,
    ObjectTransform,
    PlySource,
)
from physics_sim.presets.materials import rubber


def wolf(
    material: dict | None = None,
    role: str = "dynamic",
    transform: ObjectTransform | None = None,
    **kw,
) -> ObjectConfig:
    return ObjectConfig(
        name=kw.pop("name", "wolf"),
        source=kw.pop("source", PlySource(
            ply_path="model/wolf_whitebg-trained/point_cloud/iteration_30000/point_cloud.ply",
        )),
        role=role,
        material=material if material is not None else rubber(),
        transform=transform or ObjectTransform(),
        **kw,
    )


def bread(
    material: dict | None = None,
    role: str = "dynamic",
    transform: ObjectTransform | None = None,
    **kw,
) -> ObjectConfig:
    return ObjectConfig(
        name=kw.pop("name", "bread"),
        source=kw.pop("source", PlySource(
            ply_path="model/bread-trained/point_cloud/iteration_30000/point_cloud.ply",
        )),
        role=role,
        material=material if material is not None else rubber(),
        transform=transform or ObjectTransform(),
        **kw,
    )
