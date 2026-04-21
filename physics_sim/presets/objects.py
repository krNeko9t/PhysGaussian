"""Part preset factories.

Each function returns a fully-formed ``PartConfig``.  The caller can
override any field via keyword arguments.
"""

from __future__ import annotations

from physics_sim.config.models import ObjectTransform, PlySource
from physics_sim.config.scene import PartConfig
from physics_sim.presets.materials import rubber


def wolf(
    material: dict | None = None,
    transform: ObjectTransform | None = None,
    **kw,
) -> PartConfig:
    return PartConfig(
        name=kw.pop("name", "wolf"),
        source=kw.pop("source", PlySource(
            ply_path="model/wolf_whitebg-trained/point_cloud/iteration_30000/point_cloud.ply",
        )),
        material=material if material is not None else rubber(),
        transform=transform or ObjectTransform(),
        **kw,
    )


def bread(
    material: dict | None = None,
    transform: ObjectTransform | None = None,
    **kw,
) -> PartConfig:
    return PartConfig(
        name=kw.pop("name", "bread"),
        source=kw.pop("source", PlySource(
            ply_path="model/bread-trained/point_cloud/iteration_30000/point_cloud.ply",
        )),
        material=material if material is not None else rubber(),
        transform=transform or ObjectTransform(),
        **kw,
    )
