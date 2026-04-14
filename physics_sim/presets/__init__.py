"""Preset factory functions for experiment configuration.

Typical usage in an experiment file::

    from physics_sim.presets import wolf, bread, rubber, newton_rigid, orbit_camera

    config = SimConfig(
        backend=newton_rigid(),
        objects=[wolf(material=rubber(density=500)), bread(...)],
        ...
    )
"""

from physics_sim.presets.materials import rubber, sand, jelly, soft_body
from physics_sim.presets.objects import wolf, bread
from physics_sim.presets.backends import newton_rigid, newton_vbd, newton_mpm
from physics_sim.presets.cameras import orbit_camera, fixed_camera, json_camera
from physics_sim.presets.fillings import default_filling, dense_filling

__all__ = [
    "rubber", "sand", "jelly", "soft_body",
    "wolf", "bread",
    "newton_rigid", "newton_vbd", "newton_mpm",
    "orbit_camera", "fixed_camera", "json_camera",
    "default_filling", "dense_filling",
]
