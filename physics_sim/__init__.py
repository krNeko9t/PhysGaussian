# physics_sim - 3DGS Physics Simulation Module
# Extracted and refactored from PhysGaussian
# https://github.com/XPandora/PhysGaussian

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.config.loader import load_config
from physics_sim.coord import SourceAxes, UpAxis
from physics_sim.scene.objects import SceneObject
from physics_sim.scene.assembler import assemble_scene
