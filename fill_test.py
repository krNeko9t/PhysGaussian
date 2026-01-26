import taichi as ti
import torch
from particle_filling.filling_progress import fill_particles

ti.init(arch=ti.cuda)
pos = torch.rand(200000, 3, device="cuda")
opacity = torch.rand(200000, 1, device="cuda")
cov = torch.rand(200000, 6, device="cuda")
out = fill_particles(
    pos, opacity, cov,
    grid_n=16, max_samples=200000, grid_dx=1.0/16,
    density_thres=100.0, search_thres=1.0,
    max_particles_per_cell=1, search_exclude_dir=2, ray_cast_dir=4,
    boundary=[0.0,1.0,0.0,1.0,0.0,1.0],
    smooth=False, progress=True, grid_chunk_size=50000
)
print(out.shape)