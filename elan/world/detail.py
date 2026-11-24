import torch

from elan.config import WorldCfg


class DetailBaker:
	def __init__(self, device):
		self.device = device

	def bake_rock(self, noise, amount=1.0):
		cfg = WorldCfg
		n = cfg.DETAIL_N
		voxel = 2.0 * cfg.DETAIL_HALF / n
		dims = (n, n, n)
		layered = noise.fractal(dims, voxel, 4.0, 26.0, 1.5, aniso=[1.0, 0.36, 1.0])
		cracks = noise.fractal(dims, voxel, 3.5, 14.0, 1.2)
		blocks = 1.0 - torch.exp(-(cracks / 0.28) ** 2)
		relief = (torch.sigmoid(layered * 1.6) * 0.62 + blocks * 0.38) * cfg.DETAIL_AMP
		micro = noise.fractal(dims, voxel, 1.1, 3.4, 1.1, aniso=[1.0, 0.6, 1.0])
		bump = relief + micro * 0.26
		veins = noise.band(dims, voxel, 17.0, 0.4)
		patches = noise.band(dims, voxel, 34.0, 0.4)
		vein = torch.exp(-(veins / 0.09) ** 2) * torch.clamp(patches * 1.5 - 0.45, 0.0, 1.0)
		return torch.stack((relief * amount, vein, bump * amount), 0).unsqueeze(0).contiguous()

	def bake_ripple(self, noise):
		cfg = WorldCfg
		n = cfg.RIPPLE_N
		voxel = 2.0 * cfg.RIPPLE_HALF / n
		height = noise.fractal((n, n), voxel, 1.4, 14.0, 1.4)
		dz = (torch.roll(height, -1, 0) - torch.roll(height, 1, 0)) / (2.0 * voxel)
		dx = (torch.roll(height, -1, 1) - torch.roll(height, 1, 1)) / (2.0 * voxel)
		slope = torch.stack((dx, dz), 0)
		slope = slope / slope.std().clamp_min(1e-6)
		return slope.unsqueeze(0).contiguous()
