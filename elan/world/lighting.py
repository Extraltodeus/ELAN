import math
import torch

from elan.config import WorldCfg


class LightBaker:
	def __init__(self, device):
		self.device = device

	def bake(self, sampler, sun_dir, breathe):
		cfg = WorldCfg
		div = cfg.AUX_DIV
		step = cfg.VOXEL * div
		cz, cy, cx = cfg.NZ // div, cfg.NY // div, cfg.NX // div
		zs = (torch.arange(cz, device=self.device, dtype=torch.float32) + 0.5) * step
		ys = (torch.arange(cy, device=self.device, dtype=torch.float32) + 0.5) * step
		xs = (torch.arange(cx, device=self.device, dtype=torch.float32) + 0.5) * step
		grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing='ij')
		points = torch.stack((grid_x, grid_y, grid_z), -1).view(-1, 3)
		del grid_x, grid_y, grid_z
		depth = sampler.sample(points)
		normal = sampler.gradient(points)
		lift = torch.clamp(1.6 - depth, min=0.0, max=9.0)
		points = points + normal * lift.unsqueeze(-1)
		del depth, normal, lift
		breathe()
		sun = self._sun_visibility(sampler, points, sun_dir, breathe)
		sky = self._sky_openness(sampler, points, breathe)
		aux = torch.stack((sky, sun), 0).view(2, cz, cy, cx)
		return aux

	def _sun_visibility(self, sampler, points, sun_dir, breathe):
		sun = torch.tensor(sun_dir, device=self.device, dtype=torch.float32)
		top = WorldCfg.SIZE_Y
		reach = WorldCfg.SHADOW_RANGE
		vis = torch.ones(points.shape[0], device=self.device)
		t = torch.full((points.shape[0],), 2.5, device=self.device)
		for i in range(52):
			probe = torch.addcmul(points, sun, t.unsqueeze(-1))
			d = sampler.sample(probe)
			d = torch.where(probe[:, 1] > top, 80.0, d)
			d = torch.where(t > reach, 80.0, d)
			vis = torch.minimum(vis, (d * 6.0 / t.clamp(max=48.0)).clamp_(0.0, 1.0))
			t = t + d.clamp_(1.2, 60.0)
			if i % 8 == 7:
				breathe()
		return vis

	def _directions(self):
		dirs = []
		count = 26
		golden = math.pi * (3.0 - math.sqrt(5.0))
		for i in range(count):
			y = 1.0 - 2.0 * (i + 0.5) / count
			if y < -0.2:
				continue
			r = math.sqrt(max(1.0 - y * y, 0.0))
			dirs.append((math.cos(golden * i) * r, y, math.sin(golden * i) * r))
		return dirs

	def _sky_openness(self, sampler, points, breathe):
		acc = torch.zeros(points.shape[0], device=self.device)
		reference = 0.0
		for radius in (5.0, 12.0, 28.0):
			for direction in self._directions():
				weight = 0.35 + 0.65 * max(direction[1], 0.0)
				offset = torch.tensor(direction, device=self.device, dtype=torch.float32) * radius
				d = sampler.sample(points + offset)
				acc += (d * (1.3 / radius)).clamp_(0.0, 1.0) * weight
				reference += min(max(1.3 * (1.6 + radius * direction[1]) / radius, 0.0), 1.0) * weight
			breathe()
		return (acc / reference).clamp_(0.0, 1.0)
