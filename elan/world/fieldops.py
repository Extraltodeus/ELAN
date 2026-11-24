import torch
import torch.nn.functional as F

from elan.config import WorldCfg


class FieldOps:
	@staticmethod
	def grad_norm(field, voxel, wrap_y=True):
		acc = torch.roll(field, -1, 0)
		acc -= torch.roll(field, 1, 0)
		acc.square_()
		tmp = torch.roll(field, -1, 2)
		tmp -= torch.roll(field, 1, 2)
		acc.addcmul_(tmp, tmp)
		if wrap_y:
			tmp = torch.roll(field, -1, 1)
			tmp -= torch.roll(field, 1, 1)
		else:
			tmp[:, 1:-1] = field[:, 2:] - field[:, :-2]
			tmp[:, 0] = (field[:, 1] - field[:, 0]) * 2.0
			tmp[:, -1] = (field[:, -1] - field[:, -2]) * 2.0
		acc.addcmul_(tmp, tmp)
		del tmp
		return acc.sqrt_().mul_(0.5 / voxel)

	@staticmethod
	def smooth_min(a, b, k):
		h = torch.clamp(0.5 + (0.5 / k) * (b - a), 0.0, 1.0)
		out = torch.lerp(b, a, h)
		out.addcmul_(h, 1.0 - h, value=-k)
		return out

	@staticmethod
	def edt_pass(a, dim, radius, circular, breathe):
		out = a.clone()
		for k in range(1, radius + 1):
			kk = float(k * k)
			if circular:
				out = torch.minimum(out, torch.roll(a, k, dim) + kk)
				out = torch.minimum(out, torch.roll(a, -k, dim) + kk)
			else:
				out[:, k:] = torch.minimum(out[:, k:], a[:, :-k] + kk)
				out[:, :-k] = torch.minimum(out[:, :-k], a[:, k:] + kk)
			if k % 6 == 0:
				breathe()
		return out

	@staticmethod
	def distance_to_solid(solid, radius, breathe):
		a = torch.where(solid, 0.0, 1.0e6)
		a = FieldOps.edt_pass(a, 2, radius, True, breathe)
		a = FieldOps.edt_pass(a, 0, radius, True, breathe)
		a = FieldOps.edt_pass(a, 1, radius, False, breathe)
		return a.sqrt_().clamp_max_(float(radius))

	@staticmethod
	def upsample_periodic(coarse, factor):
		padded = F.pad(coarse[None, None], (1, 1, 0, 0, 1, 1), mode='circular')
		up = F.interpolate(padded, scale_factor=factor, mode='trilinear', align_corners=False)[0, 0]
		return up[factor:-factor, :, factor:-factor]


class TileSampler:
	def __init__(self, field):
		self.nz, self.ny, self.nx = field.shape
		self.voxel = WorldCfg.VOXEL
		self.volume = F.pad(field.view(1, 1, self.nz, self.ny, self.nx), (1, 1, 0, 0, 1, 1), mode='circular')
		self.device = field.device
		offsets = torch.tensor([[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]], device=self.device)
		self.tetra = offsets

	def sample(self, points):
		v = self.voxel
		gx = (torch.remainder(points[..., 0] / v, self.nx) + 1.0) * (2.0 / (self.nx + 2)) - 1.0
		gz = (torch.remainder(points[..., 2] / v, self.nz) + 1.0) * (2.0 / (self.nz + 2)) - 1.0
		gy = points[..., 1] * (2.0 / (self.ny * v)) - 1.0
		grid = torch.stack((gx, gy, gz), -1).view(1, 1, 1, -1, 3)
		out = F.grid_sample(self.volume, grid, mode='bilinear', padding_mode='border', align_corners=False)
		return out.view(points.shape[:-1])

	def gradient(self, points, eps=1.6):
		acc = torch.zeros_like(points)
		for i in range(4):
			k = self.tetra[i]
			acc += self.sample(points + k * eps).unsqueeze(-1) * k
		return acc / acc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
