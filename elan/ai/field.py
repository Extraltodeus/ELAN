import numpy as np
import torch

from elan.config import WorldCfg


class BatchField:
	def __init__(self, fields, device):
		self.nz, self.ny, self.nx = fields[0].shape
		self.data = torch.cat([torch.from_numpy(np.ascontiguousarray(field)).to(device).view(-1) for field in fields], 0)
		self.inv = 1.0 / WorldCfg.VOXEL

	def _cell(self, points, shift):
		inv = self.inv
		fx = points[..., 0] * inv - 0.5
		fy = points[..., 1].abs() * inv - 0.5
		fz = points[..., 2] * inv - 0.5
		x0 = torch.floor(fx)
		y0 = torch.floor(fy)
		z0 = torch.floor(fz)
		tx = fx - x0
		ty = torch.where(y0 < 0, torch.zeros_like(fy), torch.where(y0 >= self.ny - 1, torch.ones_like(fy), fy - y0))
		tz = fz - z0
		y0 = y0.clamp_(0, self.ny - 2).long()
		x0 = torch.remainder(x0, self.nx).long()
		z0 = torch.remainder(z0, self.nz).long()
		x1 = torch.remainder(x0 + 1, self.nx)
		z1 = torch.remainder(z0 + 1, self.nz)
		row0 = ((z0 + shift) * self.ny + y0) * self.nx
		row1 = ((z1 + shift) * self.ny + y0) * self.nx
		data = self.data
		step = self.nx
		c000 = data[row0 + x0].float()
		c100 = data[row0 + x1].float()
		c010 = data[row0 + step + x0].float()
		c110 = data[row0 + step + x1].float()
		c001 = data[row1 + x0].float()
		c101 = data[row1 + x1].float()
		c011 = data[row1 + step + x0].float()
		c111 = data[row1 + step + x1].float()
		return (c000, c100, c010, c110, c001, c101, c011, c111), tx, ty, tz

	def distance(self, points, shift):
		(c000, c100, c010, c110, c001, c101, c011, c111), tx, ty, tz = self._cell(points, shift)
		c00 = torch.lerp(c000, c100, tx)
		c10 = torch.lerp(c010, c110, tx)
		c01 = torch.lerp(c001, c101, tx)
		c11 = torch.lerp(c011, c111, tx)
		return torch.lerp(torch.lerp(c00, c10, ty), torch.lerp(c01, c11, ty), tz)

	def sample(self, points, shift):
		(c000, c100, c010, c110, c001, c101, c011, c111), tx, ty, tz = self._cell(points, shift)
		sx = 1.0 - tx
		sy = 1.0 - ty
		sz = 1.0 - tz
		c00 = c000 * sx + c100 * tx
		c10 = c010 * sx + c110 * tx
		c01 = c001 * sx + c101 * tx
		c11 = c011 * sx + c111 * tx
		c0 = c00 * sy + c10 * ty
		c1 = c01 * sy + c11 * ty
		dist = c0 * sz + c1 * tz
		gx = ((c100 - c000) * sy * sz + (c110 - c010) * ty * sz + (c101 - c001) * sy * tz + (c111 - c011) * ty * tz) * self.inv
		gy = ((c10 - c00) * sz + (c11 - c01) * tz) * self.inv
		gz = (c1 - c0) * self.inv
		gy = torch.where(points[..., 1] < 0.0, -gy, gy)
		return dist, torch.stack((gx, gy, gz), -1)
