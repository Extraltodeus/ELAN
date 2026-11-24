import math

from elan.config import WorldCfg


class FieldSampler:
	def __init__(self, field):
		self.field = field
		self.nz, self.ny, self.nx = field.shape
		self.inv = self.nx / WorldCfg.SIZE_X

	def sample(self, x, y, z):
		inv = self.inv
		fx = x * inv - 0.5
		fy = abs(y) * inv - 0.5
		fz = z * inv - 0.5
		x0 = math.floor(fx)
		y0 = math.floor(fy)
		z0 = math.floor(fz)
		tx = fx - x0
		ty = fy - y0
		tz = fz - z0
		x0 %= self.nx
		z0 %= self.nz
		x1 = (x0 + 1) % self.nx
		z1 = (z0 + 1) % self.nz
		if y0 < 0:
			y0 = 0
			ty = 0.0
		elif y0 >= self.ny - 1:
			y0 = self.ny - 2
			ty = 1.0
		y1 = y0 + 1
		f = self.field
		c000 = float(f[z0, y0, x0])
		c100 = float(f[z0, y0, x1])
		c010 = float(f[z0, y1, x0])
		c110 = float(f[z0, y1, x1])
		c001 = float(f[z1, y0, x0])
		c101 = float(f[z1, y0, x1])
		c011 = float(f[z1, y1, x0])
		c111 = float(f[z1, y1, x1])
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
		gx = ((c100 - c000) * sy * sz + (c110 - c010) * ty * sz + (c101 - c001) * sy * tz + (c111 - c011) * ty * tz) * inv
		gy = ((c10 - c00) * sz + (c11 - c01) * tz) * inv
		gz = (c1 - c0) * inv
		if y < 0.0:
			gy = -gy
		return dist, gx, gy, gz

	def distance(self, x, y, z):
		return self.sample(x, y, z)[0]

	def raycast(self, origin, direction, reach):
		t = 1.0
		for _ in range(56):
			d = self.sample(origin[0] + direction[0] * t, origin[1] + direction[1] * t, origin[2] + direction[2] * t)[0]
			if d < 0.35:
				return t
			t += d * 0.9
			if t > reach:
				return None
		return None
