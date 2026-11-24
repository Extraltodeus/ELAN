import math
import torch


class Wormhole:
	BEND = 1.7
	REACH = 0.92

	def __init__(self, device):
		self.device = device
		self.h = 0
		self.w = 0
		self.xs = None
		self.ys = None
		self.tints = {}

	def allocate(self, h, w):
		self.h = h
		self.w = w
		self.xs = torch.arange(w, device=self.device, dtype=torch.float32) + 0.5
		self.ys = torch.arange(h, device=self.device, dtype=torch.float32) + 0.5

	def locate(self, view, target, radius):
		w, h = self.w, self.h
		rx = target[0] - view.pos[0]
		ry = target[1] - view.pos[1]
		rz = target[2] - view.pos[2]
		r, u, f = view.right, view.up, view.forward
		x = rx * r[0] + ry * r[1] + rz * r[2]
		y = rx * u[0] + ry * u[1] + rz * u[2]
		z = rx * f[0] + ry * f[1] + rz * f[2]
		dist = math.sqrt(rx * rx + ry * ry + rz * rz)
		if z < 1.0 and dist > radius:
			return None
		focal = 0.5 * w / math.tan(math.radians(view.fov) * 0.5)
		depth = max(z, 1.0)
		cx = 0.5 * w + x / depth * focal
		cy = 0.5 * h - y / depth * focal
		size = min(focal * radius / max(math.sqrt(max(dist * dist - radius * radius, 1.0)), 1.0), self.REACH * max(w, h))
		if size < 5.0 or cx + size < 0.0 or cx - size > w or cy + size < 0.0 or cy - size > h:
			return None
		return cx, cy, size, dist

	def apply(self, color, depth, view, target, radius, tint):
		spot = self.locate(view, target, radius)
		if spot is None:
			return
		cx, cy, size, dist = spot
		w, h = self.w, self.h
		x0, x1 = max(int(cx - size), 0), min(int(cx + size) + 1, w)
		y0, y1 = max(int(cy - size), 0), min(int(cy + size) + 1, h)
		if x1 - x0 < 2 or y1 - y0 < 2:
			return
		image = color.view(1, 3, h, w)
		region = image[:, :, y0:y1, x0:x1]
		dx = (self.xs[x0:x1] - cx).unsqueeze(0)
		dy = (self.ys[y0:y1] - cy).unsqueeze(1)
		fall = torch.add(dx * dx, dy * dy).mul_(-1.0 / (size * size)).add_(1.0).clamp_(min=0.0)
		deep = fall * fall
		scale = torch.mul(deep, -self.BEND).add_(1.0)
		gx = torch.mul(dx, scale).mul_(2.0 / w).add_(cx * 2.0 / w - 1.0)
		gy = torch.mul(dy, scale).mul_(2.0 / h).add_(cy * 2.0 / h - 1.0)
		warped = torch.grid_sampler(image, torch.stack((gx, gy), -1).unsqueeze(0), 0, 1, False)
		band = torch.mul(fall, torch.rsub(fall, 1.0)).mul_(4.0).pow_(4)
		glow = torch.add(band, deep, alpha=0.06).unsqueeze(0)
		warped.mul_(torch.mul(deep, -0.62).add_(1.0))
		warped.add_(self._tint(tint) * glow)
		mask = torch.gt(depth.view(h, w)[y0:y1, x0:x1], dist - radius).mul(torch.gt(fall, 0.0))
		region.copy_(torch.where(mask, warped, region))

	def _tint(self, tint):
		key = tuple(tint)
		if key not in self.tints:
			self.tints[key] = torch.tensor((tint[2] * 0.75, tint[1] * 0.75, tint[0] * 0.75), device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
		return self.tints[key]
