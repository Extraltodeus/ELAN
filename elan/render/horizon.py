import math
import torch


class HorizonLens:
	SHADOW = 0.6
	SPLIT = (1.07, 1.035, 1.0)
	RING = 7.0
	HALO = 0.4
	TWIST = 0.9
	OPEN = 0.75
	SPIN = 2.6
	RIM = 0.8
	BEAM = 9.0
	VEIL = 1.2

	def __init__(self, device):
		self.device = device
		self.h = 0
		self.w = 0
		self.aspect = 1.0
		self.corner = 1.0
		self.x = None
		self.y = None
		self.warm = torch.tensor((0.78, 0.92, 1.0), device=device).view(1, 3, 1, 1)

	def allocate(self, h, w):
		self.h = h
		self.w = w
		self.aspect = h / w
		self.corner = math.sqrt(1.0 + self.aspect * self.aspect)
		self.x = ((torch.arange(w, device=self.device, dtype=torch.float32) + 0.5) / w * 2.0 - 1.0).view(1, 1, w).contiguous()
		self.y = ((torch.arange(h, device=self.device, dtype=torch.float32) + 0.5) / h * 2.0 - 1.0).view(1, h, 1).contiguous()

	def apply(self, hdr, view):
		kind, amount, spot, lens = view.horizon
		amount = min(max(float(amount), 0.0), 1.0)
		if kind == 'fall':
			reach = math.tan(math.radians(view.fov) * 0.5)
			return self._fall(hdr, amount, spot[0] / reach, -spot[1] / (reach * self.aspect), lens, view.time)
		return self._rise(hdr, amount, lens)

	def _grid(self, qx, qy, cx, cy, gain, turn):
		if turn is None:
			sx = qx * gain
			sy = qy * gain
		else:
			cos = torch.cos(turn)
			sin = torch.sin(turn)
			sx = (qx * cos - qy * sin) * gain
			sy = (qx * sin + qy * cos) * gain
		return torch.stack((sx + cx, sy / self.aspect + cy), -1)

	def _fall(self, hdr, amount, cx, cy, lens, clock):
		einstein = 0.03 + 0.5 * amount * amount + 1.95 * amount ** 7
		shadow = einstein * self.SHADOW
		qx = self.x - cx
		qy = (self.y - cy) * self.aspect
		r2 = (qx * qx + qy * qy).clamp_min(1.0e-6)
		r = r2.sqrt()
		if lens:
			turn = (einstein / r).clamp(max=2.5).pow(1.5).mul_(self.TWIST * amount)
			parts = []
			for channel, split in enumerate(self.SPLIT):
				gain = 1.0 - (einstein * split) ** 2 / r2
				parts.append(torch.grid_sampler(hdr[:, channel:channel + 1], self._grid(qx, qy, cx, cy, gain, turn), 0, 1, False))
			out = torch.cat(parts, 1)
		else:
			out = hdr.clone()
		edge = ((r - shadow) / (0.012 + 0.05 * shadow)).clamp(0.0, 1.0)
		out.mul_(edge.unsqueeze(1))
		ring = torch.exp(-((r - shadow * 1.05) / (0.006 + 0.07 * shadow)) ** 2)
		angle = torch.atan2(qy, qx)
		beam = 0.7 + 0.3 * torch.sin(angle * 3.0 + clock * 2.2 - r * (5.0 / einstein))
		lean = 1.0 + 0.45 * torch.cos(angle - clock * 0.6)
		halo = (shadow / r).clamp(max=1.0).pow(4).mul_(self.HALO).mul_(edge)
		glow = (ring * beam * lean * self.RING + halo) * min(0.25 + 2.0 * amount, 1.0)
		out.add_(self.warm * glow.unsqueeze(1))
		return out

	def _rise(self, hdr, amount, lens):
		spread = min((1.0 - amount) / self.OPEN, 1.0)
		port = max(self.corner * 1.15 * spread * spread, 1.0e-4)
		qx = self.x
		qy = self.y * self.aspect
		r = (qx * qx + qy * qy).sqrt()
		u = (r / port).clamp(max=1.0)
		if lens:
			gain = (1.0 + self.RIM * amount * u.pow(6)) * max(self.corner / port, 1.0)
			turn = (1.0 - u * u) * (self.SPIN * amount * amount)
			out = torch.grid_sampler(hdr, self._grid(qx, qy, 0.0, 0.0, gain, turn), 0, 1, False)
		else:
			out = hdr.clone()
		inside = ((port - r) / (0.01 + 0.04 * port)).clamp(0.0, 1.0)
		out.mul_(inside.unsqueeze(1))
		rim = torch.exp(-((r - port) / (0.008 + 0.06 * port)) ** 2) * (self.BEAM * math.sqrt(amount))
		veil = inside * (self.VEIL * amount ** 3)
		out.add_((rim + veil).unsqueeze(1))
		return out
