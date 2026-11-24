import numpy as np
import torch

from elan.config import FlightCfg


class CraftPoints:
	TRAIL = 256
	FRESH = 14
	EDGE_POINTS = 56
	FILL_POINTS = 520
	FLAME_POINTS = 16
	SHEATH_POINTS = 240
	KITE = {'nose': (0.0, 0.0, 1.55), 'tail': (0.0, 0.05, -0.7), 'keel': (0.0, -0.42, 0.2), 'fin': (0.0, 0.62, -1.0), 'left': (-1.75, 0.42, -1.05), 'right': (1.75, 0.42, -1.05)}
	SWEPT = {'nose': (0.0, 0.0, 1.95), 'tail': (0.0, 0.05, -0.95), 'keel': (0.0, -0.36, 0.05), 'fin': (0.0, 0.5, -1.55), 'left': (-1.2, 0.22, -2.15), 'right': (1.2, 0.22, -2.15)}
	DART = {'nose': (0.0, 0.0, 2.3), 'tail': (0.0, 0.0, -1.0), 'keel': (0.0, -0.34, -0.1), 'fin': (0.0, 0.58, -1.8), 'left': (-0.52, -0.36, -1.95), 'right': (0.52, -0.36, -1.95)}
	EDGES = (('nose', 'left', 1.0), ('nose', 'right', 1.0), ('left', 'tail', 1.0), ('right', 'tail', 1.0), ('nose', 'tail', 0.6), ('tail', 'fin', 0.8), ('fin', 'nose', 0.35), ('nose', 'keel', 0.5), ('keel', 'tail', 0.5), ('keel', 'left', 0.25), ('keel', 'right', 0.25))

	def __init__(self, device, uniforms):
		self.device = device
		self.pos = uniforms.view('craft_pos')
		self.basis = uniforms.view('craft_basis', 3, 3)
		self.tips = uniforms.view('tips', 2, 2, 3)
		self.fresh_from = uniforms.view('fresh_from', 1, 2, 3)
		self.fresh_step = uniforms.view('fresh_step', 1, 2, 3)
		rng = np.random.default_rng(5)
		fills = []
		for _ in range(2):
			u = rng.uniform(0.0, 1.0, (self.FILL_POINTS, 2)).astype(np.float32)
			flip = u.sum(1) > 1.0
			u[flip] = 1.0 - u[flip]
			fills.append(u)
		jitter = rng.normal(0.0, 0.1, (self.FLAME_POINTS, 3)).astype(np.float32) + np.array((0.0, 0.0, -0.3), dtype=np.float32)
		sheath, sheath_weight = self._sheath(rng)
		kite = np.concatenate((self._shape(self.KITE, fills, jitter), sheath), 0)
		dart = np.concatenate((self._shape(self.DART, fills, jitter), sheath), 0)
		weights = [np.full(self.EDGE_POINTS, 0.058 * weight, dtype=np.float32) for _, _, weight in self.EDGES]
		weights += [np.full(self.FILL_POINTS, 0.023, dtype=np.float32)] * 2
		self.body_count = sum(w.shape[0] for w in weights)
		self.solid_count = self.body_count + self.FLAME_POINTS
		self.model = torch.from_numpy(kite).to(device)
		self.solid = self.model[:self.solid_count].contiguous()
		self.delta = torch.from_numpy(dart - kite).to(device)
		swept = self._shape(self.SWEPT, fills, jitter)
		self.sweep = torch.from_numpy(swept - kite[:swept.shape[0]]).to(device)
		self.weights = torch.from_numpy(np.concatenate(weights, 0)).to(device).view(-1, 1)
		self.sheath_color = (torch.tensor((1.0, 0.85, 0.55), device=device).view(1, 3) * torch.from_numpy(sheath_weight).to(device).view(-1, 1)).contiguous()
		self.body_color = None
		self.dart_color = None
		self.flame_color = torch.tensor((0.2, 0.55, 1.0), device=device).view(1, 3) * 0.3
		ember = torch.tensor((0.1, 0.12, 1.0), device=device)
		self.alarm_color = (ember.view(1, 3) * 1.5 * self.weights).contiguous()
		self.alarm_trail = (ember * 0.09).view(1, 1, 3).contiguous()
		self.alarm = 0.0
		self.trail = torch.zeros((self.TRAIL, 2, 3), device=device)
		age = torch.arange(self.TRAIL, device=device, dtype=torch.float32) / self.TRAIL
		self.ramp = (age ** 2).view(-1, 1, 1).contiguous()
		self.fresh_u = torch.linspace(0.0, 1.0, self.FRESH, device=device).view(-1, 1, 1)
		self.trail_color = None
		self.head = 0
		self.visible = False
		self.thrust = 0.0
		self.heat = 0.0
		self.morph = 0.0
		self.sheath = 0.0
		self.surge = 0.0
		self.rush = 0.0
		self.swept = 0.0
		self.scale = (1.0, 1.0, 1.0)
		self.last_tips = None
		self.stale = True

	def _shape(self, shape, fills, jitter):
		corner = {name: np.asarray(value, dtype=np.float32) for name, value in shape.items()}
		blend = np.linspace(0.0, 1.0, self.EDGE_POINTS, dtype=np.float32)[:, None]
		points = [corner[a] + (corner[b] - corner[a]) * blend for a, b, _ in self.EDGES]
		for side, u in zip(('left', 'right'), fills):
			points.append(corner['nose'] + (corner[side] - corner['nose']) * u[:, :1] + (corner['tail'] - corner['nose']) * u[:, 1:])
		points.append(corner['tail'] + jitter)
		return np.concatenate(points, 0)

	def _sheath(self, rng):
		count = self.SHEATH_POINTS
		along = rng.uniform(0.0, 1.0, count).astype(np.float32) ** 1.4
		angle = rng.uniform(0.0, 2.0 * np.pi, count).astype(np.float32)
		radius = 0.12 + 1.05 * np.sqrt(along)
		points = np.stack((np.cos(angle) * radius, np.sin(angle) * radius, 2.75 - along * 5.2), 1).astype(np.float32)
		weight = (0.008 + 0.05 * (1.0 - along) ** 2).astype(np.float32)
		return points, weight

	def set_palette(self, color):
		tint = torch.tensor((color[2], color[1], color[0]), device=self.device, dtype=torch.float32).view(1, 3)
		deep = tint * 0.6 + torch.tensor((1.0, 0.82, 0.42), device=self.device).view(1, 3) * 1.25
		self.body_color = (tint * self.weights).contiguous()
		self.dart_color = (deep * self.weights).contiguous()
		self.trail_color = (tint * 0.07).view(1, 1, 3).contiguous()
		self.stale = True

	def _tip(self, craft, side):
		a = self.KITE[side]
		b = self.DART[side]
		m = self.morph
		sx, sy, sz = self.scale
		c = self.SWEPT[side]
		k = self.swept * (1.0 - m)
		local = ((a[0] + (b[0] - a[0]) * m + (c[0] - a[0]) * k) * sx, (a[1] + (b[1] - a[1]) * m + (c[1] - a[1]) * k) * sy, (a[2] + (b[2] - a[2]) * m + (c[2] - a[2]) * k) * sz)
		r, u, f, p = craft['right'], craft['up'], craft['forward'], craft['pos']
		return (p[0] + r[0] * local[0] + u[0] * local[1] + f[0] * local[2], p[1] + r[1] * local[0] + u[1] * local[1] + f[1] * local[2], p[2] + r[2] * local[0] + u[2] * local[1] + f[2] * local[2])

	def prepare(self, craft, uniforms):
		self.visible = craft is not None and craft['visible']
		if craft is None:
			self.stale = True
			return
		self.morph = craft.get('morph', 0.0)
		self.sheath = craft.get('sheath', 0.0)
		self.surge = craft.get('surge', 0.0)
		self.rush = craft.get('rush', 0.0)
		self.alarm = craft.get('alarm', 0.0)
		r, u, f = craft['right'], craft['up'], craft['forward']
		amount = craft.get('stretch', 0.0)
		self.swept = min(amount * FlightCfg.SWEEP, 1.0)
		sx, sy, sz = (1.0 + k * amount for k in FlightCfg.STRETCH)
		self.scale = (sx, sy, sz)
		uniforms.set('craft_pos', craft['pos'])
		uniforms.set('craft_basis', (r[0] * sx, r[1] * sx, r[2] * sx, u[0] * sy, u[1] * sy, u[2] * sy, f[0] * sz, f[1] * sz, f[2] * sz))
		left = self._tip(craft, 'left')
		right = self._tip(craft, 'right')
		if self.last_tips is None or self.stale or craft['jump']:
			self.last_tips = (left, right)
			self.stale = True
		old_left, old_right = self.last_tips
		mid_left = tuple((a + b) * 0.5 for a, b in zip(old_left, left))
		mid_right = tuple((a + b) * 0.5 for a, b in zip(old_right, right))
		uniforms.set('tips', mid_left + mid_right + left + right)
		uniforms.set('fresh_from', left + right)
		uniforms.set('fresh_step', tuple(a - b for a, b in zip(old_left, left)) + tuple(a - b for a, b in zip(old_right, right)))
		self.last_tips = (left, right)
		self.thrust = craft['thrust']
		self.heat = craft['heat']

	def gather(self, points, colors):
		if self.body_color is None:
			return
		if self.stale:
			self.trail[:] = self.tips[1]
			self.stale = False
		self.trail[self.head:self.head + 2] = self.tips
		self.head = (self.head + 2) % self.TRAIL
		fade = torch.roll(self.ramp, self.head, 0)
		strength = 0.25 + 1.75 * self.heat + 1.3 * self.sheath + 1.5 * self.rush
		alarm = self.alarm
		wake = self.trail_color if alarm < 0.01 else torch.lerp(self.trail_color, self.alarm_trail, alarm)
		body = self.body_color if alarm < 0.01 else torch.lerp(self.body_color, self.alarm_color, alarm)
		points.append(self.trail.view(-1, 3))
		colors.append((wake * fade).mul_(strength).expand(self.TRAIL, 2, 3).reshape(-1, 3))
		points.append(torch.addcmul(self.fresh_from, self.fresh_u, self.fresh_step).view(-1, 3))
		colors.append((wake * strength).expand(self.FRESH, 2, 3).reshape(-1, 3))
		if not self.visible:
			return
		morph = self.morph
		flame = self.flame_color.expand(self.FLAME_POINTS, 3) * (0.35 + 1.4 * self.thrust + 1.2 * self.surge * morph)
		if morph < 0.004:
			solid = torch.add(self.solid, self.sweep, alpha=self.swept) if self.swept > 0.02 else self.solid
			points.append(torch.matmul(solid, self.basis).add_(self.pos))
			colors.append(body * (1.0 + 1.6 * self.rush) if self.rush > 0.02 else body)
			colors.append(flame)
			return
		pulse = 4.0 * morph * (1.0 - morph)
		points.append(torch.matmul(torch.add(self.model, self.delta, alpha=morph), self.basis).add_(self.pos))
		colors.append(torch.lerp(body, self.dart_color, morph).mul_(1.0 + 2.4 * pulse + 1.3 * self.surge * morph))
		colors.append(flame)
		colors.append(self.sheath_color * (self.sheath * (0.6 + 0.9 * self.surge)))
