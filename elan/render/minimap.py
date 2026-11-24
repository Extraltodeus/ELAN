import math
import numpy as np
import torch
import torch.nn.functional as F

from elan.config import RenderCfg, WorldCfg
from elan.layout import minimap_box


class Minimap:
	LINK_DOTS = 22
	LEAD_DOTS = 5
	LEVEL = 70.0

	def __init__(self, device, uniforms, space):
		self.device = device
		self.space = space
		self.cam = uniforms.view('cam')
		self.origin = uniforms.view('origin')
		self.slice = uniforms.view('mini_slice', 2, 3)
		self.turn = uniforms.view('mini_turn', 3, 2)
		self.route = uniforms.view('mini_route', 4, 3)
		self.state = uniforms.view('mini_state')
		self.velocity = uniforms.view('mini_vel', 1, 3)
		self.enabled = True
		self.shown = True
		self.position = 'bottom_right'
		self.range = RenderCfg.MINIMAP_RANGE
		self.heading = (0.0, 0.0, 1.0)
		self.out_h = 0
		self.out_w = 0
		self.size = 0
		self.radius = 0
		self.rect = (0, 0, 0, 0)
		self.base = None
		self.disc = None
		self.glass = None
		self.frame = None
		self.buffer = None
		self.center = None
		self.flip = None
		self.stride = None
		self.points = None
		self.shift = None
		self.source = None
		self.colors = None
		self.tone = None
		self.gate = None
		self.gate_stamp = None
		self.gate_color = None
		self.gate_source = 0
		self.high = None
		self.low = None
		self.rock = None
		self.edge = None
		self.dots = None
		self.dot_colors = None
		self.groups = torch.tensor((self.LINK_DOTS, self.LINK_DOTS, self.LEAD_DOTS), device=device)
		self.blend = torch.linspace(0.0, 1.0, self.LINK_DOTS, device=device).view(-1, 1)
		self.leads = torch.linspace(0.5, 2.5, self.LEAD_DOTS, device=device).view(-1, 1)
		self.terrain = None
		self.cached = None
		self.tick = 0
		self.world = None
		self.layout = None

	def _bgr(self, rgb, scale):
		return torch.tensor((rgb[2] * scale, rgb[1] * scale, rgb[0] * scale), device=self.device, dtype=torch.float32).view(3, 1)

	def _stamp(self, radius, hollow):
		offsets = []
		for dy in range(-radius, radius + 1):
			for dx in range(-radius, radius + 1):
				reach = math.sqrt(dx * dx + dy * dy)
				if reach <= radius + 0.3 and (not hollow or reach >= radius - 0.9):
					offsets.append(dy * self.size + dx)
		return offsets

	def set_world(self, world, layout):
		self.world = world
		self.layout = layout
		if self.size:
			self._bind()

	def _bind(self):
		world = self.world
		layout = self.layout
		palette = world.palette
		ribbons = np.asarray(world.ribbons, dtype=np.float32)
		count, steps = ribbons.shape[0], ribbons.shape[1]
		cluster = layout.cluster
		motes = np.arange(count * steps) * cluster
		lines = np.concatenate([motes] + [motes + cluster - layout.RAILS + k for k in range(layout.RAILS)], 0)
		line_shift = np.zeros(lines.shape[0], dtype=np.int64)
		dot = self._stamp(max(int(self.radius * 0.022), 2), False)
		ends = np.concatenate((np.arange(count) * steps, np.arange(count) * steps + steps - 1), 0) * cluster
		end_index = np.repeat(ends, len(dot))
		end_shift = np.tile(np.asarray(dot, dtype=np.int64), ends.shape[0])
		rings = np.asarray(world.rings_pos).shape[0]
		hoops = (layout.ring_start + np.arange(rings)[:, None] * layout.RING + np.arange(0, 72, 3)[None, :]).reshape(-1)
		index = np.concatenate((lines, end_index, hoops), 0)
		shift = np.concatenate((line_shift, end_shift, np.zeros(hoops.shape[0], dtype=np.int64)), 0)
		positions = layout.positions[index]
		colors = np.zeros((index.shape[0], 3), dtype=np.float32)
		mote = np.asarray(palette['mote'], dtype=np.float32)[::-1]
		gate = np.asarray(palette['gate'], dtype=np.float32)[::-1]
		colors[:lines.shape[0]] = 150.0
		colors[lines.shape[0]:lines.shape[0] + end_index.shape[0]] = 120.0
		colors[lines.shape[0] + end_index.shape[0]:] = gate * 46.0
		tone = np.zeros(index.shape[0], dtype=np.float32)
		tone[:lines.shape[0] + end_index.shape[0]] = 1.0 / self.LEVEL
		self.points = torch.from_numpy(positions.astype(np.float32)).to(self.device)
		self.shift = torch.from_numpy(shift).to(self.device)
		self.source = torch.from_numpy(index).to(self.device)
		self.colors = torch.from_numpy(colors.T.copy()).to(self.device)
		self.tone = torch.from_numpy(tone).to(self.device)
		self.high = self._bgr((1.0, 0.8, 0.22), 130.0)
		self.low = self._bgr((0.5, 0.38, 1.0), 100.0)
		ring = self._stamp(max(int(self.radius * 0.06), 4), True) + self._stamp(1, False)
		self.gate = torch.tensor(np.asarray(world.gate_pos, dtype=np.float32), device=self.device).view(1, 3)
		self.gate_stamp = torch.tensor(ring, device=self.device, dtype=torch.long)
		self.gate_color = self._bgr(palette['gate'], 255.0)
		self.gate_source = layout.gate_start
		self.rock = self._bgr(palette['vein'], 34.0) + self._bgr((1.0, 1.0, 1.0), 12.0)
		self.edge = self._bgr(palette['vein'], 120.0) + self._bgr((1.0, 1.0, 1.0), 40.0)
		white = torch.tensor((255.0, 255.0, 255.0), device=self.device).view(3, 1)
		link = torch.cat((white.expand(3, self.LINK_DOTS) * 0.9, white.expand(3, self.LINK_DOTS) * 0.45), 1)
		lead = (self._bgr((0.6, 1.0, 0.75), 200.0)).expand(3, self.LEAD_DOTS)
		self.dot_colors = torch.cat((link, lead), 1).contiguous()
		self.terrain = None
		self.cached = None

	def set_position(self, position):
		if position == self.position:
			return
		self.position = position
		if self.out_h:
			self.allocate(self.out_h, self.out_w, True)

	def allocate(self, out_h, out_w, force=False):
		if not force and self.size and (out_h, out_w) == (self.out_h, self.out_w):
			return
		self.out_h = out_h
		self.out_w = out_w
		y0, x0, size, radius = minimap_box(out_h, out_w, self.position)
		self.radius = radius
		self.size = size
		self.rect = (y0, y0 + size, x0, x0 + size)
		self.buffer = torch.zeros((3, size * size), device=self.device)
		self.center = torch.tensor((size * 0.5, size * 0.5), device=self.device)
		self.flip = torch.tensor((float(radius), -float(radius)), device=self.device)
		self.stride = torch.tensor((1, size), device=self.device, dtype=torch.long)
		axis = (torch.arange(size, device=self.device, dtype=torch.float32) + 0.5 - size * 0.5) / radius
		gy, gx = torch.meshgrid(axis, axis, indexing='ij')
		self.base = torch.stack((gx, -gy), -1).view(-1, 2).contiguous()
		dist = torch.sqrt(gx * gx + gy * gy) * radius
		inside = torch.clamp(radius + 0.5 - dist, 0.0, 1.0)
		self.disc = inside.view(1, size * size).contiguous()
		self.glass = (1.0 - 0.8 * inside).view(1, 1, size, size).contiguous()
		rim = torch.clamp(1.0 - (dist - radius).abs() / 1.4, 0.0, 1.0) * 0.6
		half = torch.clamp(1.0 - (dist - radius * 0.5).abs() / 0.8, 0.0, 1.0) * 0.1
		px = gx * radius
		py = gy * radius
		reach = max(radius * 0.06, 4.0)
		arrow = torch.clamp((py + reach * 1.5) * 0.42 - px.abs() + 0.8, 0.0, 1.0) * torch.clamp(reach * 0.9 - py + (px.abs() * 0.5), 0.0, 1.0) * (py > -reach * 1.6) * 1.6
		shade = ((rim + half) * inside.clamp(min=0.0) + rim * (1.0 - inside) + arrow).view(1, 1, size, size)
		tint = torch.tensor((255.0, 235.0, 190.0), device=self.device).view(1, 3, 1, 1)
		self.frame = (shade * tint).contiguous()
		if self.world is not None:
			self._bind()

	def prepare(self, view, uniforms):
		if not self.enabled:
			return
		fx, fz = view.forward[0], view.forward[2]
		flat = math.sqrt(fx * fx + fz * fz)
		if flat > 0.05:
			self.heading = (fx / flat, 0.0, fz / flat)
		hx, _, hz = self.heading
		reach = self.range
		sx = reach / WorldCfg.SIZE_X
		sz = reach / WorldCfg.SIZE_Z
		uniforms.set('mini_slice', (hz * sx, 0.0, -hx * sz, hx * sx, 0.0, hz * sz))
		k = 1.0 / reach
		uniforms.set('mini_turn', (hz * k, hx * k, 0.0, 0.0, -hx * k, hz * k))
		route = view.route
		links = [0.0, 0.0]
		flat_route = [0.0] * 12
		if route is not None:
			for k, link in enumerate(route[:2]):
				a, b = link
				flat_route[k * 6:k * 6 + 6] = [a[0], a[1], a[2], b[0], b[1], b[2]]
				links[k] = 1.0
		uniforms.set('mini_route', flat_route)
		pulse = 0.65 + 0.35 * math.sin(view.time * 7.0)
		uniforms.set('mini_state', (links[0], links[1], pulse, 0.0))
		uniforms.set('mini_vel', view.velocity)

	def _project(self, rel):
		flat = torch.matmul(rel, self.turn)
		cell = torch.addcmul(self.center, flat, self.flip).long()
		return flat, (cell * self.stride).sum(1)

	def draw(self, scene):
		self.tick += 1
		if self.tick % 2 == 0 and self.cached is not None:
			return self.cached
		size = self.size
		last = size * size - 1
		state = self.state
		gain = scene.glow_gain
		route = self.route
		links = torch.cat((torch.addcmul(route[0:1], self.blend, route[1:2] - route[0:1]), torch.addcmul(route[2:3], self.blend, route[3:4] - route[2:3])), 0)
		leads = torch.addcmul(self.cam, self.leads, self.velocity)
		world = torch.cat((self.points, links, leads), 0)
		rel = world - self.cam
		rel.addcmul_(torch.round(rel * self.space.inv_period), self.space.period, value=-1.0)
		flat, index = self._project(rel)
		inside = (flat * flat).sum(1) < 0.98
		count = self.points.shape[0]
		rise = torch.mul(rel[:count, 1], self.tone).clamp_(-1.0, 1.0)
		level = torch.rsub(rise.abs(), 1.0)
		tint = self.colors * level
		tint.addcmul_(self.high, torch.relu(rise))
		tint.addcmul_(self.low, torch.relu(rise.neg()))
		tint.mul_(gain[self.source] * inside[:count])
		index[:count] += self.shift
		extra = self.dot_colors * (torch.repeat_interleave(state[:3].clamp(max=1.0), self.groups) * inside[count:])
		self.buffer.zero_()
		self.buffer.index_add_(1, index.clamp_(0, last), torch.cat((tint, extra), 1))
		spot = self.gate - self.cam
		spot.addcmul_(torch.round(spot * self.space.inv_period), self.space.period, value=-1.0)
		where = torch.matmul(spot, self.turn)
		where = where * (0.93 / torch.linalg.vector_norm(where).clamp(min=0.93))
		cell = torch.addcmul(self.center, where, self.flip).long()
		ring = ((cell * self.stride).sum(1) + self.gate_stamp).clamp_(0, last)
		power = torch.maximum(gain[self.gate_source] * state[2] * 1.6, torch.full_like(state[2], 0.3))
		self.buffer.index_add_(1, ring, (self.gate_color * power).expand(3, ring.shape[0]))
		image = self.buffer.view(1, 3, size, size)
		image = torch.add(image, F.avg_pool2d(image, 3, 1, 1), alpha=2.0)
		if self.terrain is None or self.tick % 4 == 1:
			grid = torch.addmm(self.origin.view(1, 3), self.base, self.slice).view(1, 1, size, size, 3)
			depth = torch.grid_sampler(self.world.sdf, grid, 0, 2, False).view(1, size * size)
			rock = torch.lt(depth, 0.0) * self.disc
			shore = torch.sub(depth, 2.0).abs_().mul_(-0.5).add_(1.0).clamp_(min=0.0).mul_(self.disc)
			self.terrain = (self.rock * rock).addcmul_(self.edge, shore).view(1, 3, size, size)
		self.cached = (self.glass, image.add_(self.terrain).add_(self.frame), self.rect)
		return self.cached
