import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from elan.config import TunnelCfg, WorldCfg
from elan.world.fieldops import FieldOps


class TunnelPath:
	def __init__(self, seed, hyper):
		cfg = TunnelCfg
		rng = random.Random(seed * 9973 + 5)
		calm = cfg.CALM if hyper else 1.0
		self.hyper = hyper
		self.length = cfg.LENGTH
		self.radius = cfg.RADIUS_HYPER if hyper else cfg.RADIUS
		self.sway = [(amp * calm, order, rng.uniform(0.0, math.tau)) for amp, order in cfg.SWAY]
		self.heave = [(amp * calm, order, rng.uniform(0.0, math.tau)) for amp, order in cfg.HEAVE]
		self.pulse = [(cfg.PULSE[0], cfg.PULSE[1], rng.uniform(0.0, math.tau))]
		self.turn = rng.uniform(0.0, math.tau)
		self.breath = rng.uniform(0.0, math.tau)
		self.spin = 1.0 if rng.random() < 0.5 else -1.0

	def wave(self, terms, s, sin=math.sin):
		k = math.tau / self.length
		total = 0.0
		for amp, order, phase in terms:
			total = total + sin(s * (k * order) + phase) * amp
		return total

	def rate(self, terms, s, cos=math.cos):
		k = math.tau / self.length
		total = 0.0
		for amp, order, phase in terms:
			total = total + cos(s * (k * order) + phase) * (amp * k * order)
		return total

	def curve(self, terms, s, sin=math.sin):
		k = math.tau / self.length
		total = 0.0
		for amp, order, phase in terms:
			total = total - sin(s * (k * order) + phase) * (amp * (k * order) ** 2)
		return total

	def axis(self, s):
		return (s / TunnelCfg.PASSES + self.wave(self.sway, s), TunnelCfg.CENTER_Y + self.wave(self.heave, s), s)

	def heading(self, s):
		x = 1.0 / TunnelCfg.PASSES + self.rate(self.sway, s)
		y = self.rate(self.heave, s)
		norm = math.sqrt(x * x + y * y + 1.0)
		return (x / norm, y / norm, 1.0 / norm)

	def station(self, x, z):
		pitch = WorldCfg.SIZE_X / TunnelCfg.PASSES
		z = z % WorldCfg.SIZE_Z
		index = round((x - z / TunnelCfg.PASSES) / pitch)
		return (z + index * WorldCfg.SIZE_Z) % self.length

	def spot(self, s, lane):
		cfg = TunnelCfg
		k = math.tau / self.length
		angle = self.turn + self.spin * k * cfg.TWIST * s
		if lane == 0:
			reach = cfg.LINE_RADIUS[0] + cfg.LINE_RADIUS[1] * math.sin(k * 4.0 * s + self.breath)
		else:
			reach = cfg.SIDE_RADIUS
			angle += math.pi
		x, y, z = self.axis(s)
		return (x + reach * math.cos(angle), y + reach * math.sin(angle), z)

	def course(self, s, lane):
		a = self.spot(s - 4.0, lane)
		b = self.spot(s + 4.0, lane)
		d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
		norm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
		return (d[0] / norm, d[1] / norm, d[2] / norm)

	def lanes(self, steps):
		cfg = TunnelCfg
		ribbons = []
		rings = []
		axes = []
		plan = [(self.length * k / cfg.MAIN + 60.0, 0) for k in range(cfg.MAIN)]
		plan += [(self.length * (k + 0.5) / cfg.SIDE, 1) for k in range(cfg.SIDE)]
		for start, lane in plan:
			ribbons.append([self.spot(start + i * cfg.MOTE_STEP, lane) for i in range(steps)])
			rings.append(self.spot(start - cfg.RING_LEAD, lane))
			axes.append(self.course(start - cfg.RING_LEAD, lane))
		ribbons = np.asarray(ribbons, dtype=np.float32)
		ribbons[..., 0] -= np.floor(ribbons[:, :1, 0] / WorldCfg.SIZE_X) * WorldCfg.SIZE_X
		ribbons[..., 2] -= np.floor(ribbons[:, :1, 2] / WorldCfg.SIZE_Z) * WorldCfg.SIZE_Z
		rings = np.asarray(rings, dtype=np.float32)
		rings[:, 0] %= WorldCfg.SIZE_X
		rings[:, 2] %= WorldCfg.SIZE_Z
		return ribbons, rings, np.asarray(axes, dtype=np.float32)


class TunnelBaker:
	def __init__(self, device):
		self.device = device

	def field(self, path):
		cfg = TunnelCfg
		nz, ny, nx = cfg.NZ, cfg.NY, cfg.NX
		voxel = WorldCfg.SIZE_X / nx
		xs = ((torch.arange(nx, device=self.device, dtype=torch.float32) + 0.5) * voxel).view(1, 1, nx)
		ys = ((torch.arange(ny, device=self.device, dtype=torch.float32) + 0.5) * voxel).view(1, ny, 1)
		zs = ((torch.arange(nz, device=self.device, dtype=torch.float32) + 0.5) * voxel).view(nz, 1, 1)
		pitch = WorldCfg.SIZE_X / cfg.PASSES
		lane = xs - zs / cfg.PASSES
		index = torch.round(lane / pitch)
		s = zs + index * WorldCfg.SIZE_Z
		drift = 1.0 / cfg.PASSES + path.rate(path.sway, s, torch.cos)
		climb = path.rate(path.heave, s, torch.cos)
		dx = (lane - index * pitch - path.wave(path.sway, s, torch.sin)) / torch.sqrt(1.0 + drift * drift)
		dy = (ys - cfg.CENTER_Y - path.wave(path.heave, s, torch.sin)) / torch.sqrt(1.0 + climb * climb)
		radius = path.radius * (1.0 + path.wave(path.pulse, s, torch.sin))
		dist = radius - torch.sqrt(dx * dx + dy * dy)
		norm = FieldOps.grad_norm(dist, voxel, wrap_y=False)
		dist = (dist / norm.clamp_min_(1.0)).clamp_min_(-12.0)
		turn = torch.atan2(dy, dx)
		phase = (s * (math.tau / path.length)).expand(nz, ny, nx)
		coords = torch.stack((torch.cos(phase), torch.sin(phase), torch.cos(turn), torch.sin(turn)), 0)
		coords = F.avg_pool3d(coords.unsqueeze(0), 2)[0]
		return dist, coords.contiguous()

	def detail(self, amount):
		cfg = WorldCfg
		n = cfg.DETAIL_N
		u = (torch.arange(n, device=self.device, dtype=torch.float32) + 0.5) / n * math.tau
		plate = (torch.cos(u * 2.0) * 2.5).clamp(-1.0, 1.0) * 0.5 + 0.5
		seam = (1.0 - torch.sin(u * 2.0).abs() * 5.0).clamp(0.0, 1.0)
		cut = torch.maximum(seam.view(1, n, 1), seam.view(1, 1, n))
		relief = (0.55 * plate.view(n, 1, 1) + 0.45 * (1.0 - cut)).expand(n, n, n) * (cfg.DETAIL_AMP * amount)
		return torch.stack((relief, torch.zeros_like(relief), relief), 0).unsqueeze(0).contiguous()

	def bake(self, world, palette, seed, sector):
		path = TunnelPath(seed * 131 + sector, palette['hyper'])
		relief = palette['style']['relief']
		dist, coords = self.field(path)
		ribbons, rings, axes = path.lanes(WorldCfg.RIBBON_STEPS)
		start = path.spot(0.0, 0)
		world.kind = 'tunnel'
		world.path = path
		world.palette = palette
		world.style = palette['style']
		world.sun_dir = (0.0, 1.0, 0.0)
		world.solid_fraction = float((dist < 0.0).float().mean())
		world.field = dist.half().cpu().numpy()
		world.tile = dist.sub_(WorldCfg.DETAIL_AMP * 0.5 * relief)
		world.coords_tile = coords
		world.aux_tile = torch.tensor((1.0, 0.0), device=self.device).view(2, 1, 1, 1)
		world.detail = self.detail(relief)
		world.sky = torch.zeros((1, 3, 2, 4), device=self.device)
		world.fog = torch.zeros((1, 3, 2, 4), device=self.device)
		world.ripple = torch.zeros((1, 2, 4, 4), device=self.device)
		world.star_dirs = torch.tensor(((0.0, 1.0, 0.0),), device=self.device)
		world.star_colors = torch.zeros((1, 3), device=self.device)
		world.ribbons = ribbons
		world.rings_pos = rings
		world.rings_dir = axes
		world.spawn_pos = np.asarray((start[0] % WorldCfg.SIZE_X, start[1], start[2] % WorldCfg.SIZE_Z), dtype=np.float32)
		world.spawn_dir = np.asarray(path.course(0.0, 0), dtype=np.float32)
		world.gate_pos = np.asarray((start[0] % WorldCfg.SIZE_X, 20.0, start[2]), dtype=np.float32)
		world.gate_dir = world.spawn_dir
		return world
