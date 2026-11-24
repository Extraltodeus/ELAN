import math
import torch
import torch.nn.functional as F

from elan.config import WorldCfg
from elan.world.fieldops import FieldOps


class Forms:
	def __init__(self, device, noise, shape, ys, warp_x, warp_z):
		self.device = device
		self.noise = noise
		self.shape = shape
		self.ys = ys
		self.warp_x = warp_x
		self.warp_z = warp_z
		self.land = None

	def layer(self, name):
		return getattr(self, name)()

	def _drape(self, textures, amp):
		cfg = WorldCfg
		nz, ny, nx = cfg.NZ, cfg.GEN_NY, cfg.NX
		pad = 4
		tex = F.pad(textures.unsqueeze(0), (pad, pad, pad, pad), mode='circular')
		reach = amp / cfg.VOXEL
		xi = torch.arange(nx, device=self.device, dtype=torch.float32).view(1, 1, nx) + 0.5
		zi = torch.arange(nz, device=self.device, dtype=torch.float32).view(nz, 1, 1) + 0.5
		gx = (torch.remainder(xi + self.warp_x * reach, nx) + pad) * (2.0 / (nx + 2 * pad)) - 1.0
		gz = (torch.remainder(zi + self.warp_z * reach, nz) + pad) * (2.0 / (nz + 2 * pad)) - 1.0
		grid = torch.stack((gx, gz), -1).view(1, nz * ny, nx, 2)
		del gx, gz
		return F.grid_sample(tex, grid, mode='bilinear', padding_mode='border', align_corners=False).view(textures.shape[0], nz, ny, nx)

	def _slope(self, height):
		v = WorldCfg.VOXEL
		gz = (torch.roll(height, -1, 0) - torch.roll(height, 1, 0)) / (2.0 * v)
		gx = (torch.roll(height, -1, 1) - torch.roll(height, 1, 1)) / (2.0 * v)
		return torch.sqrt(gx * gx + gz * gz)

	def towers(self):
		cfg = WorldCfg
		v = cfg.VOXEL
		shape = self.shape
		nz, nx = cfg.NZ, cfg.NX
		foot = 0.8 * self.noise.band((nz, nx), v, shape['tower_wave'], 0.45) + 0.6 * self.noise.band((nz, nx), v, shape['tower_wave'] * 0.42, 0.45)
		crown = self.noise.band((nz, nx), v, 430.0, 0.5)
		sampled = self._drape(torch.stack((foot, crown), 0), shape['warp_amp'])
		above = self.ys.clamp_min(0.0)
		threshold = shape['tower_thr'] + shape['tower_taper'] * (above / 300.0) - shape['tower_flare'] * torch.exp(-above / 22.0)
		wall = threshold - sampled[0]
		norm = FieldOps.grad_norm(wall, v)
		wall = wall / norm.clamp_min_(0.004)
		del norm
		top = shape['tower_base'] + shape['tower_rise'] * sampled[1] + 55.0 * (sampled[0] - shape['tower_thr'])
		cap = self.ys - top
		del sampled, top
		return -FieldOps.smooth_min(-wall, -cap, 8.0)

	def strands(self):
		cfg = WorldCfg
		v = cfg.VOXEL
		shape = self.shape
		dims = (cfg.NZ, cfg.GEN_NY, cfg.NX)
		f1 = self.noise.band(dims, v, shape['strand_wave'], 0.3, aniso=[1.0, 0.72, 1.0])
		f2 = self.noise.band(dims, v, shape['strand_wave'], 0.3, aniso=[1.0, 0.72, 1.0])
		rho = f1.square_().addcmul_(f2, f2).sqrt_()
		del f2
		norm = FieldOps.grad_norm(rho, v)
		presence = self.noise.band(dims, v, 430.0, 0.4)
		keep = torch.clamp((presence - shape['strand_keep']) / 0.5, 0.0, 1.0)
		del presence
		keep = keep * torch.clamp((cfg.CEILING - 45.0 - self.ys) / 55.0, 0.0, 1.0)
		if shape['strand_floor'] > 0.0:
			keep = keep * torch.clamp((self.ys - shape['strand_floor']) / 40.0, 0.0, 1.0)
		radius = shape['strand_r'] * (1.0 + 0.25 * self.warp_x) * keep - 0.15 * (1.0 - keep)
		return (rho - radius) / norm.clamp_min_(0.004)

	def islands(self):
		cfg = WorldCfg
		v = cfg.VOXEL
		shape = self.shape
		dims = (cfg.NZ, cfg.GEN_NY, cfg.NX)
		blobs = self.noise.band(dims, v, shape['island_wave'], 0.4, aniso=[1.0, shape['island_flat'], 1.0])
		threshold = shape['island_thr'] + ((self.ys - shape['island_alt']) / shape['island_band']) ** 2
		field = threshold - blobs
		del blobs
		norm = FieldOps.grad_norm(field, v)
		return field / norm.clamp_min_(0.004)

	def canyon(self):
		cfg = WorldCfg
		v = cfg.VOXEL
		shape = self.shape
		nz, nx = cfg.NZ, cfg.NX
		course = self.noise.band((nz, nx), v, shape['canyon_wave'], 0.35)
		reach = course.abs() / self._slope(course).clamp_min(1.0 / shape['canyon_wave'])
		width = shape['canyon_width'] * (1.0 + 0.35 * self.noise.band((nz, nx), v, 310.0, 0.5)).clamp(0.55, 1.8)
		climb = ((reach - width) / shape['canyon_bank']).clamp(0.0, 1.0)
		steps = shape['canyon_steps']
		stair = climb * steps
		tread = torch.floor(stair)
		riser = ((stair - tread - 0.5) / shape['canyon_riser']).clamp(-0.5, 0.5) + 0.5
		profile = (tread + riser) / steps
		mesa = shape['mesa_base'] + shape['mesa_rise'] * self.noise.band((nz, nx), v, 480.0, 0.5)
		height = shape['canyon_floor'] + (mesa - shape['canyon_floor']) * profile
		lean = 1.0 / torch.sqrt(1.0 + self._slope(height).clamp(max=3.0) ** 2)
		sampled = self._drape(torch.stack((height, lean), 0), shape['warp_amp'])
		return (self.ys - sampled[0]) * sampled[1]

	def _sites(self, cells, spread):
		cfg = WorldCfg
		v = cfg.VOXEL
		nz, nx = cfg.NZ, cfg.NX
		pitch = cfg.SIZE_X / cells
		jitter = (self.noise.uniform((cells, cells, 2)) - 0.5) * spread
		index = torch.arange(cells, device=self.device, dtype=torch.float32) + 0.5
		seed_x = ((index.view(1, cells) + jitter[..., 0]) * pitch).reshape(-1)
		seed_z = ((index.view(cells, 1) + jitter[..., 1]) * pitch).reshape(-1)
		px = (torch.arange(nx, device=self.device, dtype=torch.float32) + 0.5) * v
		pz = (torch.arange(nz, device=self.device, dtype=torch.float32) + 0.5) * v
		dx = px.view(nx, 1) - seed_x.view(1, -1)
		dx = (dx - cfg.SIZE_X * torch.round(dx / cfg.SIZE_X)).square_()
		nearest = torch.empty((nz, nx), device=self.device, dtype=torch.long)
		edge = torch.empty((nz, nx), device=self.device)
		reach = torch.empty((nz, nx), device=self.device)
		for start in range(0, nz, 32):
			dz = pz[start:start + 32].view(-1, 1) - seed_z.view(1, -1)
			dz = (dz - cfg.SIZE_Z * torch.round(dz / cfg.SIZE_Z)).square_()
			both = torch.topk(dz.unsqueeze(1) + dx.unsqueeze(0), 2, dim=2, largest=False)
			nearest[start:start + 32] = both.indices[..., 0]
			reach[start:start + 32] = both.values[..., 0].sqrt()
			edge[start:start + 32] = (both.values[..., 1].sqrt() - both.values[..., 0].sqrt()) * 0.5
		col = (seed_x / v).long().clamp_(0, nx - 1)
		row = (seed_z / v).long().clamp_(0, nz - 1)
		return nearest, edge, reach, row, col

	def _cells(self):
		nearest, edge, _, row, col = self._sites(int(self.shape['cells']), self.shape['cell_jitter'])
		return nearest, edge, row, col

	def _land(self):
		if self.land is None:
			cfg = WorldCfg
			self.land = self.noise.band((cfg.NZ, cfg.NX), cfg.VOXEL, self.shape['isle_wave'], 0.5)
		return self.land

	def _core(self, row, col):
		shape = self.shape
		return ((self._land()[row, col] - shape['isle_thr']) / shape['city_core']).clamp_(0.0, 1.0)

	def isles(self):
		shape = self.shape
		height = ((self._land() - shape['isle_thr']) * shape['isle_gain']).clamp_(max=shape['isle_top'])
		return self.ys - height.unsqueeze(1)

	def _towers(self, nearest, reach, tall, girth, decks, bulb=None):
		shape = self.shape
		height = tall[nearest].unsqueeze(1)
		radius = girth[nearest].unsqueeze(1)
		reach = reach.unsqueeze(1)
		rise = (self.ys / height.clamp_min(1.0)).clamp_(0.0, 1.0)
		profile = (1.0 - rise).pow_(shape['city_taper']).mul_(1.0 - shape['city_tip']).add_(shape['city_tip'])
		profile.add_(torch.exp(self.ys.clamp_min(0.0) / -shape['city_foot']), alpha=shape['city_flare'])
		body = -FieldOps.smooth_min(profile.mul_(radius).sub_(reach), height - self.ys, 4.0)
		del profile, rise
		for level, span, half in decks:
			deck = torch.maximum((self.ys - height * level).abs_().sub_(half), reach - radius * span)
			body = FieldOps.smooth_min(body, deck, 3.0)
			del deck
		if bulb is not None:
			dome = (reach * reach + (self.ys - shape['isle_top']) ** 2).sqrt_().sub_(bulb[nearest].unsqueeze(1))
			body = FieldOps.smooth_min(body, dome, 3.0)
		return body

	def citadel(self):
		shape = self.shape
		nearest, _, reach, row, col = self._sites(int(shape['city_cells']), shape['city_jitter'])
		core = self._core(row, col)
		luck = self.noise.uniform((row.shape[0], 4))
		built = (core > shape['city_shore']) & (luck[:, 0] < shape['city_fill'])
		domed = built & (luck[:, 3] < shape['city_domes'])
		low, high = shape['city_height']
		tall = low + (high - low) * core ** shape['city_peak'] * (0.35 + 0.65 * luck[:, 1])
		tall = torch.where(domed, tall * 0.45, tall)
		tall = torch.where(built, tall, torch.full_like(tall, -80.0))
		thin, wide = shape['city_girth']
		girth = thin + (wide - thin) * (0.5 * core + 0.5 * luck[:, 2])
		bulb = torch.where(domed, girth * shape['dome_scale'], torch.zeros_like(girth))
		return self._towers(nearest, reach, tall, girth, (), bulb)

	def spires(self):
		shape = self.shape
		nearest, _, reach, row, col = self._sites(int(shape['spire_cells']), shape['spire_jitter'])
		core = self._core(row, col)
		luck = self.noise.uniform((row.shape[0], 2))
		built = (core > shape['spire_shore']) & (luck[:, 0] < shape['spire_fill'])
		low, high = shape['spire_height']
		tall = torch.where(built, low + (high - low) * luck[:, 1], torch.full_like(core, -80.0))
		girth = torch.full_like(core, shape['spire_girth'])
		return self._towers(nearest, reach, tall, girth, shape['spire_decks'])

	def basalt(self):
		cfg = WorldCfg
		v = cfg.VOXEL
		shape = self.shape
		nz, nx = cfg.NZ, cfg.NX
		nearest, edge, row, col = self._cells()
		count = row.shape[0]
		step = shape['organ_step']
		lift = self.noise.band((nz, nx), v, shape['organ_wave'], 0.45)[row, col]
		tall = (lift - shape['organ_thr']) * shape['organ_gain'] + (self.noise.uniform((count,)) - 0.5) * shape['organ_jitter']
		tall = torch.round(tall / step) * step
		groove = shape['groove'] * (1.0 - edge / 1.6).clamp_(0.0, 1.0)
		ground = self.ys - (tall[nearest] - groove).unsqueeze(1)
		mask = self.noise.band((nz, nx), v, shape['hang_wave'], 0.45)[row, col]
		drop = (mask - shape['hang_thr']) * shape['hang_gain'] + (self.noise.uniform((count,)) - 0.5) * shape['organ_jitter']
		drop = torch.round(drop / step) * step
		bottom = torch.where(drop > 0.0, shape['hang_top'] - drop, torch.full_like(drop, 1.0e4))
		roof = shape['hang_top'] + torch.round((self.noise.uniform((count,)) * shape['organ_jitter']) / step) * step
		low = (bottom[nearest] + groove).unsqueeze(1) - self.ys
		high = self.ys - roof[nearest].unsqueeze(1)
		return torch.minimum(ground, torch.maximum(low, high))

	def lattice(self):
		cfg = WorldCfg
		v = cfg.VOXEL
		shape = self.shape
		nz, nx = cfg.NZ, cfg.NX
		k = math.tau * shape['lattice_n'] / cfg.SIZE_X
		amp = shape['lattice_warp']
		xs = ((torch.arange(nx, device=self.device, dtype=torch.float32) + 0.5) * v).view(1, 1, nx)
		zs = ((torch.arange(nz, device=self.device, dtype=torch.float32) + 0.5) * v).view(nz, 1, 1)
		phase = self.noise.uniform((3,)) * math.tau
		x = (xs + self.warp_x * amp) * k + phase[0]
		z = (zs + self.warp_z * amp) * k + phase[2]
		y = self.ys * (math.tau / shape['lattice_y']) + phase[1] + self.warp_x * (amp * 0.5 * k)
		field = torch.sin(x) * torch.cos(y)
		field.addcmul_(torch.sin(y), torch.cos(z))
		field.addcmul_(torch.sin(z), torch.cos(x))
		del x, y, z
		rise = ((self.ys - shape['lattice_alt']) / shape['lattice_band']).clamp_min(0.0) ** 2
		field = shape['lattice_thr'] + rise - field
		norm = FieldOps.grad_norm(field, v)
		return field / norm.clamp_min_(0.004)

	def tori(self, dist):
		cfg = WorldCfg
		v = cfg.VOXEL
		shape = self.shape
		nz, ny, nx = dist.shape
		for _ in range(int(shape['tori'])):
			u = self.noise.uniform((8,)).tolist()
			center = (u[0] * cfg.SIZE_X, shape['tori_alt'][0] + u[1] * (shape['tori_alt'][1] - shape['tori_alt'][0]), u[2] * cfg.SIZE_Z)
			major = shape['tori_major'][0] + u[3] * (shape['tori_major'][1] - shape['tori_major'][0])
			minor = shape['tori_minor'][0] + u[4] * (shape['tori_minor'][1] - shape['tori_minor'][0])
			turn = u[5] * math.tau
			tilt = (u[6] - 0.5) * 0.7
			axis = (math.cos(turn) * math.cos(tilt), math.sin(tilt), math.sin(turn) * math.cos(tilt))
			half = major + minor + 16.0
			ix = torch.arange(int((center[0] - half) / v), int((center[0] + half) / v) + 1, device=self.device)
			iz = torch.arange(int((center[2] - half) / v), int((center[2] + half) / v) + 1, device=self.device)
			iy = torch.arange(max(int((center[1] - half) / v) + cfg.SEA_ROW, 0), min(int((center[1] + half) / v) + cfg.SEA_ROW + 1, ny), device=self.device)
			px = ((ix.float() + 0.5) * v - center[0]).view(1, 1, -1)
			pz = ((iz.float() + 0.5) * v - center[2]).view(-1, 1, 1)
			py = ((iy.float() - cfg.SEA_ROW + 0.5) * v - center[1]).view(1, -1, 1)
			along = px * axis[0] + py * axis[1] + pz * axis[2]
			radial = torch.sqrt((px * px + py * py + pz * pz - along * along).clamp_min(0.0))
			ring = torch.sqrt((radial - major) ** 2 + along * along) - minor
			spot = (torch.remainder(iz, nz).view(-1, 1, 1), iy.view(1, -1, 1), torch.remainder(ix, nx).view(1, 1, -1))
			dist[spot] = FieldOps.smooth_min(dist[spot], ring, 6.0)
		return dist

	def fold(self, dist):
		period = int(self.shape['fold'])
		if period <= 0:
			return dist
		nz, ny, nx = dist.shape
		index = torch.arange(nx, device=self.device) % period
		index = torch.minimum(index, period - 1 - index)
		high = torch.maximum(index.view(-1, 1), index.view(1, -1))
		low = torch.minimum(index.view(-1, 1), index.view(1, -1))
		pick = (low * nx + high).view(-1)
		flat = dist.permute(1, 0, 2).reshape(ny, nz * nx)
		return flat[:, pick].view(ny, nz, nx).permute(1, 0, 2).contiguous()
