import torch
import torch.nn.functional as F

from elan.config import WorldCfg


class CloudDeck:
	def __init__(self):
		self.volume = None
		self.base = 0.0
		self.top = 0.0
		self.nx = 0
		self.nz = 0
		self.density = 0.0
		self.wind = (0.0, 0.0)
		self.sun_dir = (0.0, 1.0, 0.0)
		self.sun = (1.0, 1.0, 1.0)
		self.sky = (1.0, 1.0, 1.0)
		self.haze = (1.0, 1.0, 1.0)
		self.haze_sun = (0.0, 0.0, 0.0)
		self.cover = 0.0

	def to(self, device):
		self.volume = self.volume.to(device)
		return self


class CloudBaker:
	NX = 160
	NY = 24
	NZ = 160
	SUN_STEPS = 18
	SUN_STEP = 13.0
	ABSORB = 0.013

	def __init__(self, device):
		self.device = device

	def _grid(self, x, y, z, base, top):
		nx, nz = self.NX, self.NZ
		gx = (torch.remainder(x / WorldCfg.SIZE_X, 1.0) * nx + 1.0) * (2.0 / (nx + 2)) - 1.0
		gz = (torch.remainder(z / WorldCfg.SIZE_Z, 1.0) * nz + 1.0) * (2.0 / (nz + 2)) - 1.0
		gy = (y - base) * (2.0 / (top - base)) - 1.0
		return torch.stack((gx.expand_as(y), gy, gz.expand_as(y)), -1).unsqueeze(0)

	def _pad(self, field):
		return F.pad(field.view(1, 1, self.NZ, self.NY, self.NX), (1, 1, 0, 0, 1, 1), mode='circular')

	def bake(self, noise, palette, sun_dir):
		spec = palette['style']['cloud_deck']
		if not spec:
			return None
		nz, ny, nx = self.NZ, self.NY, self.NX
		base, top = spec['band']
		cell = WorldCfg.SIZE_X / nx
		sheet = noise.band((nz, nx), cell, spec['wave'], 0.5)
		puffs = noise.fractal((nz, ny, nx), cell, spec['detail'][0], spec['detail'][1], 1.6)
		rise = ((torch.arange(ny, device=self.device, dtype=torch.float32) + 0.5) / ny).view(1, ny, 1)
		floor = (0.18 - rise).clamp_min(0.0) * (2.2 / 0.18)
		crown = (rise - 0.3).clamp_min(0.0) ** 1.4 * spec['crown']
		field = sheet.unsqueeze(1) * 0.9 + puffs * 0.55 - spec['thr'] - floor - crown
		density = (field * spec['gain']).clamp_(0.0, 1.0)
		del field, puffs
		padded = self._pad(density)
		xs = ((torch.arange(nx, device=self.device, dtype=torch.float32) + 0.5) * cell).view(1, 1, nx)
		zs = ((torch.arange(nz, device=self.device, dtype=torch.float32) + 0.5) * cell).view(nz, 1, 1)
		ys = (base + rise * (top - base)).expand(nz, ny, nx)
		depth = torch.zeros((nz, ny, nx), device=self.device)
		for k in range(1, self.SUN_STEPS + 1):
			reach = k * self.SUN_STEP
			grid = self._grid(xs + sun_dir[0] * reach, ys + sun_dir[1] * reach, zs + sun_dir[2] * reach, base, top)
			depth += F.grid_sample(padded, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0, 0]
		light = torch.exp(depth * (-self.ABSORB * self.SUN_STEP * spec['density'] / 0.045))
		deck = CloudDeck()
		deck.volume = torch.cat((padded, self._pad(light)), 1).contiguous()
		deck.base = float(base)
		deck.top = float(top)
		deck.nx = nx
		deck.nz = nz
		deck.density = float(spec['density'])
		deck.wind = tuple(spec['wind'])
		deck.sun_dir = tuple(float(v) for v in sun_dir)
		power = spec['sun']
		deck.sun = tuple(c * power for c in palette['sun_color'])
		deck.sky = tuple(c * spec['sky'] for c in palette['amb_sky'])
		deck.haze = tuple(0.45 * z + 0.5 * h for z, h in zip(palette['zenith'], palette['horizon']))
		deck.haze_sun = tuple(c * 0.75 for c in palette['glow'])
		deck.cover = float((density.amax(1) > 0.25).float().mean())
		return deck
