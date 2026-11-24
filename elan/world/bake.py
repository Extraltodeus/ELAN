import time
import torch

from elan.config import WorldCfg
from elan.world.noise import SpectralNoise
from elan.world.palette import PaletteBook
from elan.world.terrain import TerrainBuilder
from elan.world.fieldops import TileSampler
from elan.world.lighting import LightBaker
from elan.world.sky import SkyBaker
from elan.world.detail import DetailBaker
from elan.world.paths import PathPlanner
from elan.world.tunnel import TunnelBaker

STAGES = (
	'raising spires', 'weaving arches', 'lifting islands', 'eroding rock', 'measuring distances',
	'casting light', 'charting ribbons', 'painting sky', 'carving detail', 'uploading',
)


class World:
	def __init__(self):
		self.seed = 0
		self.sector = 0
		self.kind = 'sector'
		self.pool = 0
		self.path = None
		self.sight = None
		self.coords_tile = None
		self.coords = None
		self.palette = None
		self.style = None
		self.sun_dir = None
		self.solid_fraction = 0.0
		self.field = None
		self.ribbons = None
		self.rings_pos = None
		self.rings_dir = None
		self.spawn_pos = None
		self.spawn_dir = None
		self.gate_pos = None
		self.gate_dir = None
		self.tile = None
		self.aux_tile = None
		self.sdf = None
		self.aux = None
		self.detail = None
		self.sky = None
		self.fog = None
		self.ripple = None
		self.star_dirs = None
		self.star_colors = None
		self.device = None

	def _move(self, tensor, device, breathe):
		if tensor.device == device:
			return tensor
		out = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
		rows = tensor.shape[0]
		row_bytes = max(tensor[0].numel() * tensor.element_size(), 1)
		step = max(int(12.0e6 // row_bytes), 1)
		for start in range(0, rows, step):
			out[start:start + step].copy_(tensor[start:start + step])
			breathe()
		return out

	def upload(self, device, breathe):
		device = torch.device(device)
		with torch.no_grad():
			tile = self._move(self.tile, device, breathe)
			nz, ny, nx = tile.shape
			sdf = torch.empty((1, 1, 2 * nz, ny, 2 * nx), dtype=torch.float32, device=device)
			step = 32
			for start in range(0, nz, step):
				slab = tile[start:start + step]
				for offset in (0, nz):
					sdf[0, 0, offset + start:offset + start + step, :, :nx] = slab
					sdf[0, 0, offset + start:offset + start + step, :, nx:] = slab
				breathe()
			aux_tile = self._move(self.aux_tile, device, breathe)
			self.sdf = sdf
			self.aux = aux_tile.repeat(1, 2, 1, 2).unsqueeze(0).contiguous()
			if self.coords_tile is not None:
				self.coords = self._move(self.coords_tile, device, breathe).repeat(1, 2, 1, 2).unsqueeze(0).contiguous()
				self.coords_tile = None
			self.detail = self.detail.to(device)
			self.sky = self.sky.to(device)
			self.fog = self.fog.to(device)
			self.ripple = self.ripple.to(device)
			self.star_dirs = self.star_dirs.to(device)
			self.star_colors = self.star_colors.to(device)
			self.tile = None
			self.aux_tile = None
			self.device = device
		return self


class WorldBaker:
	def __init__(self, device, throttle=0.0, owner=None):
		self.device = torch.device(device)
		self.throttle = throttle
		self.owner = owner
		self.book = PaletteBook()

	def _breathe(self):
		pause = self.owner.throttle if self.owner is not None else self.throttle
		if pause > 0.0:
			time.sleep(pause)

	def bake(self, seed, sector, progress=None):
		def tick(label):
			self._breathe()
			if progress is not None:
				progress(STAGES.index(label) / len(STAGES), label)

		world = World()
		world.seed = seed
		world.sector = sector
		with torch.no_grad():
			palette = self.book.pick(sector, seed)
			sun_dir = self.book.sun_direction(palette)
			noise = SpectralNoise(self.device, seed * 1000003 + sector * 7919 + 17)
			dist, solid_fraction = TerrainBuilder(self.device).build(noise, palette['shape'], tick, self._breathe)
			tick('casting light')
			sampler = TileSampler(dist)
			aux = LightBaker(self.device).bake(sampler, sun_dir, self._breathe)
			tick('charting ribbons')
			plan = PathPlanner(self.device).plan(sampler, noise, palette['style'])
			del sampler
			tick('painting sky')
			sky, fog, star_dirs, star_colors = SkyBaker(self.device).bake(noise, palette, sun_dir)
			tick('carving detail')
			details = DetailBaker(self.device)
			relief = palette['style']['relief']
			world.detail = details.bake_rock(noise, relief)
			world.ripple = details.bake_ripple(noise)
			world.field = dist.half().cpu().numpy()
			tile = dist.sub_(WorldCfg.DETAIL_AMP * 0.5 * relief)
			boost = torch.clamp(tile / 1.5, 0.0, 1.0) * torch.clamp((8.0 - tile) / 4.0, 0.0, 1.0)
			world.tile = tile.addcmul_(tile, boost, value=0.5)
			world.aux_tile = aux
			world.sky = sky
			world.fog = fog
			world.star_dirs = star_dirs
			world.star_colors = star_colors
			world.palette = palette
			world.style = palette['style']
			world.sun_dir = sun_dir
			world.solid_fraction = solid_fraction
			world.ribbons = plan['ribbons']
			world.rings_pos = plan['rings_pos']
			world.rings_dir = plan['rings_dir']
			world.spawn_pos = plan['spawn_pos']
			world.spawn_dir = plan['spawn_dir']
			world.gate_pos = plan['gate_pos']
			world.gate_dir = plan['gate_dir']
			world.pool = plan['pool']
			world.sight = plan['sight']
			tick('uploading')
		return world

	def bake_tunnel(self, seed, sector, progress=None):
		world = World()
		world.seed = seed
		world.sector = sector
		if progress is not None:
			progress(0.1, 'folding space')
		with torch.no_grad():
			TunnelBaker(self.device).bake(world, self.book.tunnel(sector, seed), seed, sector)
		if progress is not None:
			progress(0.9, 'uploading')
		return world

	def bake_and_upload(self, kind, seed, sector, render_device, progress=None):
		world = self.bake_tunnel(seed, sector, progress) if kind == 'tunnel' else self.bake(seed, sector, progress)
		world.upload(render_device, self._breathe)
		if progress is not None:
			progress(1.0, 'ready')
		return world
