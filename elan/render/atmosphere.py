import math
import torch

from elan.config import RenderCfg, WorldCfg
from elan.render.space import Resampler


class Atmosphere:
	def __init__(self, device, uniforms, space):
		self.device = device
		self.space = space
		self.uv_scale = torch.tensor((1.0 / math.pi, 2.0 / math.pi), device=device)
		self.ripple_amp = torch.tensor((0.004, 0.024), device=device)
		self.ripple_offset = uniforms.view('ripple')
		self.samples = RenderCfg.SHAFT_SAMPLES
		start = 3.0
		self.ratio = (RenderCfg.SHAFT_RANGE / start) ** (1.0 / (self.samples - 1))
		self.distances = start * self.ratio ** torch.arange(self.samples, device=device, dtype=torch.float32)
		self.steps = self.distances.view(1, self.samples, 1, 1, 1).contiguous()
		self.sky_floor = torch.tensor(0.58, device=device)
		self.murk_floor = torch.tensor(0.3, device=device)
		self.water_sigma = 0.012
		self.water_color = None
		self.sigma = RenderCfg.FOG_DENSITY
		self.core = uniforms.view('flow_core', 3, 1)
		self.fog_ambient = 0.42
		self.fog_sun = 0.58
		self.weights = None
		self.sea_color = None
		self.sunvis = None
		self.dither = []
		self.h = 0
		self.w = 0
		self.n = 0
		self.hs = 0
		self.ws = 0

	def set_world(self, world):
		palette = world.palette
		self.sigma = RenderCfg.FOG_DENSITY * palette['fog_density']
		spacing = self.distances * (self.ratio - 1.0)
		weights = torch.exp(-self.sigma * self.distances) * spacing
		self.weights = (weights / weights.sum()).view(1, self.samples).contiguous()
		sea = palette['sea']
		self.sea_color = torch.tensor((sea[2], sea[1], sea[0]), device=self.device, dtype=torch.float32).view(3, 1)
		self.water_color = (self.sea_color * 7.0 + torch.tensor((0.05, 0.035, 0.012), device=self.device).view(3, 1)).contiguous()
		self.sunvis = world.aux[:, 1:2]

	def allocate(self, h, w):
		div = RenderCfg.SHAFT_DIV
		self.h = h
		self.w = w
		self.n = h * w
		self.hs = len(range(0, h, div))
		self.ws = len(range(0, w, div))
		gen = torch.Generator(device=self.device)
		gen.manual_seed(7)
		self.dither = [self.ratio ** torch.rand((1, 1, self.hs, self.ws, 1), generator=gen, device=self.device) for _ in range(4)]

	def shafts(self, origin, rays_n, t, frame):
		div = RenderCfg.SHAFT_DIV
		rays_low = rays_n[:, :, ::div, ::div]
		t_low = t[:, :, ::div, ::div]
		steps = torch.mul(self.steps, self.dither[frame & 3])
		grid = torch.addcmul(origin, rays_low, steps)
		lit = torch.grid_sampler(self.sunvis, grid, 0, 2, False)
		lit.mul_(torch.lt(steps, t_low).view(lit.shape))
		total = torch.matmul(self.weights, lit.view(self.samples, self.hs * self.ws))
		return Resampler.resize(total.view(1, 1, self.hs, self.ws), (self.h, self.w)).view(self.n)

	def tunnel(self, color, t):
		trans = torch.mul(t.view(self.n), -self.sigma).exp_()
		color.mul_(trans)
		color.addcmul_(self.core, trans.neg_().add_(1.0))
		return color

	def compose(self, world, color, cover, sea, t, rays_world, shaft, altitude, submerged):
		sampler = torch.grid_sampler
		n = self.n
		rays = rays_world.view(n, 3)
		ry = rays[:, 1]
		azimuth = torch.atan2(rays[:, 0], rays[:, 2])
		elevation = torch.clamp(ry, -0.99999, 0.99999).asin_()
		if not submerged:
			sea = torch.where(ry < 0.0, torch.maximum(sea, torch.rsub(cover, 1.0)), sea)
		uv = torch.stack((azimuth, elevation), 1).mul_(self.uv_scale)
		if submerged:
			inv = torch.clamp(ry, min=1.0e-3).reciprocal_()
		else:
			inv = torch.clamp(ry, max=-1.0e-3).reciprocal_()
		grid = torch.addcmul(self.ripple_offset, rays[:, 0::2], inv.unsqueeze(1), value=-altitude / WorldCfg.RIPPLE_HALF)
		ripple = sampler(world.ripple, grid.view(1, 1, n, 2), 0, 2, False)
		if submerged:
			wobble = torch.mul(ry, 7.0).clamp_(0.0, 1.0)
		else:
			wobble = torch.mul(ry, -7.0).clamp_(0.0, 1.0).mul_(sea)
		uv.addcmul_(ripple.view(2, n).t(), wobble.unsqueeze(1) * self.ripple_amp)
		lookup = uv.view(1, 1, n, 2)
		sky = sampler(world.sky, lookup, 0, 1, False).view(3, n)
		fog = sampler(world.fog, lookup, 0, 1, False).view(3, n)
		trans = torch.mul(t.view(n), -self.sigma).exp_()
		haze = torch.mul(shaft, self.fog_sun).add_(trans, alpha=-self.fog_ambient).add_(self.fog_ambient)
		color.mul_(trans)
		color.addcmul_(fog, haze)
		sky.mul_(torch.add(self.sky_floor, shaft, alpha=0.42))
		out = torch.lerp(sky, color, cover)
		if submerged:
			reach = torch.minimum(t.view(n), torch.mul(inv, -altitude))
			clear = reach.mul_(-self.water_sigma).exp_()
			murk = torch.add(self.murk_floor, shaft, alpha=0.7).mul_(torch.rsub(clear, 1.0))
			out.mul_(clear)
			out.addcmul_(self.water_color, murk, value=math.exp(altitude / 110.0))
			return out
		depth = torch.mul(ry, -3.5).clamp_(0.0, 0.9).mul_(sea)
		out.lerp_(self.sea_color, depth)
		return out
