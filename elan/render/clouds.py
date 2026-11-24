import torch
import torch.nn.functional as F

from elan.config import RenderCfg, WorldCfg
from elan.render.space import Resampler


class CloudLayer:
	PATTERN = (0.0, 0.5, 0.75, 0.25)
	SPREAD = 420.0
	LIFT = 8.0

	def __init__(self, device):
		self.device = device
		self.steps = RenderCfg.CLOUD_STEPS
		self.div = RenderCfg.CLOUD_DIV
		self.range = RenderCfg.CLOUD_RANGE
		u = torch.arange(self.steps + 1, device=device, dtype=torch.float32) / self.steps
		self.even = u.view(1, 1, -1).contiguous()
		self.bias = (u * u - u).view(1, 1, -1).contiguous()
		self.one = torch.ones(1, device=device)
		self.running = torch.triu(torch.ones((self.steps, self.steps), device=device)).contiguous()
		self.period = torch.tensor((1.0, 1.0e6, 1.0), device=device)
		self.deck = None
		self.volume = None
		self.levels = None
		self.scale = None
		self.offset = None
		self.pitch = None
		self.paint = None
		self.sun_dir = None
		self.dither = []
		self.ones = None
		self.h = 0
		self.w = 0
		self.hs = 0
		self.ws = 0

	def allocate(self, h, w):
		self.h = h
		self.w = w
		self.hs = h // self.div
		self.ws = w // self.div
		rows = torch.arange(self.hs, device=self.device).view(-1, 1) % 2
		cols = torch.arange(self.ws, device=self.device).view(1, -1) % 2
		cell = rows * 2 + cols
		pattern = torch.tensor(self.PATTERN, device=self.device)
		self.dither = [pattern[(cell + shift) % 4].unsqueeze(-1).contiguous() for shift in range(4)]
		self.ones = torch.ones((self.hs, self.ws, 1), device=self.device)

	def _bgr(self, rgb, scale=1.0):
		return (rgb[2] * scale, rgb[1] * scale, rgb[0] * scale)

	def set_world(self, world):
		self.deck = world.clouds
		if self.deck is None:
			self.volume = None
			return
		deck = self.deck
		thick = deck.top - deck.base
		nx, nz = deck.nx, deck.nz
		self.volume = torch.cat((deck.volume[:, 0:1] * -deck.density, deck.volume[:, 1:2]), 1).contiguous()
		self.levels = torch.tensor((deck.base, deck.top), device=self.device)
		self.pitch = torch.tensor((1.0 / WorldCfg.SIZE_X, 1.0 / thick, 1.0 / WorldCfg.SIZE_Z), device=self.device)
		self.scale = torch.tensor((nx * 2.0 / (nx + 2), 2.0, nz * 2.0 / (nz + 2)), device=self.device)
		self.offset = torch.tensor((2.0 / (nx + 2) - 1.0, -1.0 - 2.0 * self.LIFT, 2.0 / (nz + 2) - 1.0), device=self.device)
		columns = (self._bgr(deck.sun, -1.0), self._bgr(deck.sky), self._bgr(deck.haze, RenderCfg.CLOUD_HAZE), self._bgr(deck.haze_sun, RenderCfg.CLOUD_HAZE))
		self.paint = torch.tensor(columns, device=self.device, dtype=torch.float32).t().contiguous()
		self.sun_dir = torch.tensor(deck.sun_dir, device=self.device, dtype=torch.float32)

	def _march(self, view, rays, limit, frame):
		deck = self.deck
		height = view.pos[1]
		crossing = torch.sub(self.levels, height) * torch.reciprocal(rays[..., 1:2] + 1.0e-5)
		near = crossing.amin(-1).clamp_(min=0.0, max=self.range)
		span = torch.minimum(crossing.amax(-1), limit).sub_(near).clamp_(min=0.0)
		curve = (span * (1.0 / self.SPREAD)).clamp_(max=1.0).unsqueeze(-1)
		edges = torch.addcmul(self.even, self.bias, curve).mul_(span.unsqueeze(-1)).add_(near.unsqueeze(-1))
		width = edges[..., 1:] - edges[..., :-1]
		t = torch.addcmul(edges[..., :-1], width, self.dither[frame & 3])
		drift = view.time
		start = torch.tensor(((view.pos[0] + deck.wind[0] * drift) / WorldCfg.SIZE_X % 1.0, (height - deck.base) / (deck.top - deck.base) + self.LIFT, (view.pos[2] + deck.wind[1] * drift) / WorldCfg.SIZE_Z % 1.0), device=self.device)
		grid = torch.addcmul(start, t.unsqueeze(-1), (rays * self.pitch).unsqueeze(-2))
		grid = torch.addcmul(self.offset, torch.remainder(grid, self.period), self.scale)
		sample = torch.grid_sampler(self.volume, grid.unsqueeze(0), 0, 0, False)[0]
		after = torch.matmul(width.mul_(sample[0]), self.running).exp_()
		change = torch.diff(after, dim=-1, prepend=self.ones)
		sums = torch.stack((sample[1], grid[..., 1], t), -1).mul_(change.unsqueeze(-1)).sum(-2)
		return near, after[..., -1], sums

	def apply(self, color, world, view, rays_world, blocker, frame, haze):
		h, w, div = self.h, self.w, self.div
		hs, ws = self.hs, self.ws
		rays = rays_world.view(h, w, 3)[::div, ::div]
		limit = F.max_pool2d(blocker.view(1, 1, h, w), div).view(hs, ws).clamp_(max=self.range)
		near, through, sums = self._march(view, rays, limit, frame)
		veil = through - (1.0 + 1.0e-4)
		depth = sums[..., 2] / veil
		clear = torch.mul(depth, -haze).exp_()
		facing = torch.matmul(rays, self.sun_dir).clamp_(min=0.0).square_()
		glare = facing.pow(4).mul_(RenderCfg.CLOUD_GLARE).add_(1.0)
		ambient = torch.mul(veil, -0.625).add_(sums[..., 1], alpha=-0.375).clamp_(min=0.0)
		mist = (clear - 1.0).mul_(veil)
		parts = torch.stack((sums[..., 0] * glare * clear, ambient * clear, mist, mist * facing), 0)
		paint = torch.matmul(self.paint, parts.view(4, hs * ws)).view(3, hs, ws)
		slope = torch.sub(near, depth).clamp_(max=-1.0).reciprocal_()
		layer = torch.cat((paint, veil.unsqueeze(0), slope.unsqueeze(0), (near * slope).unsqueeze(0)), 0).unsqueeze(0)
		layer = F.avg_pool2d(F.pad(layer, (0, 1, 0, 1), mode='replicate'), 2, 1)
		layer = Resampler.resize(layer, (h, w)).view(6, h * w)
		front = torch.addcmul(layer[5], blocker, layer[4], value=-1.0).clamp_(0.0, 1.0)
		color.mul_(torch.addcmul(self.one, layer[3], front))
		color.addcmul_(layer[:3], front)
		return color
