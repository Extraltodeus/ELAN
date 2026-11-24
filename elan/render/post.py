import math
import torch

from elan.config import RenderCfg, TunnelCfg
from elan.render.space import Resampler


class PostFX:
	def __init__(self, device, uniforms):
		self.device = device
		self.exposure = uniforms.view('exposure', 1, 3, 1, 1)
		self.damage_color = torch.tensor((0.05, 0.08, 1.0), device=device).view(1, 3, 1, 1)
		self.h = 0
		self.w = 0
		self.out_h = 0
		self.out_w = 0
		self.sizes = []
		self.base = None
		self.warp = None
		self.tunnel = None
		self.radius2 = None
		self.radius4 = None
		self.bend = None
		self.swirl = None
		self.vignette = None
		self.edge = None
		self.grain = []
		self.history = None
		self.out = None

	def allocate(self, h, w, out_h, out_w):
		self.h = h
		self.w = w
		self.out_h = out_h
		self.out_w = out_w
		self.sizes = []
		ch, cw = h, w
		for _ in range(RenderCfg.BLOOM_LEVELS):
			ch = max((ch + 1) // 2, 2)
			cw = max((cw + 1) // 2, 2)
			self.sizes.append((ch, cw))
		xs = (torch.arange(w, device=self.device, dtype=torch.float32) + 0.5) / w * 2.0 - 1.0
		ys = (torch.arange(h, device=self.device, dtype=torch.float32) + 0.5) / h * 2.0 - 1.0
		grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
		self.base = torch.stack((grid_x, grid_y), -1).unsqueeze(0).contiguous()
		radius2 = (grid_x ** 2 + (grid_y * h / w) ** 2).view(1, 1, h, w)
		self.warp = (self.base * radius2.view(1, h, w, 1)).contiguous()
		self.tunnel = (self.base * (radius2 * radius2).view(1, h, w, 1)).contiguous()
		self.radius2 = radius2.view(1, h, w, 1).contiguous()
		self.radius4 = (self.radius2 * self.radius2).contiguous()
		self.bend = (self.radius2 / (self.radius2 + 0.3)).contiguous()
		self.swirl = (torch.stack((grid_y * (-h / w), grid_x * (w / h)), -1).unsqueeze(0) * self.bend).contiguous()
		self.vignette = ((1.0 - 0.3 * radius2 ** 1.3).clamp(0.25, 1.0) * 255.0).contiguous()
		self.edge = (radius2 ** 1.5).contiguous()
		gen = torch.Generator(device=self.device)
		gen.manual_seed(11)
		self.grain = [(torch.rand((1, 1, h, w), generator=gen, device=self.device) - 0.5) * 2.2 + 0.5 for _ in range(4)]
		self.history = None
		self.out = torch.full((out_h, out_w, 4), 255, dtype=torch.uint8, device=self.device)

	def reset(self):
		self.history = None

	def _bloom(self, hdr):
		resize = Resampler.resize
		current = torch.clamp(hdr, max=28.0)
		levels = []
		for size in self.sizes:
			current = resize(current, size)
			levels.append(current)
		acc = levels[-1]
		for i in range(len(levels) - 2, -1, -1):
			acc = resize(acc, self.sizes[i]).add_(levels[i])
		return resize(acc, (self.h, self.w))

	def _grid(self, shift, pull):
		grid = torch.add(self.base, self.warp, alpha=-shift)
		if pull > 0.0:
			grid.add_(self.tunnel, alpha=-pull)
		return grid

	def _bent(self, shift, pull, burst, twist):
		factor = torch.mul(self.radius2, -shift).add_(self.radius4, alpha=-pull).add_(self.bend, alpha=-burst).add_(1.0).clamp_(min=0.12)
		return torch.add(self.base, self.swirl, alpha=twist).mul_(factor)

	def _rush(self, image, amount, drive, surge):
		sampler = torch.grid_sampler
		shift = 0.05 * amount + 0.07 * drive
		pull = 0.1 * drive
		reach = 1.6 + 1.6 * drive
		if surge > 0.01:
			burst = TunnelCfg.SURGE_BURST * surge
			twist = TunnelCfg.SURGE_TWIST * surge
			blue = sampler(image[:, 0:1], self._bent(shift * 0.35, pull * 0.5, burst * 0.55, twist * 0.8), 0, 1, False)
			green = sampler(image[:, 1:2], self._bent(shift * 0.7, pull * 0.75, burst * 0.78, twist * 0.9), 0, 1, False)
			red = sampler(image[:, 2:3], self._bent(shift, pull, burst, twist), 0, 1, False)
			smear = sampler(image, self._bent(shift * reach, pull * reach, burst * 1.6, twist * 1.4), 0, 1, False)
			return torch.cat((blue, green, red), 1).lerp_(smear, min(0.4 + 0.18 * drive + 0.2 * surge, 0.75))
		blue = sampler(image[:, 0:1], self._grid(shift * 0.35, pull * 0.5), 0, 1, False)
		green = sampler(image[:, 1:2], self._grid(shift * 0.7, pull * 0.75), 0, 1, False)
		red = sampler(image[:, 2:3], self._grid(shift, pull), 0, 1, False)
		smear = sampler(image, self._grid(shift * reach, pull * reach), 0, 1, False)
		return torch.cat((blue, green, red), 1).lerp_(smear, 0.4 + 0.18 * drive)

	def process(self, hdr, view, frame, overlay=None):
		if self.history is not None and view.persistence > 0.0:
			hdr = torch.lerp(hdr, self.history, view.persistence)
		self.history = torch.clamp(hdr, max=5.0)
		image = torch.add(hdr, self._bloom(hdr), alpha=RenderCfg.BLOOM_STRENGTH / len(self.sizes) * 3.0)
		if view.rush > 0.02 or view.surge > 0.01:
			image = self._rush(image, view.rush, view.warp, view.surge)
		image.mul_(self.exposure).neg_().exp_()
		image = torch.rsub(image, 1.0).pow_(0.4545)
		image.mul_(self.vignette)
		if view.damage > 0.01:
			image.addcmul_(self.edge, self.damage_color, value=view.damage * 255.0)
		image.add_(self.grain[frame & 3])
		if self.out_h != self.h or self.out_w != self.w:
			image = Resampler.sharp(image, (self.out_h, self.out_w))
		if overlay is not None:
			glass, mini, rect = overlay
			region = image[:, :, rect[0]:rect[1], rect[2]:rect[3]]
			region.mul_(glass)
			region.add_(mini)
		image.clamp_(0.0, 255.0)
		self.out[..., :3] = image[0].permute(1, 2, 0)
		return self.out
