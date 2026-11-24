import torch
import torch.nn.functional as F

from elan.config import RenderCfg


class ConeLevel:
	def __init__(self, device, h, w, step, offset, steps, cone):
		self.step = step
		self.offset = offset
		self.steps = steps
		self.cone = cone
		self.h = len(range(offset, h, step))
		self.w = len(range(offset, w, step))
		self.shape = (1, 1, self.h, self.w, 1)
		self.pos = torch.empty((1, 1, self.h, self.w, 3), device=device)


class RayMarcher:
	def __init__(self, device, space):
		self.device = device
		self.unscale = (1.0 / space.scale).contiguous()
		self.pos = None
		self.dpos = None
		self.shape = None
		self.levels = []
		self.h = 0
		self.w = 0
		self.fast_nearest = None

	def allocate(self, h, w):
		self.h = h
		self.w = w
		self.shape = (1, 1, h, w, 1)
		self.pos = torch.empty((1, 1, h, w, 3), device=self.device)
		self.dpos = torch.empty((1, 1, h, w, 3), device=self.device)
		self.levels = [
			ConeLevel(self.device, h, w, 4, 2, RenderCfg.CONE_STEPS[0], 3.7),
			ConeLevel(self.device, h, w, 2, 1, RenderCfg.CONE_STEPS[1], 0.0),
		]

	def _expand(self, t, shape_from, size):
		flat = t.view(1, 1, shape_from[2], shape_from[3])
		if self.fast_nearest is None:
			try:
				torch._C._nn.upsample_nearest2d(flat, size, None)
				self.fast_nearest = True
			except Exception:
				self.fast_nearest = False
		if self.fast_nearest:
			return torch._C._nn.upsample_nearest2d(flat, size, None)
		return F.interpolate(flat, size=size, mode='nearest')

	def march(self, sdf, relief, origin, rays, detail_offset, detail_scale, pixel):
		sampler = torch.grid_sampler
		addcmul = torch.addcmul
		relax = RenderCfg.RELAX_COARSE
		cone = self.levels[0]
		low = rays[:, :, cone.offset::cone.step, cone.offset::cone.step].contiguous()
		t = torch.full(cone.shape, RenderCfg.NEAR, device=self.device)
		shrink = 1.0 / (1.0 + cone.cone * pixel)
		scale = 1.0
		pos = cone.pos
		for _ in range(cone.steps):
			addcmul(origin, low, t, value=scale, out=pos)
			t.add_(sampler(sdf, pos, 0, 2, False).view(cone.shape), alpha=relax / scale)
			scale *= shrink
		t.mul_(scale)

		half = self.levels[1]
		low = rays[:, :, half.offset::half.step, half.offset::half.step].contiguous()
		t = self._expand(t, cone.shape, (half.h, half.w)).view(half.shape)
		pos = addcmul(origin, low, t)
		for _ in range(half.steps):
			pos.addcmul_(low, sampler(sdf, pos, 0, 2, False).view(half.shape), value=relax)
		t = torch.linalg.vector_norm(torch.sub(pos, origin).mul_(self.unscale), dim=-1, keepdim=True)
		own = t
		t = F.max_pool2d(own.view(1, 1, half.h, half.w).neg(), 3, 1, 1).neg_()

		shape = self.shape
		pos = self.pos
		dpos = self.dpos
		t = self._expand(t, half.shape, (self.h, self.w)).view(shape)
		t.mul_(1.0 - 3.0 * pixel)
		addcmul(origin, rays, t, out=pos)
		for _ in range(RenderCfg.COARSE_STEPS):
			pos.addcmul_(rays, sampler(sdf, pos, 0, 2, False).view(shape), value=relax)
		stalled = sampler(sdf, pos, 0, 2, False).view(shape) > t.mul_(2.5 * pixel).add_(0.25)
		far = self._expand(own, half.shape, (self.h, self.w)).view(shape)
		far.mul_(1.0 - 3.0 * pixel)
		pos = torch.where(stalled, addcmul(origin, rays, far), pos)
		relax = RenderCfg.RELAX_DETAIL
		for _ in range(RenderCfg.DETAIL_STEPS):
			addcmul(detail_offset, pos, detail_scale, out=dpos)
			step = torch.add(sampler(sdf, pos, 0, 2, False), sampler(relief, dpos, 0, 2, False))
			pos.addcmul_(rays, step.view(shape), value=relax)
		t = torch.linalg.vector_norm(torch.sub(pos, origin).mul_(self.unscale), dim=-1, keepdim=True)
		return t, pos
