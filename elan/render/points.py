import torch
import torch.nn.functional as F

from elan.config import RenderCfg


class PointSplatter:
	REFERENCE = 55.0 ** 2
	SOFTNESS = 60.0
	NEAR_FULL = 25.0
	NEAR_NONE = 120.0
	FADE_START = 340.0
	STAMP_RADIUS = 4

	def __init__(self, device, uniforms, space):
		self.device = device
		self.space = space
		self.cam = uniforms.view('cam')
		self.viewmat = uniforms.view('viewmat', 3, 3)
		self.focal = uniforms.view('focal')
		self.center = uniforms.view('center')
		near_span = self.NEAR_NONE ** 2 - self.NEAR_FULL ** 2
		fade_span = RenderCfg.FAR ** 2 - self.FADE_START ** 2
		self.slope = torch.tensor((-1.0 / near_span, -1.0 / fade_span), device=device)
		self.shift = torch.tensor((self.NEAR_NONE ** 2 / near_span, RenderCfg.FAR ** 2 / fade_span), device=device)
		self.buffer = None
		self.marks = None
		self.stamp = None
		self.stamp_base = None
		self.stamp_near = None
		self.limit = None
		self.corners = None
		self.h = 0
		self.w = 0
		self.n = 0

	def allocate(self, h, w):
		self.h = h
		self.w = w
		self.n = h * w
		self.buffer = torch.zeros((self.n, 3), device=self.device)
		self.marks = torch.zeros((4, self.n), device=self.device)
		offsets = []
		base = []
		near = []
		reach = self.STAMP_RADIUS
		for dy in range(-reach, reach + 1):
			for dx in range(-reach, reach + 1):
				r2 = dx * dx + dy * dy
				if r2 > reach * reach:
					continue
				offsets.append(dy * w + dx)
				base.append(1.0 if r2 <= 1 else (0.8 if r2 == 2 else (0.45 if r2 == 4 else 0.0)))
				near.append(0.0 if r2 <= 1 else (0.2 if r2 == 2 else (0.55 if r2 == 4 else (1.0 if r2 <= 10 else 0.6))))
		self.stamp = torch.tensor(offsets, device=self.device, dtype=torch.long)
		self.stamp_base = torch.tensor(base, device=self.device)
		self.stamp_near = torch.tensor(near, device=self.device)
		self.limit = torch.tensor((w - 1.0, h - 1.0), device=self.device)
		self.corners = torch.tensor((0, 1, w, w + 1), device=self.device, dtype=torch.long)

	def _motes(self, motes, index, dist2, visible, gain, tint):
		a, b = motes.range
		reach = dist2[a:b]
		level = torch.addcmul(self.shift, reach.unsqueeze(1), self.slope).clamp_(0.0, 1.0)
		near = level[:, 0]
		step = torch.log(reach).mul_(motes.scale).add_(motes.offset).clamp_(0.0, motes.last).long()
		tint[a:b] = motes.glow[step] * (motes.white * motes.amp * gain[a:b]).unsqueeze(1)
		power = torch.addcmul(motes.far, motes.near, near).mul_(motes.amp).mul_(level[:, 1]).mul_(visible[a:b])
		weights = torch.addcmul(self.stamp_base, near.unsqueeze(1), self.stamp_near)
		stamped = (motes.mark[:, step] * power).unsqueeze(2) * weights
		spots = (index[a:b].unsqueeze(1) + self.stamp).clamp_(0, self.n - 1)
		self.marks.zero_()
		self.marks.index_add_(1, spots.view(-1), stamped.view(4, -1))

	def splat(self, points, colors, far_points, far_colors, depth, motes):
		near = RenderCfg.NEAR
		rel = points - self.cam
		rel.addcmul_(torch.round(rel * self.space.inv_period), self.space.period, value=-1.0)
		rel = torch.cat((rel, far_points), 0)
		colors = torch.cat((colors, far_colors), 0)
		cam = torch.matmul(rel, self.viewmat)
		z = cam[:, 2]
		inv_z = torch.clamp(z, min=near).reciprocal_()
		pix = torch.addcmul(self.center, cam[:, :2] * inv_z.unsqueeze(1), self.focal)
		base = torch.floor(pix)
		frac = pix - base
		inside = ((base >= 0.0) & (base < self.limit)).all(1) & (z > near)
		index = (base[:, 0] + base[:, 1] * self.w).long().clamp_(0, self.n - self.w - 2)
		dist2 = (rel * rel).sum(1)
		visible = inside & (torch.sqrt(dist2) < depth[index] + 0.5)
		gain = torch.div(self.REFERENCE, dist2 + self.SOFTNESS).mul_(visible)
		tint = colors * gain.unsqueeze(1)
		self._motes(motes, index, dist2, visible, gain, tint)
		fx = frac[:, 0:1]
		fy = frac[:, 1:2]
		wx = torch.cat((1.0 - fx, fx), 1)
		wy = torch.cat((1.0 - fy, fy), 1)
		weights = (wy.unsqueeze(2) * wx.unsqueeze(1)).view(-1, 4)
		targets = index.unsqueeze(1) + self.corners
		values = tint.unsqueeze(1) * weights.unsqueeze(2)
		self.buffer.zero_()
		self.buffer.index_add_(0, targets.view(-1), values.view(-1, 3))
		return self.buffer

	def compose(self, color):
		image = color.view(1, 3, self.h, self.w)
		marks = self.marks.view(1, 4, self.h, self.w)
		halo = F.avg_pool2d(marks[:, 3:4], 7, 1, 3)
		image.mul_(halo.mul_(-RenderCfg.MARK_INK_GAIN).clamp_(min=-RenderCfg.MARK_INK).add_(1.0))
		color.add_(self.buffer.t())
		image.add_(marks[:, :3].clamp(max=RenderCfg.MARK_CAP))
