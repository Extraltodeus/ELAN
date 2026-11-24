import math
import torch
import torch.nn.functional as F

from elan.config import RenderCfg, TunnelCfg, WorldCfg


class SurfaceShader:
	def __init__(self, device, uniforms, space):
		self.device = device
		self.uniforms = uniforms
		self.space = space
		tetra = torch.tensor([[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]], device=device)
		self.tetra_n = (tetra * space.scale).view(1, 4, 1, 1, 3).contiguous()
		self.tetra_k = tetra.t().contiguous()
		self.lift = (space.scale * RenderCfg.AUX_LIFT).contiguous()
		self.eps0 = torch.tensor([RenderCfg.NORMAL_EPS], device=device)
		self.hit0 = torch.tensor([0.2], device=device)
		self.one = torch.tensor(1.0, device=device)
		self.half = torch.tensor(0.5, device=device)
		self.moss0 = torch.tensor(-1.9, device=device)
		self.cavity0 = torch.tensor(0.36, device=device)
		self.under0 = torch.tensor(0.3, device=device)
		self.strata_k = torch.tensor(WorldCfg.SIZE_Y * 0.5 * math.tau / 7.5, device=device)
		self.neon_k = torch.tensor(1.0, device=device)
		self.pix = uniforms.view('pix')
		self.hitk = uniforms.view('hitk')
		self.flow_ring = uniforms.view('flow_ring', 3, 1)
		self.flow_streak = uniforms.view('flow_streak', 3, 1)
		self.flow_phase = uniforms.view('flow_phase')
		self.iris_phase = uniforms.view('iris_phase', 3, 1)
		self.sun = None
		self.grain_source = None
		self.grain_scale = (space.detail_scale * 6.0).contiguous()
		self.colors = {}
		self.moss = 1.0
		self.neon = 0.0
		self.iris = 0.0
		self.iris_y = 0.0
		self.headlight = 0.0
		self.head_k = 1.0 / (55.0 * 55.0)
		self.glow_field = None
		self.h = 0
		self.w = 0
		self.n = 0

	def allocate(self, h, w):
		self.h = h
		self.w = w
		self.n = h * w

	def _color(self, rgb, scale=1.0):
		return torch.tensor((rgb[2] * scale, rgb[1] * scale, rgb[0] * scale), device=self.device, dtype=torch.float32).view(3, 1)

	def set_world(self, world):
		p = world.palette
		style = world.style
		rock_a = self._color(p['rock_a'])
		rock_b = self._color(p['rock_b'])
		self.colors = {
			'rock_mid': (rock_a + rock_b) * 0.5,
			'rock_delta': (rock_b - rock_a) * 0.5,
			'top': self._color(p['top']),
			'amb_ground': self._color(p['amb_ground']),
			'amb_delta': self._color(p['amb_sky']) - self._color(p['amb_ground']),
			'sun': self._color(p['sun_color'], p['sun_power']),
			'rim': self._color(p['rim'], 0.3),
			'vein': self._color(p['vein'], p['vein_power']),
			'neon': self._color(style['neon_color'], style['neon'] / 0.04),
			'head': self._color(p['mote'], 0.45 * style['headlight']) + self._color((1.0, 1.0, 1.0), 0.55 * style['headlight']),
		}
		self.sun = torch.tensor(world.sun_dir, device=self.device, dtype=torch.float32)
		self.grain_source = world.detail[:, 2:3]
		self.under0 = torch.tensor(float(style['vein_spread']), device=self.device)
		self.moss = float(style['moss'])
		self.neon = float(style['neon'])
		self.neon_k = torch.tensor(WorldCfg.SIZE_Y * 0.5 * math.tau / style['neon_period'], device=self.device)
		self.iris = float(style['iridescence'])
		self.iris_y = WorldCfg.SIZE_Y * 0.5 * math.tau / 150.0
		self.headlight = float(style['headlight'])

	def shade(self, world, pos, t, rays_world, submerged, time):
		sampler = torch.grid_sampler
		n = self.n
		colors = self.colors
		pos_y = pos.view(n, 3)[:, 1]
		t_flat = t.view(n)
		eps = torch.addcmul(self.eps0, t, self.pix)
		grid = torch.addcmul(pos, self.tetra_n, eps)
		coarse = sampler(world.sdf, grid, 0, 2, False)
		dgrid = torch.addcmul(self.space.detail_offset, grid, self.space.detail_scale)
		fine = sampler(world.detail, dgrid, 0, 2, False)
		field = torch.add(coarse, fine[:, 2:3]).view(4, n)
		normal = F.normalize(torch.matmul(self.tetra_k, field), dim=0)
		resid = field.mean(0)
		detail = fine.view(3, 4, n).mean(1)

		threshold = torch.addcmul(self.hit0, t_flat, self.hitk)
		cover = torch.div(resid, threshold).neg_().add_(1.5).clamp_(0.0, 1.0)
		cover.mul_(pos_y <= 1.0)
		sea = (pos_y < -1.0).to(torch.float32)
		facing = (normal * rays_world.view(n, 3).t()).sum(0)
		if world.coords is not None:
			return self._tunnel(world, pos, normal, facing, detail, t_flat, cover, sea)
		sign = torch.add(self.one, sea, alpha=-2.0)
		lifted = torch.addcmul(pos, normal.t().view(1, 1, self.h, self.w, 3), self.lift)
		aux = sampler(world.aux, lifted, 0, 2, False)
		openness = aux[0, 0].view(n)
		sunlit = aux[0, 1].view(n)
		if not submerged:
			normal[1].mul_(sign)
		normal_y = normal[1]
		lambert = torch.matmul(self.sun, normal).clamp_min_(0.0)
		if submerged:
			scale = WorldCfg.SIZE_X / 11.0
			cells = torch.mul(pos.view(n, 3)[:, 0::2], scale).add_(scale + time * 0.13)
			caustic = sampler(world.ripple, cells.view(1, 1, n, 2), 0, 2, False)[0, 0].view(n)
			caustic = torch.relu(caustic).mul_(1.5).add_(0.4)
			dim = torch.clamp(pos_y + 1.0, max=0.0).mul_(WorldCfg.SIZE_Y * 0.5 / 80.0).exp_()
			lambert.mul_(torch.addcmul(self.one, sea, caustic.mul_(dim).sub_(1.0)))

		strata = torch.addcmul(self.strata_k, pos_y, self.strata_k).add_(detail[0], alpha=2.2).cos_()
		albedo = torch.addcmul(colors['rock_mid'], colors['rock_delta'], strata)
		moss = torch.add(self.moss0, normal_y, alpha=3.4).clamp_(0.0, self.moss)
		albedo.lerp_(colors['top'], moss)
		if self.iris > 0.0:
			swirl = torch.mul(pos_y, self.iris_y).add_(facing, alpha=5.0).add_(detail[0], alpha=1.4)
			albedo.addcmul_(albedo, torch.add(self.iris_phase, swirl).cos_(), value=self.iris)
		hemi = torch.add(self.half, normal_y, alpha=0.5)
		light = torch.addcmul(colors['amb_ground'], colors['amb_delta'], hemi)
		grain = sampler(self.grain_source, torch.addcmul(self.space.detail_offset, pos, self.grain_scale), 0, 2, False).view(n)
		cavity = torch.add(self.cavity0, detail[0], alpha=0.5 / WorldCfg.DETAIL_AMP).add_(grain, alpha=0.42 / WorldCfg.DETAIL_AMP).mul_(openness)
		light.mul_(cavity)
		light.addcmul_(colors['sun'], lambert.mul_(sunlit))
		self._lamps(light, pos, facing, t_flat)
		color = albedo.mul_(light)
		rim = facing.add_(1.0).clamp_(0.0, 1.0).pow_(4.0).mul_(openness)
		color.addcmul_(colors['rim'], rim)
		glow = torch.add(self.under0, normal_y, alpha=-1.0).clamp_(0.0, 1.0).mul_(detail[1])
		color.addcmul_(colors['vein'], glow)
		if self.neon > 0.0:
			line = torch.addcmul(self.neon_k, pos_y, self.neon_k).add_(detail[0], alpha=1.1).cos_().sub_(0.96).clamp_(min=0.0)
			color.addcmul_(colors['neon'], line.mul_(torch.mul(t_flat, -0.004).exp_()))
		return color, cover, sea

	def _lamps(self, light, pos, facing, t_flat):
		if self.glow_field is not None:
			light.add_(torch.grid_sampler(self.glow_field, pos, 0, 2, False).view(3, self.n))
		if self.headlight > 0.0:
			beam = torch.addcmul(self.one, t_flat, t_flat, value=self.head_k)
			light.addcmul_(self.colors['head'], torch.relu(facing.neg()).div_(beam))

	def _tunnel(self, world, pos, normal, facing, detail, t_flat, cover, sea):
		n = self.n
		colors = self.colors
		coords = torch.grid_sampler(world.coords, pos, 0, 2, False).view(4, n)
		along = torch.atan2(coords[1], coords[0])
		around = torch.atan2(coords[3], coords[2])
		phase = self.flow_phase
		rings = torch.mul(along, float(TunnelCfg.RINGS)).sub_(phase[0]).cos_().sub_(0.8).clamp_(min=0.0)
		streak = torch.mul(around, float(TunnelCfg.STREAKS)).add_(along, alpha=7.0).cos_().sub_(0.84).clamp_(min=0.0)
		dash = torch.mul(along, 96.0).add_(around, alpha=3.0).sub_(phase[1]).cos_().mul_(0.5).add_(0.5)
		streak.mul_(dash)
		light = torch.add(colors['amb_ground'], colors['amb_delta'], alpha=0.5).mul(torch.add(self.cavity0, detail[0], alpha=0.5 / WorldCfg.DETAIL_AMP))
		self._lamps(light, pos, facing, t_flat)
		color = light.mul_(colors['rock_mid'])
		color.addcmul_(self.flow_ring, rings)
		color.addcmul_(self.flow_streak, streak)
		rim = facing.add_(1.0).clamp_(0.0, 1.0).pow_(3.0)
		color.addcmul_(self.flow_streak, rim, value=0.05)
		return color, cover, sea
