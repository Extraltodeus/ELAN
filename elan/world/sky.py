import math
import torch
import torch.nn.functional as F

from elan.config import WorldCfg


class SkyBaker:
	def __init__(self, device):
		self.device = device

	def _vec(self, rgb, scale=1.0):
		return torch.tensor(rgb, device=self.device, dtype=torch.float32).view(3, 1, 1) * scale

	def _unit(self, az_deg, el_deg):
		az = math.radians(az_deg)
		el = math.radians(el_deg)
		return torch.tensor((math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)), device=self.device, dtype=torch.float32)

	def bake(self, noise, palette, sun_dir):
		width = WorldCfg.SKY_W
		half = WorldCfg.SKY_H // 2
		az = ((torch.arange(width, device=self.device, dtype=torch.float32) + 0.5) / width * 2.0 - 1.0) * math.pi
		el = (torch.arange(half, device=self.device, dtype=torch.float32) + 0.5) / half * (math.pi * 0.5)
		az = az.view(1, width)
		el = el.view(half, 1)
		dx = torch.sin(az) * torch.cos(el)
		dy = torch.sin(el).expand(half, width)
		dz = torch.cos(az) * torch.cos(el)
		dirs = torch.stack((dx, dy, dz), 0)
		sun = torch.tensor(sun_dir, device=self.device, dtype=torch.float32)
		cos_sun = (dirs * sun.view(3, 1, 1)).sum(0)
		cos_pos = cos_sun.clamp_min(0.0)
		sun_az = math.atan2(sun_dir[0], sun_dir[2])
		toward = (0.5 + 0.5 * torch.cos(az - sun_az)).expand(half, width)

		zenith = self._vec(palette['zenith'])
		horizon = self._vec(palette['horizon'])
		glow = self._vec(palette['glow'])
		sun_color = self._vec(palette['sun_color'])
		falloff = torch.exp(-dy * 4.2)
		base = zenith + (horizon * (0.35 + 0.65 * toward ** 2) - zenith) * falloff
		base = base + glow * (torch.exp(-dy * 9.0) * toward ** 6 * 0.9)
		base = base + glow * (cos_pos ** 8 * 0.3 + cos_pos ** 48 * 0.8)
		fog = self._fog_texture(base)

		style = palette['style']
		sky = base
		night = (1.0 - (base.mean(0) * 1.6).clamp(0.0, 1.0)) ** 2
		sky = sky + self._fold(self._nebula(noise, dirs, palette), style['sky_fold']) * night * (2.2 if style['sky_fold'] else 1.0)
		if style['aurora'] > 0.0:
			sky = sky + self._aurora(noise, az, el, half, width) * night * style['aurora']
		sky = self._clouds(noise, sky, palette, el, toward, half, width)
		sky = self._planet(noise, sky, base, dirs, sun, palette)
		size = style['sun_disc']
		outer = 1.7 * size ** 0.8
		disc = torch.clamp((cos_sun - math.cos(math.radians(outer))) / (math.cos(math.radians(outer - 0.5)) - math.cos(math.radians(outer))), 0.0, 1.0)
		sky = sky + sun_color * (cos_pos ** 900 * 5.0 / size + disc * 55.0 / size ** 2.7)
		full = torch.cat((torch.flip(sky, (1,)), sky), 1)
		star_dirs, star_colors = self._stars(noise, palette, night, half, width)
		return full.flip(0).unsqueeze(0).contiguous(), fog, star_dirs, star_colors

	def _fold(self, layer, folds):
		if not folds:
			return layer
		width = layer.shape[-1]
		span = width // folds
		index = torch.arange(width, device=self.device) % (2 * span)
		index = torch.where(index >= span, 2 * span - 1 - index, index)
		return layer[..., index]

	def _aurora(self, noise, az, el, half, width):
		total = torch.zeros((3, half, width), device=self.device)
		low = torch.tensor((0.08, 1.0, 0.45), device=self.device).view(3, 1, 1)
		high = torch.tensor((0.5, 0.18, 0.95), device=self.device).view(3, 1, 1)
		for k in range(3):
			phase = noise.uniform((2,)) * math.tau
			wander = noise.band((width,), 1.0, 260.0, 0.5).view(1, width)
			rays = noise.fractal((width,), 1.0, 3.0, 60.0, 1.2).view(1, width)
			veil = noise.band((width,), 1.0, 420.0, 0.5).view(1, width)
			base = 0.2 + 0.13 * k + 0.09 * torch.sin(az * 2.0 + phase[0]) + 0.05 * wander
			above = el - base
			rise = torch.clamp(above / 0.035, 0.0, 1.0) * torch.exp(-above.clamp_min(0.0) / (0.2 + 0.06 * k))
			strength = torch.clamp(0.55 + 0.45 * rays, 0.0, 1.5) * torch.clamp(0.4 + 0.8 * veil, 0.0, 1.4)
			tint = low + (high - low) * torch.clamp(above / 0.4, 0.0, 1.0)
			total = total + tint * (rise * strength * (0.5 - 0.12 * k)).unsqueeze(0)
		return total

	def _fog_texture(self, base):
		full = torch.cat((torch.flip(base, (1,)), base), 1).unsqueeze(0)
		small = F.adaptive_avg_pool2d(full, (32, 64))
		small = torch.cat((small[..., -1:], small, small[..., :1]), 3)
		small = F.avg_pool2d(small, (1, 3), stride=1)
		small = small / (1.0 + 0.55 * small)
		return small[0].flip(0).unsqueeze(0).contiguous()

	def _stars(self, noise, palette, night, half, width):
		count = max(int(2200 * palette['stars']), 16)
		sin_el = noise.uniform((count,)) * 0.985 + 0.015
		el = torch.asin(sin_el)
		az = (noise.uniform((count,)) * 2.0 - 1.0) * math.pi
		dirs = torch.stack((torch.sin(az) * torch.cos(el), sin_el, torch.cos(az) * torch.cos(el)), 1)
		row = (el / (math.pi * 0.5) * half).long().clamp_(0, half - 1)
		col = ((az / math.pi * 0.5 + 0.5) * width).long().clamp_(0, width - 1)
		dark = night[row, col] * torch.clamp((el - 0.06) * 6.0, 0.0, 1.0)
		power = (noise.uniform((count,)) ** 10 * 14.0 + 0.16) * dark
		tint = 0.7 + 0.6 * noise.uniform((count, 3))
		colors = tint * power.unsqueeze(1)
		mirrored = dirs * torch.tensor((1.0, -1.0, 1.0), device=self.device)
		dirs = torch.cat((dirs, mirrored), 0)
		colors = torch.cat((colors, colors * 0.16), 0)
		return dirs.contiguous(), colors.flip(1).contiguous()

	def _nebula(self, noise, dirs, palette):
		half, width = dirs.shape[1], dirs.shape[2]
		field = noise.fractal((half, width), 1.0, 12.0, 420.0, 1.9)
		field2 = noise.fractal((half, width), 1.0, 30.0, 700.0, 2.0)
		axis = noise.normal((3,))
		axis = axis / axis.norm()
		lane = torch.exp(-((dirs * axis.view(3, 1, 1)).sum(0) / 0.32) ** 2)
		density = torch.clamp(field * 0.45 + 0.35, 0.0, 1.5) ** 2 * lane
		tint = self._vec(palette['nebula'])
		warm = self._vec(palette['glow']) * 0.35
		color = tint + (warm - tint) * torch.clamp(field2 * 0.5 + 0.5, 0.0, 1.0)
		return color * density * 0.55

	def _clouds(self, noise, sky, palette, el, toward, half, width):
		amount = palette['cloud']
		if amount <= 0.0:
			return sky
		field = noise.fractal((half, width), 1.0, 6.0, 90.0, 1.8, aniso=[0.3, 2.6])
		band = torch.exp(-((el - 0.16) / 0.11) ** 2)
		density = torch.clamp(field * 0.5 + 0.1, 0.0, 1.0) * band * amount * 0.8
		dark = self._vec(palette['zenith']) * 1.8 + self._vec(palette['horizon']) * 0.12
		lit = self._vec(palette['glow']) * 1.5
		color = dark + (lit - dark) * (toward ** 3)
		return sky + (color - sky) * density.clamp(0.0, 0.85)

	def _planet(self, noise, sky, base, dirs, sun, palette):
		center = self._unit(palette['planet_az'], palette['planet_el'])
		radius = math.sin(math.radians(palette['planet_size']))
		up = torch.tensor((0.0, 1.0, 0.0), device=self.device)
		right = torch.cross(up, center, dim=0)
		right = right / right.norm()
		up = torch.cross(center, right, dim=0)
		tilt = math.radians(palette['ring_tilt'])
		axis = up * math.cos(tilt) + right * math.sin(tilt)
		lean = math.radians(24.0)
		axis = axis * math.cos(lean) - center * math.sin(lean)
		axis = axis / axis.norm()

		along = (dirs * center.view(3, 1, 1)).sum(0)
		disc = along ** 2 - (1.0 - radius * radius)
		hit = (disc > 0.0) & (along > 0.0)
		t_near = along - torch.sqrt(disc.clamp_min(0.0))
		normal = (dirs * t_near - center.view(3, 1, 1)) / radius
		lambert = (normal * sun.view(3, 1, 1)).sum(0).clamp_min(0.0)
		latitude = (normal * axis.view(3, 1, 1)).sum(0)
		bands = 0.5 + 0.5 * torch.sin(latitude * 15.0 + 2.2 * torch.sin(latitude * 6.0 + 1.0))
		color = self._vec(palette['planet_b']) + (self._vec(palette['planet_a']) - self._vec(palette['planet_b'])) * bands
		facing = (-(normal * dirs).sum(0)).clamp(0.0, 1.0)
		surface = color * (lambert ** 0.8 * 1.5 + 0.015) * (0.35 + 0.65 * facing ** 0.5)
		surface = surface + self._vec(palette['glow']) * ((1.0 - facing) ** 3 * lambert * 0.6)

		denom = (dirs * axis.view(3, 1, 1)).sum(0)
		t_ring = float((center * axis).sum()) / torch.where(denom.abs() < 1e-5, torch.full_like(denom, 1e-5), denom)
		point = dirs * t_ring - center.view(3, 1, 1)
		rr = point.norm(dim=0) / radius
		ring = torch.clamp((rr - 1.35) / 0.08, 0.0, 1.0) * torch.clamp((2.35 - rr) / 0.08, 0.0, 1.0)
		ring = ring * (0.55 + 0.45 * torch.sin(rr * 46.0)) * (1.0 - 0.85 * torch.exp(-((rr - 1.93) / 0.035) ** 2))
		ring = ring * (t_ring > 0.0)
		ring = ring * torch.where(hit & (t_ring > t_near), 0.0, 1.0)
		sun_along = (point * sun.view(3, 1, 1)).sum(0)
		lateral = (point - sun.view(3, 1, 1) * sun_along).norm(dim=0)
		shadowed = (sun_along < 0.0) & (lateral < radius)
		ring_light = torch.where(shadowed, 0.04, 1.0)
		ring_color = self._vec(palette['ring']) * 0.85

		haze = base * 0.35
		out = torch.where(hit, surface * 0.9 + haze, sky)
		out = out + (ring_color * ring_light + haze * 0.5 - out) * (ring * 0.8)
		return out
