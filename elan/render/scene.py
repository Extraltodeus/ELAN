import math
import torch

from elan.config import RenderCfg
from elan.render.glow import GlowLayout
from elan.render.craft import CraftPoints
from elan.render.ramp import ProximityRamp


class MoteStyle:
	def __init__(self, device, layout, ramp, start):
		self.range = (start, start + layout.mote_count)
		self.far = torch.from_numpy(layout.mote_far).to(device)
		self.near = torch.from_numpy(layout.mote_near).to(device)
		self.white = torch.from_numpy(layout.mote_white).to(device)
		self.glow = torch.from_numpy(ramp.glow).to(device)
		self.mark = torch.from_numpy(ramp.mark).to(device)
		self.scale = ramp.scale
		self.offset = ramp.offset
		self.last = ramp.STEPS - 1.0
		self.amp = None


class PointScene:
	ROPE_POINTS = 320

	def __init__(self, device, uniforms):
		self.device = device
		gen = torch.Generator(device=device)
		gen.manual_seed(4242)
		self.gen = gen
		count = RenderCfg.DUST_COUNT
		self.box = RenderCfg.DUST_BOX
		self.dust = torch.rand((count, 1, 3), generator=gen, device=device) * self.box
		self.dust_tone = torch.rand((count, 1, 1), generator=gen, device=device) ** 3 * 0.9 + 0.1
		self.dust_color = None
		self.dust_base = uniforms.view('dust_base')
		self.dust_shift = uniforms.view('dust_shift')
		self.streak = uniforms.view('streak', 1, 3, 3)
		sparks = RenderCfg.SPARK_COUNT
		self.spark_count = sparks
		self.spark_pos = torch.zeros((sparks, 3), device=device)
		self.spark_vel = torch.zeros((sparks, 3), device=device)
		self.spark_life = torch.zeros(sparks, device=device)
		self.spark_color = torch.zeros((sparks, 3), device=device)
		self.pool_a = torch.randn((sparks, 3), generator=gen, device=device)
		self.pool_b = torch.randn((sparks, 3), generator=gen, device=device)
		self.pool_u = torch.rand(sparks, generator=gen, device=device)
		self.gravity = torch.tensor((0.0, -9.0, 0.0), device=device)
		self.emit_pos = uniforms.view('emit_pos')
		self.emit_vel = uniforms.view('emit_vel')
		self.emit_color = uniforms.view('emit_color')
		self.cursor = 0
		self.spark_time = 0.0
		self.pending = None
		self.hand = uniforms.view('hand')
		self.rope = uniforms.view('rope')
		spread = torch.linspace(0.0, 1.0, self.ROPE_POINTS, device=device)
		self.rope_u = ((1.0 / (1.0 - spread * (1.0 - 1.0 / 120.0)) - 1.0) / 119.0).view(-1, 1).contiguous()
		self.rope_gain = 1.0
		self.rope_color = None
		self.rope_on = False
		self.glow_pos = None
		self.glow_color = None
		self.glow_gain = None
		self.glow_base = None
		self.glow_phase = None
		self.layout = GlowLayout()
		self.craft = CraftPoints(device, uniforms)
		self.far_points = None
		self.far_colors = None
		self.glow_tint = None
		self.motes = None
		self.changed = True

	def _bgr(self, rgb, scale=1.0):
		return torch.tensor((rgb[2] * scale, rgb[1] * scale, rgb[0] * scale), device=self.device, dtype=torch.float32)

	def set_world(self, world):
		palette = world.palette
		layout = self.layout.build(world)
		self.glow_pos = torch.from_numpy(layout.positions).to(self.device)
		self.glow_color = torch.from_numpy(layout.colors).to(self.device)
		self.motes = MoteStyle(self.device, layout, ProximityRamp(palette), self.dust.shape[0] * self.streak.shape[1])
		self.glow_phase = torch.from_numpy(layout.phase).to(self.device)
		self.glow_gain = torch.from_numpy(layout.gain).to(self.device)
		self.glow_base = self.glow_gain * 0.75
		weights = torch.tensor((0.5, 0.3, 0.2), device=self.device).view(1, 3, 1)
		self.dust_color = (self.dust_tone * weights * self._bgr(palette['dust'], 0.5).view(1, 1, 3)).view(-1, 3).contiguous()
		self.rope_color = self._bgr(palette['mote'], 1.5 / 3025.0).view(1, 3).contiguous()
		self.far_points = (world.star_dirs * 1.0e6).contiguous()
		self.far_colors = (world.star_colors * (1.0e12 / 3025.0)).contiguous()
		self.craft.set_palette(palette['mote'])
		self.spark_life.zero_()
		self.spark_time = 0.0
		self.pending = None
		self.changed = True

	def _set_range(self, start, stop, value):
		self.glow_gain[start:stop] = value
		self.glow_base[start:stop] = value * 0.75
		self.changed = True

	def set_motes(self, indices, value):
		if len(indices) == 0:
			return
		size = self.layout.cluster
		index = (torch.tensor(indices, device=self.device, dtype=torch.long).view(-1, 1) * size + torch.arange(size, device=self.device)).view(-1)
		self.glow_gain[index] = value
		self.glow_base[index] = value * 0.75
		self.changed = True

	def set_beacons(self, ribbon, value):
		start, stop = self.layout.beacon_slice(ribbon)
		self._set_range(start, stop, value)

	def set_ring(self, ring, value):
		start, stop = self.layout.ring_slice(ring)
		self._set_range(start, stop, value)

	def set_gate(self, level):
		self._set_range(self.layout.gate_start, self.layout.total, level)

	def emit(self, pos, vel, color, count, spread, speed, life):
		if self.pending is not None and self.pending[3] >= count:
			return
		self.pending = (pos, vel, color, int(count), spread, speed, life)

	def prepare(self, view, uniforms):
		pos = view.pos
		wind = view.time * 1.5
		half = self.box * 0.5
		uniforms.set('dust_base', (pos[0] - wind, pos[1] - wind * 0.2, pos[2] - wind * 0.6))
		uniforms.set('dust_shift', (pos[0] - half, pos[1] - half, pos[2] - half))
		vel = view.velocity
		stretch = 0.011 * (1.0 + 2.5 * view.warp)
		uniforms.set('streak', (0.0, 0.0, 0.0, vel[0] * stretch * 0.5, vel[1] * stretch * 0.5, vel[2] * stretch * 0.5, vel[0] * stretch, vel[1] * stretch, vel[2] * stretch))
		self.rope_on = view.tether is not None
		if self.rope_on:
			hand, anchor = view.tether
			uniforms.set('hand', hand)
			uniforms.set('rope', (anchor[0] - hand[0], anchor[1] - hand[1], anchor[2] - hand[2]))
			self.rope_gain = math.sqrt((anchor[0] - hand[0]) ** 2 + (anchor[1] - hand[1]) ** 2 + (anchor[2] - hand[2]) ** 2)
		self.craft.prepare(view.craft, uniforms)
		if self.pending is not None:
			uniforms.set('emit_pos', self.pending[0])
			uniforms.set('emit_vel', self.pending[1])
			color = self.pending[2]
			uniforms.set('emit_color', (color[2], color[1], color[0]))

	def simulate(self, dt):
		if self.pending is not None:
			count = min(self.pending[3], self.spark_count)
			spread, speed, life = self.pending[4], self.pending[5], self.pending[6]
			self.pending = None
			if self.cursor + count > self.spark_count:
				self.cursor = 0
			a = self.cursor
			b = a + count
			self.cursor = b
			self.spark_pos[a:b] = torch.add(self.emit_pos, self.pool_a[a:b], alpha=spread)
			self.spark_vel[a:b] = torch.add(self.emit_vel, self.pool_b[a:b], alpha=speed)
			self.spark_life[a:b] = torch.mul(self.pool_u[a:b], life).add_(life * 0.35)
			self.spark_color[a:b] = self.emit_color
			self.spark_time = max(self.spark_time, life * 1.4)
		if self.spark_time <= 0.0:
			return
		self.spark_vel.add_(self.gravity, alpha=dt)
		self.spark_vel.mul_(max(1.0 - 1.2 * dt, 0.0))
		self.spark_pos.add_(self.spark_vel, alpha=dt)
		self.spark_life.sub_(dt)
		self.spark_time -= dt

	def gather(self, time):
		dust = torch.remainder(torch.sub(self.dust, self.dust_base), self.box).add_(self.dust_shift)
		points = [torch.add(dust, self.streak).view(-1, 3), self.glow_pos]
		twinkle = torch.sin(torch.add(self.glow_phase, time * 3.0))
		amp = torch.addcmul(self.glow_base, self.glow_gain, twinkle, value=0.25)
		glow = self.glow_color * amp.unsqueeze(1)
		self.glow_tint = torch.mul(glow, 0.045).t()
		self.motes.amp = amp[:self.layout.mote_count]
		colors = [self.dust_color, glow]
		if self.rope_on:
			points.append(torch.addcmul(self.hand, self.rope_u, self.rope))
			reach = torch.mul(self.rope_u, self.rope_gain).add_(1.0).square_().add_(60.0)
			colors.append(self.rope_color * reach)
		if self.spark_time > 0.0:
			points.append(self.spark_pos)
			colors.append(self.spark_color * self.spark_life.clamp(0.0, 1.0).unsqueeze(1))
		self.craft.gather(points, colors)
		return torch.cat(points, 0), torch.cat(colors, 0)
