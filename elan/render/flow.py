import colorsys
import math


class TunnelFlow:
	RING_POWER = 6.5
	STREAK_POWER = 3.2
	CORE_POWER = 0.55
	RING_RATE = 9.0
	DASH_RATE = 15.0

	def __init__(self):
		self.active = False
		self.plunge = False
		self.iris_flow = 0.0
		self.source = (1.0, 1.0, 1.0)
		self.target = (1.0, 1.0, 1.0)
		self.core_from = (1.0, 1.0, 1.0)
		self.core_to = (1.0, 1.0, 1.0)

	def set_world(self, world):
		palette = world.palette
		self.active = world.kind == 'tunnel'
		self.plunge = bool(palette.get('plunge'))
		self.iris_flow = float(world.style['iris_flow'])
		if not self.active:
			return
		self.source = palette['flow_from']
		self.target = palette['flow_to']
		self.core_from = palette['core_from']
		self.core_to = palette['core_to']

	def _mix(self, a, b, t):
		return tuple(x + (y - x) * t for x, y in zip(a, b))

	def _bgr(self, rgb, scale):
		return (rgb[2] * scale, rgb[1] * scale, rgb[0] * scale)

	def prepare(self, view, uniforms):
		clock = view.time
		drift = clock * self.iris_flow
		uniforms.set('iris_phase', (4.19 + drift, 2.09 + drift, drift))
		if not self.active:
			return
		progress, hyper = view.tunnel if view.tunnel is not None else (0.0, False)
		blend = progress * progress * (3.0 - 2.0 * progress)
		ring = self._mix(self.source, self.target, blend)
		core = self._mix(self.core_from, self.core_to, blend)
		pace = 1.0
		if hyper:
			pace = 1.8
			ring = self._mix(ring, colorsys.hsv_to_rgb((clock * 0.19) % 1.0, 0.8, 1.0), 0.65)
			core = self._mix(core, colorsys.hsv_to_rgb((clock * 0.19 + 0.45) % 1.0, 0.45, 1.0), 0.5)
		if self.plunge:
			pace = 1.0 + 2.4 * progress * progress
		streak = self._mix(ring, (1.0, 1.0, 1.0), 0.6)
		surge = 0.0 if self.plunge else max((progress - 0.82) / 0.18, 0.0)
		core = self._mix(core, (1.0, 1.0, 1.0), 0.6 * surge)
		pulse = 1.0 + 0.18 * math.sin(clock * 5.0)
		uniforms.set('flow_ring', self._bgr(ring, self.RING_POWER * pulse))
		uniforms.set('flow_streak', self._bgr(streak, self.STREAK_POWER))
		uniforms.set('flow_core', self._bgr(core, self.CORE_POWER * (1.0 + 2.5 * surge)))
		uniforms.set('flow_phase', ((-clock * self.RING_RATE * pace) % math.tau, (-clock * self.DASH_RATE * pace) % math.tau))
