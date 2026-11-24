import math
import random

from elan.config import FlightCfg, WorldCfg


class Camera:
	def __init__(self):
		self.mode = 'chase'
		self.trail = None
		self.pos = (0.0, 100.0, 0.0)
		self.right = (1.0, 0.0, 0.0)
		self.up = (0.0, 1.0, 0.0)
		self.forward = (0.0, 0.0, 1.0)
		self.craft = None
		self.hand = (0.0, 0.0, 0.0)
		self.jump = True

	def cut(self):
		self.trail = None
		self.jump = True

	def _unit(self, v):
		norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
		if norm < 1.0e-6:
			return (0.0, 0.0, 1.0)
		return (v[0] / norm, v[1] / norm, v[2] / norm)

	def _cross(self, a, b):
		return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

	def _roll(self, right, up, angle):
		c = math.cos(angle)
		s = math.sin(angle)
		return (
			(right[0] * c - up[0] * s, right[1] * c - up[1] * s, right[2] * c - up[2] * s),
			(up[0] * c + right[0] * s, up[1] * c + right[1] * s, up[2] * c + right[2] * s),
		)

	def _shake(self, r, u, f, amount):
		if amount <= 0.0:
			return r, u, f
		sx = random.uniform(-amount, amount)
		sy = random.uniform(-amount, amount)
		f = self._unit((f[0] + r[0] * sx + u[0] * sy, f[1] + r[1] * sx + u[1] * sy, f[2] + r[2] * sx + u[2] * sy))
		r = self._unit((r[0] - f[0] * sx, r[1] - f[1] * sx, r[2] - f[2] * sx))
		return r, self._cross(f, r), f

	def _craft_axes(self, kite, look):
		if kite.speed > 8.0:
			v = kite.vel
			heading = self._unit((v[0] * 0.75 / kite.speed + look[0] * 0.25, v[1] * 0.75 / kite.speed + look[1] * 0.25, v[2] * 0.75 / kite.speed + look[2] * 0.25))
		else:
			heading = look
		flat = math.sqrt(heading[0] * heading[0] + heading[2] * heading[2])
		if flat < 1.0e-4:
			right = (math.cos(kite.yaw), 0.0, -math.sin(kite.yaw))
		else:
			right = (heading[2] / flat, 0.0, -heading[0] / flat)
		up = self._cross(heading, right)
		lean = max(-1.15, min(1.15, kite.roll * 1.7)) + math.tau * kite.form
		right, up = self._roll(right, up, lean)
		return right, up, heading

	def update(self, kite, field, shake, dt, thrust, heat):
		r, u, f = kite.axes()
		r, u, f = self._shake(r, u, f, shake * 0.012)
		limit = WorldCfg.SIZE_Y - 4.0
		chase = self.mode == 'chase'
		craft_right, craft_up, craft_forward = self._craft_axes(kite, f)
		scale = 0.0
		if chase:
			if kite.speed > 10.0:
				v = kite.vel
				target = self._unit((f[0] * 0.78 + v[0] * 0.22 / kite.speed, f[1] * 0.78 + v[1] * 0.22 / kite.speed, f[2] * 0.78 + v[2] * 0.22 / kite.speed))
			else:
				target = f
			if self.trail is None:
				self.trail = target
			blend = min(dt * 11.0, 1.0)
			self.trail = self._unit((self.trail[0] + (target[0] - self.trail[0]) * blend, self.trail[1] + (target[1] - self.trail[1]) * blend, self.trail[2] + (target[2] - self.trail[2]) * blend))
			back = self.trail
			pos = None
			for scale in (1.0, 0.72, 0.48, 0.26):
				reach = FlightCfg.CHASE_DISTANCE * scale * (1.0 + FlightCfg.CHASE_DART * kite.form)
				lift = FlightCfg.CHASE_HEIGHT * scale
				candidate = (kite.pos[0] - back[0] * reach + u[0] * lift, kite.pos[1] - back[1] * reach + u[1] * lift, kite.pos[2] - back[2] * reach + u[2] * lift)
				if field.sample(candidate[0], candidate[1], candidate[2])[0] > 1.1:
					pos = candidate
					break
			if pos is None:
				scale = 0.0
				pos = (kite.pos[0], kite.pos[1], kite.pos[2])
			right, up = self._roll(r, u, kite.roll * 0.35)
			self.hand = (kite.pos[0] + craft_forward[0] * 1.2, kite.pos[1] + craft_forward[1] * 1.2, kite.pos[2] + craft_forward[2] * 1.2)
		else:
			pos = (kite.pos[0], kite.pos[1], kite.pos[2])
			right, up = self._roll(r, u, kite.roll)
			self.hand = (pos[0] + right[0] * 0.9 - up[0] * 0.75 + f[0] * 0.8, pos[1] + right[1] * 0.9 - up[1] * 0.75 + f[1] * 0.8, pos[2] + right[2] * 0.9 - up[2] * 0.75 + f[2] * 0.8)
		height = max(-limit, min(limit, pos[1]))
		if abs(height) < 0.35:
			height = 0.35 if height >= 0.0 else -0.35
		self.pos = (pos[0], height, pos[2])
		self.right = right
		self.up = up
		self.forward = f
		self.craft = {
			'pos': (kite.pos[0], kite.pos[1], kite.pos[2]),
			'right': craft_right, 'up': craft_up, 'forward': craft_forward,
			'visible': chase and scale > 0.3,
			'jump': self.jump,
			'thrust': thrust,
			'heat': heat,
			'morph': kite.form,
			'sheath': kite.form * min(max((kite.speed - 50.0) / 70.0, 0.0), 1.0),
			'surge': kite.surge,
			'rush': kite.rush,
			'stretch': min(kite.speed / FlightCfg.STRETCH_SPEED, 1.3),
		}
		self.jump = False

	def submerged(self):
		return self.pos[1] < 0.0
