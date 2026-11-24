import math
import numpy as np

from elan.config import WorldCfg
from elan.physics import Controls


class Autopilot:
	GAIN = 5.0
	TURN_MAX = 2.6
	PITCH_MAX = 2.2
	ACCEL = 16.0
	ACCEL_DANGER = 16.0

	def __init__(self):
		self.ribbon = -1
		self.direction = 1
		self.rate = [0.0, 0.0]

	def _ease(self, index, want, limit):
		step = max(-limit, min(limit, want - self.rate[index]))
		self.rate[index] += step
		return self.rate[index]

	def _wrap(self, dx, dz):
		dx -= WorldCfg.SIZE_X * round(dx / WorldCfg.SIZE_X)
		dz -= WorldCfg.SIZE_Z * round(dz / WorldCfg.SIZE_Z)
		return dx, dz

	def _target(self, game):
		kite = game.kite
		if game.phase == 'tunnel':
			return game.path.spot(game.station + 70.0 + kite.speed * 0.45, 0)
		if game.dry:
			delta = game._wrapped(game.motes - np.asarray(kite.pos, dtype=np.float32))
			cost = np.sqrt((delta * delta).sum(1)) + 4.0 * np.maximum(delta[:, 1], 0.0)
			cost[~game.alive | game.deep] = 1.0e12
			return game.motes[int(np.argmin(cost))]
		if game.gate_open:
			return game.world.gate_pos
		size = game.ribbon_size
		if self.ribbon >= len(game.ribbon_left):
			self.ribbon = -1
		if self.ribbon < 0 or game.ribbon_left[self.ribbon] <= 0:
			if game.nearest_dry is None or not game.alive[game.nearest_dry] or game.deep[game.nearest_dry]:
				return None
			self.ribbon = game.nearest_dry // size
			self.direction = 1 if (game.nearest_dry % size) < size // 2 else -1
		base = self.ribbon * size
		order = range(size) if self.direction > 0 else range(size - 1, -1, -1)
		for i in order:
			if game.alive[base + i]:
				ahead = i + self.direction
				if 0 <= ahead < size and game.alive[base + ahead]:
					near = game.motes[base + i]
					far = game.motes[base + ahead]
					return ((near[0] + far[0]) * 0.5, (near[1] + far[1]) * 0.5, (near[2] + far[2]) * 0.5)
				return game.motes[base + i]
		self.ribbon = -1
		return None

	def drive(self, game, dt):
		kite = game.kite
		controls = Controls()
		target = self._target(game)
		f = kite.forward()
		if target is None:
			dx, dy, dz = f
		else:
			dx, dz = self._wrap(target[0] - kite.pos[0], target[2] - kite.pos[2])
			dy = target[1] - kite.pos[1]
		norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-6
		dx, dy, dz = dx / norm, dy / norm, dz / norm
		if game.dry and kite.speed < 45.0 and kite.pos[1] > 30.0:
			dy = min(dy, -0.5)
		facing = dx * f[0] + dy * f[1] + dz * f[2]
		orbiting = target is not None and norm < 60.0 and facing < 0.75
		speed = max(kite.speed, 20.0)
		danger = 0.0
		for reach in (0.35, 0.8, 1.4):
			px = kite.pos[0] + kite.vel[0] * reach
			py = kite.pos[1] + kite.vel[1] * reach
			pz = kite.pos[2] + kite.vel[2] * reach
			dist, gx, gy, gz = game.field.sample(px, py, pz)
			margin = 7.0 + speed * 0.045
			if dist < margin:
				push = (margin - dist) / margin * 2.4
				dx += gx * push
				dy += gy * push
				dz += gz * push
				danger = max(danger, (margin - dist) / margin)
		low = kite.pos[1] + kite.vel[1] * 1.2
		if low < 22.0:
			dy += (22.0 - low) / 22.0 * 1.5
		if kite.pos[1] > WorldCfg.CEILING - 60.0:
			dy -= 0.8
		norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-6
		dx, dy, dz = dx / norm, dy / norm, dz / norm
		yaw = math.atan2(dx, dz)
		pitch = math.asin(max(-0.95, min(0.95, dy)))
		turn = (yaw - kite.yaw + math.pi) % math.tau - math.pi
		surge = (self.ACCEL + self.ACCEL_DANGER * danger) * dt
		controls.look_x = self._ease(0, max(-self.TURN_MAX, min(self.TURN_MAX, turn * self.GAIN)), surge) * dt
		controls.look_y = self._ease(1, max(-self.PITCH_MAX, min(self.PITCH_MAX, (kite.pitch - pitch) * self.GAIN)), surge) * dt
		controls.thrust = 1.0 if kite.speed < 78.0 and not (orbiting and kite.speed > 26.0) else 0.0
		controls.brake = 1.0 if danger > 0.75 or (orbiting and kite.speed > 30.0) else 0.0
		if game.phase == 'tunnel':
			controls.thrust = 1.0
			controls.brake = 0.0
		controls.boost = False
		return controls
