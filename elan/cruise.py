import math

from elan.config import WorldCfg
from elan.physics import Controls


def wrap(angle):
	return (angle + math.pi) % math.tau - math.pi


class Cruise:
	OFFSETS = (-1.1, -0.75, -0.45, -0.2, 0.0, 0.2, 0.45, 0.75, 1.1)
	REACH = (30.0, 70.0, 130.0, 200.0)
	SAFE = 60.0
	BEND = 22.0
	DRIFT = 14.0
	TURN = 0.5
	CLIMB = 0.3
	TILT = 0.35
	EASE = 0.9
	SPEED = 62.0
	FLOOR = 45.0
	ROOF = 90.0
	WANDER = 0.09

	def __init__(self):
		self.world = None
		self.heading = 0.0
		self.height = 0.0
		self.rates = [0.0, 0.0]
		self.clock = 0.0

	def _enter(self, game):
		self.world = game.world
		self.heading = game.kite.yaw
		self.height = min(max(float(game.world.spawn_pos[1]), self.FLOOR + 20.0), WorldCfg.CEILING - self.ROOF)
		self.rates = [0.0, 0.0]

	def _clearance(self, sample, pos, yaw, pitch):
		cp = math.cos(pitch)
		dx, dy, dz = math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp
		low = 1.0e9
		for reach in self.REACH:
			y = pos[1] + dy * reach
			room = min(sample(pos[0] + dx * reach, y, pos[2] + dz * reach)[0], y - 6.0, WorldCfg.CEILING - y)
			low = min(low, room + reach * 0.15)
		return low

	def _pitch(self, kite):
		return min(max((self.height - kite.pos[1]) / 160.0, -self.TILT), self.TILT)

	def _choose(self, game, pitch):
		kite = game.kite
		sample = game.field.sample
		best = None
		for offset in self.OFFSETS:
			yaw = kite.yaw + offset
			room = self._clearance(sample, kite.pos, yaw, pitch)
			score = min(room, self.SAFE) - self.BEND * abs(offset) - self.DRIFT * abs(wrap(yaw - self.heading))
			if best is None or score > best[0]:
				best = (score, offset, room)
		return best

	def drive(self, game, dt):
		if game.world is not self.world:
			self._enter(game)
		kite = game.kite
		self.clock += dt
		self.heading = wrap(self.heading + self.WANDER * math.sin(self.clock * 0.11) * dt)
		self.heading = wrap(self.heading + wrap(kite.yaw - self.heading) * min(dt * 0.15, 1.0))
		pitch = self._pitch(kite)
		_, offset, room = self._choose(game, pitch)
		if room < self.SAFE * 0.5:
			pitch = max(pitch, self.TILT * (1.0 - room / (self.SAFE * 0.5)))
		turn = min(max(offset * 0.8, -self.TURN), self.TURN)
		climb = min(max((pitch - kite.pitch) * 1.2, -self.CLIMB), self.CLIMB)
		blend = 1.0 - math.exp(-max(dt, 1.0e-4) / self.EASE)
		self.rates[0] += (turn - self.rates[0]) * blend
		self.rates[1] += (climb - self.rates[1]) * blend
		controls = Controls()
		controls.look_x = self.rates[0] * dt
		controls.look_y = -self.rates[1] * dt
		controls.thrust = 0.7 if kite.speed < self.SPEED else 0.0
		controls.brake = 1.0 if room < 12.0 and kite.speed > 35.0 else 0.0
		return controls
