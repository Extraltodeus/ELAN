import math
import os

import numpy as np

from elan.ai.observe import Observer
from elan.ai.shield import Shield
from elan.config import RulesCfg, WorldCfg
from elan.field import FieldSampler
from elan.physics import Controls
from elan.route import Route


class LearnedPilot:
	SECTORS = 25
	RESERVE = 0.25
	ENDURANCE = 14.0
	APPROACH = (320.0, 0.75, 60.0, 0.25, 0.3, 90.0)
	CAUTION = (0.35, 8.0, 0.8, 6.0, 0.3, 60.0, 90.0)
	SHIELD = True

	def __init__(self, path):
		self.ready = False
		self.layers = []
		self.center = None
		self.scale = None
		self.turn = 2.4
		self.pitch = 2.0
		self.ease = 0.1
		self.rates = [0.0, 0.0]
		self.thrust = 1.0
		self.world = None
		self.field = None
		self.route = None
		self.tunnel = False
		self.hyper = False
		self.observation = None
		self.door = None
		self.reserve = self.RESERVE
		self.shield = None
		self.rays = Observer.directions()
		self.distant = [self.rays[k] for k in Observer.DISTANT]
		if path and os.path.isfile(path):
			self._load(path)

	def _load(self, path):
		try:
			import torch
			saved = torch.load(path, map_location='cpu')
			state = saved['state']
			if saved.get('version') != 2 or int(saved['size']) != Observer.SIZE:
				return
			index = 0
			while 'actor.%d.weight' % index in state:
				self.layers.append((state['actor.%d.weight' % index].numpy().astype(np.float64), state['actor.%d.bias' % index].numpy().astype(np.float64)))
				index += 2
			self.center = state['center'].numpy().astype(np.float64)
			self.scale = 1.0 / np.sqrt(state['spread'].numpy().astype(np.float64) + 1.0e-5)
			self.turn = float(saved.get('turn', self.turn))
			self.pitch = float(saved.get('pitch', self.pitch))
			self.ease = float(saved.get('ease', self.ease))
			self.shield = Shield(self.turn, self.pitch, self.ease)
			self.ready = True
		except Exception:
			self.ready = False

	def _enter(self, world):
		self.world = world
		self.field = FieldSampler(world.field)
		self.route = Route(world)
		self.tunnel = world.kind == 'tunnel'
		self.hyper = self.tunnel and bool(world.palette['hyper'])
		self.rates = [0.0, 0.0]

	def _cap(self, vector, limit):
		norm = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
		gain = min(limit / max(norm, 1.0e-6), 1.0)
		return [vector[0] * gain, vector[1] * gain, vector[2] * gain]

	def observe(self, game):
		kite = game.kite
		route = self.route
		pos = np.asarray(kite.pos, dtype=np.float64)
		vel = np.asarray(kite.vel, dtype=np.float64)
		cp = math.cos(kite.pitch)
		f = (math.sin(kite.yaw) * cp, math.sin(kite.pitch), math.cos(kite.yaw) * cp)
		r = (math.cos(kite.yaw), 0.0, -math.sin(kite.yaw))
		u = (f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0])
		axes = (r, u, f)
		self.reserve = max(self.RESERVE, self.ENDURANCE * (RulesCfg.FLUX_DRAIN + game.sector * RulesCfg.FLUX_DRAIN_SECTOR))
		route.urgent = not self.tunnel and not game.attract and game.flux < self.reserve
		route.advance(pos, vel, f, game.alive)
		way, valid, left = route.waypoints()
		ahead = route.upcoming(pos)
		gate = game.gate_open and not self.tunnel
		self.door = None
		if gate:
			door = route.wrap(np.asarray(self.world.gate_pos, dtype=np.float64) - pos)
			self.door = door
			way = np.tile(door, (len(valid), 1))
			valid = [True] * len(valid)
			ahead = None
		speed = math.sqrt(float(vel @ vel))
		height = kite.pos[1]
		drain = 0.0 if self.tunnel else RulesCfg.FLUX_DRAIN + min(game.sector, self.SECTORS) * RulesCfg.FLUX_DRAIN_SECTOR
		out = [float(vel @ np.asarray(axis)) / 100.0 for axis in axes]
		out += [
			speed / 150.0,
			kite.pitch,
			height / 200.0,
			min(max((WorldCfg.CEILING - abs(height)) / 100.0, 0.0), 1.0),
			float(game.flow),
			float(kite.rush),
			1.0 if gate else 0.0,
			1.0 if height < 0.0 else 0.0,
			self.rates[0] / self.turn,
			self.rates[1] / self.pitch,
			self.thrust,
			1.0 if kite.boosting else 0.0,
			float(game.flux),
			drain * 20.0,
			1.0 if self.tunnel else 0.0,
			1.0 if self.hyper else 0.0,
		]
		spans = []
		for k in range(len(valid)):
			x, y, z = way[k]
			gain = 1.0 if valid[k] else 0.0
			local = self._cap([(x * axis[0] + y * axis[1] + z * axis[2]) / 100.0 for axis in axes], 3.0)
			out += [value * gain for value in local]
			spans.append(min(math.sqrt(x * x + y * y + z * z) / 200.0, 3.0) * gain)
		out += spans
		out += [1.0 if v else 0.0 for v in valid]
		out.append(left)
		if ahead is None:
			out += [0.0] * 8
		else:
			target, tangent = ahead
			x, y, z = target
			out += self._cap([(x * axis[0] + y * axis[1] + z * axis[2]) / 100.0 for axis in axes], 4.0)
			out += [tangent[0] * axis[0] + tangent[1] * axis[1] + tangent[2] * axis[2] for axis in axes]
			out += [min(math.sqrt(x * x + y * y + z * z) / 200.0, 4.0), 1.0]
		sample = self.field.sample
		px, py, pz = kite.pos
		for x, y, z in self.rays:
			dx = x * r[0] + y * u[0] + z * f[0]
			dy = x * r[1] + y * u[1] + z * f[1]
			dz = x * r[2] + y * u[2] + z * f[2]
			for reach in Observer.RANGES:
				out.append(min(max(sample(px + dx * reach, py + dy * reach, pz + dz * reach)[0] / Observer.SIGHT, -0.25), 1.0))
		for lead in Observer.LEADS:
			out.append(min(max(sample(px + vel[0] * lead, py + vel[1] * lead, pz + vel[2] * lead)[0] / Observer.SIGHT, -0.25), 1.0))
		for x, y, z in self.distant:
			dx = x * r[0] + y * u[0] + z * f[0]
			dy = x * r[1] + y * u[1] + z * f[1]
			dz = x * r[2] + y * u[2] + z * f[2]
			out.append(min(max(sample(px + dx * Observer.FAR, py + dy * Observer.FAR, pz + dz * Observer.FAR)[0] / Observer.FAR_SIGHT, -0.25), 1.0))
		return np.asarray(out, dtype=np.float64)

	def decide(self, observation):
		x = np.clip((observation - self.center) * self.scale, -8.0, 8.0)
		for weight, bias in self.layers[:-1]:
			x = np.tanh(weight @ x + bias)
		weight, bias = self.layers[-1]
		return np.tanh(weight @ x + bias)

	def drive(self, game, dt):
		controls = Controls()
		if not self.ready or game.world is None:
			return controls
		if game.world is not self.world:
			self._enter(game.world)
		self.observation = self.observe(game)
		action = self.decide(self.observation)
		blend = 1.0 - math.exp(-max(dt, 1.0e-4) / self.ease)
		steer = (float(action[0]), float(action[1]))
		blocked = False
		if self.SHIELD and not self.tunnel:
			steer, blocked = self.shield.guard(self.field, game.kite, self.rates, steer)
		self.rates[0] += (steer[0] * self.turn - self.rates[0]) * blend
		self.rates[1] += (steer[1] * self.pitch - self.rates[1]) * blend
		self.thrust += ((float(action[2]) + 1.0) * 0.5 - self.thrust) * blend
		controls.look_x = self.rates[0] * dt
		controls.look_y = -self.rates[1] * dt
		controls.thrust = self.thrust
		controls.boost = float(action[3]) > 0.0 and (self.hyper or game.flux > self.reserve) and not (game.gate_open and not self.tunnel)
		if not self.tunnel:
			soon, tight, later, close, idle, brisk, swift = self.CAUTION
			kite = game.kite
			sample = self.field.sample
			first = sample(kite.pos[0] + kite.vel[0] * soon, kite.pos[1] + kite.vel[1] * soon, kite.pos[2] + kite.vel[2] * soon)[0]
			second = sample(kite.pos[0] + kite.vel[0] * later, kite.pos[1] + kite.vel[1] * later, kite.pos[2] + kite.vel[2] * later)[0]
			if first < tight and kite.speed > brisk:
				controls.thrust = 0.0
				controls.brake = 1.0
				controls.boost = False
			elif second < close and kite.speed > swift:
				controls.thrust = min(controls.thrust, idle)
				controls.boost = False
		if self.door is not None and controls.brake <= 0.0:
			near, aligned, slow, idle, away, fast = self.APPROACH
			kite = game.kite
			span = math.sqrt(float(self.door @ self.door))
			closing = float(self.door @ np.asarray(kite.vel, dtype=np.float64)) / (max(span, 1.0e-6) * max(kite.speed, 1.0))
			if span < near and closing < aligned and kite.speed > slow:
				controls.thrust = min(controls.thrust, idle)
				controls.brake = 1.0 if closing < away and kite.speed > fast else 0.0
		return controls
