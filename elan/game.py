import math
import numpy as np

from elan.config import FlightCfg, RenderCfg, RulesCfg, StoryCfg, TunnelCfg, WorldCfg
from elan.camera import Camera
from elan.field import FieldSampler
from elan.modes import MODES
from elan.physics import Kite
from elan.rail import TunnelRail
from elan.render import View
from elan.route import Route


class Message:
	def __init__(self, text, kind, ttl):
		self.text = text
		self.kind = kind
		self.ttl = ttl
		self.age = 0.0


class Marker:
	def __init__(self, kind):
		self.kind = kind
		self.x = 0.0
		self.y = 0.0
		self.onscreen = False
		self.angle = 0.0
		self.dist = 0.0


class HudState:
	def __init__(self):
		self.time = 0.0
		self.speed = 0.0
		self.speed_ref = FlightCfg.SPEED_REF
		self.flux = 1.0
		self.danger = 0.0
		self.flux_gain = 0.0
		self.score = 0
		self.combo = 1
		self.combo_time = 0.0
		self.sector = 1
		self.quota = 0
		self.quota_need = 1
		self.gate_open = False
		self.boost = False
		self.drift = False
		self.graze = 0.0
		self.tether_state = 0
		self.vel_marker = None
		self.markers = []
		self.messages = []
		self.perf = None
		self.hint = None
		self.tier = 0
		self.tier_name = ''
		self.depth = 0.0
		self.surge = 0.0
		self.dry = False
		self.stall = 0.0
		self.rush = 0.0
		self.rush_ready = False
		self.rush_on = False
		self.tunnel = None
		self.hyper = False
		self.grip = 1.0
		self.dev = None
		self.stage = ''
		self.plunge = False
		self.clock = None
		self.finale = False


class Summary:
	def __init__(self):
		self.score = 0
		self.best = 0
		self.new_best = False
		self.sector = 1
		self.motes = 0
		self.top_speed = 0.0
		self.duration = 0.0
		self.deepest = 0
		self.rank = ''
		self.frontier = False
		self.mode = 'chill'
		self.unit = 'SECTOR'
		self.levels = 0
		self.timed = False
		self.completed = False
		self.best_time = 0.0
		self.new_time = False


class Mix:
	def __init__(self):
		self.speed = 0.0
		self.thrust = 0.0
		self.boost = 0.0
		self.graze = 0.0
		self.tether = 0.0
		self.skim = 0.0
		self.intensity = 0.0
		self.danger = 0.0
		self.menu = 0.0
		self.paused = 0.0
		self.submerged = 0.0
		self.serene = 0.0
		self.dry = 0.0
		self.credits = 0.0


class Game:
	def __init__(self, scene, audio):
		self.scene = scene
		self.audio = audio
		self.kite = Kite()
		self.camera = Camera()
		self.world = None
		self.field = None
		self.motes = None
		self.deep = None
		self.alive = None
		self.ribbon_size = 1
		self.ribbon_left = None
		self.ribbon_idle = None
		self.rings_pos = None
		self.rings_dir = None
		self.ring_side = None
		self.ring_cool = None
		self.time = 0.0
		self.sector = 0
		self.flux = RulesCfg.FLUX_START
		self.flux_gain = 0.0
		self.score = 0.0
		self.combo = 1
		self.combo_time = 0.0
		self.quota = 0
		self.quota_need = RulesCfg.QUOTA_BASE
		self.gate_open = False
		self.gate_side = 0.0
		self.warp_requested = False
		self.over = False
		self.attract = False
		self.messages = []
		self.motes_total = 0
		self.top_speed = 0.0
		self.duration = 0.0
		self.fov = RenderCfg.FOV_BASE
		self.shake = 0.0
		self.damage = 0.0
		self.flash = 0.0
		self.aim_valid = False
		self.aim_depth = 0.0
		self.nearest_mote = None
		self.nearest_dry = None
		self.hint_time = 0.0
		self.hint_text = ''
		self.drifting = False
		self.tier = 0
		self.skips = 0
		self.skip_time = 0.0
		self.sling_speed = 0.0
		self.flow = 0.0
		self.shake_scale = 1.0
		self.dry = False
		self.stall = 0.0
		self.warp_enabled = True
		self.warp = 0.0
		self.scrape = 0.0
		self.rush_charge = 0.0
		self.rush_ready = False
		self.rush_on = False
		self.rush_told = False
		self.phase = 'sector'
		self.path = None
		self.travel = 0.0
		self.travel_need = 1.0
		self.station = 0.0
		self.hyper = False
		self.arrival = False
		self.respawn = RulesCfg.RIBBON_RESPAWN
		self.before = None
		self.quota_limit = 0
		self.route = None
		self.links = []
		self.career = None
		self.odometer = 0.0
		self.impacts = 0
		self.touched = -1
		self.reach = RulesCfg.MOTE_RADIUS
		self.surge = 0.0
		self.limp = 0.0
		self.grip = 1.0
		self.step = 0.0
		self.guide = None
		self.breath = RulesCfg.BREATH_PERIOD
		self.alarm = 0.0
		self.mode = MODES[0]
		self.finale = False
		self.plunge = False
		self.sealed = None
		self.finale_time = 0.0
		self.horizon = 0.0

	def say(self, text, kind='info', ttl=2.6):
		if self.attract:
			return
		self.messages.append(Message(text, kind, ttl))
		if len(self.messages) > 4:
			self.messages.pop(0)

	def cheer(self, text, kind='good', ttl=2.0):
		if self.finale:
			return
		self.say(text, kind, ttl)

	def drain(self):
		return self.mode.drain(self.sector)

	def homing(self):
		if not self.gate_open or self.phase != 'sector':
			return False
		return not self.finale or (not self.attract and self.finale_time > StoryCfg.PORTAL_DELAY)

	def course(self):
		return self.guide if self.finale else self.alive

	def progress(self):
		return min(max(self.travel / self.travel_need, 0.0), 1.0)

	def bonus(self):
		return RulesCfg.TIER_BONUS[self.tier]

	def start_run(self, sector=0, mode=None):
		self.mode = mode or MODES[0]
		self.finale = False
		self.plunge = False
		self.sealed = None
		self.finale_time = 0.0
		self.horizon = 0.0
		self.breath = RulesCfg.BREATH_PERIOD
		self.alarm = 0.0
		self.time = 0.0
		self.sector = sector
		self.phase = 'sector'
		self.arrival = False
		self.odometer = 0.0
		self.impacts = 0
		self.flux = RulesCfg.FLUX_START
		self.flux_gain = 0.0
		self.score = 0.0
		self.combo = 1
		self.combo_time = 0.0
		self.over = False
		self.warp_requested = False
		self.messages = []
		self.motes_total = 0
		self.top_speed = 0.0
		self.duration = 0.0
		self.damage = 0.0
		self.shake = 0.0
		self.tier = 0
		self.skips = 0
		self.flow = 0.0
		self.dry = False
		self.stall = 0.0
		self.surge = 0.0
		self.limp = 0.0
		self.grip = 1.0
		self.hint_time = 16.0

	def _bind(self, world):
		self.world = world
		self.route = Route(world)
		self.links = []
		self.field = FieldSampler(world.field)
		self.kite.field = self.field
		ribbons = world.ribbons
		self.ribbon_size = ribbons.shape[1]
		self.motes = ribbons.reshape(-1, 3).astype(np.float32)
		self.deep = self.motes[:, 1] < 0.0
		self.alive = np.ones(self.motes.shape[0], dtype=bool)
		self.guide = np.ones(self.motes.shape[0], dtype=bool)
		self.nearest_mote = None
		self.nearest_dry = None
		self.ribbon_left = np.full(ribbons.shape[0], self.ribbon_size, dtype=np.int32)
		self.ribbon_idle = np.zeros(ribbons.shape[0], dtype=np.float32)
		self.rings_pos = world.rings_pos.astype(np.float32)
		self.rings_dir = world.rings_dir.astype(np.float32)
		self.ring_side = np.zeros(self.rings_pos.shape[0], dtype=np.float32)
		self.ring_cool = np.zeros(self.rings_pos.shape[0], dtype=np.float32)
		self.quota = 0
		self.gate_open = False
		self.gate_side = 0.0
		self.warp_requested = False
		self.arrival = False
		self.rush_charge = 0.0
		self.rush_ready = False
		self.rush_on = False
		self.rush_told = False
		self.before = None
		self.touched = -1
		self.camera.cut()

	def enter_world(self, world, fresh):
		self._bind(world)
		self.phase = 'sector'
		self.path = None
		self.hyper = False
		self.plunge = False
		self.horizon = 0.0
		self.finale = bool(world.palette.get('final'))
		self.respawn = RulesCfg.RIBBON_RESPAWN
		self.reach = RulesCfg.MOTE_RADIUS
		self.kite.warp = 0.0
		self.kite.rail = None
		self.surge = 0.0 if fresh else 1.0
		self.quota_need = min(RulesCfg.QUOTA_BASE + self.sector * RulesCfg.QUOTA_STEP, RulesCfg.QUOTA_MAX, int(self.motes.shape[0] * 0.5))
		if self.quota_limit > 0:
			self.quota_need = min(self.quota_need, self.quota_limit)
		self.quota_need = max(self.quota_need, 1)
		self.scene.set_gate(0.04)
		speed = 52.0 if fresh else min(max(self.kite.speed, 75.0), TunnelCfg.EXIT_SPEED)
		self.kite.reset(world.spawn_pos, world.spawn_dir, speed)
		if self.finale:
			self._settle(fresh)
			return
		self.say('%s  -  %s' % (self.mode.stage(self.sector), world.palette['name']), 'big', 3.5)
		if self.career is not None and not self.attract:
			for text, kind in self.career.enter(self.sector + 1, world.palette['key'], world.palette['name']):
				self.say(text, kind, 4.0)

	def _settle(self, fresh):
		if self.sealed is None:
			self.sealed = (self.score, self.duration)
		self.finale_time = 0.0
		self.horizon = 0.0 if fresh else 1.0
		self.flux = 1.0
		self.dry = False
		self.stall = 0.0
		self.combo = 1
		self.combo_time = 0.0
		self.alive[:] = False
		self.ribbon_left[:] = 0
		self.respawn = 1.0e9
		self.ring_cool[:] = 1.0e9
		self.quota = self.quota_need
		self.gate_open = True
		self.scene.clear_course()
		self.scene.set_gate(1.0)
		if not fresh:
			self.kite.reset(self.world.spawn_pos, self.world.spawn_dir, StoryCfg.ARRIVAL_SPEED)
		self.messages = []
		self.say('WELCOME TO THE SINGULARITY', 'big', 8.0)
		self.say('NO QUOTA  -  NO DRAIN  -  NOTHING LEFT TO CHASE', 'good', 8.0)
		self.say('THE WHITE PORTAL IS OPEN  -  CROSS IT WHENEVER YOU ARE READY', 'info', 11.0)

	def enter_tunnel(self, world):
		self._bind(world)
		self.phase = 'tunnel'
		self.path = world.path
		self.hyper = bool(world.palette['hyper'])
		self.plunge = bool(world.palette.get('plunge'))
		self.finale = False
		self.horizon = 0.0
		self.respawn = TunnelCfg.RESPAWN
		self.reach = TunnelCfg.MOTE_RADIUS
		self.surge = 0.0
		self.kite.rail = TunnelRail(world.path)
		self.travel = 0.0
		self.travel_need = TunnelCfg.TRAVEL_HYPER if self.hyper else TunnelCfg.TRAVEL
		if self.plunge:
			self.travel_need = TunnelCfg.TRAVEL_PLUNGE
		self.quota_need = 10 ** 9
		self.scene.set_gate(0.0)
		self.kite.reset(world.spawn_pos, world.spawn_dir, max(self.kite.speed, TunnelCfg.ENTRY_SPEED))
		self.kite.warp = 1.0 if self.hyper else 0.0
		self.station = self.path.station(self.kite.pos[0], self.kite.pos[2])
		self.flash = 1.0
		self.say(world.palette['name'], 'big', 3.0)
		if self.hyper:
			self.say('INFINITE FLUX  -  FULL BURN', 'good', 4.0)
		if self.plunge:
			self.say('NO WAY BACK  -  LET IT PULL YOU IN', 'good', 5.0)

	def _journey(self):
		if self.phase != 'tunnel':
			if self.finale:
				self.finale_time += self.step
				self.horizon = max(self.horizon - self.step / StoryCfg.EMERGE_TIME, 0.0)
			return
		kite = self.kite
		length = self.path.length
		station = self.path.station(kite.pos[0], kite.pos[2])
		step = (station - self.station + length * 0.5) % length - length * 0.5
		self.station = station
		if abs(step) < 120.0:
			self.travel += step
		if self.plunge:
			self.horizon = self.progress()
			kite.warp = TunnelCfg.PLUNGE_THRUST * self.horizon
		if self.travel >= self.travel_need and not self.over:
			self.arrival = True

	def _wrapped(self, delta):
		delta[..., 0] -= WorldCfg.SIZE_X * np.round(delta[..., 0] / WorldCfg.SIZE_X)
		delta[..., 2] -= WorldCfg.SIZE_Z * np.round(delta[..., 2] / WorldCfg.SIZE_Z)
		return delta

	def _nearest(self, target):
		p = self.kite.pos
		dx = target[0] - p[0]
		dz = target[2] - p[2]
		dx -= WorldCfg.SIZE_X * round(dx / WorldCfg.SIZE_X)
		dz -= WorldCfg.SIZE_Z * round(dz / WorldCfg.SIZE_Z)
		return (p[0] + dx, float(target[1]), p[2] + dz)

	def set_probe(self, probe):
		depth, cover, sea = probe
		self.aim_depth = depth
		self.aim_valid = cover > 0.5 and depth < FlightCfg.TETHER_RANGE and (sea < 0.5 or self.camera.submerged())

	def update(self, dt, controls):
		kite = self.kite
		self.time += dt
		self.step = dt
		self.before = (kite.pos[0], kite.pos[1], kite.pos[2])
		self.drifting = controls.drift
		if not self.over and self.sealed is None:
			self.duration += dt
		self.flow = max(min(self.flow + RulesCfg.FLOW_GRAZE * kite.graze * dt, 1.0) - RulesCfg.FLOW_DECAY * dt, 0.0)
		kite.slip = self.flow
		self._tether(controls)
		self._rush(dt, controls)
		events = []
		self.scrape = 0.0
		powered = not self.dry and not self.over
		if not powered:
			controls.thrust = 0.0
			controls.strafe = 0.0
		kite.grip = self._hold(dt)
		kite.step(dt, controls, powered and self.flux > 0.03, events)
		self.top_speed = max(self.top_speed, kite.speed)
		if not self.over:
			self.odometer += kite.speed * dt
		self._events(events)
		self._journey()
		self._tier()
		self._effects(dt)
		self._motes(dt)
		self._chart()
		self._rings(dt)
		self._gate()
		self._flux(dt)
		self._breathe(dt)
		if self.sealed is not None:
			self.score = self.sealed[0]
		for message in self.messages:
			message.age += dt
		self.messages = [m for m in self.messages if m.age < m.ttl]
		self.hint_time = max(self.hint_time - dt, 0.0)
		self.skip_time = max(self.skip_time - dt, 0.0)
		if self.skip_time <= 0.0:
			self.skips = 0

	def _hold(self, dt):
		rules = RulesCfg
		fading = self.dry and not self.over
		self.limp = min(max(self.limp + (rules.DRY_FADE if fading else -rules.DRY_RECOVER) * dt, 0.0), 1.0)
		slack = min(max((rules.DRY_GRIP_SPEED - self.kite.speed) / (rules.DRY_GRIP_SPEED - rules.DRY_DRIFT_SPEED), 0.0), 1.0)
		self.grip = 1.0 - self.limp * slack ** rules.DRY_CURVE
		return self.grip

	def _rush(self, dt, controls):
		kite = self.kite
		engaged = False
		self.rush_ready = False
		if self.phase == 'tunnel':
			self.rush_on = True
			kite.rush += (1.0 - kite.rush) * min(dt * 5.0, 1.0)
			return
		if self.gate_open and self.rush_charge > 0.0 and not self.over and not self.warp_requested:
			gate = self._nearest(self.world.gate_pos)
			dx, dy, dz = gate[0] - kite.pos[0], gate[1] - kite.pos[1], gate[2] - kite.pos[2]
			f = kite.forward()
			facing = (dx * f[0] + dy * f[1] + dz * f[2]) / (math.sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-6)
			self.rush_ready = facing > FlightCfg.RUSH_FACING
			engaged = self.rush_ready and controls.thrust > 0.3
		if engaged and not self.rush_on:
			self.flash = max(self.flash, 0.22)
			self.fov += 8.0
			self.shake = max(self.shake, 0.4)
			self.audio.play('start', 0.8)
			if not self.rush_told:
				self.rush_told = True
				self.say('GATE RUSH', 'big', 1.8)
		self.rush_on = engaged
		if engaged:
			self.rush_charge = max(self.rush_charge - dt, 0.0)
		kite.rush += ((1.0 if engaged else 0.0) - kite.rush) * min(dt * 5.0, 1.0)

	def _tether(self, controls):
		kite = self.kite
		if controls.tether_pressed and kite.anchor is None:
			if self.aim_valid:
				o, d, t = self.camera.pos, self.camera.forward, self.aim_depth
				kite.attach((o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t))
				self.sling_speed = kite.speed
				self.audio.play('tether_fire', 0.8)
			else:
				self.audio.play('ui_move', 0.5)
		if kite.anchor is not None and not controls.tether:
			kite.release()
			self._slingshot()
			self.audio.play('tether_release', 0.6)

	def _slingshot(self):
		gain = self.kite.speed - self.sling_speed
		if gain < RulesCfg.SLING_GAIN:
			return
		points = int(RulesCfg.SLING_SCORE * gain / RulesCfg.SLING_GAIN * self.bonus())
		self.score += points
		self.flux += RulesCfg.FLUX_SLING
		self.flux_gain = 1.0
		self.cheer('SLINGSHOT  +%d' % points, 'good', 2.0)
		self.audio.play('ribbon', 0.5)

	def _burst(self, pos, vel, color, count, spread, speed, life):
		self.scene.emit(pos, vel, color, count, spread, speed, life)

	def _events(self, events):
		kite = self.kite
		rules = RulesCfg
		for name, amount in events:
			if name == 'crash':
				self.flux -= rules.FLUX_CRASH * (0.25 if self.phase == 'tunnel' else 1.0)
				self.impacts += 1
				self.combo = 1
				self.combo_time = 0.0
				self.damage = 1.0
				self.shake = 1.0
				self.audio.play('crash', 1.0)
				self.audio.music_event('crash')
				self._burst(tuple(kite.pos), (kite.vel[0] * 0.3, kite.vel[1] * 0.3, kite.vel[2] * 0.3), (1.0, 0.6, 0.3), 220, 1.0, 26.0, 0.9)
				self.say('IMPACT', 'bad', 1.6)
			elif name == 'skip':
				self.skips += 1
				self.skip_time = 2.6
				self.flow = min(self.flow + rules.FLOW_SKIP, 1.0)
				points = int(rules.SKIP_SCORE * self.skips * self.bonus())
				self.score += points
				self.flux += rules.FLUX_SKIP
				self.flux_gain = 1.0
				self.shake = max(self.shake, 0.4)
				self.audio.play('skim_hit', min(0.45 + amount / 200.0, 1.0))
				self._burst((kite.pos[0], 0.4, kite.pos[2]), (kite.vel[0] * 0.6, 9.0, kite.vel[2] * 0.6), (0.6, 0.8, 1.0), 120, 1.2, 9.0, 0.8)
				self.cheer('SKIP x%d  +%d' % (self.skips, points), 'good', 1.6)
			elif name == 'dive':
				self.shake = max(self.shake, 0.4)
				self.flash = max(self.flash, 0.06)
				self.audio.play('skim_hit', 0.9)
				self.audio.play('tether_fire', 0.45)
				self._burst((kite.pos[0], 0.5, kite.pos[2]), (kite.vel[0] * 0.25, 16.0, kite.vel[2] * 0.25), (0.7, 0.9, 1.0), 260, 1.4, 13.0, 1.0)
				self.say('DART FORM', 'info', 1.3)
			elif name == 'breach':
				full = amount > 0.85
				points = int(rules.BREACH_SCORE * (0.5 + rules.SURGE_SCORE * amount) * self.bonus())
				self.score += points
				self.flux += rules.FLUX_BREACH * (0.4 + rules.SURGE_FLUX * amount)
				self.flux_gain = 1.0
				self.flow = min(self.flow + rules.FLOW_SURGE * amount, 1.0)
				self.shake = max(self.shake, 0.35 + 0.45 * amount)
				self.flash = max(self.flash, 0.05 + 0.2 * amount)
				self.fov += 9.0 * amount
				self.audio.play('skim_hit', 1.0)
				if amount > 0.3:
					self.audio.play('start', 0.35 + 0.45 * amount)
				self._burst((kite.pos[0], 0.5, kite.pos[2]), (kite.vel[0] * 0.5, max(kite.vel[1], 0.0) * 0.6 + 8.0, kite.vel[2] * 0.5), (0.7, 0.9, 1.0), 260 + int(340 * amount), 1.4 + amount, 12.0 + 10.0 * amount, 1.0 + 0.4 * amount)
				self.cheer(('SURGE BREACH  +%d' if full else 'BREACH  +%d') % points, 'big' if full else 'good', 2.2 if full else 1.8)
			elif name == 'scrape':
				self.shake = max(self.shake, 0.35)
				self.damage = max(self.damage, 0.3)
				self.scrape = max(self.scrape, 0.5 + 0.5 * min(amount / FlightCfg.CRASH_SPEED, 1.0))
			elif name == 'tether_auto':
				self._slingshot()
				self.audio.play('tether_release', 0.6)

	def _tier(self):
		speed = self.kite.speed
		speeds = RulesCfg.TIER_SPEEDS
		tier = self.tier
		while tier < len(speeds) and speed >= speeds[tier]:
			tier += 1
		while tier > 0 and speed < speeds[tier - 1] - 9.0:
			tier -= 1
		if tier > self.tier and not self.over:
			self.say(RulesCfg.TIER_NAMES[tier], 'good', 1.8)
			self.audio.play('ui_select', 0.8)
			self.flash = max(self.flash, 0.1)
		self.tier = tier

	def _effects(self, dt):
		kite = self.kite
		if kite.graze > 0.12:
			vel = (kite.vel[0] * 0.55, kite.vel[1] * 0.55, kite.vel[2] * 0.55)
			dust = self.world.palette['dust']
			self._burst(kite.graze_point, vel, (dust[0] * 0.9, dust[1] * 0.7, dust[2] * 0.5), 3 + int(kite.graze * 16), 0.5, 9.0, 0.5)
			self.shake = max(self.shake, kite.graze * 0.22)
		elif kite.skim > 0.25 and 0.0 <= kite.pos[1] < 5.5:
			vel = (kite.vel[0] * 0.5, 5.0, kite.vel[2] * 0.5)
			self._burst((kite.pos[0], 0.4, kite.pos[2]), vel, (0.45, 0.6, 0.75), 4 + int(kite.skim * 14), 0.8, 6.0, 0.6)
		elif kite.depth > 0.0 and kite.speed > 25.0:
			rate = min(kite.speed / 110.0, 1.4)
			vel = (kite.vel[0] * 0.3, 7.0, kite.vel[2] * 0.3)
			glow = 0.55 + 0.6 * kite.surge
			self._burst(tuple(kite.pos), vel, (0.5 * glow, 0.85 * glow, glow), 3 + int(9 * rate), 0.7, 3.0 + 3.0 * rate, 0.9)
		if kite.rush > 0.3:
			gate = self.world.palette['gate']
			self._burst(tuple(kite.pos), (kite.vel[0] * 0.35, kite.vel[1] * 0.35, kite.vel[2] * 0.35), (gate[0], gate[1], gate[2]), 10 + int(10 * kite.rush), 0.6, 11.0, 0.7)
			self.shake = max(self.shake, 0.12 * kite.rush)
		if kite.boosting:
			self.shake = max(self.shake, 0.07 if kite.depth > 0.0 else 0.16)
		self.shake = max(self.shake - dt * 2.2, 0.0)
		self.damage = max(self.damage - dt * 1.4, 0.0)
		self.flash = max(self.flash - dt * 1.3, 0.0)
		self.flux_gain = max(self.flux_gain - dt * 2.5, 0.0)
		ratio = min(kite.speed / FlightCfg.SPEED_REF, 1.3)
		drive = max(min(max((kite.speed - RenderCfg.WARP_SPEED) / RenderCfg.WARP_SPAN, 0.0), 1.0), 0.4 * kite.rush) if self.warp_enabled else 0.0
		if self.phase == 'tunnel':
			ramp = min(max((self.progress() - TunnelCfg.SURGE_START) / (1.0 - TunnelCfg.SURGE_START), 0.0), 1.0)
			self.surge = max(self.surge, ramp * ramp)
			if self.warp_enabled:
				drive = max(drive, TunnelCfg.WARP + (1.0 - TunnelCfg.WARP) * self.surge)
		else:
			self.surge = max(self.surge - dt * TunnelCfg.SURGE_DECAY, 0.0)
		self.warp += (drive - self.warp) * min(dt * 4.0, 1.0)
		target = RenderCfg.FOV_BASE + RenderCfg.FOV_KICK * min(ratio * ratio, 1.2) + (5.0 if kite.boosting else 0.0) + 6.0 * self.warp + 5.0 * kite.rush
		self.fov += (target - self.fov) * min(dt * 4.0, 1.0)

	def _collect(self, index):
		rules = RulesCfg
		kite = self.kite
		self.alive[index] = False
		ribbon = int(index) // self.ribbon_size
		self.ribbon_left[ribbon] -= 1
		self.ribbon_idle[ribbon] = 0.0
		self.combo = min(self.combo + 1, rules.COMBO_MAX)
		self.combo_time = max(self.combo_time, rules.COMBO_TIME)
		self.flow = min(self.flow + rules.FLOW_MOTE, 1.0)
		depth_bonus = rules.DEEP_BONUS if self.deep[index] else 1.0
		if self.deep[index] and kite.depth > 0.0:
			kite.surge = min(kite.surge + rules.SURGE_MOTE, 1.0)
		speed_bonus = 1.0 + 2.0 * min(kite.speed / FlightCfg.SPEED_REF, 1.5)
		self.score += rules.MOTE_SCORE * self.combo * speed_bonus * self.bonus() * depth_bonus
		self.flux += rules.FLUX_MOTE * depth_bonus
		self.flux_gain = 1.0
		self.quota += 1
		self.motes_total += 1
		picked = self.ribbon_size - int(self.ribbon_left[ribbon])
		self.audio.play_orb(min((picked - 1) % 16, 15))
		if self.ribbon_left[ribbon] > 0:
			return
		points = int(rules.RIBBON_SCORE * self.combo * self.bonus() * depth_bonus)
		self.score += points
		self.flux += rules.FLUX_RIBBON
		self.combo_time = rules.CHAIN_TIME
		self.scene.set_beacons(ribbon, 0.0)
		self.audio.play_pickup('ribbon', 0.9)
		self.say('RIBBON COMPLETE  +%d' % points, 'good', 2.4)

	def _motes(self, dt):
		if self.finale:
			return
		kite = self.kite
		rules = RulesCfg
		delta = self._wrapped(self.motes - np.asarray(kite.pos, dtype=np.float32))
		dist2 = (delta * delta).sum(1)
		dist2[~self.alive] = 1.0e12
		self.nearest_mote = int(np.argmin(dist2))
		self.nearest_dry = int(np.argmin(np.where(self.deep, 1.0e12, dist2)))
		hits = np.nonzero(self._sweep(delta, dist2) < self.reach ** 2)[0]
		if hits.size:
			self.touched = int(hits[-1])
			for index in hits:
				self._collect(int(index))
			self.scene.set_motes([int(i) for i in hits], 0.0)
			self.flash = max(self.flash, 0.035)
			mote = self.motes[int(hits[-1])]
			color = self.world.palette['mote']
			self._burst(self._nearest(mote), (kite.vel[0] * 0.7, kite.vel[1] * 0.7, kite.vel[2] * 0.7), (color[0] * 0.8, color[1] * 0.8, color[2] * 0.8), 30, 0.6, 12.0, 0.5)
			if self.quota >= self.quota_need:
				self.open_gate()
		touched = self.ribbon_left < self.ribbon_size
		self.ribbon_idle[touched] += dt
		for ribbon in np.nonzero(self.ribbon_idle > self.respawn)[0]:
			a = int(ribbon) * self.ribbon_size
			b = a + self.ribbon_size
			self.alive[a:b] = True
			self.ribbon_left[ribbon] = self.ribbon_size
			self.ribbon_idle[ribbon] = 0.0
			self.scene.set_motes(list(range(a, b)), 1.0)
			self.scene.set_beacons(int(ribbon), 1.0)
		if self.combo > 1:
			self.combo_time -= dt * (0.45 if kite.graze > 0.2 else 1.0)
			if self.combo_time <= 0.0:
				self.combo = 1
		if kite.graze > 0.0:
			self.score += rules.GRAZE_SCORE * kite.graze * self.combo * self.bonus() * dt

	def open_gate(self):
		if self.phase != 'sector' or self.gate_open:
			return
		kite = self.kite
		self.gate_open = True
		self.scene.set_gate(1.0)
		self.audio.play('gate_open', 1.0)
		self.audio.music_event('gate')
		self.say('GATE OPEN', 'big', 3.2)
		self.say('RUSH READY  -  FACE THE GATE AND THRUST', 'good', 4.0)
		self.rush_charge = RulesCfg.RUSH_TIME
		self.rush_told = False
		self.flash = max(self.flash, 0.4)
		self.fov += 9.0
		gate = self.world.palette['gate']
		self._burst(tuple(kite.pos), (kite.vel[0] * 0.5, kite.vel[1] * 0.5, kite.vel[2] * 0.5), (gate[0], gate[1], gate[2]), 420, 2.0, 34.0, 1.2)

	def _chart(self):
		kite = self.kite
		pos = np.asarray(kite.pos, dtype=np.float64)
		vel = np.asarray(kite.vel, dtype=np.float64)
		if self.touched >= 0:
			self.route.adopt(self.touched, vel)
			self.touched = -1
		self.route.advance(pos, vel, kite.forward(), self.course())
		self.links = self.route.links(pos, self.alive) if self.phase == 'sector' and not self.gate_open else []

	def _sweep(self, delta, dist2):
		if self.before is None:
			return dist2
		kite = self.kite
		back = np.asarray((self.before[0] - kite.pos[0], self.before[1] - kite.pos[1], self.before[2] - kite.pos[2]), dtype=np.float32)
		back[0] -= WorldCfg.SIZE_X * round(float(back[0]) / WorldCfg.SIZE_X)
		back[2] -= WorldCfg.SIZE_Z * round(float(back[2]) / WorldCfg.SIZE_Z)
		span = float(back @ back)
		if span < 1.0 or span > 3600.0:
			return dist2
		along = np.clip((delta @ back) / span, 0.0, 1.0)
		gap = delta - along[:, None] * back
		swept = (gap * gap).sum(1)
		swept[~self.alive] = 1.0e12
		return swept

	def _rings(self, dt):
		kite = self.kite
		rules = RulesCfg
		offset = -self._wrapped(self.rings_pos - np.asarray(kite.pos, dtype=np.float32))
		side = (offset * self.rings_dir).sum(1)
		lateral2 = (offset * offset).sum(1) - side * side
		previous = self.ring_side
		crossed = (side * previous < 0.0) & (np.abs(side) < 30.0) & (np.abs(previous) < 30.0) & (lateral2 < (WorldCfg.RING_RADIUS * 1.12) ** 2) & (self.ring_cool <= 0.0)
		self.ring_side = side
		cooling = self.ring_cool > 0.0
		if cooling.any():
			self.ring_cool[cooling] -= dt
			for ring in np.nonzero(cooling & (self.ring_cool <= 0.0))[0]:
				self.scene.set_ring(int(ring), 1.0)
		for ring in np.nonzero(crossed)[0]:
			self.ring_cool[ring] = rules.RING_COOLDOWN
			self.flow = min(self.flow + rules.FLOW_RING, 1.0)
			self.scene.set_ring(int(ring), 0.1)
			speed = max(kite.speed, 1.0)
			scale = min(speed + rules.RING_BOOST, max(speed, FlightCfg.SPEED_MAX)) / speed
			kite.vel[0] *= scale
			kite.vel[1] *= scale
			kite.vel[2] *= scale
			kite.speed = speed * scale
			self.combo = min(self.combo + 1, rules.COMBO_MAX)
			self.combo_time = max(self.combo_time, rules.COMBO_TIME)
			points = int(rules.RING_SCORE * self.combo * self.bonus())
			self.score += points
			self.flux += rules.FLUX_RING
			self.flux_gain = 1.0
			self.flash = max(self.flash, 0.16)
			self.fov += 7.0
			self.shake = max(self.shake, 0.3)
			color = self.world.palette['gate']
			self._burst(tuple(kite.pos), (kite.vel[0] * 0.8, kite.vel[1] * 0.8, kite.vel[2] * 0.8), (color[0], color[1], color[2]), 90, 1.5, 16.0, 0.6)
			self.audio.play('start', 0.55)
			self.say('RING  +%d' % points, 'good', 1.5)

	def _gate(self):
		if self.phase != 'sector' or not self.gate_open or self.warp_requested or self.over:
			return
		kite = self.kite
		center = self._nearest(self.world.gate_pos)
		rx = kite.pos[0] - center[0]
		ry = kite.pos[1] - center[1]
		rz = kite.pos[2] - center[2]
		reach = WorldCfg.GATE_RADIUS * WorldCfg.GATE_CAPTURE
		if rx * rx + ry * ry + rz * rz < reach * reach:
			self.warp_requested = True

	def complete_warp(self):
		self.sector += 1
		self.score += RulesCfg.WARP_SCORE * self.sector
		self.flux = max(self.flux, TunnelCfg.FLUX_FLOOR)
		self.flux_gain = 1.0
		self.flash = 1.0
		self.shake = max(self.shake, 0.7)

	def _flux(self, dt):
		if self.attract or self.over:
			self.flux = min(max(self.flux, 0.5), 1.0)
			self.dry = False
			self.stall = 0.0
			return
		if self.finale:
			self.flux = 1.0
			self.dry = False
			self.stall = 0.0
			return
		kite = self.kite
		rules = RulesCfg
		gain = rules.FLUX_GRAZE * kite.graze * (0.0 if self.scrape > 0.0 else 1.0) + rules.FLUX_SKIM * kite.skim
		if self.phase == 'tunnel':
			endless = self.hyper or self.plunge
			self.dry = False
			self.stall = 0.0
			drain = rules.FLUX_BOOST if kite.boosting and not endless else 0.0
			self.flux = 1.0 if endless else min(max(self.flux + (gain - drain) * dt, 0.04), 1.0)
			return
		drain = 0.0 if self.dry else self.drain()
		drain += rules.FLUX_SCRAPE * self.scrape
		if kite.boosting:
			drain += rules.FLUX_BOOST
		if gain > 0.02:
			self.flux_gain = max(self.flux_gain, 0.5)
		self.flux = min(max(self.flux + (gain - drain) * dt, 0.0), 1.0)
		if not self.dry and self.flux <= 0.0:
			self.dry = True
			self.audio.music_event('crash')
			self.say('FLUX DEPLETED', 'bad', 2.6)
		elif self.dry and self.flux >= rules.FLUX_RELIGHT:
			self.dry = False
			self.flash = max(self.flash, 0.12)
			self.audio.play('start', 0.7)
			self.say('FLUX RELIT', 'good', 2.0)
		self._strand(dt)

	def _breathe(self, dt):
		rules = RulesCfg
		if not self.dry or self.over or self.attract:
			self.breath = rules.BREATH_PERIOD
			self.alarm = max(self.alarm - dt * 2.0, 0.0)
			return
		self.breath += dt
		if self.breath >= rules.BREATH_PERIOD:
			self.breath = 0.0
			self.audio.play('flux_dry', rules.BREATH_VOLUME)
		swell = math.sin(math.pi * min(self.breath / rules.BREATH_SWELL, 1.0)) ** 2
		target = rules.BREATH_FLOOR + (1.0 - rules.BREATH_FLOOR) * swell
		self.alarm += (target - self.alarm) * min(dt * 10.0, 1.0)

	def _strand(self, dt):
		kite = self.kite
		rules = RulesCfg
		stuck = self.dry and kite.speed < rules.STALL_SPEED
		if stuck and kite.pos[1] > rules.STALL_HEIGHT:
			stuck = self.field.sample(kite.pos[0], kite.pos[1], kite.pos[2])[0] < FlightCfg.RADIUS + rules.STALL_GROUND
		if not stuck:
			self.stall = max(self.stall - dt * 2.0, 0.0)
			return
		self.stall = min(self.stall + dt / rules.STALL_TIME, 1.0)
		if self.stall < 1.0:
			return
		self.over = True
		kite.release()
		self.audio.play('game_over', 1.0)
		self.audio.music_event('over')

	def summary(self, best):
		out = Summary()
		out.score = int(self.score)
		out.best = max(int(best), out.score)
		out.new_best = out.score > int(best)
		out.sector = self.sector + 1
		out.motes = self.motes_total
		out.top_speed = self.top_speed
		out.duration = self.duration
		out.mode = self.mode.KEY
		out.unit = self.mode.UNIT
		out.levels = self.mode.LEVELS
		out.timed = self.mode.TIMED
		out.completed = self.sealed is not None
		if out.levels:
			out.sector = min(out.sector, out.levels)
		if self.career is not None:
			out.deepest = self.career.deepest()
			out.rank = self.career.rank()
			out.frontier = self.career.frontier(out.sector)
		return out

	def build_view(self, dt, time):
		kite = self.kite
		camera = self.camera
		heat = max(self.tier / 3.0, kite.rush)
		camera.update(kite, self.field, self.shake * self.shake_scale, dt, kite.thrusting * 0.5 + (0.5 if kite.boosting else 0.0) + kite.rush, heat)
		view = View()
		view.pos = camera.pos
		view.right = camera.right
		view.up = camera.up
		view.forward = camera.forward
		view.velocity = tuple(kite.vel)
		view.fov = self.fov
		view.time = time
		view.dt = dt
		view.rush = min(max((kite.speed - 75.0) / 120.0, 0.0), 1.0) * 0.85 + (0.15 if kite.boosting else 0.0)
		view.warp = self.warp
		view.surge = self.surge * self.surge * (3.0 - 2.0 * self.surge) if self.warp_enabled else 0.0
		view.flash = self.flash
		view.damage = self.damage
		view.persistence = min(math.exp(-dt / 0.0065), 0.5)
		camera.craft['alarm'] = self.alarm
		view.craft = camera.craft
		view.submerged = camera.submerged()
		if view.submerged:
			view.tint = (0.72, 0.98, 1.12)
		else:
			gate = self.world.palette['gate']
			glow = 0.3 * kite.rush
			view.tint = (1.0 + 0.05 * self.tier - 0.1 * self.warp + glow * (gate[0] - 0.6), 1.0 + 0.02 * self.tier + glow * (gate[1] - 0.6), 1.0 + 0.16 * self.warp + glow * (gate[2] - 0.6))
		if kite.anchor is not None:
			view.tether = (camera.hand, self._nearest(kite.anchor))
		view.route = self.links
		if self.phase == 'tunnel':
			view.tunnel = (self.progress(), self.hyper)
		if self.horizon > 0.0:
			view.horizon = self._horizon()
		if self.gate_open:
			view.wormhole = (self._nearest(self.world.gate_pos), WorldCfg.GATE_RADIUS * WorldCfg.GATE_LENS, self.world.palette['gate'])
		return view

	def _horizon(self):
		if not self.plunge:
			return ('rise', self.horizon, (0.0, 0.0), self.warp_enabled)
		camera = self.camera
		spot = self._nearest(self.path.axis(self.station + StoryCfg.FALL_LEAD))
		rx, ry, rz = spot[0] - camera.pos[0], spot[1] - camera.pos[1], spot[2] - camera.pos[2]
		depth = rx * camera.forward[0] + ry * camera.forward[1] + rz * camera.forward[2]
		if depth < 1.0:
			return ('fall', self.horizon, (0.0, 0.0), self.warp_enabled)
		hold = 1.0 - self.horizon ** 3
		across = (rx * camera.right[0] + ry * camera.right[1] + rz * camera.right[2]) / depth * hold
		above = (rx * camera.up[0] + ry * camera.up[1] + rz * camera.up[2]) / depth * hold
		return ('fall', self.horizon, (across, above), self.warp_enabled)

	def _project(self, target, size, marker):
		camera = self.camera
		right, up, f = camera.right, camera.up, camera.forward
		rx = target[0] - camera.pos[0]
		ry = target[1] - camera.pos[1]
		rz = target[2] - camera.pos[2]
		rx -= WorldCfg.SIZE_X * round(rx / WorldCfg.SIZE_X)
		rz -= WorldCfg.SIZE_Z * round(rz / WorldCfg.SIZE_Z)
		x = rx * right[0] + ry * right[1] + rz * right[2]
		y = rx * up[0] + ry * up[1] + rz * up[2]
		z = rx * f[0] + ry * f[1] + rz * f[2]
		w, h = size
		focal = 0.5 * w / math.tan(math.radians(self.fov) * 0.5)
		margin = h * 0.07
		marker.dist = math.sqrt(rx * rx + ry * ry + rz * rz)
		marker.angle = math.atan2(-y, x)
		if z > 1.0:
			sx = w * 0.5 + x / z * focal
			sy = h * 0.5 - y / z * focal
			if margin < sx < w - margin and margin < sy < h - margin:
				marker.x = sx
				marker.y = sy
				marker.onscreen = True
				return marker
		marker.onscreen = False
		cx = math.cos(marker.angle)
		cy = math.sin(marker.angle)
		reach = min((w * 0.5 - margin) / max(abs(cx), 1.0e-4), (h * 0.5 - margin) / max(abs(cy), 1.0e-4))
		marker.x = w * 0.5 + cx * reach
		marker.y = h * 0.5 + cy * reach
		return marker

	def hud_state(self, size, perf):
		kite = self.kite
		hs = HudState()
		hs.time = self.time
		hs.speed = kite.speed
		hs.flux = self.flux
		hs.danger = min(max((0.25 - self.flux) / 0.2, 0.0), 1.0)
		hs.flux_gain = self.flux_gain
		hs.score = int(self.score)
		hs.combo = self.combo
		hs.combo_time = min(max(self.combo_time / RulesCfg.COMBO_TIME, 0.0), 1.0)
		hs.sector = self.sector + 1
		hs.quota = min(self.quota, self.quota_need)
		hs.quota_need = self.quota_need
		hs.gate_open = self.gate_open
		hs.boost = kite.boosting
		hs.drift = self.drifting or self.grip < 0.5
		hs.grip = self.grip
		hs.graze = max(kite.graze, self.flow * 0.85)
		hs.tether_state = 2 if kite.anchor is not None else (1 if self.aim_valid else 0)
		hs.messages = self.messages
		hs.perf = perf
		hs.tier = self.tier
		hs.tier_name = RulesCfg.TIER_NAMES[self.tier]
		hs.depth = kite.depth
		hs.surge = kite.surge
		hs.rush = self.rush_charge / RulesCfg.RUSH_TIME if self.gate_open else 0.0
		hs.rush_ready = self.rush_ready
		hs.rush_on = self.rush_on
		hs.dry = self.dry and not self.over
		hs.stall = self.stall
		hs.hyper = self.hyper
		hs.finale = self.finale
		hs.stage = 'THE SINGULARITY' if self.finale else self.mode.stage(self.sector)
		if self.mode.TIMED:
			hs.clock = self.sealed[1] if self.sealed is not None else self.duration
		if self.phase == 'tunnel':
			hs.tunnel = self.progress()
			hs.stage = '%s %d  >  %s' % (self.mode.UNIT, self.sector + 1, '?' if self.plunge else str(self.sector + 2))
			hs.plunge = self.plunge
		if kite.speed > 8.0:
			ahead = (kite.pos[0] + kite.vel[0] * 4.0, kite.pos[1] + kite.vel[1] * 4.0, kite.pos[2] + kite.vel[2] * 4.0)
			marker = self._project(ahead, size, Marker('velocity'))
			if marker.onscreen:
				hs.vel_marker = (marker.x, marker.y)
		if self.gate_open:
			hs.markers.append(self._project(self.world.gate_pos, size, Marker('gate')))
		if self.phase == 'sector' and not self.gate_open:
			route = self.route
			if route.ribbon >= 0 and route.ready is not None and route.ready.any():
				hits = np.nonzero(route.ready)[0]
				spot = route.lanes[route.ribbon, int(hits[0]) if route.direction > 0 else int(hits[-1])]
				marker = self._project(spot, size, Marker('ribbon'))
				if marker.dist > 45.0:
					hs.markers.append(marker)
			if self.links:
				marker = self._project(self.links[0][1], size, Marker('next'))
				if marker.dist > 45.0:
					hs.markers.append(marker)
		if self.hint_time > 0.0 and self.hint_text:
			hs.hint = self.hint_text
		return hs

	def mix(self, menu, paused):
		kite = self.kite
		mix = Mix()
		wet = kite.depth > 0.0
		mix.speed = min(kite.speed / FlightCfg.SPEED_REF * 1.15, 1.5) * (0.35 if wet else 1.0)
		mix.thrust = kite.thrusting
		mix.boost = 1.0 if kite.boosting or kite.rush > 0.5 else 0.0
		mix.graze = kite.graze
		mix.tether = kite.tension
		mix.skim = max(kite.skim if 0.0 <= kite.pos[1] < 7.0 else 0.0, min(0.3 + kite.speed / 190.0, 1.0) if wet and kite.speed > 20.0 else 0.0)
		mix.intensity = min(kite.speed / 150.0 * 0.6 + min(self.combo, 16) / 16.0 * 0.3 + self.tier * 0.12, 1.0) * (0.3 if self.dry else 1.0)
		mix.danger = 0.0 if (self.attract or self.over) else min(max((0.25 - self.flux) / 0.2, 0.0), 1.0)
		mix.menu = 1.0 if menu else 0.0
		mix.paused = 1.0 if paused else 0.0
		mix.submerged = 1.0 if self.camera.submerged() else 0.0
		mix.serene = 1.0 if self.finale else 0.0
		mix.dry = 1.0 if self.dry and not self.over and not self.attract else 0.0
		return mix
