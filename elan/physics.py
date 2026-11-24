import math

from elan.config import FlightCfg, TunnelCfg, WorldCfg


class Controls:
	def __init__(self):
		self.look_x = 0.0
		self.look_y = 0.0
		self.thrust = 0.0
		self.brake = 0.0
		self.strafe = 0.0
		self.boost = False
		self.tether = False
		self.tether_pressed = False
		self.drift = False


class Kite:
	def __init__(self):
		self.field = None
		self.pos = [0.0, 120.0, 0.0]
		self.vel = [0.0, 0.0, 0.0]
		self.yaw = 0.0
		self.pitch = 0.0
		self.roll = 0.0
		self.turn = 0.0
		self.speed = 0.0
		self.anchor = None
		self.rope = 0.0
		self.tension = 0.0
		self.graze = 0.0
		self.graze_point = (0.0, 0.0, 0.0)
		self.skim = 0.0
		self.boosting = False
		self.thrusting = 0.0
		self.load = 0.0
		self.taut = 0.0
		self.depth = 0.0
		self.slip = 0.0
		self.skip_cool = 0.0
		self.deepest = 0.0
		self.surge = 0.0
		self.form = 0.0
		self.folded = False
		self.reel = 1.0
		self.rush = 0.0
		self.warp = 0.0
		self.rail = None
		self.grip = 1.0

	def reset(self, pos, direction, speed):
		self.pos = [float(pos[0]), float(pos[1]), float(pos[2])]
		self.yaw = math.atan2(direction[0], direction[2])
		self.pitch = math.asin(max(-1.0, min(1.0, float(direction[1]))))
		forward = self.forward()
		self.vel = [forward[0] * speed, forward[1] * speed, forward[2] * speed]
		self.roll = 0.0
		self.turn = 0.0
		self.speed = speed
		self.anchor = None
		self.tension = 0.0
		self.graze = 0.0
		self.skim = 0.0
		self.boosting = False
		self.load = 0.0
		self.depth = max(-self.pos[1], 0.0)
		self.deepest = self.depth
		self.skip_cool = 0.0
		self.surge = 0.0
		self.rush = 0.0
		self.folded = self.pos[1] < 0.0
		self.form = 1.0 if self.folded else 0.0

	def forward(self):
		cp = math.cos(self.pitch)
		return (math.sin(self.yaw) * cp, math.sin(self.pitch), math.cos(self.yaw) * cp)

	def axes(self):
		f = self.forward()
		r = (math.cos(self.yaw), 0.0, -math.sin(self.yaw))
		u = (f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0])
		return r, u, f

	def rope_vector(self):
		dx = self.anchor[0] - self.pos[0]
		dy = self.anchor[1] - self.pos[1]
		dz = self.anchor[2] - self.pos[2]
		dx -= WorldCfg.SIZE_X * round(dx / WorldCfg.SIZE_X)
		dz -= WorldCfg.SIZE_Z * round(dz / WorldCfg.SIZE_Z)
		return dx, dy, dz

	def attach(self, anchor):
		self.anchor = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
		dx, dy, dz = self.rope_vector()
		self.rope = math.sqrt(dx * dx + dy * dy + dz * dz) * 0.94

	def release(self):
		self.anchor = None
		self.tension = 0.0

	def look(self, dx, dy, dt):
		self.yaw += dx
		self.pitch = max(-1.52, min(1.52, self.pitch - dy))
		rate = dx / max(dt, 1.0e-4)
		blend = min(dt * 9.0, 1.0)
		self.turn += (rate - self.turn) * blend

	def step(self, dt, controls, boost_ok, events):
		cfg = FlightCfg
		rail = self.rail
		if rail is None:
			self.look(controls.look_x, controls.look_y, dt)
		else:
			turn, tilt = rail.assist(self, dt)
			self.look(controls.look_x + turn, controls.look_y + tilt, dt)
			rail.confine(self)
		count = max(1, min(cfg.MAX_SUBSTEPS, int(math.ceil(dt / cfg.SUBSTEP))))
		h = dt / count
		self.boosting = bool(controls.boost and boost_ok)
		self.thrusting = controls.thrust
		self.graze = 0.0
		self.skim = 0.0
		self.tension = 0.0
		self.load = 0.0
		if self.anchor is None:
			self.taut = 0.0
		r, u, f = self.axes()
		for _ in range(count):
			self._substep(h, controls, r, f, events)
		self.pos[0] %= WorldCfg.SIZE_X
		self.pos[2] %= WorldCfg.SIZE_Z
		v = self.vel
		self.speed = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
		lean = self.turn * self.speed * cfg.ROLL_GAIN + controls.strafe * 0.22
		lean = max(-cfg.ROLL_MAX, min(cfg.ROLL_MAX, lean))
		self.roll += (lean - self.roll) * min(dt * 6.0, 1.0)
		wet = self.pos[1] < 0.0
		if wet:
			reach = 0.35 + 0.65 * min(self.depth / cfg.SURGE_DEPTH, 1.0)
			self.surge = min(self.surge + self.speed * reach * dt / cfg.SURGE_DISTANCE, 1.0)
		else:
			self.surge = max(self.surge - cfg.SURGE_DECAY * dt, 0.0)
		if self.pos[1] < -1.0:
			self.folded = True
		elif self.pos[1] > 0.5:
			self.folded = False
		self.form += ((1.0 if self.folded else 0.0) - self.form) * min(dt * cfg.MORPH_RATE, 1.0)

	def _substep(self, h, controls, r, f, events):
		cfg = FlightCfg
		p = self.pos
		v = self.vel
		speed = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
		if speed > 1.0e-3:
			hx, hy, hz = v[0] / speed, v[1] / speed, v[2] / speed
		else:
			hx, hy, hz = f
		wet = p[1] < 0.0
		before = p[1]
		thrust = cfg.THRUST * controls.thrust + (cfg.BOOST_THRUST if self.boosting else 0.0) + cfg.RUSH_THRUST * self.rush + TunnelCfg.HYPER_THRUST * self.warp
		ceiling = (abs(p[1]) - (WorldCfg.CEILING - cfg.CEILING_SOFT)) / cfg.CEILING_SOFT
		if ceiling > 0.0:
			thrust *= max(1.0 - ceiling, 0.0)
		ax = f[0] * thrust + r[0] * cfg.STRAFE * controls.strafe
		ay = f[1] * thrust + (cfg.BUOYANCY if wet else -cfg.GRAVITY)
		az = f[2] * thrust + r[2] * cfg.STRAFE * controls.strafe
		if ceiling > 0.0:
			ay += ceiling * (40.0 if wet else -40.0)
		if self.rail is not None:
			gx, gy, gz = self.rail.pull(self)
			ax += gx
			ay += gy
			az += gz

		along = f[0] * hx + f[1] * hy + f[2] * hz
		lx, ly, lz = f[0] - hx * along, f[1] - hy * along, f[2] - hz * along
		bite = min(cfg.DRIFT_LIFT if controls.drift else 1.0, cfg.DRIFT_LIFT + (1.0 - cfg.DRIFT_LIFT) * self.grip)
		gain = cfg.LIFT_GAIN * speed * min(speed / cfg.LIFT_STALL, 1.0) ** 2 * bite * (1.0 - 0.88 * self.taut) * (cfg.WATER_GRIP if wet else 1.0)
		lx *= gain
		ly *= gain
		lz *= gain
		lift = math.sqrt(lx * lx + ly * ly + lz * lz)
		limit = cfg.LIFT_MAX * (cfg.WATER_LIFT if wet else 1.0)
		if lift > limit:
			scale = limit / lift
			lx *= scale
			ly *= scale
			lz *= scale
			lift = limit
		self.load = max(self.load, lift / limit)
		sleek = cfg.WATER_DRAG + (1.0 - cfg.WATER_DRAG) * self.slip if wet else 1.0
		drag = cfg.DRAG * speed * speed * sleek * (1.0 - 0.45 * self.graze) * (1.0 - cfg.SLIP_DRAG * self.slip) * (1.0 - cfg.RUSH_DRAG * self.rush) + cfg.BRAKE_DRAG * speed * speed * controls.brake + cfg.INDUCED_DRAG * lift * (cfg.WATER_INDUCED if wet else 1.0)
		ax += lx - hx * drag
		ay += ly - hy * drag
		az += lz - hz * drag

		if self.anchor is not None:
			dx, dy, dz = self.rope_vector()
			dist = math.sqrt(dx * dx + dy * dy + dz * dz)
			if dist < cfg.TETHER_MIN * 0.8:
				self.release()
				events.append(('tether_auto', 0.0))
			else:
				self.rope = max(self.rope - cfg.TETHER_REEL * self.reel * h, cfg.TETHER_MIN)
				nx, ny, nz = dx / dist, dy / dist, dz / dist
				stretch = dist - self.rope
				if stretch > 0.0:
					away = -(v[0] * nx + v[1] * ny + v[2] * nz)
					pull = min(cfg.TETHER_STIFF * stretch + cfg.TETHER_DAMP * max(away, 0.0), cfg.TETHER_MAX)
					ax += nx * pull
					ay += ny * pull
					az += nz * pull
					self.tension = max(self.tension, pull / cfg.TETHER_MAX)
					self.taut += (pull / cfg.TETHER_MAX - self.taut) * min(h * 12.0, 1.0)
				else:
					self.taut += (0.0 - self.taut) * min(h * 12.0, 1.0)

		v[0] += ax * h
		v[1] += ay * h
		v[2] += az * h
		p[0] += v[0] * h
		p[1] += v[1] * h
		p[2] += v[2] * h
		if self.rail is not None:
			self.rail.hold(self)
		self.skip_cool = max(self.skip_cool - h, 0.0)
		self._surface(before, events)
		self._rock(h, events)

	def _surface(self, before, events):
		cfg = FlightCfg
		p = self.pos
		v = self.vel
		if 0.0 <= p[1] < 7.0:
			flat = math.sqrt(v[0] * v[0] + v[2] * v[2])
			self.skim = max(self.skim, (1.0 - p[1] / 7.0) * min(max(flat - cfg.SKIM_MIN_SPEED, 0.0) / 45.0, 1.0))
		self.depth = max(-p[1], 0.0)
		self.deepest = max(self.deepest, self.depth) if p[1] < 0.0 else self.deepest
		if before >= 0.0 and p[1] < 0.0:
			speed = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
			sink = -v[1]
			angle = max(math.asin(max(0.0, min(sink / max(speed, 1.0e-3), 1.0))), -self.pitch * 0.75)
			if speed > cfg.SKIP_SPEED and angle < cfg.SKIP_ANGLE:
				p[1] = 0.02
				v[1] = sink * cfg.SKIP_BOUNCE + 1.0
				if sink > 3.0 and self.skip_cool <= 0.0:
					self.skip_cool = cfg.SKIP_COOLDOWN
					flat = math.sqrt(v[0] * v[0] + v[2] * v[2])
					scale = (flat + cfg.SKIP_GAIN) / max(flat, 1.0)
					v[0] *= scale
					v[2] *= scale
					events.append(('skip', speed))
				return
			self.deepest = 0.0
			if speed > cfg.SPLASH_SPEED and sink > cfg.DIVE_SINK and self.skip_cool <= 0.0:
				self.skip_cool = cfg.SKIP_COOLDOWN
				events.append(('dive', speed))
			return
		if before < 0.0 and p[1] >= 0.0:
			speed = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
			if speed < cfg.SPLASH_SPEED or self.deepest < cfg.BREACH_DEPTH:
				return
			charge = self.surge
			self.surge = 0.0
			self.deepest = 0.0
			scale = min(speed + cfg.BREACH_KICK + cfg.SURGE_KICK * charge, max(speed, cfg.SPEED_MAX)) / speed
			v[0] *= scale
			v[1] = v[1] * scale + 2.0 + cfg.SURGE_LIFT * charge
			v[2] *= scale
			events.append(('breach', charge))

	def _rock(self, h, events):
		cfg = FlightCfg
		p = self.pos
		v = self.vel
		dist, gx, gy, gz = self.field.sample(p[0], p[1], p[2])
		norm = math.sqrt(gx * gx + gy * gy + gz * gz)
		if norm < 1.0e-5:
			gx, gy, gz, norm = 0.0, 1.0, 0.0, 1.0
		nx, ny, nz = gx / norm, gy / norm, gz / norm
		gap = dist - cfg.RADIUS
		speed = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
		if gap < cfg.GRAZE_RANGE and speed > cfg.GRAZE_MIN_SPEED:
			amount = (1.0 - max(gap, 0.0) / cfg.GRAZE_RANGE) * min((speed - cfg.GRAZE_MIN_SPEED) / 50.0, 1.0)
			if amount > self.graze:
				self.graze = amount
				reach = max(dist, 0.0)
				self.graze_point = (p[0] - nx * reach, p[1] - ny * reach, p[2] - nz * reach)
		if gap >= 0.0:
			return
		p[0] -= nx * gap
		p[1] -= ny * gap
		p[2] -= nz * gap
		into = v[0] * nx + v[1] * ny + v[2] * nz
		if into >= 0.0:
			return
		impact = -into
		if impact > cfg.CRASH_SPEED:
			bounce = into * (1.0 + cfg.RESTITUTION)
			v[0] = (v[0] - nx * bounce) * 0.62
			v[1] = (v[1] - ny * bounce) * 0.62
			v[2] = (v[2] - nz * bounce) * 0.62
			events.append(('crash', impact))
			return
		keep = max(1.0 - 0.5 * h, 0.0)
		v[0] = (v[0] - nx * into) * keep
		v[1] = (v[1] - ny * into) * keep
		v[2] = (v[2] - nz * into) * keep
		events.append(('scrape', impact))
