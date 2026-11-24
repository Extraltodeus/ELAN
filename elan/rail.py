import math

from elan.config import FlightCfg, TunnelCfg, WorldCfg


class TunnelRail:
	def __init__(self, path):
		self.path = path
		self.limit = path.radius * TunnelCfg.RAIL_LIMIT
		self.station = 0.0
		self.heading = (0.0, 0.0, 1.0)
		self.offset = (0.0, 0.0, 0.0)
		self.aim = None

	def _turn(self, angle):
		return (angle + math.pi) % math.tau - math.pi

	def locate(self, pos):
		path = self.path
		s = path.station(pos[0], pos[2])
		axis = path.axis(s)
		h = path.heading(s)
		dx = pos[0] - axis[0]
		dx -= WorldCfg.SIZE_X * round(dx / WorldCfg.SIZE_X)
		dy = pos[1] - axis[1]
		along = dx * h[0] + dy * h[1]
		self.station = s
		self.heading = h
		self.offset = (dx - h[0] * along, dy - h[1] * along, -h[2] * along)
		return s

	def assist(self, kite, dt):
		cfg = TunnelCfg
		self.locate(kite.pos)
		h = self.heading
		o = self.offset
		lead = cfg.RAIL_LEAD + cfg.RAIL_PREVIEW * kite.speed
		dx = h[0] - o[0] / lead
		dy = h[1] - o[1] / lead
		dz = h[2] - o[2] / lead
		yaw = math.atan2(dx, dz)
		pitch = math.atan2(dy, math.sqrt(dx * dx + dz * dz))
		turn = 0.0
		tilt = 0.0
		if self.aim is not None:
			turn = self._turn(yaw - self.aim[0])
			tilt = pitch - self.aim[1]
		self.aim = (yaw, pitch)
		blend = min(dt * cfg.RAIL_STEER, 1.0)
		turn += self._turn(yaw - kite.yaw - turn) * blend
		tilt += (pitch - kite.pitch - tilt) * blend
		return turn, -tilt

	def confine(self, kite):
		cone = TunnelCfg.RAIL_CONE
		h = self.heading
		along = math.atan2(h[0], h[2])
		climb = math.asin(max(-1.0, min(1.0, h[1])))
		kite.yaw = along + max(-cone, min(cone, self._turn(kite.yaw - along)))
		kite.pitch = climb + max(-cone, min(cone, kite.pitch - climb))

	def pull(self, kite):
		cfg = TunnelCfg
		path = self.path
		s = self.locate(kite.pos)
		o = self.offset
		h = self.heading
		v = kite.vel
		along = v[0] * h[0] + v[1] * h[1] + v[2] * h[2]
		reach = math.sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]) / self.limit
		stiff = cfg.RAIL_PULL * (1.0 + cfg.RAIL_STIFFEN * reach * reach)
		damp = cfg.RAIL_DAMP
		pace = v[2] * v[2]
		bx = path.curve(path.sway, s) * pace
		by = path.curve(path.heave, s) * pace
		lean = bx * h[0] + by * h[1]
		return (
			bx - h[0] * lean - o[0] * stiff - (v[0] - h[0] * along) * damp,
			by - h[1] * lean - o[1] * stiff - (v[1] - h[1] * along) * damp + FlightCfg.GRAVITY,
			-h[2] * lean - o[2] * stiff - (v[2] - h[2] * along) * damp,
		)

	def hold(self, kite):
		self.locate(kite.pos)
		p = kite.pos
		v = kite.vel
		o = self.offset
		h = self.heading
		dist = math.sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
		if dist > self.limit:
			nx, ny, nz = o[0] / dist, o[1] / dist, o[2] / dist
			excess = dist - self.limit
			p[0] -= nx * excess
			p[1] -= ny * excess
			p[2] -= nz * excess
			out = v[0] * nx + v[1] * ny + v[2] * nz
			if out > 0.0:
				v[0] -= nx * out
				v[1] -= ny * out
				v[2] -= nz * out
		along = v[0] * h[0] + v[1] * h[1] + v[2] * h[2]
		if along >= TunnelCfg.RAIL_SPEED:
			return
		gap = TunnelCfg.RAIL_SPEED - along
		v[0] += h[0] * gap
		v[1] += h[1] * gap
		v[2] += h[2] * gap
