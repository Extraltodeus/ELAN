import math
import torch

from elan.ai.observe import Observer
from elan.ai.shield import BatchShield
from elan.config import FlightCfg, RulesCfg, TunnelCfg, WorldCfg


class Reward:
	MOTE = 1.0
	MOTE_SPEED = 120.0
	RIBBON = 1.0
	RING = 1.0
	ENTRY = 2.0
	GATE = 10.0
	SPEED = 1.5
	SPEED_CAP = 9.0
	CURVE = 2.0
	PROGRESS = 0.03
	CRASH = 6.0
	SCRAPE = 0.3
	NEAR = 0.03
	JERK = 0.15
	TURN = 0.01
	THROTTLE = 0.1
	SLOW = 0.06
	CRUISE = 60.0
	STALL = 8.0
	DRY = 0.6
	RUSH = 1.0
	SHIELD = 0.05
	WINDOW = 0.0


class FlightSim:
	STEP = 1.0 / 30.0
	SUBSTEPS = 4
	TURN_MAX = 2.4
	PITCH_MAX = 2.0
	EASE = 0.1
	EPISODE = 60.0
	QUOTA = (25, 80)
	SECTORS = 26
	SAFE = 0.2
	RESERVE = 0.25
	ENDURANCE = 14.0
	APPROACH = (320.0, 0.75, 60.0, 0.25, 0.3, 90.0)
	CAUTION = (0.35, 8.0, 0.8, 6.0, 0.3, 60.0, 90.0)
	STATS = 14

	def __init__(self, course, device, seed=0):
		count = course.count
		self.device = device
		self.count = count
		self.course = course
		self.observer = Observer(device)
		self.gen = torch.Generator(device=device)
		self.gen.manual_seed(seed)
		self.limits = torch.tensor((self.TURN_MAX, self.PITCH_MAX), device=device)
		self.shield = BatchShield(course.field, self.limits, self.EASE)
		zeros = lambda *shape: torch.zeros(shape, device=device)
		self.pos = zeros(count, 3)
		self.vel = zeros(count, 3)
		self.yaw = zeros(count)
		self.pitch = zeros(count)
		self.rates = zeros(count, 2)
		self.thrust = zeros(count)
		self.boost = zeros(count)
		self.flux = zeros(count)
		self.drain = zeros(count)
		self.flow = zeros(count)
		self.graze = zeros(count)
		self.skip_cool = zeros(count)
		self.deepest = zeros(count)
		self.surge = zeros(count)
		self.rush = zeros(count)
		self.limp = zeros(count)
		self.bite = torch.ones(count, device=device)
		self.span = max(int(round(Reward.WINDOW / self.STEP)), 1)
		self.trail = zeros(count, self.span)
		self.total = zeros(count)
		self.beat = 0
		self.charge = zeros(count)
		self.clock = zeros(count)
		self.previous = zeros(count, 4)
		self.quota = zeros(count)
		self.taken = zeros(count)
		self.gate = torch.zeros(count, device=device, dtype=torch.bool)
		self.dry = torch.zeros(count, device=device, dtype=torch.bool)
		self.safe = torch.zeros(count, device=device, dtype=torch.bool)
		self.ribbon = torch.full((count,), -1, device=device, dtype=torch.long)
		self.direction = torch.ones(count, device=device, dtype=torch.long)
		self.cursor = torch.zeros(count, device=device, dtype=torch.long)
		self.alive = course.real.clone()
		self.idle = zeros(count, course.ribbons)
		self.respawn = torch.where(course.tunnel, torch.full_like(self.flux, TunnelCfg.RESPAWN), torch.full_like(self.flux, RulesCfg.RIBBON_RESPAWN))
		self.warp = course.hyper.float() * TunnelCfg.HYPER_THRUST
		self.ring_side = torch.zeros((count, course.rings.shape[1]), device=device)
		self.ring_cool = torch.zeros((count, course.rings.shape[1]), device=device)
		self.stats = None
		self.hardness = 1.0
		self.pool = self._spawns()

	def rand(self, *shape):
		return torch.rand(shape, device=self.device, generator=self.gen)

	def _spawns(self):
		course = self.course
		worlds = int(course.world.max()) + 1
		size = course.size
		free_all, head_all, tilt_all, start_all, yaw_all, pitch_all = [], [], [], [], [], []
		spread = torch.tensor((WorldCfg.SIZE_X, 220.0, WorldCfg.SIZE_Z), device=self.device)
		lift = torch.tensor((0.0, 35.0, 0.0), device=self.device)
		for w in range(worlds):
			shift = w * course.field.nz
			env = int((course.world == w).nonzero()[0])
			lanes = course.motes[env].view(course.ribbons, size, 3)
			real = course.real[env].view(course.ribbons, size)[:, 0]
			good = lanes[real & course.usable[env]]
			ends = torch.cat((good[:, 0], good[:, -1]), 0)
			after = torch.cat((good[:, 3], good[:, -4]), 0)
			along = torch.nn.functional.normalize(after - ends, dim=1)
			starts = ends - along * (45.0 + 120.0 * self.rand(ends.shape[0], 1))
			keep = course.field.distance(starts, shift) > 12.0
			if not bool(keep.any()):
				keep = torch.ones_like(keep)
			starts = starts[keep]
			along = along[keep]
			order = torch.arange(64, device=self.device) % starts.shape[0]
			lane_yaw = torch.atan2(along[:, 0], along[:, 2])[order]
			lane_pitch = torch.asin(along[:, 1].clamp(-0.9, 0.9))[order]
			if bool(course.tunnel[env]):
				pick = torch.arange(2048, device=self.device) % good.shape[0]
				slot = torch.randint(2, size - 2, (2048,), device=self.device, generator=self.gen)
				free = good[pick, slot]
				flow = torch.nn.functional.normalize(good[pick, slot + 1] - good[pick, slot - 1], dim=1)
				head = torch.atan2(flow[:, 0], flow[:, 2])
				tilt = torch.asin(flow[:, 1].clamp(-0.9, 0.9))
			else:
				points = self.rand(40000, 3) * spread + lift
				free = points[course.field.distance(points, shift) > 28.0]
				free = free[torch.arange(2048, device=self.device) % free.shape[0]]
				head = self.rand(2048) * math.tau
				tilt = torch.zeros(2048, device=self.device)
			free_all.append(free)
			head_all.append(head)
			tilt_all.append(tilt)
			start_all.append(starts[order])
			yaw_all.append(lane_yaw)
			pitch_all.append(lane_pitch)
		return torch.stack(free_all), torch.stack(head_all), torch.stack(tilt_all), torch.stack(start_all), torch.stack(yaw_all), torch.stack(pitch_all)

	def start(self):
		self.reset(torch.ones(self.count, device=self.device, dtype=torch.bool))
		self.clock = self.rand(self.count) * self.EPISODE
		return self.observe()

	def reset(self, mask):
		number = int(mask.sum())
		if number == 0:
			return
		free, heading, tilt, starts, lane_yaw, lane_pitch = self.pool
		world = self.course.world[mask]
		tunnel = self.course.tunnel[mask]
		pick = torch.randint(0, free.shape[1], (number,), device=self.device, generator=self.gen)
		lane = torch.randint(0, starts.shape[1], (number,), device=self.device, generator=self.gen)
		on_lane = (self.rand(number) < 0.5) & ~tunnel
		pos = torch.where(on_lane.unsqueeze(1), starts[world, lane], free[world, pick])
		yaw = torch.where(on_lane, lane_yaw[world, lane], heading[world, pick])
		pitch = torch.where(on_lane, lane_pitch[world, lane], tilt[world, pick])
		speed = torch.where(tunnel, 150.0 + 110.0 * self.rand(number), 45.0 + 130.0 * self.rand(number) ** 2)
		cp = torch.cos(pitch)
		forward = torch.stack((torch.sin(yaw) * cp, torch.sin(pitch), torch.cos(yaw) * cp), 1)
		self.pos[mask] = pos
		self.vel[mask] = forward * speed.unsqueeze(1)
		self.trail[mask] = speed.unsqueeze(1)
		self.total[mask] = speed * self.span
		self.yaw[mask] = yaw
		self.pitch[mask] = pitch
		self.rates[mask] = 0.0
		self.thrust[mask] = 1.0
		self.previous[mask] = torch.tensor((0.0, 0.0, 1.0, 0.0), device=self.device)
		for state in (self.boost, self.flow, self.graze, self.skip_cool, self.deepest, self.surge, self.rush, self.limp, self.clock, self.taken):
			state[mask] = 0.0
		self.flux[mask] = 0.35 + 0.65 * self.rand(number)
		sector = torch.floor(self.rand(number) * (1.0 + (self.SECTORS - 1.0) * self.hardness))
		self.drain[mask] = torch.where(tunnel, torch.zeros(number, device=self.device), RulesCfg.FLUX_DRAIN + sector * RulesCfg.FLUX_DRAIN_SECTOR)
		self.safe[mask] = self.rand(number) < 1.0 - (1.0 - self.SAFE) * self.hardness
		self.dry[mask] = False
		low, high = self.QUOTA
		self.quota[mask] = torch.where(tunnel, torch.full((number,), 1.0e9, device=self.device), low + (high - low) * self.rand(number))
		opened = (self.rand(number) < 0.1) & ~tunnel
		self.gate[mask] = opened
		self.charge[mask] = opened.float() * RulesCfg.RUSH_TIME
		self.ribbon[mask] = -1
		self.direction[mask] = 1
		self.cursor[mask] = 0
		self.alive[mask] = self.course.real[mask]
		self.idle[mask] = 0.0
		self.ring_side[mask] = 0.0
		self.ring_cool[mask] = 0.0

	def forward(self):
		cp = torch.cos(self.pitch)
		return torch.stack((torch.sin(self.yaw) * cp, torch.sin(self.pitch), torch.cos(self.yaw) * cp), 1)

	def _substep(self, h, drive, boosting, brake, crashed, scraped, skipped, burst):
		cfg = FlightCfg
		pos = self.pos
		vel = self.vel
		self.yaw += self.rates[:, 0] * h
		self.pitch = (self.pitch + self.rates[:, 1] * h).clamp(-1.52, 1.52)
		f = self.forward()
		speed = vel.norm(dim=1)
		heading = torch.where((speed > 1.0e-3).unsqueeze(1), vel / speed.clamp(min=1.0e-3).unsqueeze(1), f)
		before = pos[:, 1].clone()
		wet = before < 0.0
		push = cfg.THRUST * drive + cfg.BOOST_THRUST * boosting + cfg.RUSH_THRUST * self.rush + self.warp
		ceiling = (before.abs() - (WorldCfg.CEILING - cfg.CEILING_SOFT)) / cfg.CEILING_SOFT
		high = ceiling > 0.0
		push = torch.where(high, push * (1.0 - ceiling).clamp(min=0.0), push)
		acc = f * push.unsqueeze(1)
		acc[:, 1] += torch.where(wet, torch.full_like(speed, cfg.BUOYANCY), torch.full_like(speed, -cfg.GRAVITY))
		acc[:, 1] += torch.where(high, ceiling * torch.where(wet, torch.full_like(speed, 40.0), torch.full_like(speed, -40.0)), torch.zeros_like(speed))
		along = (f * heading).sum(1, keepdim=True)
		lift = f - heading * along
		stall = (speed / cfg.LIFT_STALL).clamp(max=1.0)
		gain = cfg.LIFT_GAIN * speed * stall * stall * self.bite * torch.where(wet, torch.full_like(speed, cfg.WATER_GRIP), torch.ones_like(speed))
		lift = lift * gain.unsqueeze(1)
		size = lift.norm(dim=1)
		limit = cfg.LIFT_MAX * torch.where(wet, torch.full_like(speed, cfg.WATER_LIFT), torch.ones_like(speed))
		over = size > limit
		lift = torch.where(over.unsqueeze(1), lift * (limit / size.clamp(min=1.0e-6)).unsqueeze(1), lift)
		size = torch.minimum(size, limit)
		sleek = torch.where(wet, cfg.WATER_DRAG + (1.0 - cfg.WATER_DRAG) * self.flow, torch.ones_like(speed))
		drag = cfg.DRAG * speed * speed * sleek * (1.0 - 0.45 * self.graze) * (1.0 - cfg.SLIP_DRAG * self.flow) * (1.0 - cfg.RUSH_DRAG * self.rush)
		drag = drag + cfg.INDUCED_DRAG * size * torch.where(wet, torch.full_like(speed, cfg.WATER_INDUCED), torch.ones_like(speed)) + cfg.BRAKE_DRAG * speed * speed * brake
		acc = acc + lift - heading * drag.unsqueeze(1)
		vel += acc * h
		pos += vel * h
		self.skip_cool = (self.skip_cool - h).clamp(min=0.0)
		self._surface(before, skipped, burst)
		self._rock(h, crashed, scraped)

	def _surface(self, before, skipped, burst):
		cfg = FlightCfg
		pos = self.pos
		vel = self.vel
		height = pos[:, 1]
		under = height < 0.0
		self.deepest = torch.where(under, torch.maximum(self.deepest, -height), self.deepest)
		speed = vel.norm(dim=1)
		sink = -vel[:, 1]
		entering = (before >= 0.0) & under
		angle = torch.maximum(torch.asin((sink / speed.clamp(min=1.0e-3)).clamp(0.0, 1.0)), -self.pitch * 0.75)
		skip = entering & (speed > cfg.SKIP_SPEED) & (angle < cfg.SKIP_ANGLE)
		scoring = skip & (sink > 3.0) & (self.skip_cool <= 0.0)
		flat = torch.sqrt(vel[:, 0] * vel[:, 0] + vel[:, 2] * vel[:, 2])
		scale = torch.where(scoring, (flat + cfg.SKIP_GAIN) / flat.clamp(min=1.0), torch.ones_like(flat))
		pos[:, 1] = torch.where(skip, torch.full_like(height, 0.02), height)
		vel[:, 1] = torch.where(skip, sink * cfg.SKIP_BOUNCE + 1.0, vel[:, 1])
		vel[:, 0] *= scale
		vel[:, 2] *= scale
		self.skip_cool = torch.where(scoring, torch.full_like(speed, cfg.SKIP_COOLDOWN), self.skip_cool)
		skipped |= scoring
		self.deepest = torch.where(entering & ~skip, torch.zeros_like(speed), self.deepest)
		leaving = (before < 0.0) & (pos[:, 1] >= 0.0) & (speed >= cfg.SPLASH_SPEED) & (self.deepest >= cfg.BREACH_DEPTH)
		kick = torch.minimum(speed + cfg.BREACH_KICK + cfg.SURGE_KICK * self.surge, speed.clamp(min=cfg.SPEED_MAX)) / speed.clamp(min=1.0)
		kick = torch.where(leaving, kick, torch.ones_like(kick))
		vel *= kick.unsqueeze(1)
		vel[:, 1] += torch.where(leaving, 2.0 + cfg.SURGE_LIFT * self.surge, torch.zeros_like(speed))
		burst += torch.where(leaving, 0.4 + self.surge, torch.zeros_like(speed))
		self.surge = torch.where(leaving, torch.zeros_like(speed), self.surge)
		self.deepest = torch.where(leaving, torch.zeros_like(speed), self.deepest)

	def _rock(self, h, crashed, scraped):
		cfg = FlightCfg
		pos = self.pos
		vel = self.vel
		dist, grad = self.course.sample(pos)
		size = grad.norm(dim=1, keepdim=True)
		normal = torch.where(size > 1.0e-5, grad / size.clamp(min=1.0e-5), torch.tensor((0.0, 1.0, 0.0), device=self.device).expand_as(grad))
		gap = dist - cfg.RADIUS
		speed = vel.norm(dim=1)
		amount = (1.0 - gap.clamp(min=0.0) / cfg.GRAZE_RANGE) * ((speed - cfg.GRAZE_MIN_SPEED) / 50.0).clamp(max=1.0)
		near = (gap < cfg.GRAZE_RANGE) & (speed > cfg.GRAZE_MIN_SPEED)
		self.graze = torch.where(near, torch.maximum(self.graze, amount), self.graze)
		touch = gap < 0.0
		pos -= normal * torch.where(touch, gap, torch.zeros_like(gap)).unsqueeze(1)
		into = (vel * normal).sum(1)
		hit = touch & (into < 0.0)
		crash = hit & (-into > cfg.CRASH_SPEED)
		slide = hit & ~crash
		bounced = (vel - normal * (into * (1.0 + cfg.RESTITUTION)).unsqueeze(1)) * 0.62
		slid = (vel - normal * into.unsqueeze(1)) * max(1.0 - 0.5 * h, 0.0)
		vel.copy_(torch.where(crash.unsqueeze(1), bounced, torch.where(slide.unsqueeze(1), slid, vel)))
		crashed |= crash
		scraped |= slide

	def _sweep(self, offsets, origin):
		back = self.course.wrap(origin - self.pos)
		span = (back * back).sum(1)
		fine = (span > 1.0) & (span < 3600.0)
		along = ((offsets * back.unsqueeze(1)).sum(2) / span.clamp(min=1.0).unsqueeze(1)).clamp(0.0, 1.0) * fine.unsqueeze(1)
		gap = offsets - along.unsqueeze(2) * back.unsqueeze(1)
		return (gap * gap).sum(2)

	def _cruise(self, speed):
		slot = self.beat % self.span
		self.beat += 1
		if slot == 0:
			self.trail[:, 0] = speed
			self.total = self.trail.sum(1)
			return self.total / self.span
		self.total += speed - self.trail[:, slot]
		self.trail[:, slot] = speed
		return self.total / self.span

	def _goal(self, rel, valid, target, found):
		lane = valid[:, 0]
		goal = torch.where(lane.unsqueeze(1), rel[:, 0], target)
		return goal.norm(dim=1), lane | found, lane

	def step(self, action):
		dt = self.STEP
		cfg = FlightCfg
		rules = RulesCfg
		course = self.course
		count = self.count
		tunnel = course.tunnel
		action = action.clamp(-1.0, 1.0)
		blend = 1.0 - math.exp(-dt / self.EASE)
		pace = self.vel.norm(dim=1)
		steer, blocked = self.shield.guard(self.pos, self.vel, self.yaw, self.pitch, self.rates, action[:, :2], course.shift, ~tunnel)
		shielded = (steer != action[:, :2]).any(1)
		self.rates += (steer * self.limits - self.rates) * blend
		self.thrust += ((action[:, 2] + 1.0) * 0.5 - self.thrust) * blend
		powered = ~self.dry
		boosting = (action[:, 3] > 0.0) & powered & ((self.flux > (self.drain * self.ENDURANCE).clamp(min=self.RESERVE)) | course.hyper) & ~self.gate
		self.boost = boosting.float()
		drive = self.thrust * powered
		door = course.door(self.pos)
		span = door.norm(dim=1)
		facing = (door * self.forward()).sum(1) / span.clamp(min=1.0e-6)
		closing =(door * self.vel).sum(1) / (span.clamp(min=1.0e-6) * pace.clamp(min=1.0))
		near, aligned, slow, idle, away, fast = self.APPROACH
		astray = self.gate & (span < near) & (closing < aligned) & (pace > slow)
		soon, tight, later, close, coast, brisk, swift = self.CAUTION
		alarm = ((course.distance(self.pos + self.vel * soon) < tight) | blocked) & ~tunnel & (pace > brisk)
		wary = (course.distance(self.pos + self.vel * later) < close) & ~tunnel & ~alarm & (pace > swift)
		astray = astray & ~alarm
		throttle = torch.where(astray, self.thrust.clamp(max=idle), self.thrust)
		throttle = torch.where(wary, throttle.clamp(max=coast), throttle)
		throttle = torch.where(alarm, torch.zeros_like(throttle), throttle)
		boosting = boosting & ~alarm & ~wary
		self.boost = boosting.float()
		drive = throttle * powered
		brake = ((astray & (closing < away) & (pace > fast)) | alarm).float()
		engaged = self.gate & (self.charge > 0.0) & (facing > cfg.RUSH_FACING) & (throttle > 0.3)
		self.charge = torch.where(engaged, (self.charge - dt).clamp(min=0.0), self.charge)
		self.rush += ((engaged | tunnel).float() - self.rush) * min(dt * 5.0, 1.0)
		self.limp = (self.limp + torch.where(self.dry, rules.DRY_FADE, -rules.DRY_RECOVER) * dt).clamp(0.0, 1.0)
		slack = ((rules.DRY_GRIP_SPEED - pace) / (rules.DRY_GRIP_SPEED - rules.DRY_DRIFT_SPEED)).clamp(0.0, 1.0)
		self.bite = cfg.DRIFT_LIFT + (1.0 - cfg.DRIFT_LIFT) * (1.0 - self.limp * slack ** rules.DRY_CURVE)
		self.flow =((self.flow + rules.FLOW_GRAZE * self.graze * dt).clamp(max=1.0) - rules.FLOW_DECAY * dt).clamp(min=0.0)
		self.graze = torch.zeros_like(self.graze)
		crashed = torch.zeros(count, device=self.device, dtype=torch.bool)
		scraped = torch.zeros(count, device=self.device, dtype=torch.bool)
		skipped = torch.zeros(count, device=self.device, dtype=torch.bool)
		burst = torch.zeros(count, device=self.device)
		rel, valid, _, target, _, found = course.preview(self.pos, self.vel, self.forward(), self.alive, self.ribbon, self.direction, self.cursor, self.gate)
		reach_before, aimed_before, lane_before = self._goal(rel, valid, target, found)
		origin = self.pos.clone()
		h = dt / self.SUBSTEPS
		for _ in range(self.SUBSTEPS):
			self._substep(h, drive, boosting.float(), brake, crashed, scraped, skipped, burst)
		self.pos[:, 0] = torch.remainder(self.pos[:, 0], WorldCfg.SIZE_X)
		self.pos[:, 2] = torch.remainder(self.pos[:, 2], WorldCfg.SIZE_Z)
		speed = self.vel.norm(dim=1)
		height = self.pos[:, 1]
		wet = height < 0.0
		depth = (-height).clamp(min=0.0)
		charge = speed * (0.35 + 0.65 * (depth / cfg.SURGE_DEPTH).clamp(max=1.0)) * dt / cfg.SURGE_DISTANCE
		self.surge = torch.where(wet, (self.surge + charge).clamp(max=1.0), (self.surge - cfg.SURGE_DECAY * dt).clamp(min=0.0))
		self.flow = (self.flow + rules.FLOW_SKIP * skipped.float() + rules.FLOW_SURGE * (burst - 0.4).clamp(min=0.0)).clamp(max=1.0)
		offsets = course.offsets(self.pos)
		lanes = self.alive.view(count, course.ribbons, course.size)
		grab = torch.where(tunnel, TunnelCfg.MOTE_RADIUS ** 2, rules.MOTE_RADIUS ** 2)
		hits = (self._sweep(offsets, origin) < grab.unsqueeze(1)) & self.alive
		whole = lanes.all(2)
		lanes_before = lanes.any(2)
		self.alive &= ~hits
		lanes = self.alive.view(count, course.ribbons, course.size)
		lanes_after = lanes.any(2)
		struck = hits.view(count, course.ribbons, course.size)
		touched = struck.any(2)
		motes = hits.sum(1).float()
		sunk = (hits & course.deep).sum(1).float()
		finished = (lanes_before & ~lanes_after).sum(1).float()
		entered = touched & whole
		lead = struck[:, :, :course.size // 2].any(2)
		heading = course.heading(self.vel, self.forward())
		axis = torch.where(lead.unsqueeze(2), course.tangent[:, :, 0], course.tangent[:, :, 1])
		aligned = ((axis * heading.unsqueeze(1)).sum(2).clamp(min=0.0) * entered).sum(1)
		self.taken += motes
		self.flow = (self.flow + rules.FLOW_MOTE * motes).clamp(max=1.0)
		self.idle = torch.where(touched, torch.zeros_like(self.idle), self.idle + dt * (~lanes.all(2)).float())
		back = self.idle > self.respawn.unsqueeze(1)
		self.alive |= back.unsqueeze(2).expand(count, course.ribbons, course.size).reshape(count, -1) & course.real
		self.idle = torch.where(back, torch.zeros_like(self.idle), self.idle)
		opening = ~self.gate & (self.taken >= self.quota)
		self.gate |= opening
		self.charge = torch.where(opening, torch.full_like(self.charge, rules.RUSH_TIME), self.charge)
		ring_offset = -course.wrap(course.rings - self.pos.unsqueeze(1))
		side = (ring_offset * course.ring_dirs).sum(2)
		lateral2 = (ring_offset * ring_offset).sum(2) - side * side
		crossed = (side * self.ring_side < 0.0) & (side.abs() < 30.0) & (self.ring_side.abs() < 30.0) & (lateral2 < (WorldCfg.RING_RADIUS * 1.12) ** 2) & (self.ring_cool <= 0.0)
		self.ring_side = side
		self.ring_cool = torch.where(crossed, torch.full_like(side, rules.RING_COOLDOWN), (self.ring_cool - dt).clamp(min=0.0))
		rings = crossed.any(1)
		boost = torch.where(rings, torch.minimum(speed + rules.RING_BOOST, speed.clamp(min=cfg.SPEED_MAX)) / speed.clamp(min=1.0), torch.ones_like(speed))
		self.vel *= boost.unsqueeze(1)
		speed = speed * boost
		self.flow = (self.flow + rules.FLOW_RING * rings.float()).clamp(max=1.0)
		clear = course.distance(self.pos)
		flat = torch.sqrt(self.vel[:, 0] * self.vel[:, 0] + self.vel[:, 2] * self.vel[:, 2])
		skim = torch.where((height >= 0.0) & (height < 7.0), (1.0 - height / 7.0) * ((flat - cfg.SKIM_MIN_SPEED).clamp(min=0.0) / 45.0).clamp(max=1.0), torch.zeros_like(height))
		gain = rules.FLUX_GRAZE * self.graze * (~scraped).float() + rules.FLUX_SKIM * skim
		loss = torch.where(self.dry, torch.zeros_like(self.drain), self.drain) + rules.FLUX_SCRAPE * 0.6 * scraped.float() + rules.FLUX_BOOST * boosting.float() * (~course.hyper).float()
		loss = torch.where(tunnel, rules.FLUX_BOOST * boosting.float() * (~course.hyper).float(), loss)
		flux = self.flux + (gain - loss) * dt
		flux = flux + rules.FLUX_MOTE * (motes + (rules.DEEP_BONUS - 1.0) * sunk) + rules.FLUX_RIBBON * finished + rules.FLUX_RING * rings.float() + rules.FLUX_SKIP * skipped.float()
		flux = flux + rules.FLUX_BREACH * (burst > 0.0).float() * (0.4 + rules.SURGE_FLUX * (burst - 0.4).clamp(min=0.0)) - rules.FLUX_CRASH * crashed.float() * torch.where(tunnel, 0.25, 1.0)
		flux = torch.where(self.safe, flux.clamp(min=0.5), flux)
		flux = torch.where(tunnel, flux.clamp(min=0.04), flux)
		flux = torch.where(course.hyper, torch.ones_like(flux), flux)
		self.flux = flux.clamp(0.0, 1.0)
		self.dry = torch.where(self.dry, self.flux < rules.FLUX_RELIGHT, self.flux <= 0.0)
		stalled = self.dry & (speed < rules.STALL_SPEED) & ((height < rules.STALL_HEIGHT) | (clear < cfg.RADIUS + rules.STALL_GROUND))
		self.ribbon, self.direction, self.cursor = course.advance(self.pos, self.vel, self.forward(), self.alive, self.ribbon, self.direction, self.cursor)
		rel, valid, _, target, _, found = course.preview(self.pos, self.vel, self.forward(), self.alive, self.ribbon, self.direction, self.cursor, self.gate)
		reach, aimed, lane = self._goal(rel, valid, target, found)
		same = aimed & aimed_before & (lane == lane_before) & ((reach - reach_before).abs() < 40.0)
		progress = torch.where(same, reach_before - reach, torch.zeros_like(reach))
		arrived = self.gate & (course.door(self.pos).norm(dim=1) < WorldCfg.GATE_RADIUS * WorldCfg.GATE_CAPTURE)
		applied = torch.cat((self.rates / self.limits, self.thrust.unsqueeze(1), self.boost.unsqueeze(1)), 1)
		change = applied - self.previous
		self.previous = applied
		weight = Reward
		cruise = self._cruise(speed)
		pace = 0.5 + cruise / weight.MOTE_SPEED
		reward = weight.MOTE * motes * pace + weight.RIBBON * finished + weight.RING * rings.float() + weight.GATE * arrived.float() + weight.ENTRY * aligned * cruise / 100.0
		reward = reward + weight.SPEED * dt * (speed / 100.0).pow(weight.CURVE).clamp(max=weight.SPEED_CAP) + weight.PROGRESS * progress + weight.RUSH * dt * self.rush * (~tunnel).float()
		reward = reward - weight.CRASH * crashed.float() - weight.SCRAPE * scraped.float() - weight.NEAR * (1.0 - clear / 5.0).clamp(min=0.0)
		reward = reward - weight.JERK * (change[:, :2] * change[:, :2]).sum(1) - weight.TURN * (applied[:, :2] * applied[:, :2]).sum(1) - weight.THROTTLE * change[:, 2] * change[:, 2]
		reward = reward - weight.SLOW * (1.0 - cruise / weight.CRUISE).clamp(min=0.0) - weight.STALL * stalled.float() - weight.DRY * dt * self.dry.float() - weight.SHIELD * shielded.float()
		self.clock += dt
		done = arrived | stalled | (self.clock >= self.EPISODE)
		calm = (~tunnel).float()
		self.stats = torch.stack((
			(motes * calm).sum(), (crashed.float() * calm).sum(), (scraped.float() * calm).sum(), (speed * calm).sum(), change[:, :2].abs().sum(), arrived.float().sum(), wet.float().sum(), reward.sum(),
			(self.boost * calm).sum(), (self.flux * calm).sum(), stalled.float().sum(), calm.sum(), (speed * (1.0 - calm)).sum(), (crashed.float() * (1.0 - calm)).sum(),
		))
		self.reset(done)
		return self.observe(), reward, done

	def observe(self):
		course = self.course
		rel, valid, left, target, axis, found = course.preview(self.pos, self.vel, self.forward(), self.alive, self.ribbon, self.direction, self.cursor, self.gate)
		return self.observer.build(course, self, rel, valid, left, target, axis, found)
