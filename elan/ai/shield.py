import math

import torch

from elan.config import FlightCfg


class Shield:
	STEP = 0.07
	STEPS = 14
	URGENT = 7
	EASED = (0.5, 0.0)
	ROOM = 6.0
	SPEED = 50.0
	YAWS = (-1.0, -0.5, 0.0, 0.5, 1.0)
	PITCHES = (-0.6, 0.0, 0.6)

	def __init__(self, turn, pitch, ease):
		self.turn = turn
		self.pitch = pitch
		self.blend = 1.0 - math.exp(-self.STEP / ease)

	def arc(self, field, kite, rates, command):
		cfg = FlightCfg
		h = self.STEP
		blend = self.blend
		room = self.ROOM
		sample = field.sample
		yaw = kite.yaw
		pitch = kite.pitch
		spin = rates[0]
		tilt = rates[1]
		spin_goal = command[0] * self.turn
		tilt_goal = command[1] * self.pitch
		px, py, pz = kite.pos
		speed = max(kite.speed, 1.0)
		dx, dy, dz = kite.vel[0] / speed, kite.vel[1] / speed, kite.vel[2] / speed
		gain = cfg.LIFT_GAIN * speed * min(speed / cfg.LIFT_STALL, 1.0) ** 2
		reach = speed * h
		clearance = 1.0e9
		for k in range(self.STEPS):
			spin += (spin_goal - spin) * blend
			tilt += (tilt_goal - tilt) * blend
			yaw += spin * h
			pitch = max(-1.52, min(1.52, pitch + tilt * h))
			cp = math.cos(pitch)
			fx, fy, fz = math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp
			along = fx * dx + fy * dy + fz * dz
			lx, ly, lz = (fx - dx * along) * gain, (fy - dy * along) * gain, (fz - dz * along) * gain
			size = math.sqrt(lx * lx + ly * ly + lz * lz)
			if size > cfg.LIFT_MAX:
				scale = cfg.LIFT_MAX / size
				lx *= scale
				ly *= scale
				lz *= scale
			vx = dx * speed + lx * h
			vy = dy * speed + (ly - cfg.GRAVITY) * h
			vz = dz * speed + lz * h
			norm = math.sqrt(vx * vx + vy * vy + vz * vz)
			dx, dy, dz = vx / norm, vy / norm, vz / norm
			px += dx * reach
			py += dy * reach
			pz += dz * reach
			dist = sample(px, py, pz)[0]
			if dist < clearance:
				clearance = dist
			if dist < room:
				return k, clearance
		return self.STEPS, clearance

	def guard(self, field, kite, rates, command):
		if kite.speed <= self.SPEED:
			return command, False
		first, clearance = self.arc(field, kite, rates, command)
		if first >= self.STEPS:
			return command, False
		if first >= self.URGENT:
			for ease in self.EASED:
				if self.arc(field, kite, rates, (command[0] * ease, command[1] * ease))[0] >= self.STEPS:
					return command, False
		best = command
		score = first * 100.0 + clearance
		safe = None
		gap = 1.0e9
		for spin in self.YAWS:
			for tilt in self.PITCHES:
				first, clearance = self.arc(field, kite, rates, (spin, tilt))
				if first >= self.STEPS:
					away = (spin - command[0]) ** 2 + (tilt - command[1]) ** 2
					if away < gap:
						gap = away
						safe = (spin, tilt)
				elif safe is None and first * 100.0 + clearance > score:
					score = first * 100.0 + clearance
					best = (spin, tilt)
		if safe is not None:
			return safe, False
		return best, True


class BatchShield:
	def __init__(self, field, limits, ease):
		self.field = field
		self.limits = limits
		self.blend = 1.0 - math.exp(-Shield.STEP / ease)
		grid = [(spin, tilt) for spin in Shield.YAWS for tilt in Shield.PITCHES]
		self.options = torch.tensor(grid, device=limits.device)

	def arc(self, pos, vel, yaw, pitch, rates, command, shift):
		cfg = FlightCfg
		h = Shield.STEP
		steps = Shield.STEPS
		speed = vel.norm(dim=1).clamp(min=1.0)
		heading = vel / speed.unsqueeze(1)
		gain = (cfg.LIFT_GAIN * speed * (speed / cfg.LIFT_STALL).clamp(max=1.0) ** 2).unsqueeze(1)
		reach = (speed * h).unsqueeze(1)
		goal = command * self.limits
		spin = rates.clone()
		point = pos.clone()
		first = torch.full_like(speed, steps, dtype=torch.long)
		clearance = torch.full_like(speed, 1.0e9)
		for k in range(steps):
			spin = spin + (goal - spin) * self.blend
			yaw = yaw + spin[:, 0] * h
			pitch = (pitch + spin[:, 1] * h).clamp(-1.52, 1.52)
			cp = torch.cos(pitch)
			forward = torch.stack((torch.sin(yaw) * cp, torch.sin(pitch), torch.cos(yaw) * cp), 1)
			along = (forward * heading).sum(1, keepdim=True)
			lift = (forward - heading * along) * gain
			size = lift.norm(dim=1, keepdim=True)
			lift = lift * (cfg.LIFT_MAX / size.clamp(min=1.0e-6)).clamp(max=1.0)
			drift = heading * speed.unsqueeze(1) + lift * h
			drift[:, 1] -= cfg.GRAVITY * h
			heading = torch.nn.functional.normalize(drift, dim=1)
			point = point + heading * reach
			dist = self.field.distance(point, shift)
			live = first == steps
			clearance = torch.where(live, torch.minimum(clearance, dist), clearance)
			first = torch.where(live & (dist < Shield.ROOM), torch.full_like(first, k), first)
		return first, clearance

	def guard(self, pos, vel, yaw, pitch, rates, command, shift, active):
		steps = Shield.STEPS
		blocked = torch.zeros_like(active)
		first, clearance = self.arc(pos, vel, yaw, pitch, rates, command, shift)
		rows = (active & (vel.norm(dim=1) > Shield.SPEED) & (first < steps)).nonzero().squeeze(1)
		if rows.numel() == 0:
			return command, blocked
		spare = torch.zeros_like(rows, dtype=torch.bool)
		for ease in Shield.EASED:
			spare |= self.arc(pos[rows], vel[rows], yaw[rows], pitch[rows], rates[rows], command[rows] * ease, shift[rows])[0] >= steps
		rows = rows[~(spare & (first[rows] >= Shield.URGENT))]
		if rows.numel() == 0:
			return command, blocked
		options = self.options
		count = options.shape[0]
		spread = rows.repeat_interleave(count)
		trial = options.repeat(rows.numel(), 1)
		reach, room = self.arc(pos[spread], vel[spread], yaw[spread], pitch[spread], rates[spread], trial, shift[spread])
		reach = reach.view(-1, count)
		room = room.view(-1, count)
		wanted = command[rows]
		away = ((options.unsqueeze(0) - wanted.unsqueeze(1)) ** 2).sum(2)
		safe = reach >= steps
		found = safe.any(1)
		near = torch.where(safe, away, torch.full_like(away, 1.0e9)).argmin(1)
		score = reach.float() * 100.0 + room
		top, best = score.max(1)
		better = top > first[rows].float() * 100.0 + clearance[rows]
		chosen = torch.where(found.unsqueeze(1), options[near], torch.where(better.unsqueeze(1), options[best], wanted))
		command = command.clone()
		command[rows] = chosen
		blocked[rows] = ~found
		return command, blocked
