import math
import numpy as np
import torch
import torch.nn.functional as F

from elan.ai.field import BatchField
from elan.config import WorldCfg
from elan.route import Route


class Course:
	FAR = 1.0e12
	AWAY = 1.0e6

	def __init__(self, worlds, device, count):
		self.device = device
		self.count = count
		shape = next(world.field.shape for world in worlds if world.kind == 'sector')
		self.field = BatchField([self._fit(world.field, shape) for world in worlds], device)
		self.world = torch.arange(count, device=device) % len(worlds)
		self.shift = self.world * self.field.nz
		self.size = int(worlds[0].ribbons.shape[1])
		self.ribbons = max(int(world.ribbons.shape[0]) for world in worlds)
		total = self.ribbons * self.size
		rings = max(int(np.asarray(world.rings_pos).shape[0]) for world in worlds)
		motes = np.zeros((len(worlds), total, 3), dtype=np.float32)
		motes[:, :, 1] = -self.AWAY
		real = np.zeros((len(worlds), total), dtype=bool)
		usable = np.zeros((len(worlds), self.ribbons), dtype=bool)
		entry = np.zeros((len(worlds), self.ribbons, 2, 3), dtype=np.float32)
		tangent = np.zeros((len(worlds), self.ribbons, 2, 3), dtype=np.float32)
		block = np.zeros((len(worlds), self.ribbons * 2, self.ribbons, 2), dtype=np.float32)
		tangent[..., 2] = 1.0
		ring_pos = np.zeros((len(worlds), rings, 3), dtype=np.float32)
		ring_pos[:, :, 1] = self.AWAY
		ring_dir = np.zeros((len(worlds), rings, 3), dtype=np.float32)
		ring_dir[:, :, 2] = 1.0
		gates = np.zeros((len(worlds), 3), dtype=np.float32)
		tunnel = np.zeros(len(worlds), dtype=bool)
		hyper = np.zeros(len(worlds), dtype=bool)
		for k, world in enumerate(worlds):
			flat = np.asarray(world.ribbons, dtype=np.float32).reshape(-1, 3)
			motes[k, :flat.shape[0]] = flat
			real[k, :flat.shape[0]] = True
			route = Route(world)
			usable[k, :route.count] = route.usable
			entry[k, :route.count] = route.entry
			tangent[k, :route.count] = route.tangent
			block[k, :route.count * 2, :route.count] = route.block
			found = np.asarray(world.rings_pos, dtype=np.float32)
			ring_pos[k, :found.shape[0]] = found
			ring_dir[k, :found.shape[0]] = np.asarray(world.rings_dir, dtype=np.float32)
			gates[k] = np.asarray(world.gate_pos, dtype=np.float32)
			tunnel[k] = world.kind == 'tunnel'
			hyper[k] = tunnel[k] and bool(world.palette['hyper'])
		pick = self.world.cpu().numpy()
		self.motes = torch.from_numpy(motes[pick]).to(device)
		self.real = torch.from_numpy(real[pick]).to(device)
		self.deep = self.motes[:, :, 1] < 0.0
		self.usable = torch.from_numpy(usable[pick]).to(device)
		self.entry = torch.from_numpy(entry[pick]).to(device)
		self.tangent = torch.from_numpy(tangent[pick]).to(device)
		self.block = torch.from_numpy(block).to(device)
		self.rings = torch.from_numpy(ring_pos[pick]).to(device)
		self.ring_dirs = torch.from_numpy(ring_dir[pick]).to(device)
		self.gate = torch.from_numpy(gates[pick]).to(device)
		self.tunnel = torch.from_numpy(tunnel[pick]).to(device)
		self.hyper = torch.from_numpy(hyper[pick]).to(device)
		self.period = torch.tensor((WorldCfg.SIZE_X, 1.0e9, WorldCfg.SIZE_Z), device=device)
		self.order = torch.arange(self.size, device=device)
		self.rows = torch.arange(count, device=device)
		self.lanes = torch.arange(self.ribbons, device=device).view(1, -1)
		self.look = torch.tensor(Route.LOOK, device=device)

	def _fit(self, field, shape):
		if field.shape == shape:
			return field
		source = torch.from_numpy(np.ascontiguousarray(field).astype(np.float32))[None, None]
		padded = F.pad(source, (1, 1, 0, 0, 1, 1), mode='circular')
		scale = shape[0] // field.shape[0]
		grown = F.interpolate(padded, scale_factor=scale, mode='trilinear', align_corners=False)[0, 0]
		return grown[scale:-scale, :, scale:-scale].numpy().astype(np.float16)

	def wrap(self, delta):
		return delta - self.period * torch.round(delta / self.period)

	def distance(self, points):
		shift = self.shift if points.dim() == 2 else self.shift.unsqueeze(1)
		return self.field.distance(points, shift)

	def sample(self, points):
		return self.field.sample(points, self.shift)

	def offsets(self, pos):
		return self.wrap(self.motes - pos.unsqueeze(1))

	def door(self, pos):
		return self.wrap(self.gate - pos)

	def spots(self, pos, ribbon):
		count = pos.shape[0]
		size = self.size
		pick = ribbon.clamp(min=0)
		spots = self.motes.view(count, self.ribbons, size, 3).gather(1, pick.view(count, 1, 1, 1).expand(count, 1, size, 3)).squeeze(1)
		return self.wrap(spots - pos.unsqueeze(1))

	def track(self, pos, vel, ribbon, direction, cursor):
		count = pos.shape[0]
		rel = self.spots(pos, ribbon)
		ahead = (self.order.unsqueeze(0) - cursor.unsqueeze(1)) * direction.unsqueeze(1) >= 0
		reach = torch.where(ahead, (rel * rel).sum(2), torch.full_like(rel[..., 0], self.FAR))
		best, near = reach.min(1)
		closest = rel.gather(1, near.view(count, 1, 1).expand(count, 1, 3)).squeeze(1)
		passed = (best < Route.REACH ** 2) & ((closest * vel).sum(1) < 0.0) & (ribbon >= 0)
		return torch.where(passed, near + direction, cursor)

	def lane(self, pos, alive, ribbon, direction, cursor):
		count = alive.shape[0]
		size = self.size
		pick = ribbon.clamp(min=0)
		row = alive.view(count, self.ribbons, size).gather(1, pick.view(count, 1, 1).expand(count, 1, size)).squeeze(1)
		ahead = (self.order.unsqueeze(0) - cursor.unsqueeze(1)) * direction.unsqueeze(1) >= 0
		return row & ahead & (ribbon >= 0).unsqueeze(1), self.spots(pos, ribbon)

	def plan(self, pos, heading, alive, exclude, origin=None):
		count = alive.shape[0]
		lanes = alive.view(count, self.ribbons, self.size)
		full = (lanes.sum(2) >= Route.FULL * self.size) & self.usable & (self.lanes != exclude.view(count, 1))
		ok = torch.stack((lanes[:, :, 0], lanes[:, :, -1]), 2) & full.unsqueeze(2)
		rel = self.wrap(self.entry - pos.view(count, 1, 1, 3))
		dist = rel.norm(dim=3)
		to = rel / dist.clamp(min=1.0).unsqueeze(3)
		cost = dist + Route.TURN * (1.0 - (to * heading.view(count, 1, 1, 3)).sum(3)) + Route.ALIGN * (1.0 - (to * self.tangent).sum(3)) + Route.CLIMB * rel[..., 1].clamp(min=0.0)
		if origin is not None:
			cost = cost + self.block[self.world, origin.clamp(min=0)] * (origin >= 0).view(count, 1, 1)
		cost = torch.where(ok, cost, torch.full_like(cost, self.FAR))
		best, index = cost.view(count, -1).min(1)
		return index // 2, index % 2, best < self.FAR

	def heading(self, vel, forward):
		speed = vel.norm(dim=1, keepdim=True)
		return torch.where(speed > 1.0e-3, vel / speed.clamp(min=1.0e-3), forward)

	def advance(self, pos, vel, forward, alive, ribbon, direction, cursor):
		cursor = self.track(pos, vel, ribbon, direction, cursor)
		ready, _ = self.lane(pos, alive, ribbon, direction, cursor)
		need = (ribbon < 0) | ~ready.any(1)
		fresh, end, found = self.plan(pos, self.heading(vel, forward), alive, ribbon)
		ribbon = torch.where(need, torch.where(found, fresh, torch.full_like(fresh, -1)), ribbon)
		direction = torch.where(need & found, 1 - 2 * end, direction)
		cursor = torch.where(need & found, end * (self.size - 1), cursor)
		return ribbon, direction, cursor

	def preview(self, pos, vel, forward, alive, ribbon, direction, cursor, gate_open):
		count = alive.shape[0]
		size = self.size
		ready, lane = self.lane(pos, alive, ribbon, direction, cursor)
		weight = ready.to(torch.int8)
		first = weight.argmax(1)
		last = size - 1 - weight.flip(1).argmax(1)
		start = torch.where(direction > 0, first, last)
		steps = start.unsqueeze(1) + direction.unsqueeze(1) * self.look
		inside = (steps >= 0) & (steps < size)
		steps = steps.clamp(0, size - 1)
		looks = self.look.shape[0]
		valid = ready.gather(1, steps) & inside
		rel = lane.gather(1, steps.unsqueeze(2).expand(count, looks, 3))
		door = self.door(pos)
		rel = torch.where(gate_open.view(count, 1, 1), door.unsqueeze(1).expand(count, looks, 3), rel)
		valid = valid | gate_open.unsqueeze(1)
		left = ready.sum(1).float() / size
		has = ribbon >= 0
		pick = ribbon.clamp(min=0)
		end = (direction > 0).long()
		origin = torch.where(has.unsqueeze(1), self.entry[self.rows, pick, end], pos)
		course = torch.where(has.unsqueeze(1), -self.tangent[self.rows, pick, end], self.heading(vel, forward))
		after, side, found = self.plan(origin, course, alive, ribbon, torch.where(has, pick * 2 + end, torch.full_like(pick, -1)))
		target = self.wrap(self.entry[self.rows, after, side] - pos)
		axis = self.tangent[self.rows, after, side]
		return rel, valid, left, target, axis, found & ~gate_open
