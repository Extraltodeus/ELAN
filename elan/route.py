import math
import numpy as np

from elan.config import WorldCfg


class Route:
	TURN = 120.0
	ALIGN = 160.0
	CLIMB = 1.0
	FULL = 0.75
	LOOK = (0, 1, 2, 4, 7)
	REACH = 80.0
	BLOCK = 350.0
	ROOM = 12.0
	FAR = 1.0e12

	def __init__(self, world):
		lanes = np.asarray(world.ribbons, dtype=np.float64)
		self.lanes = lanes
		self.count = lanes.shape[0]
		self.size = lanes.shape[1]
		self.order = np.arange(self.size)
		self.usable = np.ones(self.count, dtype=bool) if world.kind == 'tunnel' else lanes[:, 0, 1] > 0.0
		self.entry = np.stack((lanes[:, 0], lanes[:, -1]), 1)
		inward = np.stack((lanes[:, 3] - lanes[:, 0], lanes[:, -4] - lanes[:, -1]), 1)
		self.tangent = inward / np.linalg.norm(inward, axis=2, keepdims=True)
		self.block = np.zeros((self.count * 2, self.count, 2))
		if world.sight is not None:
			self.block = (self.BLOCK * np.clip(1.0 - np.asarray(world.sight, dtype=np.float64) / self.ROOM, 0.0, 1.0)).reshape(self.count * 2, self.count, 2)
		self.ribbon = -1
		self.direction = 1
		self.cursor = 0
		self.urgent = False
		self.ready = None
		self.rel = None
		self.next = None

	def wrap(self, delta):
		delta[..., 0] -= WorldCfg.SIZE_X * np.round(delta[..., 0] / WorldCfg.SIZE_X)
		delta[..., 2] -= WorldCfg.SIZE_Z * np.round(delta[..., 2] / WorldCfg.SIZE_Z)
		return delta

	def plan(self, pos, heading, alive, exclude, origin=-1):
		lanes = alive.reshape(self.count, self.size)
		full = (lanes.sum(1) >= self.FULL * self.size) & self.usable
		for index in (exclude if isinstance(exclude, tuple) else (exclude,)):
			if index >= 0:
				full[index] = False
		ok = np.stack((lanes[:, 0], lanes[:, -1]), 1) & full[:, None]
		rel = self.wrap(self.entry - pos)
		dist = np.sqrt((rel * rel).sum(2))
		to = rel / np.maximum(dist, 1.0)[..., None]
		ease = 0.2 if self.urgent else 1.0
		cost = dist + ease * self.TURN * (1.0 - to @ heading) + ease * self.ALIGN * (1.0 - (to * self.tangent).sum(2)) + self.CLIMB * np.maximum(rel[..., 1], 0.0)
		if origin >= 0:
			cost = cost + ease * self.block[origin]
		cost = np.where(ok, cost, self.FAR)
		index = int(np.argmin(cost))
		if cost.flat[index] >= self.FAR:
			return None
		return index // 2, index % 2

	def _track(self, pos, vel):
		rel = self.wrap(self.lanes[self.ribbon] - pos)
		ahead = (self.order - self.cursor) * self.direction >= 0
		reach = np.where(ahead, (rel * rel).sum(1), self.FAR)
		near = int(np.argmin(reach))
		if reach[near] < self.REACH ** 2 and float(rel[near] @ vel) < 0.0:
			self.cursor = near + self.direction

	def _lane(self, pos, alive):
		rel = self.wrap(self.lanes[self.ribbon] - pos)
		ready = alive[self.ribbon * self.size:(self.ribbon + 1) * self.size] & ((self.order - self.cursor) * self.direction >= 0)
		return ready, rel

	def heading(self, vel, forward):
		speed = math.sqrt(float(vel @ vel))
		if speed < 1.0e-3:
			return np.asarray(forward, dtype=np.float64)
		return vel / speed

	def advance(self, pos, vel, forward, alive):
		heading = self.heading(vel, forward)
		self.ready = None
		if self.ribbon >= 0:
			self._track(pos, vel)
			self.ready, self.rel = self._lane(pos, alive)
		if self.ribbon < 0 or not self.ready.any():
			choice = self.plan(pos, heading, alive, self.ribbon)
			if choice is None:
				self.ribbon = -1
				self.ready = None
			else:
				self.ribbon = choice[0]
				self.direction = 1 if choice[1] == 0 else -1
				self.cursor = 0 if choice[1] == 0 else self.size - 1
				self.ready, self.rel = self._lane(pos, alive)
		if self.ribbon < 0:
			self.next = self.plan(pos, heading, alive, -1)
			return
		end = 1 if self.direction > 0 else 0
		self.next = self.plan(self.entry[self.ribbon, end], -self.tangent[self.ribbon, end], alive, self.ribbon, self.ribbon * 2 + end)

	def waypoints(self):
		count = len(self.LOOK)
		way = np.zeros((count, 3))
		valid = [False] * count
		if self.ribbon < 0 or self.ready is None:
			return way, valid, 0.0
		ready = self.ready
		hits = np.nonzero(ready)[0]
		if hits.size:
			start = int(hits[0]) if self.direction > 0 else int(hits[-1])
		else:
			start = 0 if self.direction > 0 else self.size - 1
		for k, look in enumerate(self.LOOK):
			step = start + self.direction * look
			inside = 0 <= step < self.size
			step = min(max(step, 0), self.size - 1)
			valid[k] = bool(ready[step]) and inside
			way[k] = self.rel[step]
		return way, valid, float(hits.size) / self.size

	def adopt(self, mote, vel):
		ribbon = mote // self.size
		if ribbon == self.ribbon or not self.usable[ribbon]:
			return
		slot = mote % self.size
		lane = self.lanes[ribbon]
		tangent = lane[min(slot + 1, self.size - 1)] - lane[max(slot - 1, 0)]
		self.ribbon = ribbon
		self.direction = 1 if float(tangent @ vel) >= 0.0 else -1
		self.cursor = slot

	def upcoming(self, pos):
		if self.next is None:
			return None
		ribbon, end = self.next
		return self.wrap(self.entry[ribbon, end] - pos), self.tangent[ribbon, end]

	def links(self, pos, alive):
		if self.next is None:
			return []
		ribbon, end = self.next
		start = np.asarray(pos, dtype=np.float64)
		if self.ribbon >= 0:
			start = self.entry[self.ribbon, 1 if self.direction > 0 else 0]
		entry = self.entry[ribbon, end]
		links = [(start, start + self.wrap(entry - start))]
		leave = self.entry[ribbon, 1 - end]
		after = self.plan(leave, -self.tangent[ribbon, 1 - end], alive, (self.ribbon, ribbon), ribbon * 2 + 1 - end)
		if after is not None:
			links.append((leave, leave + self.wrap(self.entry[after[0], after[1]] - leave)))
		return links
