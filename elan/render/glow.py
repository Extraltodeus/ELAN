import math
import numpy as np

from elan.config import WorldCfg


class GlowLayout:
	RAILS = 3
	BEACON = 26
	RING = 108

	def __init__(self):
		golden = (1.0 + math.sqrt(5.0)) * 0.5
		shell = []
		for a in (-1.0, 1.0):
			for b in (-golden, golden):
				shell += [(0.0, a, b), (a, b, 0.0), (b, 0.0, a)]
		shell = np.asarray(shell, dtype=np.float32)
		self.shell = shell / np.linalg.norm(shell, axis=1, keepdims=True) * 0.85
		self.inner = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), dtype=np.float32) * 0.42
		self.cluster = 1 + self.shell.shape[0] + self.inner.shape[0] + self.RAILS
		self.positions = None
		self.colors = None
		self.mote_far = None
		self.mote_near = None
		self.mote_white = None
		self.phase = None
		self.gain = None
		self.mote_count = 0
		self.beacon_start = 0
		self.ring_start = 0
		self.gate_start = 0
		self.total = 0

	def _motes(self, ribbons, color):
		count, steps = ribbons.shape[0], ribbons.shape[1]
		pos = np.zeros((count, steps, self.cluster, 3), dtype=np.float32)
		weight = np.zeros((count, steps, self.cluster), dtype=np.float32)
		pos[:, :, 0] = ribbons
		weight[:, :, 0] = 1.0
		shell = 1 + self.shell.shape[0]
		rails = shell + self.inner.shape[0]
		pos[:, :, 1:shell] = ribbons[:, :, None, :] + self.shell[None, None]
		weight[:, :, 1:shell] = 0.3
		pos[:, :, shell:rails] = ribbons[:, :, None, :] + self.inner[None, None]
		weight[:, :, shell:rails] = 0.12
		following = np.concatenate((ribbons[:, 1:], ribbons[:, -1:]), 1)
		for rail in range(self.RAILS):
			blend = (rail + 1.0) / (self.RAILS + 1.0)
			pos[:, :, rails + rail] = ribbons + (following - ribbons) * blend
			weight[:, :-1, rails + rail] = 0.16
		phase = np.broadcast_to((np.arange(steps, dtype=np.float32) * -0.6)[None, :, None], weight.shape)
		colors = weight[..., None] * (np.asarray(color, dtype=np.float32) * 24.0)
		far = np.zeros_like(weight)
		far[:, :, 0] = 1.0
		far[:, :-1, rails:] = 0.4
		near = np.zeros_like(weight)
		near[:, :, 0] = 0.3
		near[:, :, 1:shell] = 0.6
		near[:, :, shell:rails] = 0.75
		near[:, :-1, rails:] = 0.2
		self.mote_far = far.reshape(-1)
		self.mote_near = near.reshape(-1)
		self.mote_white = (weight * 24.0).reshape(-1)
		return pos.reshape(-1, 3), colors.reshape(-1, 3), phase.reshape(-1).copy()

	def _beacons(self, ribbons, color):
		count = ribbons.shape[0]
		pos = np.zeros((count, 2, self.BEACON, 3), dtype=np.float32)
		ramp = np.linspace(0.0, 1.0, self.BEACON, dtype=np.float32)
		for index in range(count):
			for side, end in enumerate((0, -1)):
				base = ribbons[index, end]
				column = np.tile(base, (self.BEACON, 1))
				if base[1] >= 0.0:
					top = min(base[1] + 75.0, WorldCfg.CEILING - 4.0)
					column[:, 1] = base[1] + 5.0 + (top - base[1] - 5.0) * ramp
				else:
					column[:, 1] = base[1] + 5.0 + (-2.0 - base[1] - 5.0) * ramp
				pos[index, side] = column
		fade = (1.0 - 0.75 * ramp)[None, None, :, None]
		colors = np.broadcast_to(fade * (np.asarray(color, dtype=np.float32) * 7.0), pos.shape)
		phase = np.broadcast_to((ramp * 9.0)[None, None, :], pos.shape[:3])
		return pos.reshape(-1, 3), colors.reshape(-1, 3).copy(), phase.reshape(-1).copy()

	def _circle(self, center, axis, radius, count):
		up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
		side = np.cross(axis, up)
		if np.linalg.norm(side) < 0.1:
			side = np.cross(axis, np.array((1.0, 0.0, 0.0), dtype=np.float32))
		side = side / np.linalg.norm(side)
		lift = np.cross(side, axis)
		angle = np.linspace(0.0, math.tau, count, endpoint=False, dtype=np.float32)
		return center + (np.cos(angle)[:, None] * side + np.sin(angle)[:, None] * lift) * radius, angle

	def _rings(self, centers, axes, color):
		outer = 72
		inner = self.RING - outer
		pos = np.zeros((centers.shape[0], self.RING, 3), dtype=np.float32)
		phase = np.zeros((centers.shape[0], self.RING), dtype=np.float32)
		weight = np.zeros(self.RING, dtype=np.float32)
		weight[:outer] = 1.0
		weight[outer:] = 0.35
		for index in range(centers.shape[0]):
			pos[index, :outer], angle = self._circle(centers[index], axes[index], WorldCfg.RING_RADIUS, outer)
			phase[index, :outer] = angle * 3.0
			pos[index, outer:], angle = self._circle(centers[index] - axes[index] * 2.5, axes[index], WorldCfg.RING_RADIUS * 0.8, inner)
			phase[index, outer:] = angle * 3.0 + 1.5
		colors = np.broadcast_to(weight[None, :, None] * (np.asarray(color, dtype=np.float32) * 13.0), pos.shape)
		return pos.reshape(-1, 3), colors.reshape(-1, 3).copy(), phase.reshape(-1)

	def _shell(self, center, radius, count, twist):
		index = np.arange(count, dtype=np.float32) + 0.5
		height = 1.0 - 2.0 * index / count
		around = index * (math.pi * (3.0 - math.sqrt(5.0)))
		flat = np.sqrt(np.maximum(1.0 - height * height, 0.0))
		points = np.stack((np.cos(around) * flat, height, np.sin(around) * flat), 1).astype(np.float32) * radius
		phase = np.arctan2(points[:, 2], points[:, 0]) * 3.0 + height * twist
		return center + points, phase.astype(np.float32)

	def _gate(self, world, color):
		center = np.asarray(world.gate_pos, dtype=np.float32)
		radius = WorldCfg.GATE_RADIUS
		pieces = []
		gains = []
		phases = []
		for scale, count, gain, twist in ((1.0, 900, 0.3, 5.0), (0.62, 420, 0.18, -7.0), (0.28, 160, 0.45, 9.0)):
			shell, phase = self._shell(center, radius * scale, count, twist)
			pieces.append(shell)
			gains.append(np.full(count, gain, dtype=np.float32))
			phases.append(phase)
		count = 220
		heights = np.linspace(4.0, WorldCfg.SIZE_Y - 6.0, count, dtype=np.float32)
		pillar = np.tile(center, (count, 1))
		pillar[:, 1] = heights
		keep = np.abs(heights - center[1]) > radius * 1.15
		pieces.append(pillar[keep])
		gains.append(np.full(int(keep.sum()), 0.5, dtype=np.float32))
		phases.append(heights[keep] * 0.2)
		pos = np.concatenate(pieces, 0).astype(np.float32)
		gain = np.concatenate(gains, 0)
		colors = gain[:, None] * (np.asarray(color, dtype=np.float32) * 12.0)
		return pos, colors.astype(np.float32), np.concatenate(phases, 0).astype(np.float32)

	def build(self, world):
		palette = world.palette
		mote_pos, mote_col, mote_phase = self._motes(world.ribbons, palette['mote'])
		beacon_pos, beacon_col, beacon_phase = self._beacons(world.ribbons, palette['mote'])
		if world.kind == 'tunnel':
			beacon_col[:] = 0.0
		ring_pos, ring_col, ring_phase = self._rings(world.rings_pos, world.rings_dir, palette['gate'])
		gate_pos, gate_col, gate_phase = self._gate(world, palette['gate'])
		self.mote_count = mote_pos.shape[0]
		self.beacon_start = self.mote_count
		self.ring_start = self.beacon_start + beacon_pos.shape[0]
		self.gate_start = self.ring_start + ring_pos.shape[0]
		self.positions = np.concatenate((mote_pos, beacon_pos, ring_pos, gate_pos), 0).astype(np.float32)
		self.colors = np.concatenate((mote_col, beacon_col, ring_col, gate_col), 0).astype(np.float32)[:, ::-1].copy()
		self.phase = np.concatenate((mote_phase, beacon_phase, ring_phase, gate_phase), 0).astype(np.float32)
		self.total = self.positions.shape[0]
		self.gain = np.ones(self.total, dtype=np.float32)
		self.gain[self.gate_start:] = 0.0
		return self

	def mote_slice(self, mote):
		return mote * self.cluster, (mote + 1) * self.cluster

	def beacon_slice(self, ribbon):
		size = 2 * self.BEACON
		return self.beacon_start + ribbon * size, self.beacon_start + (ribbon + 1) * size

	def ring_slice(self, ring):
		return self.ring_start + ring * self.RING, self.ring_start + (ring + 1) * self.RING
