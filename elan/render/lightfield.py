import math
import numpy as np
import torch
import torch.nn.functional as F

from elan.config import WorldCfg


class LightField:
	CELL = 8.0
	SIGMA = 2.2
	TAPS = 6
	PERIOD = 0.18

	def __init__(self, device):
		self.device = device
		self.nx = int(WorldCfg.SIZE_X / self.CELL)
		self.ny = int(WorldCfg.SIZE_Y / self.CELL)
		self.nz = int(WorldCfg.SIZE_Z / self.CELL)
		taps = torch.arange(-self.TAPS, self.TAPS + 1, device=device, dtype=torch.float32)
		kernel = torch.exp(-0.5 * (taps / self.SIGMA) ** 2)
		self.kernel = kernel / kernel.sum()
		self.grid = torch.zeros((3, self.nz * self.ny * self.nx), device=device)
		self.cell = None
		self.source = None
		self.color = None
		self.volume = None
		self.timer = 0.0
		self.gain = 0.0

	def _bgr(self, rgb):
		return np.asarray((rgb[2], rgb[1], rgb[0]), dtype=np.float32)

	def set_world(self, world, layout):
		palette = world.palette
		self.gain = float(world.style['glow'])
		motes = np.asarray(world.ribbons, dtype=np.float32).reshape(-1, 3)
		rings = np.asarray(world.rings_pos, dtype=np.float32).reshape(-1, 3)
		gate = np.asarray(world.gate_pos, dtype=np.float32).reshape(1, 3)
		points = np.concatenate((motes, rings, gate), 0)
		source = np.concatenate((
			np.arange(motes.shape[0]) * layout.cluster,
			layout.ring_start + np.arange(rings.shape[0]) * layout.RING,
			np.asarray((layout.gate_start,)),
		), 0)
		line = 1.0 / (math.sqrt(math.tau) * self.SIGMA) ** 2 * (self.CELL / WorldCfg.RIBBON_STEP)
		color = np.zeros((points.shape[0], 3), dtype=np.float32)
		color[:motes.shape[0]] = self._bgr(palette['mote']) * (self.gain / line)
		color[motes.shape[0]:-1] = self._bgr(palette['gate']) * (self.gain / line * 5.0)
		color[-1] = self._bgr(palette['gate']) * (self.gain / line * 60.0)
		ix = np.floor(points[:, 0] / self.CELL).astype(np.int64) % self.nx
		iy = np.clip(np.floor(np.abs(points[:, 1]) / self.CELL).astype(np.int64), 0, self.ny - 1)
		iz = np.floor(points[:, 2] / self.CELL).astype(np.int64) % self.nz
		self.cell = torch.from_numpy((iz * self.ny + iy) * self.nx + ix).to(self.device)
		self.source = torch.from_numpy(source).to(self.device)
		self.color = torch.from_numpy(color.T.copy()).to(self.device)
		self.timer = 0.0
		self.volume = None

	def _blur(self, volume, axis, wrap):
		taps = self.TAPS
		shape = [1, 1, 1, 1, 1]
		shape[axis] = 2 * taps + 1
		pad = [0, 0, 0, 0, 0, 0]
		pad[(4 - axis) * 2] = taps
		pad[(4 - axis) * 2 + 1] = taps
		padded = F.pad(volume, pad, mode='circular' if wrap else 'replicate')
		return F.conv3d(padded, self.kernel.view(shape))

	def update(self, scene, dt):
		if self.gain <= 0.0:
			return None
		self.timer -= dt
		if self.volume is not None and (not scene.changed or self.timer > 0.0):
			return self.volume
		scene.changed = False
		self.timer = self.PERIOD
		self.grid.zero_()
		self.grid.index_add_(1, self.cell, self.color * scene.glow_gain[self.source])
		volume = self.grid.view(3, 1, self.nz, self.ny, self.nx)
		volume = self._blur(self._blur(self._blur(volume, 4, True), 2, True), 3, False)
		self.volume = volume.view(1, 3, self.nz, self.ny, self.nx).repeat(1, 1, 2, 1, 2)
		return self.volume
