import math
import numpy as np
import torch

from elan.config import WorldCfg


class PathPlanner:
	def __init__(self, device):
		self.device = device
		self.size = torch.tensor((WorldCfg.SIZE_X, WorldCfg.SIZE_Y, WorldCfg.SIZE_Z), device=device, dtype=torch.float32)
		self.sunk_count = 0
		self.sunk_ratio = 0.25
		self.probes = 8
		self.shallow = 0.33
		self.valid_count = 0

	def plan(self, sampler, noise, style):
		self.sunk_ratio = style['sunk']
		self.probes = int(style['corridor'])
		self.shallow = float(style['shallow'])
		ribbons = self._ribbons(sampler, noise)
		rings_pos, rings_dir = self._rings(ribbons)
		launch = self._launch(sampler, ribbons)
		ribbons, rings_pos, rings_dir = self._sink(ribbons, rings_pos, rings_dir)
		spawn_pos, spawn_dir = launch if launch is not None else self._open_spot(sampler, noise, (110.0, 230.0), 42.0, None, False)
		gate_pos, gate_dir = self._open_spot(sampler, noise, (70.0, 250.0), 46.0, spawn_pos, True)
		return {
			'ribbons': ribbons, 'rings_pos': rings_pos, 'rings_dir': rings_dir,
			'spawn_pos': spawn_pos, 'spawn_dir': spawn_dir, 'gate_pos': gate_pos, 'gate_dir': gate_dir,
			'pool': self.valid_count, 'sight': self._sight(sampler, ribbons),
		}

	def _sight(self, sampler, ribbons):
		ends = torch.from_numpy(np.stack((ribbons[:, 0], ribbons[:, -1]), 1).reshape(-1, 3)).to(self.device)
		delta = ends.unsqueeze(0) - ends.unsqueeze(1)
		period = torch.tensor((WorldCfg.SIZE_X, 1.0e9, WorldCfg.SIZE_Z), device=self.device)
		delta = delta - period * torch.round(delta / period)
		blend = torch.linspace(0.1, 0.9, 12, device=self.device).view(1, 1, -1, 1)
		points = ends.view(-1, 1, 1, 3) + delta.unsqueeze(2) * blend
		points[..., 1] = points[..., 1].abs()
		return sampler.sample(points).min(2).values.cpu().numpy().astype(np.float32)

	def _launch(self, sampler, ribbons):
		cfg = WorldCfg
		steps = ribbons.shape[1]
		best = None
		best_score = 0.0
		blend = torch.linspace(-0.25, 1.0, 16, device=self.device).view(-1, 1)
		for index in range(self.sunk_count, ribbons.shape[0]):
			for end, inner in ((0, 4), (steps - 1, steps - 5)):
				entry = ribbons[index, end]
				tangent = ribbons[index, inner] - entry
				tangent = tangent / np.linalg.norm(tangent)
				if abs(tangent[1]) > 0.4:
					continue
				for lead in (230.0, 180.0, 135.0):
					start = entry - tangent * lead
					if start[1] < 28.0 or start[1] > cfg.CEILING - 45.0:
						continue
					a = torch.from_numpy(start).to(self.device)
					b = torch.from_numpy(entry).to(self.device)
					room = float(sampler.sample(a + (b - a) * blend).min())
					score = min(room, 40.0) + lead * 0.06 - 25.0 * abs(float(tangent[1])) - 0.25 * max(float(start[1]) - 200.0, 0.0) - 0.25 * max(70.0 - float(start[1]), 0.0)
					if room > 15.0 and score > best_score:
						best_score = score
						best = (start.astype(np.float32), tangent.astype(np.float32))
					if room > 15.0:
						break
		return best

	def _random_points(self, noise, count, y_range):
		points = noise.uniform((count, 3)) * self.size
		points[:, 1] = y_range[0] + noise.uniform((count,)) * (y_range[1] - y_range[0])
		return points

	def _walk(self, sampler, noise):
		cfg = WorldCfg
		count = cfg.RIBBON_CANDIDATES
		seeds = self._random_points(noise, count * 24, (22.0, cfg.SIZE_Y * 0.8))
		depth = sampler.sample(seeds)
		good = ((depth > 15.0) & (depth < 34.0)).nonzero().flatten()
		if good.numel() == 0:
			good = torch.arange(seeds.shape[0], device=self.device)
		pick = good[torch.arange(count, device=self.device) % good.numel()]
		shallow = good[seeds[good, 1] < 85.0]
		third = int(count * self.shallow)
		cap = torch.full((count,), cfg.SIZE_Y * 0.8, device=self.device)
		if shallow.numel() > 0:
			pick[:third] = shallow[torch.arange(third, device=self.device) % shallow.numel()]
			cap[:third] = 95.0
		pos = seeds[pick]
		angle = noise.uniform((count,)) * math.tau
		heading = torch.stack((torch.cos(angle), noise.uniform((count,)) * 0.4 - 0.2, torch.sin(angle)), -1)
		heading = heading / heading.norm(dim=-1, keepdim=True)
		spin = noise.normal((count, 3)) * 0.04
		track = []
		for step in range(cfg.RIBBON_STEPS):
			track.append(pos)
			depth = sampler.sample(pos)
			normal = sampler.gradient(pos)
			steer = normal * (((18.0 - depth) / 18.0).clamp(-1.0, 1.0) * 0.5).unsqueeze(-1)
			for reach, weight in ((30.0, 1.2), (65.0, 0.8)):
				ahead = pos + heading * reach
				depth_ahead = sampler.sample(ahead)
				steer = steer + sampler.gradient(ahead) * (((24.0 - depth_ahead) / 24.0).clamp(0.0, 1.0) * weight).unsqueeze(-1)
			lift = ((30.0 - pos[:, 1]) / 30.0).clamp_min(0.0) - ((pos[:, 1] - cap) / 40.0).clamp_min(0.0)
			steer[:, 1] += lift * 0.8
			steer = steer + spin
			spin = spin * 0.9 + noise.normal((count, 3)) * 0.025
			steer = steer - heading * (steer * heading).sum(-1, keepdim=True)
			magnitude = steer.norm(dim=-1, keepdim=True)
			steer = steer * (0.068 / magnitude.clamp_min(0.068))
			heading = heading + steer
			heading[:, 1] = heading[:, 1].clamp(-0.42, 0.42)
			heading = heading / heading.norm(dim=-1, keepdim=True)
			pos = pos + heading * cfg.RIBBON_STEP
		return torch.stack(track, 1)

	def _corridor(self, sampler, origin, heading):
		cfg = WorldCfg
		reach = torch.tensor((22.0, 45.0, 70.0, 100.0, 135.0, 175.0, 220.0, 265.0)[:self.probes], device=self.device).view(1, -1, 1)
		probes = origin.unsqueeze(1) + heading.unsqueeze(1) * reach
		free = (sampler.sample(probes) - reach.view(1, -1) * 0.115).min(1).values
		height = probes[..., 1]
		inside = (height.min(1).values > 8.0) & (height.max(1).values < cfg.CEILING - 15.0)
		return torch.where(inside, free, torch.zeros_like(free))

	def _ribbons(self, sampler, noise):
		paths, valid, score, low = self._survey(sampler, noise)
		for _ in range(2):
			if int(valid.sum()) >= 90:
				break
			more = self._survey(sampler, noise)
			paths = torch.cat((paths, more[0]), 0)
			valid = torch.cat((valid, more[1]), 0)
			score = torch.cat((score, more[2]), 0)
			low = torch.cat((low, more[3]), 0)
		cfg = WorldCfg
		self.valid_count = int(valid.sum())
		order = torch.argsort(score, descending=True).cpu().numpy()
		score = score.cpu().numpy()
		low = low.cpu().numpy()
		paths = paths.cpu().numpy().astype(np.float32)
		sunk = self._pick(paths, order, score, low, int(round(cfg.RIBBON_COUNT * self.sunk_ratio)), set())
		dry = self._pick(paths, order, score, None, cfg.RIBBON_COUNT - len(sunk), set(sunk))
		if not dry and not sunk:
			dry = [int(order[0])]
		self.sunk_count = len(sunk)
		return np.stack([paths[i] for i in sunk + dry], 0)

	def _survey(self, sampler, noise):
		cfg = WorldCfg
		paths = self._walk(sampler, noise)
		blend = torch.tensor((0.0, 0.25, 0.5, 0.75), device=self.device).view(1, 1, 4, 1)
		dense = paths[:, :-1, None, :] + (paths[:, 1:, None, :] - paths[:, :-1, None, :]) * blend
		depth = sampler.sample(paths)
		lowest = torch.minimum(depth.min(1).values, sampler.sample(dense).flatten(1).min(1).values)
		y_min = paths[..., 1].min(1).values
		y_max = paths[..., 1].max(1).values
		head = paths[:, -1] - paths[:, -2]
		tail = paths[:, 0] - paths[:, 1]
		exit_free = self._corridor(sampler, paths[:, -1], head / head.norm(dim=-1, keepdim=True))
		entry_free = self._corridor(sampler, paths[:, 0], tail / tail.norm(dim=-1, keepdim=True))
		valid = (lowest > 10.0) & (y_min > 12.0) & (y_max < cfg.CEILING - 46.0) & (exit_free > 7.0) & (entry_free > 7.0)
		score = (1.0 - ((depth - 18.0).abs() / 18.0).clamp(0.0, 1.0)).mean(1)
		score = torch.where(valid, score, -1.0)
		return paths, valid, score, y_max < cfg.SUNK_DEPTH

	def _pick(self, paths, order, score, mask, limit, taken):
		cfg = WorldCfg
		period = np.array((cfg.SIZE_X, cfg.SIZE_Z), dtype=np.float32)
		chosen = []
		occupied = np.zeros((0, 3), dtype=np.float32)
		for index in order:
			if score[index] < 0.0 or len(chosen) >= limit:
				break
			if index in taken or (mask is not None and not mask[index]):
				continue
			sparse = paths[index, ::3]
			if occupied.shape[0] > 0:
				delta = sparse[:, None, :] - occupied[None, :, :]
				flat = delta[..., [0, 2]]
				flat = flat - np.round(flat / period) * period
				dist2 = (flat ** 2).sum(-1) + delta[..., 1] ** 2
				if dist2.min() < 36.0 ** 2:
					continue
			chosen.append(int(index))
			occupied = np.concatenate((occupied, sparse), 0)
		return chosen

	def _rings(self, ribbons):
		cfg = WorldCfg
		steps = ribbons.shape[1]
		slots = [i for i in cfg.RING_SLOTS if 1 <= i < steps - 1]
		centers = []
		axes = []
		for ribbon in ribbons:
			for i in slots:
				axis = ribbon[i + 1] - ribbon[i - 1]
				centers.append(ribbon[i])
				axes.append(axis / np.linalg.norm(axis))
			for end, inner in ((steps - 1, steps - 2), (0, 1)):
				axis = ribbon[end] - ribbon[inner]
				axis = axis / np.linalg.norm(axis)
				centers.append(ribbon[end] + axis * cfg.RING_EXIT)
				axes.append(axis)
		return np.asarray(centers, dtype=np.float32), np.asarray(axes, dtype=np.float32)

	def _sink(self, ribbons, rings_pos, rings_dir):
		per = rings_pos.shape[0] // ribbons.shape[0]
		ribbons = ribbons.copy()
		for index in range(self.sunk_count):
			ribbons[index, :, 1] *= -1.0
			rings_pos[index * per:(index + 1) * per, 1] *= -1.0
			rings_dir[index * per:(index + 1) * per, 1] *= -1.0
		return ribbons, rings_pos, rings_dir

	def _open_spot(self, sampler, noise, y_range, clearance, avoid, both):
		cfg = WorldCfg
		count = 6000
		points = self._random_points(noise, count, y_range)
		depth = sampler.sample(points)
		angles = torch.arange(16, device=self.device, dtype=torch.float32) * (math.tau / 16.0)
		dirs = torch.stack((torch.cos(angles), torch.zeros_like(angles), torch.sin(angles)), -1)
		reach = torch.tensor((40.0, 80.0, 120.0, 170.0, 230.0, 300.0), device=self.device)
		probes = points[:, None, None, :] + dirs[None, :, None, :] * reach[None, None, :, None]
		free = sampler.sample(probes).min(-1).values
		if both:
			free = torch.minimum(free, torch.roll(free, 8, 1))
		best, best_dir = free.max(-1)
		quality = torch.minimum(best, depth * 0.8)
		quality = torch.where(depth > clearance, quality, quality - 1000.0)
		if avoid is not None:
			delta = points - torch.tensor(avoid, device=self.device, dtype=torch.float32)
			flat = delta[:, [0, 2]]
			period = self.size[[0, 2]]
			flat = flat - torch.round(flat / period) * period
			far = flat.norm(dim=-1)
			quality = torch.where(far > 360.0, quality, quality - 500.0)
		quality = quality + noise.uniform((count,)) * 6.0
		winner = int(torch.argmax(quality))
		position = points[winner].cpu().numpy().astype(np.float32)
		direction = dirs[int(best_dir[winner])].cpu().numpy().astype(np.float32)
		return position, direction
