import torch

from elan.ai.observe import Observer


class Coach:
	WAY = 19
	VALID = 39
	GOAL = 45
	FOUND = 52
	HEIGHT = 5

	def __init__(self):
		rays = len(Observer.RAYS)
		ranges = len(Observer.RANGES)
		self.sight = Observer.SIZE - (rays * ranges + len(Observer.LEADS) + len(Observer.DISTANT))
		self.leads = self.sight + rays * ranges
		self.ranges = ranges

	def _ray(self, observation, ray, reach):
		return observation[:, self.sight + ray * self.ranges + reach]

	def act(self, observation):
		way = observation[:, self.WAY:self.WAY + 3]
		valid = observation[:, self.VALID] > 0.5
		goal = observation[:, self.GOAL:self.GOAL + 3]
		found = observation[:, self.FOUND] > 0.5
		target = torch.where(valid.unsqueeze(1), way, goal)
		has = (valid | found).float()
		side, lift, ahead = target[:, 0], target[:, 1], target[:, 2]
		yaw = torch.atan2(side, ahead.clamp(min=1.0e-3)) * 2.2
		yaw = torch.where(ahead < 0.0, torch.sign(side + 1.0e-6), yaw)
		pitch = torch.atan2(lift, torch.sqrt(side * side + ahead * ahead).clamp(min=1.0e-3)) * 2.2
		front = torch.minimum(self._ray(observation, 0, 1), torch.minimum(observation[:, self.leads], observation[:, self.leads + 1]))
		danger = ((0.5 - front) / 0.5).clamp(0.0, 1.0)
		right = self._ray(observation, 4, 1) + 0.5 * self._ray(observation, 5, 1)
		left = self._ray(observation, 3, 1) + 0.5 * self._ray(observation, 2, 1)
		up = self._ray(observation, 8, 1)
		down = self._ray(observation, 7, 1)
		yaw = yaw * has + danger * 3.0 * (right - left + 0.02)
		pitch = pitch * has + danger * 3.0 * (up - down)
		pitch = pitch + (0.12 - observation[:, self.HEIGHT]).clamp(min=0.0) * 8.0
		return torch.stack((yaw.clamp(-1.0, 1.0), pitch.clamp(-1.0, 1.0), torch.ones_like(yaw), torch.full_like(yaw, -0.25)), 1)
