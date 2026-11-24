import torch
from torch import nn


class PilotNet(nn.Module):
	ACTIONS = 4

	def __init__(self, size, hidden=384, layers=3):
		super().__init__()
		self.size = size
		self.hidden = hidden
		self.layers = layers
		self.register_buffer('center', torch.zeros(size))
		self.register_buffer('spread', torch.ones(size))
		self.register_buffer('seen', torch.zeros(()))
		self.actor = self._stack(size, hidden, layers, self.ACTIONS)
		self.critic = self._stack(size, hidden, layers, 1)
		self.register_buffer('noise', torch.full((self.ACTIONS,), 0.5))
		with torch.no_grad():
			self.actor[-1].weight.mul_(0.05)
			self.actor[-1].bias.zero_()
			self.actor[-1].bias[2] = 1.0
			self.actor[-1].bias[3] = -1.0

	def _stack(self, size, hidden, layers, out):
		parts = []
		width = size
		for _ in range(layers):
			parts += [nn.Linear(width, hidden), nn.Tanh()]
			width = hidden
		parts.append(nn.Linear(width, out))
		return nn.Sequential(*parts)

	def absorb(self, batch):
		with torch.no_grad():
			total = self.seen + batch.shape[0]
			mean = batch.mean(0)
			var = batch.var(0, unbiased=False)
			delta = mean - self.center
			weight = batch.shape[0] / total
			self.spread.copy_(self.spread * (1.0 - weight) + var * weight + delta * delta * weight * (1.0 - weight))
			self.center.add_(delta * weight)
			self.seen.copy_(total.clamp(max=5.0e7))

	def scale(self, observation):
		return ((observation - self.center) / torch.sqrt(self.spread + 1.0e-5)).clamp(-8.0, 8.0)

	def steer(self, observation):
		return torch.tanh(self.actor(self.scale(observation)))

	def judge(self, observation):
		return self.critic(self.scale(observation)).squeeze(-1)

	def spread_out(self):
		return self.noise
