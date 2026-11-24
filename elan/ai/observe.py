import math
import torch

from elan.config import WorldCfg


class Observer:
	RAYS = ((0.0, 0.0), (-80.0, 0.0), (-45.0, 0.0), (-20.0, 0.0), (20.0, 0.0), (45.0, 0.0), (80.0, 0.0), (0.0, -25.0), (0.0, 25.0), (-35.0, -25.0), (-35.0, 25.0), (35.0, -25.0), (35.0, 25.0))
	DISTANT = (0, 3, 4, 7, 8)
	RANGES = (18.0, 50.0, 120.0)
	LEADS = (0.35, 0.8, 1.4, 2.0, 2.8)
	SIGHT = 60.0
	FAR = 250.0
	FAR_SIGHT = 90.0
	LOOKS = 5
	SIZE = 102

	def __init__(self, device):
		local = self.directions()
		self.rays = torch.tensor(local, device=device)
		self.ranges = torch.tensor(self.RANGES, device=device)
		self.leads = torch.tensor(self.LEADS, device=device)
		self.far = torch.tensor([local[k] for k in self.DISTANT], device=device)
		near = len(local) * len(self.RANGES) + len(self.LEADS)
		self.reach = torch.tensor([self.SIGHT] * near + [self.FAR_SIGHT] * len(self.DISTANT), device=device)

	@staticmethod
	def directions():
		local = []
		for yaw, pitch in Observer.RAYS:
			a = math.radians(yaw)
			b = math.radians(pitch)
			local.append((math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)))
		return local

	def basis(self, yaw, pitch):
		cp = torch.cos(pitch)
		forward = torch.stack((torch.sin(yaw) * cp, torch.sin(pitch), torch.cos(yaw) * cp), -1)
		right = torch.stack((torch.cos(yaw), torch.zeros_like(yaw), -torch.sin(yaw)), -1)
		up = torch.cross(forward, right, dim=-1)
		return torch.stack((right, up, forward), 1)

	def _cap(self, vectors, limit):
		return vectors * (limit / vectors.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)).clamp(max=1.0)

	def build(self, course, state, rel, valid, left, target, axis, found):
		pos, vel = state.pos, state.vel
		count = pos.shape[0]
		basis = self.basis(state.yaw, state.pitch)
		speed = vel.norm(dim=1, keepdim=True)
		local = torch.matmul(basis, vel.unsqueeze(2)).squeeze(2) / 100.0
		height = pos[:, 1:2]
		turned = basis.transpose(1, 2)
		way = self._cap(torch.matmul(rel, turned) / 100.0, 3.0) * valid.unsqueeze(2)
		reach = (rel.norm(dim=2) / 200.0).clamp(0.0, 3.0) * valid
		seen = found.float().unsqueeze(1)
		goal = self._cap(torch.matmul(target.unsqueeze(1), turned).squeeze(1) / 100.0, 4.0) * seen
		lead = torch.matmul(axis.unsqueeze(1), turned).squeeze(1) * seen
		span = (target.norm(dim=1, keepdim=True) / 200.0).clamp(0.0, 4.0) * seen
		aim = torch.matmul(self.rays, basis)
		probes = pos.view(count, 1, 1, 3) + aim.unsqueeze(2) * self.ranges.view(1, 1, -1, 1)
		ahead = pos.unsqueeze(1) + vel.unsqueeze(1) * self.leads.view(1, -1, 1)
		distant = pos.unsqueeze(1) + torch.matmul(self.far, basis) * self.FAR
		points = torch.cat((probes.view(count, -1, 3), ahead, distant), 1)
		sight = (course.distance(points) / self.reach).clamp(-0.25, 1.0)
		return torch.cat((
			local,
			speed / 150.0,
			state.pitch.unsqueeze(1),
			height / 200.0,
			((WorldCfg.CEILING - height.abs()) / 100.0).clamp(0.0, 1.0),
			state.flow.unsqueeze(1),
			state.rush.unsqueeze(1),
			state.gate.float().unsqueeze(1),
			(height < 0.0).float(),
			state.rates / state.limits,
			state.thrust.unsqueeze(1),
			state.boost.unsqueeze(1),
			state.flux.unsqueeze(1),
			state.drain.unsqueeze(1) * 20.0,
			course.tunnel.float().unsqueeze(1),
			course.hyper.float().unsqueeze(1),
			way.view(count, -1),
			reach,
			valid.float(),
			left.unsqueeze(1),
			goal,
			lead,
			span,
			seen,
			sight,
		), 1)
