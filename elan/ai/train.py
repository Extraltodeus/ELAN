import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from elan.ai.coach import Coach
from elan.ai.course import Course
from elan.ai.observe import Observer
from elan.ai.policy import PilotNet
from elan.ai.sim import FlightSim, Reward
from elan.world import WorldBaker
from elan.world.palette import ORDER

SEEDS = (11, 23, 37, 41, 59, 67, 73, 89, 97, 101, 113, 127)
TUNNELS = ((211, 3), (223, 5), (227, 9))


class Trainer:
	HORIZON = 16
	EPOCHS = 4
	BATCHES = 16
	GAMMA = 0.995
	LAMBDA = 0.95
	CLIP = 0.2
	SMOOTH = 1.0
	NOISE = (0.3, 0.08)
	RATE = 3.0e-4

	def __init__(self, options):
		self.options = options
		self.device = torch.device('cuda:%d' % options.gpu)
		torch.manual_seed(options.seed)
		baker = WorldBaker(self.device)
		with torch.no_grad():
			worlds = []
			for k in range(options.worlds):
				sector = k % len(ORDER)
				world = baker.bake(SEEDS[sector] + 1000 * options.seed + 7 * (k // len(ORDER)), sector)
				world.tile = None
				world.aux_tile = None
				world.sky = None
				world.detail = None
				worlds.append(world)
			for seed, sector in TUNNELS[:options.tunnels]:
				worlds.append(baker.bake_tunnel(seed + options.seed, sector))
			torch.cuda.empty_cache()
			Reward.WINDOW = options.window
			if options.linear:
				Reward.SPEED, Reward.SPEED_CAP, Reward.CURVE = 3.9, 3.0, 1.0
			self.sim = FlightSim(Course(worlds, self.device, options.envs), self.device, options.seed)
		Reward.JERK *= options.grace
		Reward.TURN *= options.grace
		Reward.CRASH *= options.crash
		Reward.SCRAPE *= options.crash
		Reward.SPEED *= options.speed
		Reward.MOTE *= options.motes
		self.net = PilotNet(Observer.SIZE, options.hidden, options.layers).to(self.device)
		if options.resume:
			self.net.load_state_dict(torch.load(options.resume, map_location=self.device)['state'])
			self.NOISE = (options.noise, 0.08)
			self.RATE = 1.0e-4
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.RATE, eps=1.0e-5)
		self.samples = 0

	def save(self, path):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torch.save({
			'state': self.net.state_dict(), 'version': 2, 'size': self.net.size, 'hidden': self.net.hidden, 'layers': self.net.layers,
			'samples': self.samples, 'turn': FlightSim.TURN_MAX, 'pitch': FlightSim.PITCH_MAX, 'ease': FlightSim.EASE,
		}, path)

	def collect(self, observation):
		sim = self.sim
		net = self.net
		count = sim.count
		steps = self.HORIZON
		device = self.device
		width = PilotNet.ACTIONS
		obs = torch.zeros((steps, count, Observer.SIZE), device=device)
		acts = torch.zeros((steps, count, width), device=device)
		logs = torch.zeros((steps, count), device=device)
		rewards = torch.zeros((steps, count), device=device)
		dones = torch.zeros((steps, count), device=device)
		values = torch.zeros((steps + 1, count), device=device)
		tally = torch.zeros(FlightSim.STATS, device=device)
		with torch.no_grad():
			std = net.spread_out()
			for t in range(steps):
				mean = net.steer(observation)
				action = mean + std * torch.randn_like(mean)
				obs[t] = observation
				acts[t] = action
				logs[t] = (-0.5 * ((action - mean) / std) ** 2 - std.log() - 0.9189385).sum(1)
				values[t] = net.judge(observation)
				observation, reward, done = sim.step(action)
				rewards[t] = reward
				dones[t] = done.float()
				tally += sim.stats
			values[steps] = net.judge(observation)
			gain = torch.zeros(count, device=device)
			advantages = torch.zeros((steps, count), device=device)
			for t in range(steps - 1, -1, -1):
				alive = 1.0 - dones[t]
				delta = rewards[t] + self.GAMMA * values[t + 1] * alive - values[t]
				gain = delta + self.GAMMA * self.LAMBDA * alive * gain
				advantages[t] = gain
			returns = advantages + values[:steps]
			net.absorb(obs.view(-1, Observer.SIZE))
			after = torch.cat((obs[1:], observation.unsqueeze(0)), 0)
			keep = 1.0 - dones
		return observation, obs.view(-1, Observer.SIZE), acts.view(-1, width), logs.view(-1), advantages.view(-1), returns.view(-1), tally, after.view(-1, Observer.SIZE), keep.view(-1)

	def warm(self, observation):
		options = self.options
		sim = self.sim
		net = self.net
		coach = Coach()
		optimizer = torch.optim.Adam(net.actor.parameters(), lr=1.0e-3)
		steps = self.HORIZON
		for iteration in range(options.warm):
			seen = []
			wanted = []
			with torch.no_grad():
				for _ in range(steps):
					target = coach.act(observation)
					seen.append(observation)
					wanted.append(target)
					lead = target if iteration < options.warm // 2 else net.steer(observation)
					observation, _, _ = sim.step(lead + 0.25 * torch.randn_like(lead))
				obs = torch.cat(seen, 0)
				aim = torch.cat(wanted, 0)
				net.absorb(obs)
			total = obs.shape[0]
			size = total // 8
			for _ in range(2):
				order = torch.randperm(total, device=self.device)
				for k in range(8):
					pick = order[k * size:(k + 1) * size]
					loss = (net.steer(obs[pick]) - aim[pick] * 0.92).pow(2).mean()
					optimizer.zero_grad(set_to_none=True)
					loss.backward()
					optimizer.step()
			if iteration % 10 == 9:
				print('warm %3d  imitation loss %.4f' % (iteration + 1, float(loss.detach())), flush=True)
		net.noise.fill_(self.NOISE[0])
		for _ in range(options.settle):
			observation, obs, acts, logs, advantages, returns, counts, after, keep = self.collect(observation)
			self.learn(obs, acts, logs, advantages, returns, after, keep, False)
		return observation

	def learn(self, obs, acts, logs, advantages, returns, after, keep, steer=True):
		net = self.net
		total = obs.shape[0]
		size = total // self.BATCHES
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-6)
		for _ in range(self.EPOCHS):
			order = torch.randperm(total, device=self.device)
			for k in range(self.BATCHES):
				pick = order[k * size:(k + 1) * size]
				mean = net.steer(obs[pick])
				std = net.spread_out()
				fresh = (-0.5 * ((acts[pick] - mean) / std) ** 2 - std.log() - 0.9189385).sum(1)
				ratio = (fresh - logs[pick]).exp()
				gain = advantages[pick]
				policy = -torch.minimum(ratio * gain, ratio.clamp(1.0 - self.CLIP, 1.0 + self.CLIP) * gain).mean()
				value = (net.judge(obs[pick]) - returns[pick]).pow(2).mean()
				calm = (((mean[:, :3] - net.steer(after[pick])[:, :3]) ** 2).sum(1) * keep[pick]).mean()
				loss = policy + 0.5 * value + self.SMOOTH * self.options.grace * calm if steer else 0.5 * value
				self.optimizer.zero_grad(set_to_none=True)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
				self.optimizer.step()

	def report(self, iteration, tally, seconds, start):
		sim = self.sim
		t = tally.tolist()
		ticks = max(t[11], 1.0)
		fast = max(seconds / sim.STEP * sim.count - t[11], 1.0)
		calm = ticks * sim.STEP / 60.0
		minutes = seconds * sim.count / 60.0
		elapsed = time.time() - start
		print('it %5d  %6.1f M  %4.0f k/s  rew/s %6.2f | motes/min %6.1f  crash/min %5.2f  scrape %5.2f%%  speed %5.1f  boost %4.1f%%  flux %.2f | tunnel speed %5.1f crash/min %5.2f | jerk %.3f  gates/min %.2f  stalls/min %.3f  std %.2f  hard %.2f' % (
			iteration, self.samples / 1.0e6, self.samples / elapsed / 1000.0, t[7] / (minutes * 60.0), t[0] / calm, t[1] / calm, 100.0 * t[2] / ticks, t[3] / ticks, 100.0 * t[8] / ticks, t[9] / ticks,
			t[12] / fast, t[13] / (fast * sim.STEP / 60.0), t[4] / (minutes * 60.0 / sim.STEP) / 2.0, t[5] / minutes, t[10] / minutes, float(self.net.spread_out()[0]), sim.hardness), flush=True)

	def run(self):
		options = self.options
		sim = self.sim
		with torch.no_grad():
			observation = sim.start()
		if options.warm > 0 and not options.resume and not options.watch:
			sim.hardness = 0.0
			observation = self.warm(observation)
		if options.resume and not options.watch:
			for _ in range(options.settle):
				observation, obs, acts, logs, advantages, returns, counts, after, keep = self.collect(observation)
				self.learn(obs, acts, logs, advantages, returns, after, keep, False)
		start = time.time()
		budget = options.minutes * 60.0
		iteration = 0
		tally = torch.zeros(FlightSim.STATS, device=self.device)
		seconds = 0.0
		saved = start
		while time.time() - start < budget:
			done = min((time.time() - start) / budget, 1.0)
			for group in self.optimizer.param_groups:
				group['lr'] = self.RATE * max(1.0 - done, 0.1)
			sim.hardness = 1.0 if options.watch or options.resume else min(done / options.ramp, 1.0)
			self.net.noise.fill_(options.noise * 0.2 if options.watch else self.NOISE[0] + (self.NOISE[1] - self.NOISE[0]) * min(done / 0.85, 1.0))
			observation, obs, acts, logs, advantages, returns, counts, after, keep = self.collect(observation)
			if not options.watch:
				self.learn(obs, acts, logs, advantages, returns, after, keep)
			self.samples += obs.shape[0]
			tally += counts
			seconds += self.HORIZON * sim.STEP
			iteration += 1
			if iteration % options.report == 0:
				self.report(iteration, tally, seconds, start)
				tally.zero_()
				seconds = 0.0
			if not options.watch and time.time() - saved > 90.0:
				self.save(options.out)
				saved = time.time()
		if not options.watch:
			self.save(options.out)
		print('DONE %d iterations, %.1f M samples in %.1f min -> %s' % (iteration, self.samples / 1.0e6, (time.time() - start) / 60.0, options.out), flush=True)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=1)
	parser.add_argument('--minutes', type=float, default=20.0)
	parser.add_argument('--envs', type=int, default=32768)
	parser.add_argument('--worlds', type=int, default=24)
	parser.add_argument('--tunnels', type=int, default=3)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--hidden', type=int, default=384)
	parser.add_argument('--layers', type=int, default=3)
	parser.add_argument('--grace', type=float, default=1.0)
	parser.add_argument('--crash', type=float, default=1.0)
	parser.add_argument('--speed', type=float, default=1.0)
	parser.add_argument('--motes', type=float, default=1.0)
	parser.add_argument('--report', type=int, default=20)
	parser.add_argument('--noise', type=float, default=0.2)
	parser.add_argument('--ramp', type=float, default=0.45)
	parser.add_argument('--warm', type=int, default=70)
	parser.add_argument('--settle', type=int, default=25)
	parser.add_argument('--window', type=float, default=0.0)
	parser.add_argument('--linear', action='store_true')
	parser.add_argument('--resume', default='')
	parser.add_argument('--watch', action='store_true')
	parser.add_argument('--out', default='assets/ai/pilot.pt')
	Trainer(parser.parse_args()).run()


if __name__ == '__main__':
	main()
