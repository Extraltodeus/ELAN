import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from elan.ai.pilot import LearnedPilot
from elan.autopilot import Autopilot
from elan.game import Game
from elan.world.bake import WorldBaker

RUNS = ((4242, 0), (909, 5), (31, 18), (77, 9), (5, 2), (64, 12))


class Quiet:
	def __getattr__(self, name):
		return lambda *args, **kwargs: None


class Flight:
	def __init__(self, baker, pilot, seed, first, quota):
		self.baker = baker
		self.pilot = pilot
		self.seed = seed
		self.first = first
		self.game = Game(Quiet(), Quiet())
		self.game.attract = False
		self.game.quota_limit = quota
		self.game.start_run(first)
		self.game.enter_world(baker.bake(seed, first), True)
		self.time = {'sector': 0.0, 'tunnel': 0.0}
		self.path = {'sector': 0.0, 'tunnel': 0.0}
		self.hits = {'sector': 0, 'tunnel': 0}
		self.dry = 0.0

	def run(self, seconds):
		game = self.game
		dt = 1.0 / 60.0
		for _ in range(int(seconds / dt)):
			phase = game.phase
			before = game.impacts
			game.update(dt, self.pilot.drive(game, dt))
			self.time[phase] += dt
			self.path[phase] += game.kite.speed * dt
			self.hits[phase] += game.impacts - before
			self.dry += dt if game.dry else 0.0
			if game.warp_requested:
				game.enter_tunnel(self.baker.bake_tunnel(self.seed, game.sector))
			elif game.arrival:
				world = self.baker.bake(self.seed, game.sector + 1)
				game.complete_warp()
				game.enter_world(world, False)
			if game.over:
				break
		return self


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('pilot', nargs='?', default='assets/ai/pilot.pt')
	parser.add_argument('--gpu', type=int, default=1)
	parser.add_argument('--seconds', type=float, default=300.0)
	parser.add_argument('--quota', type=int, default=0)
	parser.add_argument('--runs', type=int, default=len(RUNS))
	options = parser.parse_args()
	torch.set_grad_enabled(False)
	baker = WorldBaker(torch.device('cuda:%d' % options.gpu if torch.cuda.is_available() else 'cpu'))
	total = [0.0, 0, 0, 0, 0]
	start = time.time()
	for seed, first in RUNS[:options.runs]:
		pilot = Autopilot() if options.pilot == 'classic' else LearnedPilot(options.pilot)
		flight = Flight(baker, pilot, seed, first, options.quota).run(options.seconds)
		game = flight.game
		flown = flight.time['sector'] + flight.time['tunnel']
		total[0] += flown
		total[1] += game.sector - first
		total[2] += int(game.over)
		total[3] += flight.hits['sector']
		total[4] += game.motes_total
		print('seed %4d  sector %2d -> %2d  score %8d  motes %4d  %3.0f s  sector speed %3.0f  tunnel speed %3.0f  impacts %2d  dry %2.0f%%  %s' % (
			seed, first + 1, game.sector + 1, game.score, game.motes_total, flown, flight.path['sector'] / max(flight.time['sector'], 1.0), flight.path['tunnel'] / max(flight.time['tunnel'], 1.0),
			flight.hits['sector'], 100.0 * flight.dry / max(flight.time['sector'], 1.0), 'GAME OVER' if game.over else 'alive'), flush=True)
	print('TOTAL %s  flight %.0f s  sectors cleared %d  game overs %d  impacts/min %.2f  motes/min %.0f  (%.0f s)' % (
		options.pilot, total[0], total[1], total[2], total[3] * 60.0 / max(total[0], 1.0), total[4] * 60.0 / max(total[0], 1.0), time.time() - start))


if __name__ == '__main__':
	main()
