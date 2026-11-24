from elan.modes import MODES
from elan.world.palette import BIOMES

RANKS = ((60, 'LEGEND'), (40, 'VOIDWALKER'), (30, 'STARCHASER'), (20, 'VOYAGER'), (14, 'STORMRIDER'), (9, 'WAYFARER'), (5, 'GLIDER'), (2, 'DRIFTER'), (0, 'FLEDGLING'))


class Career:
	def __init__(self, settings):
		self.settings = settings
		self.mode = MODES[0]
		self.mark = self.deepest()

	def deepest(self, mode=None):
		return int(self.settings.get((mode or self.mode).DEPTH_KEY))

	def seen(self):
		return [key for key in self.settings.get('worlds_seen') if key in BIOMES]

	def rank(self, sector=None):
		sector = self.deepest() if sector is None else sector
		for need, name in RANKS:
			if sector >= need:
				return name
		return RANKS[-1][1]

	def begin(self, mode=None):
		self.mode = mode or MODES[0]
		self.mark = self.deepest()

	def enter(self, sector, key, name):
		notes = []
		mode = self.mode
		seen = self.seen()
		if key in BIOMES and key not in seen:
			seen.append(key)
			self.settings.set('worlds_seen', seen)
			notes.append(('NEW WORLD  -  %s  (%d / %d)' % (name, len(seen), len(BIOMES)), 'good'))
		if mode.LEVELS and sector > mode.LEVELS:
			return notes
		if sector > self.deepest():
			before = self.rank()
			self.settings.set(mode.DEPTH_KEY, sector)
			if self.mark > 0 and sector > 1:
				notes.append(('NEW FRONTIER  -  %s %d' % (mode.UNIT, sector), 'good'))
			if self.rank() != before:
				notes.append(('RANK UP  -  %s' % self.rank(), 'big'))
		return notes

	def frontier(self, sector):
		return sector > self.mark and sector > 1

	def clears(self):
		return int(self.settings.get('story_clears'))

	def line(self, mode=None):
		mode = mode or self.mode
		deepest = self.deepest(mode)
		if deepest <= 0:
			return None
		if mode.LEVELS:
			done = 'CLEARED x%d' % self.clears() if self.clears() else '%s %d / %d' % (mode.UNIT, min(deepest, mode.LEVELS), mode.LEVELS)
			return '%s  -  %s' % (done, self.rank(deepest))
		return '%s %d  -  %s  -  WORLDS %d / %d' % (mode.UNIT, deepest, self.rank(deepest), len(self.seen()), len(BIOMES))
