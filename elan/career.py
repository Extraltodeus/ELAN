from elan.world.palette import BIOMES

RANKS = ((60, 'LEGEND'), (40, 'VOIDWALKER'), (30, 'STARCHASER'), (20, 'VOYAGER'), (14, 'STORMRIDER'), (9, 'WAYFARER'), (5, 'GLIDER'), (2, 'DRIFTER'), (0, 'FLEDGLING'))


class Career:
	def __init__(self, settings):
		self.settings = settings
		self.mark = self.deepest()

	def deepest(self):
		return int(self.settings.get('best_sector'))

	def seen(self):
		return [key for key in self.settings.get('worlds_seen') if key in BIOMES]

	def rank(self, sector=None):
		sector = self.deepest() if sector is None else sector
		for need, name in RANKS:
			if sector >= need:
				return name
		return RANKS[-1][1]

	def begin(self):
		self.mark = self.deepest()

	def enter(self, sector, key, name):
		notes = []
		seen = self.seen()
		if key in BIOMES and key not in seen:
			seen.append(key)
			self.settings.set('worlds_seen', seen)
			notes.append(('NEW WORLD  -  %s  (%d / %d)' % (name, len(seen), len(BIOMES)), 'good'))
		if sector > self.deepest():
			before = self.rank()
			self.settings.set('best_sector', sector)
			if self.mark > 0 and sector > 1:
				notes.append(('NEW FRONTIER  -  SECTOR %d' % sector, 'good'))
			if self.rank() != before:
				notes.append(('RANK UP  -  %s' % self.rank(), 'big'))
		return notes

	def frontier(self, sector):
		return sector > self.mark and sector > 1

	def line(self):
		if self.deepest() <= 0:
			return None
		return 'SECTOR %d  -  %s  -  WORLDS %d / %d' % (self.deepest(), self.rank(), len(self.seen()), len(BIOMES))
