from elan.config import RulesCfg, StoryCfg


class Mode:
	KEY = 'chill'
	NAME = 'CHILL'
	BLURB = 'endless sectors  ·  rising drain  ·  chase the score'
	UNIT = 'SECTOR'
	LEVELS = 0
	DRAIN_GAIN = 1.0
	TIMED = False
	SCORE_KEY = 'best_score'
	DEPTH_KEY = 'best_sector'
	TIME_KEY = None

	def drain(self, sector):
		return RulesCfg.FLUX_DRAIN + sector * RulesCfg.FLUX_DRAIN_SECTOR * self.DRAIN_GAIN

	def clamp(self, sector):
		return max(int(sector), 0)

	def last(self, sector):
		return False

	def tunnel_kind(self, sector):
		return 'tunnel'

	def sector_kind(self, sector):
		return 'sector'

	def stage(self, sector):
		return '%s %d' % (self.UNIT, sector + 1)

	def keys(self, seed, sector):
		return ((self.tunnel_kind(sector), seed, sector), (self.sector_kind(sector + 1), seed, sector + 1))


class ChillMode(Mode):
	pass


class StoryMode(Mode):
	KEY = 'story'
	NAME = 'STORY'
	BLURB = '%d levels  ·  drain climbs %.1fx faster  ·  timed  ·  one destination' % (StoryCfg.LEVELS, StoryCfg.DRAIN_GAIN)
	UNIT = 'LEVEL'
	LEVELS = StoryCfg.LEVELS
	DRAIN_GAIN = StoryCfg.DRAIN_GAIN
	TIMED = True
	SCORE_KEY = 'story_best_score'
	DEPTH_KEY = 'story_best_level'
	TIME_KEY = 'story_best_time'

	def clamp(self, sector):
		return min(max(int(sector), 0), self.LEVELS - 1)

	def last(self, sector):
		return sector >= self.LEVELS - 1

	def tunnel_kind(self, sector):
		return 'plunge' if self.last(sector) else 'tunnel'

	def sector_kind(self, sector):
		return 'singularity' if sector >= self.LEVELS else 'sector'

	def stage(self, sector):
		return '%s %d / %d' % (self.UNIT, min(sector + 1, self.LEVELS), self.LEVELS)


MODES = (ChillMode(), StoryMode())
MODE_KEYS = tuple(mode.KEY for mode in MODES)


def mode_of(key):
	for mode in MODES:
		if mode.KEY == key:
			return mode
	return MODES[0]
