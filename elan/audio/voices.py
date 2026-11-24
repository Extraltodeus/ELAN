import pygame


class VoicePool:
	def __init__(self, first, count):
		self.channels = [pygame.mixer.Channel(first + i) for i in range(count)]
		self.stamps = [0] * count
		self.clock = 0

	def take(self):
		if not self.channels:
			return None
		pick = 0
		for i, channel in enumerate(self.channels):
			if not channel.get_busy():
				pick = i
				break
			if self.stamps[i] < self.stamps[pick]:
				pick = i
		self.clock += 1
		self.stamps[pick] = self.clock
		return self.channels[pick]

	def stop(self):
		for channel in self.channels:
			channel.stop()
