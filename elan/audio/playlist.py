import json
import os


class Playlist:
	def __init__(self, folder):
		self.folder = folder
		self.names = []
		self.index = 0
		if not folder:
			return
		try:
			with open(os.path.join(folder, 'music.json'), 'r', encoding='utf-8') as handle:
				data = json.load(handle)
		except (OSError, ValueError):
			return
		tracks = data.get('tracks', {}) if isinstance(data, dict) else {}
		order = data.get('playlist') if isinstance(data, dict) else None
		if not isinstance(order, list):
			order = ['flight']
		self.names = [name for name in order if name in tracks and os.path.isfile(os.path.join(folder, '%s.wav' % name))]

	def current(self):
		return self.names[self.index % len(self.names)] if self.names else None

	def upcoming(self):
		return self.names[(self.index + 1) % len(self.names)] if self.names else None

	def advance(self):
		if self.names:
			self.index = (self.index + 1) % len(self.names)
		return self.current()

	def has_stems(self, name):
		stems = os.path.join(self.folder, 'stems')
		return all(os.path.isfile(os.path.join(stems, '%s_%s.wav' % (name, part))) for part in ('drums', 'bass', 'other'))
