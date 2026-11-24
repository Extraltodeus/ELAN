import json
import os

import numpy as np
import pygame

from .library import locate


class FeatureTrack:
	LEVEL = 0.95
	FADE_OUT = 900

	def __init__(self, reader, folder, name, path=None):
		self.reader = reader
		self.folder = folder
		self.name = name
		self.path = path or locate(folder, name)
		self.channel = None
		self.intro = None
		self.body = None
		self.hush = None
		self.ready = False
		self.playing = False
		self.applied = -1.0
		self.length = 0.0
		self.lead = 0.0

	def available(self):
		return os.path.isfile(self.path)

	def _meta(self):
		side = os.path.splitext(self.path)[0] + '.json'
		if not os.path.isfile(side):
			return self.reader._meta(self.folder, self.name)
		with open(side, 'r', encoding='utf-8') as handle:
			data = json.load(handle)
		return data if isinstance(data, dict) else {}

	def _bounds(self, meta, total):
		rate = self.reader.format.rate
		if 'loop_start_sample' in meta:
			scale = rate / float(meta.get('sample_rate', rate) or rate)
			start = int(float(meta['loop_start_sample']) * scale)
			end = int(float(meta.get('loop_end_sample', 0) or 0) * scale) or total
		else:
			start = int(float(meta.get('loop_start', 0.0) or 0.0) * rate)
			end = int(float(meta.get('loop_end', 0.0) or 0.0) * rate) or total
		end = min(max(end, 0), total)
		start = min(max(start, 0), end)
		if end - start < rate:
			return 0, total
		return start, end

	def load(self):
		try:
			reader = self.reader
			data = reader._read(self.path)
			start, end = self._bounds(self._meta(), data.shape[0])
			self.intro = reader._sound(data[:start]) if start > 0 else None
			self.body = reader._sound(data[start:end])
			self.hush = reader._sound(np.zeros((64, 2), dtype=np.float32))
			self.lead = start / float(reader.format.rate)
			self.length = (end - start) / float(reader.format.rate)
			self.ready = True
		except Exception:
			self.ready = False

	def bind(self, index):
		self.channel = pygame.mixer.Channel(index)

	def _halt(self, fade):
		channel = self.channel
		if channel.get_busy():
			channel.queue(self.hush)
		if fade > 0:
			channel.fadeout(fade)
		else:
			channel.stop()
		self.playing = False

	def stop(self):
		if self.channel is not None and self.playing:
			self._halt(0)

	def update(self, wanted, scale):
		if not self.ready or self.channel is None:
			return
		channel = self.channel
		if not wanted:
			if self.playing:
				self._halt(self.FADE_OUT)
			return
		volume = min(self.LEVEL * scale, 1.0)
		if not self.playing:
			channel.set_volume(volume)
			channel.play(self.intro if self.intro is not None else self.body)
			self.playing = True
			self.applied = volume
		if channel.get_queue() is None:
			channel.queue(self.body)
		if abs(volume - self.applied) > 0.004:
			channel.set_volume(volume)
			self.applied = volume
