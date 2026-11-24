import json
import math
import os
import threading
import wave

import numpy as np
import pygame

from .library import is_ogg, locate, read_ogg


class Soundtrack:
	TAU = 1.1

	def __init__(self, folder, sample_format, device):
		self.format = sample_format
		self.device = device
		self.sounds = []
		self.channels = []
		self.current = [0.0, 0.0, 0.0]
		self.applied = [0.0, 0.0, 0.0]
		self.ready = False
		self.folder = folder
		self.loaded = 'flight'
		self.wanted = 'flight'
		self.loading = False
		self.fresh = None
		try:
			self._load(folder)
		except Exception:
			self.sounds = []
			self.ready = False

	def _meta(self, folder, name):
		path = os.path.join(folder, 'music.json')
		if not os.path.isfile(path):
			return {}
		with open(path, 'r', encoding='utf-8') as handle:
			data = json.load(handle)
		if isinstance(data, dict) and isinstance(data.get('tracks'), dict):
			data = data['tracks']
		entries = data.items() if isinstance(data, dict) else enumerate(data)
		for key, entry in entries:
			if not isinstance(entry, dict):
				continue
			label = str(entry.get('file', key))
			if os.path.splitext(label)[0] == name or str(key) == name:
				return entry
		return {}

	def _read(self, path):
		if is_ogg(path):
			pcm, rate = read_ogg(path)
			return self._shape(pcm.astype(np.float32) / 32768.0, 2, rate)
		with wave.open(path, 'rb') as handle:
			channels = handle.getnchannels()
			width = handle.getsampwidth()
			rate = handle.getframerate()
			raw = handle.readframes(handle.getnframes())
		if width == 2:
			data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
		elif width == 3:
			packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
			value = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int8).astype(np.int32) << 16)
			data = value.astype(np.float32) / 8388608.0
		elif width == 4:
			data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
		else:
			raise ValueError('unsupported sample width')
		return self._shape(data, channels, rate)

	def _shape(self, data, channels, rate):
		data = data.reshape(-1, channels)
		if channels == 1:
			data = np.repeat(data, 2, axis=1)
		data = data[:, :2]
		if rate != self.format.rate:
			count = int(data.shape[0] * self.format.rate / rate)
			source = np.arange(data.shape[0], dtype=np.float64)
			target = np.linspace(0.0, data.shape[0] - 1.0, count)
			data = np.stack((np.interp(target, source, data[:, 0]), np.interp(target, source, data[:, 1])), 1).astype(np.float32)
		return data

	def _loop(self, data, meta):
		rate = self.format.rate
		start = int(float(meta.get('loop_start', 0.0) or 0.0) * rate)
		end = meta.get('loop_end', None)
		end = int(float(end) * rate) if end else data.shape[0]
		start = max(0, min(start, data.shape[0] - rate))
		end = max(start + rate, min(end, data.shape[0]))
		exact = 'loop_start_sample' in meta and 'loop_end_sample' in meta and int(meta.get('sample_rate', rate)) == rate
		if exact:
			start = max(0, int(meta['loop_start_sample']))
			end = max(start + rate, min(int(meta['loop_end_sample']), data.shape[0]))
		body = data[start:end].copy()
		fade = 0 if exact else min(int(0.06 * rate), body.shape[0] // 4)
		lead = data[max(start - fade, 0):start]
		if fade > 0 and lead.shape[0] == fade:
			ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
			body[-fade:] = body[-fade:] * np.sqrt(1.0 - ramp) + lead * np.sqrt(ramp)
		peak = float(np.abs(body).max())
		if peak > 1.0e-4:
			body = body * (0.89 / peak)
		return body

	def _deep(self, data):
		try:
			from .lowpass import LowPass
			return LowPass(self.format.rate).apply(data, 520.0, 1.2).clamp_(-1.0, 1.0).numpy()
		except Exception:
			return data * 0.5

	def _sound(self, data):
		pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
		return pygame.mixer.Sound(buffer=self.format.encode(np.ascontiguousarray(pcm)))

	def _load(self, folder):
		menu_path = locate(folder, 'menu')
		flight_path = locate(folder, 'flight')
		if not os.path.isfile(flight_path):
			return
		flight = self._loop(self._read(flight_path), self._meta(folder, 'flight'))
		menu = self._loop(self._read(menu_path), self._meta(folder, 'menu')) if os.path.isfile(menu_path) else flight
		self.sounds = [self._sound(menu), self._sound(flight), self._sound(self._deep(flight))]
		self.ready = True

	def set_track(self, name):
		if name:
			self.wanted = name

	def _fetch(self, name):
		try:
			body = self._loop(self._read(locate(self.folder, name)), self._meta(self.folder, name))
			self.fresh = (name, self._sound(body), self._sound(self._deep(body)))
		except Exception:
			self.wanted = self.loaded
		self.loading = False

	def _rotate(self, flight):
		if self.fresh is not None:
			name, bright, deep = self.fresh
			self.fresh = None
			self.loaded = name
			self.sounds[1] = bright
			self.sounds[2] = deep
			self.channels[1].play(bright, loops=-1, fade_ms=500)
			self.channels[2].play(deep, loops=-1, fade_ms=500)
			self.applied[1] = self.applied[2] = -1.0
		if flight and not self.loading and self.wanted != self.loaded:
			self.loading = True
			threading.Thread(target=self._fetch, args=(self.wanted,), daemon=True).start()

	def start(self, first_channel):
		if not self.ready:
			return
		self.channels = [pygame.mixer.Channel(first_channel + i) for i in range(3)]
		for channel in self.channels:
			channel.set_volume(0.0)
		self.channels[1].play(self.sounds[1], loops=-1)
		self.channels[2].play(self.sounds[2], loops=-1)
		self.channels[0].play(self.sounds[0], loops=-1)

	def update(self, dt, mix, scale, flight=True, menu=True):
		if not self.ready or not self.channels:
			return
		self._rotate(flight)
		duck = 0.3 if getattr(mix, 'paused', 0.0) > 0.5 else 1.0
		if getattr(mix, 'menu', 0.0) > 0.5:
			targets = (0.92 * duck, 0.0, 0.0)
		else:
			bright = min(max(getattr(mix, 'intensity', 0.0), 0.0), 1.0) ** 0.7
			if getattr(mix, 'submerged', 0.0) > 0.5:
				bright = 0.0
			targets = (0.0, 0.88 * (0.35 + 0.65 * bright) * duck, 0.7 * (1.0 - bright) * duck)
			if getattr(mix, 'serene', 0.0) > 0.5:
				targets = (0.0, 0.0, 0.85 * duck)
		if not flight:
			targets = (targets[0], 0.0, 0.0)
		if not menu:
			targets = (0.0, targets[1], targets[2])
		alpha = 1.0 - math.exp(-max(dt, 0.0) / self.TAU)
		for i in range(3):
			value = self.current[i] + (targets[i] - self.current[i]) * alpha
			self.current[i] = value
			volume = value * scale
			if volume < 0.002:
				volume = 0.0
			if abs(volume - self.applied[i]) > 0.004 or (volume == 0.0 and self.applied[i] != 0.0):
				self.channels[i].set_volume(volume)
				self.applied[i] = volume
