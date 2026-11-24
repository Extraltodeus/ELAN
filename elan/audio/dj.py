import json
import math
import os
import threading
import time
import wave

import numpy as np
import pygame

from .library import is_ogg, locate, read_ogg

STEMS = ('drums', 'bass', 'other')
LAYERS = ('drums', 'bass', 'other', 'deep')


def smooth(lo, hi, x):
	if x <= lo:
		return 0.0
	if x >= hi:
		return 1.0
	t = (x - lo) / (hi - lo)
	return t * t * (3.0 - 2.0 * t)


class StemSet:
	def __init__(self, folder, sample_format, device, keep=False, name='flight'):
		self.name = name
		self.format = sample_format
		self.device = device
		self.keep = keep
		self.ready = False
		self.arrays = {}
		self.sounds = {}
		self.edges = []
		self.count = 0
		self.loop_start = 0
		self.loop_end = 0
		self.levels = None
		self.classes = []
		self.drive_starts = []
		self.calm_starts = []
		self.peak_start = 0
		self.build_start = 0
		self.outro_start = 0
		try:
			self._load(folder)
		except Exception:
			self.ready = False

	def _read(self, path):
		if is_ogg(path):
			pcm, rate = read_ogg(path)
			if rate != self.format.rate:
				raise ValueError('unsupported stem format')
			return pcm
		with wave.open(path, 'rb') as handle:
			if handle.getnchannels() != 2 or handle.getsampwidth() != 2 or handle.getframerate() != self.format.rate:
				raise ValueError('unsupported stem format')
			raw = handle.readframes(handle.getnframes())
		return np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)

	def _grid(self, folder, total):
		with open(os.path.join(folder, 'music.json'), 'r', encoding='utf-8') as handle:
			meta = json.load(handle)['tracks'][self.name]
		start = int(meta['loop_start_sample'])
		end = int(meta['loop_end_sample'])
		bars = int(meta['loop_bars'])
		edges = meta.get('bar_samples')
		if isinstance(edges, list) and len(edges) > bars and start in edges:
			edges = [int(e) for e in edges if int(e) <= total]
			self.edges = edges
			self.count = len(edges) - 1
			self.loop_start = edges.index(start)
			self.loop_end = min(self.loop_start + bars, self.count)
			return
		length = (end - start) / bars
		lead = int(start / length)
		first = start - lead * length
		self.count = int((total - first) // length)
		self.loop_start = lead
		self.loop_end = lead + bars
		self.edges = [int(round(first + k * length)) for k in range(self.count + 1)]

	def _deep(self, drums):
		try:
			import torch
			from .lowpass import LowPass
			out = LowPass(self.format.rate).apply(drums, 380.0, 1.5)
			return out.clamp_(-32767.0, 32767.0).to(torch.int16).numpy()
		except Exception:
			return (drums // 3).astype(np.int16)

	def _analyse(self):
		drums = self.arrays['drums']
		level = np.zeros(self.count, dtype=np.float32)
		for k in range(self.count):
			chunk = drums[self.edges[k]:self.edges[k + 1]:4].astype(np.float32)
			level[k] = math.sqrt(float((chunk * chunk).mean()) + 1.0e-9)
		reference = float(np.percentile(level, 90)) + 1.0e-6
		self.levels = level / reference
		phrases = self.count // 4
		self.classes = [1 if float(self.levels[p * 4:p * 4 + 4].mean()) >= 0.45 else 0 for p in range(phrases)]
		self.drive_starts = []
		self.calm_starts = []
		best = -1.0
		for bar in range(self.loop_start, self.loop_end, 4):
			kind = self.classes[bar // 4]
			previous = self.classes[bar // 4 - 1] if bar > self.loop_start else 1 - kind
			if kind == previous:
				continue
			if kind == 1:
				self.drive_starts.append(bar)
				power = float(self.levels[bar:bar + 4].mean())
				if power > best:
					best = power
					self.peak_start = bar
			else:
				self.calm_starts.append(bar)
		if not self.drive_starts:
			self.drive_starts = [self.loop_start]
			self.peak_start = self.loop_start
		if not self.calm_starts:
			self.calm_starts = [self.loop_start]
		self.build_start = self.calm_starts[-1]
		self.outro_start = min(self.loop_end + 8, max(self.count - 4, self.loop_end))

	def _load(self, folder):
		stems = os.path.join(folder, 'stems')
		for name in STEMS:
			self.arrays[name] = self._read(locate(stems, '%s_%s' % (self.name, name)))
		total = min(a.shape[0] for a in self.arrays.values())
		self._grid(folder, total)
		self._analyse()
		self.arrays['deep'] = self._deep(self.arrays['drums'])
		if hasattr(self.format, 'encode'):
			for name in LAYERS:
				data = self.arrays[name]
				self.sounds[name] = [pygame.mixer.Sound(buffer=self.format.encode(np.ascontiguousarray(data[self.edges[k]:self.edges[k + 1]]))) for k in range(self.count)]
		if not self.keep:
			self.arrays = {}
		self.ready = True


class DJBrain:
	def __init__(self, stems):
		self.stems = stems
		self.gains = {name: 0.0 for name in LAYERS}
		self.reset()

	def reset(self):
		stems = self.stems
		self.bar = max(stems.loop_start - 4, 0)
		self.since_jump = 99
		self.energy = 0.0
		self.low_time = 0.0
		self.pending = None
		self.stun = 0.0
		self.outro = False
		self.finished = False
		self.log = []

	def event(self, name):
		stems = self.stems
		if name == 'start':
			self.reset()
		elif name == 'warp':
			self.pending = ('bar', stems.peak_start)
		elif name == 'gate':
			self.pending = ('phrase', stems.build_start)
		elif name == 'crash':
			self.stun = 2.8
		elif name == 'over':
			self.outro = True
			self.pending = ('bar', stems.outro_start)

	def _after(self, starts, bar):
		later = [s for s in starts if s >= bar]
		return later[0] if later else starts[0]

	def choose(self):
		stems = self.stems
		following = self.bar + 1
		if self.pending is not None and (self.pending[0] == 'bar' or following % 4 == 0):
			target = self.pending[1]
			self.pending = None
			self.since_jump = 0
			self.log.append((self.bar, target, 'event'))
			return target
		if self.outro:
			if following >= stems.count:
				self.finished = True
				return None
			return following
		if following >= stems.loop_end:
			following = stems.loop_start
		self.since_jump += 1
		if following % 4 != 0 or self.since_jump < 8:
			return following
		kind = stems.classes[following // 4]
		if kind == 0 and self.energy > 0.58:
			target = self._after(stems.drive_starts, following)
		elif kind == 1 and self.energy < 0.2 and self.low_time > 5.0:
			target = self._after(stems.calm_starts, following)
		else:
			return following
		if target != following:
			self.since_jump = 0
			self.log.append((self.bar, target, 'drive' if kind == 0 else 'calm'))
		return target

	def tick(self, dt, mix):
		get = getattr
		serene = get(mix, 'serene', 0.0) > 0.5
		intensity = 0.0 if serene else min(max(get(mix, 'intensity', 0.0), 0.0), 1.0)
		rate = 1.1 if intensity > self.energy else 1.8
		self.energy += (intensity - self.energy) * (1.0 - math.exp(-dt / rate))
		self.low_time = self.low_time + dt if self.energy < 0.2 else 0.0
		self.stun = max(self.stun - dt, 0.0)
		energy = self.energy
		drums = smooth(0.2, 0.45, energy)
		bass = smooth(0.06, 0.25, energy)
		other = 0.78 + 0.22 * energy
		deep = 0.0
		if self.outro:
			drums, bass, other = 1.0, 1.0, 1.0
		if serene:
			drums, bass, other = 0.0, 0.0, 1.0
		if get(mix, 'submerged', 0.0) > 0.5:
			deep = 0.9 * max(drums, 0.6)
			drums = 0.0
			other *= 0.5
		tension = smooth(0.25, 0.7, get(mix, 'tether', 0.0))
		bass *= 1.0 - 0.92 * tension
		drums *= 1.0 - 0.25 * tension
		if serene:
			self.stun = 0.0
		if self.stun > 0.0:
			drums = 0.0
			deep = 0.0
			bass *= 0.4
		other *= 1.0 - 0.25 * min(max(get(mix, 'danger', 0.0), 0.0), 1.0)
		if get(mix, 'paused', 0.0) > 0.5:
			drums, deep, bass, other = 0.0, 0.0, bass * 0.3, other * 0.3
		targets = {'drums': drums, 'bass': bass, 'other': other, 'deep': deep}
		for name in LAYERS:
			target = targets[name]
			current = self.gains[name]
			quick = 0.07 if target < current else 0.2
			if serene:
				quick = 1.6
			self.gains[name] = current + (target - current) * (1.0 - math.exp(-dt / quick))


class DJDeck:
	TRIES = 4
	TIGHT = 0.002

	def __init__(self, stems, first_channel):
		self.stems = stems
		self.repairs = 0
		self.brain = DJBrain(stems)
		self.channels = [pygame.mixer.Channel(first_channel + i) for i in range(len(LAYERS))]
		self.hush = pygame.mixer.Sound(buffer=stems.format.encode(np.zeros((64, 2), dtype=np.int16)))
		self.lock = threading.Lock()
		self.active = False
		self.playing = False
		self.queued = None
		self.scale = 0.0
		self.applied = [-1.0] * len(LAYERS)
		self.alive = True
		self.thread = threading.Thread(target=self._run, daemon=True)
		self.thread.start()

	def event(self, name):
		with self.lock:
			brain = self.brain
			brain.event(name)
			if not self.playing:
				return
			if name == 'start':
				self._halt(60)
			elif name == 'warp':
				brain.pending = None
				brain.since_jump = 0
				brain.log.append((brain.bar, self.stems.peak_start, 'cut'))
				self._start(self.stems.peak_start, True)
			elif brain.pending is not None and brain.pending[0] == 'bar':
				self._queue(brain.choose())

	def swap(self, stems, intro):
		with self.lock:
			energy = self.brain.energy
			self._halt(90)
			self.stems = stems
			self.brain = DJBrain(stems)
			self.brain.energy = energy
			if not intro:
				self.brain.bar = stems.peak_start
				self.brain.since_jump = 0
			self.applied = [-1.0] * len(LAYERS)

	def update(self, dt, mix, scale, active):
		with self.lock:
			self.scale = scale
			self.active = active
			self.brain.tick(dt, mix)

	def close(self):
		with self.lock:
			self.alive = False
			self.active = False
			self._halt(0)
		if self.thread.is_alive() and threading.current_thread() is not self.thread:
			self.thread.join(1.0)

	def _halt(self, fade):
		for channel in self.channels:
			if channel.get_busy():
				channel.queue(self.hush)
			if fade > 0:
				channel.fadeout(fade)
			else:
				channel.stop()
		self.playing = False
		self.queued = None

	def _start(self, bar, keep=False):
		if not keep:
			for channel in self.channels:
				channel.set_volume(0.0)
			self.applied = [0.0] * len(LAYERS)
		sounds = [self.stems.sounds[layer][bar] for layer in LAYERS]
		for _ in range(self.TRIES):
			begin = time.perf_counter()
			for sound, channel in zip(sounds, self.channels):
				channel.play(sound)
			if time.perf_counter() - begin < self.TIGHT:
				break
		self.brain.bar = bar
		self.playing = True
		self.queued = None

	def _aligned(self):
		bar = self.brain.bar
		sounds = self.stems.sounds
		return all(channel.get_sound() is sounds[layer][bar] for layer, channel in zip(LAYERS, self.channels))

	def _queue(self, following):
		self.queued = following
		if following is None:
			return
		for layer, channel in zip(LAYERS, self.channels):
			channel.queue(self.stems.sounds[layer][following])

	def _service(self):
		brain = self.brain
		channels = self.channels
		if not self.active or self.scale <= 0.0:
			if self.playing:
				self._halt(350)
			return
		if self.playing and not any(channel.get_busy() for channel in channels):
			self.playing = False
			if brain.finished:
				return
			if self.queued is not None:
				brain.bar = self.queued
		if brain.finished:
			return
		if not self.playing:
			self._start(brain.bar)
		if all(channel.get_queue() is None for channel in channels):
			if self.queued is not None:
				brain.bar = self.queued
			if not self._aligned():
				self.repairs += 1
				self._start(brain.bar, True)
			self._queue(brain.choose())
		for i, layer in enumerate(LAYERS):
			volume = min(brain.gains[layer] * self.scale, 1.0)
			if volume < 0.002:
				volume = 0.0
			if abs(volume - self.applied[i]) > 0.004 or (volume == 0.0 and self.applied[i] != 0.0):
				self.channels[i].set_volume(volume)
				self.applied[i] = volume

	def _run(self):
		while self.alive:
			time.sleep(0.02)
			try:
				with self.lock:
					if self.alive:
						self._service()
			except Exception:
				continue
