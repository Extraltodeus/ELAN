import math
import threading
import time

import numpy as np
import pygame

LOOPS = (
	('wind_low', 0),
	('wind_mid', 0),
	('wind_high', 0),
	('engine', 0),
	('boost', 0),
	('graze', 0),
	('tether', 0),
	('skim', 0),
	('music_pad', 1),
	('music_bass', 1),
	('music_arp', 1),
	('music_drive', 1),
)
TAUS = (0.12, 1.5)
CHANNELS = 34
RESERVED = len(LOOPS) + 9
FINALE_CHANNEL = len(LOOPS) + 7
CREDITS_CHANNEL = len(LOOPS) + 8
ORB_VOICES = 6
ORB_GAP = 0.03
FLUX_PERIOD = (1.15, 0.66)
FLUX_VOLUME = (0.6, 0.9)
SQRT2 = math.sqrt(2.0)


def clamp(x, lo=0.0, hi=1.0):
	return lo if x < lo else hi if x > hi else x


def window(x, lo, hi):
	return 0.0 if x <= lo else 1.0 if x >= hi else (x - lo) / (hi - lo)


class MixRules:
	def __init__(self):
		self.out = [0.0] * len(LOOPS)

	def compute(self, mix):
		g = getattr
		o = self.out
		duck = 0.25 if g(mix, 'paused', 0.0) > 0.5 else 1.0
		if g(mix, 'menu', 0.0) > 0.5:
			for i in range(8):
				o[i] = 0.0
			o[8] = 0.46 * duck
			o[9] = 0.05 * duck
			o[10] = 0.0
			o[11] = 0.0
			return o
		speed = clamp(g(mix, 'speed', 0.0), 0.0, 1.5)
		intensity = clamp(g(mix, 'intensity', 0.0))
		o[0] = 0.42 * speed * duck
		o[1] = 0.32 * speed ** 1.5 * duck
		o[2] = 0.3 * window(speed, 0.5, 1.2) ** 1.3 * duck
		o[3] = (0.08 + 0.1 * clamp(g(mix, 'thrust', 0.0))) * duck
		o[4] = 0.2 * clamp(g(mix, 'boost', 0.0)) * duck
		o[5] = 0.4 * clamp(g(mix, 'graze', 0.0)) * duck
		o[6] = 0.2 * clamp(g(mix, 'tether', 0.0)) * duck
		o[7] = 0.42 * clamp(g(mix, 'skim', 0.0)) * duck
		o[8] = 0.38 * duck
		o[9] = 0.22 * window(intensity, 0.15, 0.3) * duck
		o[10] = 0.3 * window(intensity, 0.4, 0.55) * duck
		o[11] = 0.44 * window(intensity, 0.7, 0.85) * duck
		if g(mix, 'serene', 0.0) > 0.5:
			o[8] = 0.46 * duck
			o[9] = 0.05 * duck
			o[10] = 0.14 * duck
			o[11] = 0.0
		return o


class SampleFormat:
	def __init__(self, rate, size, channels):
		self.rate = int(rate)
		self.size = int(size)
		self.channels = int(channels)

	def layout(self, pcm):
		if self.channels == 1:
			return ((pcm[:, 0].astype(np.int32) + pcm[:, 1]) // 2).astype(np.int16)
		if self.channels == 2:
			return pcm
		mid = ((pcm[:, 0].astype(np.int32) + pcm[:, 1]) // 2).astype(np.int16)
		return np.column_stack([pcm[:, 0], pcm[:, 1]] + [mid] * (self.channels - 2))

	def encode(self, pcm):
		data = self.layout(pcm)
		if self.size == 16:
			data = (data.astype(np.int32) + 32768).astype(np.uint16)
		elif self.size == 32:
			data = data.astype(np.float32) / 32768.0
		elif self.size == -32:
			data = data.astype(np.int32) << 16
		elif self.size == 8:
			data = ((data >> 8) + 128).astype(np.uint8)
		elif self.size == -8:
			data = (data >> 8).astype(np.int8)
		return np.ascontiguousarray(data).tobytes()


class AudioDirector:
	def __init__(self, device, master_volume=0.8, music_volume=0.7, sfx_volume=0.9):
		self.device = device
		self.enabled = False
		self.muted = False
		self.owns_mixer = False
		self.format = None
		self.sounds = {}
		self.peaks = {}
		self.build_time = 0.0
		self.channels = []
		self.kinds = [kind for _, kind in LOOPS]
		self.current = [0.0] * len(LOOPS)
		self.applied = [0.0] * len(LOOPS)
		self.rules = MixRules()
		self.flux_timer = 0.0
		self.orb_names = tuple('orb_%d' % i for i in range(16))
		self.master = clamp(master_volume)
		self.music = clamp(music_volume)
		self.sfx = clamp(sfx_volume)
		self.pickup = 0.5
		self.music_folder = None
		self.finale_path = None
		self.finale = None
		self.credits = None
		self.soundtrack = None
		self.deck = None
		self.playlist = None
		self.spare = None
		self.fetching = False
		self.played = False
		self.music_mode = 'dj'
		self.voices = None
		self.orbs = None
		self.orb_time = 0.0
		self.open_mixer()

	def open_mixer(self):
		try:
			info = pygame.mixer.get_init()
			if not info:
				pygame.mixer.init(44100, -16, 2, 1024)
				self.owns_mixer = True
				info = pygame.mixer.get_init()
			if not info:
				return
			pygame.mixer.set_num_channels(CHANNELS)
			pygame.mixer.set_reserved(RESERVED)
			from .voices import VoicePool
			self.voices = VoicePool(RESERVED, CHANNELS - RESERVED - ORB_VOICES)
			self.orbs = VoicePool(CHANNELS - ORB_VOICES, ORB_VOICES)
			self.format = SampleFormat(*info)
			self.enabled = True
		except Exception:
			self.enabled = False

	def build(self, seed=0):
		if not self.enabled:
			return False
		t0 = time.perf_counter()
		pcm = self.synthesize(seed)
		if pcm is None:
			self.enabled = False
			return False
		try:
			self.sounds = {name: pygame.mixer.Sound(buffer=self.format.encode(data)) for name, data in pcm.items()}
		except Exception:
			self.sounds = {}
			self.enabled = False
			return False
		if self.music_folder:
			from .soundtrack import Soundtrack
			self.soundtrack = Soundtrack(self.music_folder, self.format, self.device)
			threading.Thread(target=self.load_deck, daemon=True).start()
			self.finale = self.load_feature('finale', self.finale_path)
			self.credits = self.load_feature('credits')
		self.build_time = time.perf_counter() - t0
		return True

	def synthesize(self, seed):
		devices = [self.device] if str(self.device) == 'cpu' else [self.device, 'cpu']
		for device in devices:
			try:
				from .soundbank import SoundBank
				bank = SoundBank(device, self.format.rate, seed)
				data = bank.build()
				self.peaks = dict(bank.peaks)
				return data
			except Exception:
				continue
		return None

	def start(self):
		if not self.enabled or not self.sounds:
			return
		try:
			self.channels = [pygame.mixer.Channel(i) for i in range(len(LOOPS))]
			order = sorted(range(len(LOOPS)), key=lambda i: -self.kinds[i])
			for i in order:
				self.channels[i].play(self.sounds[LOOPS[i][0]], loops=-1)
			for i, ch in enumerate(self.channels):
				ch.set_volume(0.0)
				self.current[i] = 0.0
				self.applied[i] = 0.0
			if self.soundtrack is not None:
				self.soundtrack.start(len(LOOPS))
			if self.finale is not None:
				self.finale.bind(FINALE_CHANNEL)
			if self.credits is not None:
				self.credits.bind(CREDITS_CHANNEL)
		except Exception:
			self.channels = []

	def scales(self):
		if self.muted:
			return (0.0, 0.0)
		return (self.master * self.sfx, self.master * self.music)

	def update(self, dt, mix):
		if not self.enabled or not self.channels:
			return
		try:
			self.mix_loops(dt, mix)
			self.warn(dt, mix)
		except Exception:
			pass

	def mix_loops(self, dt, mix):
		targets = self.rules.compute(mix)
		dt = dt if dt > 0.0 else 0.0
		mode = self.active_mode()
		scale = self.scales()[1]
		flying = getattr(mix, 'menu', 0.0) < 0.5
		crowned = self.finale is not None and self.finale.ready and flying and getattr(mix, 'serene', 0.0) > 0.5
		rolling = self.credits is not None and self.credits.ready and getattr(mix, 'credits', 0.0) > 0.5
		if mode != 'synth' or crowned or rolling:
			targets[8] = targets[9] = targets[10] = targets[11] = 0.0
		if self.soundtrack is not None and self.soundtrack.ready:
			self.soundtrack.update(dt, mix, scale, mode == 'linear' and not crowned, mode != 'synth' and not rolling)
		if self.deck is not None:
			self.deck.update(dt, mix, scale * 0.95, mode == 'dj' and flying and not crowned)
		if self.finale is not None:
			self.finale.update(crowned, scale * (0.3 if getattr(mix, 'paused', 0.0) > 0.5 else 1.0))
		if self.credits is not None:
			self.credits.update(rolling, scale)
		alphas = (1.0 - math.exp(-dt / TAUS[0]), 1.0 - math.exp(-dt / TAUS[1]))
		scales = self.scales()
		cur = self.current
		applied = self.applied
		kinds = self.kinds
		chans = self.channels
		for i in range(len(chans)):
			k = kinds[i]
			c = cur[i] + (targets[i] - cur[i]) * alphas[k]
			cur[i] = c
			v = c * scales[k]
			if v < 0.002:
				v = 0.0
			a = applied[i]
			if v - a > 0.004 or a - v > 0.004 or (v == 0.0 and a != 0.0):
				chans[i].set_volume(v)
				applied[i] = v

	def warn(self, dt, mix):
		if getattr(mix, 'danger', 0.0) <= 0.5 or getattr(mix, 'dry', 0.0) > 0.5 or getattr(mix, 'paused', 0.0) > 0.5 or getattr(mix, 'menu', 0.0) > 0.5:
			self.flux_timer = 0.0
			return
		self.flux_timer -= dt
		if self.flux_timer > 0.0:
			return
		urgency = clamp((getattr(mix, 'danger', 0.0) - 0.5) * 2.0)
		self.flux_timer = FLUX_PERIOD[0] + (FLUX_PERIOD[1] - FLUX_PERIOD[0]) * urgency
		self.play('flux_low', FLUX_VOLUME[0] + (FLUX_VOLUME[1] - FLUX_VOLUME[0]) * urgency)

	def play(self, name, volume=1.0, pan=0.0, pool=None):
		if not self.enabled or self.muted:
			return
		snd = self.sounds.get(name)
		if snd is None:
			return
		try:
			v = clamp(volume, 0.0, 4.0) * self.master * self.sfx
			if v <= 0.0:
				return
			ch = (pool or self.voices).take()
			if ch is None:
				return
			ch.play(snd)
			if self.format.channels == 1:
				ch.set_volume(min(1.0, v))
				return
			a = (clamp(pan, -1.0, 1.0) + 1.0) * (math.pi / 4.0)
			ch.set_volume(min(1.0, v * SQRT2 * math.cos(a)), min(1.0, v * SQRT2 * math.sin(a)))
		except Exception:
			pass

	def play_orb(self, index, volume=1.0, pan=0.0):
		try:
			i = int(index)
		except Exception:
			i = 0
		now = time.perf_counter()
		if now - self.orb_time < ORB_GAP:
			return
		self.orb_time = now
		self.play(self.orb_names[0 if i < 0 else 15 if i > 15 else i], volume * self.pickup, pan, self.orbs)

	def play_pickup(self, name, volume=1.0):
		self.play(name, volume * self.pickup)

	def set_volumes(self, master, music, sfx, pickup=None):
		self.master = clamp(master)
		self.music = clamp(music)
		self.sfx = clamp(sfx)
		if pickup is not None:
			self.pickup = clamp(pickup)
		self.apply_now()

	def toggle_mute(self):
		self.muted = not self.muted
		self.apply_now()
		if self.muted and self.enabled:
			try:
				self.voices.stop()
				self.orbs.stop()
			except Exception:
				pass
		return self.muted

	def apply_now(self):
		if not self.enabled or not self.channels:
			return
		try:
			scales = self.scales()
			for i, ch in enumerate(self.channels):
				v = self.current[i] * scales[self.kinds[i]]
				v = 0.0 if v < 0.002 else v
				ch.set_volume(v)
				self.applied[i] = v
		except Exception:
			pass

	def load_feature(self, name, path=None):
		from .feature import FeatureTrack
		track = FeatureTrack(self.soundtrack, self.music_folder, name, path)
		if not track.available():
			return None
		threading.Thread(target=track.load, daemon=True).start()
		return track

	def load_deck(self):
		try:
			from .dj import DJDeck, StemSet
			from .playlist import Playlist
			self.playlist = Playlist(self.music_folder)
			name = self.playlist.current() or 'flight'
			stems = StemSet(self.music_folder, self.format, self.device, name=name)
			if stems.ready and self.enabled:
				self.deck = DJDeck(stems, len(LOOPS) + 3)
		except Exception:
			self.deck = None
		self.fetch_spare()

	def fetch_spare(self):
		playlist = self.playlist
		if playlist is None or len(playlist.names) < 2 or self.deck is None or self.fetching:
			return
		name = playlist.upcoming()
		if (self.spare is not None and self.spare.name == name) or not playlist.has_stems(name):
			return
		self.fetching = True
		threading.Thread(target=self._fetch, args=(name,), daemon=True).start()

	def _fetch(self, name):
		try:
			from .dj import StemSet
			stems = StemSet(self.music_folder, self.format, self.device, name=name)
			self.spare = stems if stems.ready else None
		except Exception:
			self.spare = None
		self.fetching = False

	def next_track(self, intro):
		playlist = self.playlist
		if playlist is None or len(playlist.names) < 2:
			return False
		name = playlist.upcoming()
		spare = self.spare
		swapped = False
		if self.deck is not None and playlist.has_stems(name):
			if spare is None or spare.name != name:
				self.fetch_spare()
				return False
			self.deck.swap(spare, intro)
			self.spare = None
			swapped = True
		playlist.advance()
		if self.soundtrack is not None:
			self.soundtrack.set_track(name)
		self.fetch_spare()
		return swapped

	def active_mode(self):
		linear = self.soundtrack is not None and self.soundtrack.ready
		if self.music_mode == 'dj' and self.deck is not None:
			return 'dj'
		if self.music_mode != 'synth' and linear:
			return 'linear'
		return 'synth'

	def set_music_mode(self, mode):
		self.music_mode = mode

	def music_event(self, name):
		swapped = False
		if name == 'warp' or (name == 'start' and self.played):
			swapped = self.next_track(name == 'start')
		if name == 'start':
			self.played = True
		if self.deck is not None and not swapped:
			self.deck.event(name)

	def shutdown(self):
		if self.deck is not None:
			self.deck.close()
		for track in (self.finale, self.credits):
			if track is not None:
				track.stop()
		if not self.enabled:
			return
		self.enabled = False
		try:
			for ch in self.channels:
				ch.stop()
			if self.owns_mixer:
				pygame.mixer.quit()
		except Exception:
			pass
		self.channels = []
		self.sounds = {}
