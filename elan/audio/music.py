import math

import torch

from .synth import Synth

CHORDS = (
	(38, (50, 53, 57, 60, 64)),
	(34, (46, 53, 57, 62, 65)),
	(41, (53, 57, 60, 64, 67)),
	(33, (45, 52, 55, 60, 62)),
)
LEAD = (
	(0, 3, 81), (3, 1, 79), (4, 4, 77), (8, 3, 76), (11, 1, 74), (12, 4, 77),
	(16, 3, 74), (19, 1, 77), (20, 4, 81), (24, 3, 77), (27, 1, 76), (28, 4, 74),
	(32, 3, 79), (35, 1, 81), (36, 4, 84), (40, 3, 81), (43, 1, 79), (44, 4, 76),
	(48, 3, 74), (51, 1, 76), (52, 4, 79), (56, 3, 76), (59, 1, 72), (60, 4, 74),
)
ARP_PATTERN = (0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 4, 3, 1)
BASS_PATTERN = (0, 0, 12, 0, 0, 12, 0, 7)
BASS_ACCENT = (1.0, 0.7, 0.85, 0.7, 0.95, 0.7, 0.85, 0.75)


def midi_hz(m):
	return 440.0 * 2.0 ** ((m - 69) / 12.0)


class Arrangement:
	def __init__(self, sr, bpm=100, bars=16, bars_per_chord=4):
		self.sr = sr
		self.beat = sr * 60.0 / bpm
		self.beats = bars * 4
		self.beats_per_chord = bars_per_chord * 4
		self.length = int(round(self.beat * self.beats))

	def chord_index(self, beat):
		return int(beat // self.beats_per_chord) % len(CHORDS)

	def chord(self, beat):
		return CHORDS[self.chord_index(beat)]

	def at(self, beat):
		return int(round(beat * self.beat)) % self.length

	def starts(self, beats):
		return [self.at(b) for b in beats]

	def seconds(self, beats):
		return beats * self.beat / self.sr


class PadLayer:
	def __init__(self, synth, arr):
		self.s = synth
		self.arr = arr

	def render(self):
		s, arr = self.s, self.arr
		block = arr.at(arr.beats_per_chord)
		length = block + s.samples(3.2)
		t = s.time(length)
		env = s.ramp(t, 0.0, 1.8) * (1.0 - s.ramp(t, block / s.sr - 0.5, 2.7))
		buf = s.zeros(2, arr.length)
		for c, (root, notes) in enumerate(CHORDS):
			mixed = self.chord(notes, length) + s.sin(s.phase(midi_hz(root + 12), length)) * 0.4
			s.place_stereo(buf, mixed * env, c * block)
		bright = s.filter(buf, Synth.lowpass(2400.0, 2))
		dark = s.filter(buf, Synth.lowpass(650.0, 2))
		lfo = 0.5 + 0.5 * s.lfo(arr.length, 2, -math.pi / 2)
		x = dark + (bright - dark) * lfo * 0.85
		x = s.space(x, s.reverb_ir(5.0, 4.2, 4500.0, 0.03), 0.9)
		return s.filter(x, Synth.highpass(60.0, 2))

	def chord(self, notes, length):
		s = self.s
		freqs, pans, weights = [], [], []
		for j, m in enumerate(notes):
			for d, p in ((-0.12, -0.75), (0.0, 0.05), (0.11, 0.7)):
				freqs.append(midi_hz(m + d))
				pans.append(p)
				weights.append(1.0 - 0.1 * j)
		f = s.tensor(freqs)
		ph = s.phase(f, length, s.rand(len(freqs)))
		voices = 0.6 * s.saw(ph, f) + 0.5 * s.sin(ph)
		return s.pan_gains(pans, weights) @ voices


class BassLayer:
	def __init__(self, synth, arr):
		self.s = synth
		self.arr = arr

	def render(self):
		s, arr = self.s, self.arr
		count = arr.beats * 2
		length = int(arr.beat / 2)
		t = s.time(length)
		f = s.tensor([midi_hz(arr.chord(i / 2)[0] + BASS_PATTERN[i % 8]) for i in range(count)])
		ph = s.phase(f, length)
		tone = 0.6 * s.sin(ph)
		for k in range(1, 11):
			tone = tone + s.sin(ph * k) * (1.0 / k) * torch.exp(-t / (0.16 / (0.6 + 0.4 * k)))
		amp = (1.0 - torch.exp(-t / 0.003)) * (0.5 + 0.5 * torch.exp(-t / 0.12)) * (1.0 - s.ramp(t, length / s.sr - 0.02, 0.02))
		vel = s.tensor([BASS_ACCENT[i % 8] for i in range(count)]).unsqueeze(-1)
		buf = s.zeros(2, arr.length)
		s.place_notes(buf, tone * amp * vel, arr.starts([i / 2 for i in range(count)]))
		tb = torch.remainder(torch.arange(arr.length, device=s.device, dtype=torch.float64), arr.beat).float() / s.sr
		buf = buf * (1.0 - 0.5 * torch.exp(-tb / 0.07))
		x = s.softclip(s.normalize(buf, 1.0), 1.6)
		x = s.filter(x, Synth.product(Synth.lowpass(1400.0, 2), Synth.highpass(30.0, 2)))
		return s.space(x, s.reverb_ir(0.8, 0.5, 2500.0), 0.08)


class ArpLayer:
	def __init__(self, synth, arr):
		self.s = synth
		self.arr = arr

	def tones(self, beat):
		pcs = set(m % 12 for m in self.arr.chord(beat)[1])
		return [m for m in range(62, 86) if m % 12 in pcs]

	def render(self):
		s, arr = self.s, self.arr
		count = arr.beats * 4
		length = s.samples(0.45)
		t = s.time(length)
		freqs, vels, pans = [], [], []
		for i in range(count):
			beat = i // 4
			tones = self.tones(beat)
			step = ARP_PATTERN[i % 16] + (beat // 4) % 2
			freqs.append(midi_hz(tones[step % len(tones)]))
			vels.append(1.0 if i % 4 == 0 else 0.72 if i % 2 == 0 else 0.58)
			pans.append(-0.4 if i % 2 == 0 else 0.4)
		f = s.tensor(freqs)
		ph = s.phase(f, length)
		tone = torch.zeros_like(ph)
		for k in range(1, 9):
			a = (1.0 if k % 2 else 0.45) / k
			cut = (f * k < 0.45 * s.sr).float().unsqueeze(-1)
			tone = tone + s.sin(ph * k) * (a * cut) * torch.exp(-t / (0.3 / (1.0 + 0.55 * (k - 1))))
		amp = (1.0 - torch.exp(-t / 0.002)) * torch.exp(-t / 0.2) * (1.0 - s.ramp(t, length / s.sr - 0.03, 0.03))
		vel = (s.tensor(vels) * (0.9 + 0.2 * s.rand(count))).unsqueeze(-1)
		buf = s.zeros(2, arr.length)
		s.place_notes(buf, tone * amp * vel, arr.starts([i / 4 for i in range(count)]), s.pan_gains(pans))
		x = s.echo(buf, arr.beat * 0.75, 0.5, 6, 0.5)
		x = s.space(x, s.reverb_ir(2.5, 2.0, 5000.0), 0.3)
		return s.filter(x, Synth.product(Synth.highpass(180.0, 2), Synth.lowpass(7000.0, 1)))


class DriveLayer:
	def __init__(self, synth, arr):
		self.s = synth
		self.arr = arr

	def render(self):
		s = self.s
		return s.normalize(self.drums(), 0.9) + s.normalize(self.lead(), 0.55)

	def drums(self):
		s, arr = self.s, self.arr
		beats = arr.beats
		buf = s.zeros(2, arr.length)
		s.place_notes(buf, self.kick().expand(beats, -1), arr.starts(range(beats)))
		s.place_notes(buf, self.hats(beats, 0.09, 0.028, 0.32), arr.starts([b + 0.5 for b in range(beats)]), s.pan_gains([0.2] * beats))
		ghosts = beats * 2
		s.place_notes(buf, self.hats(ghosts, 0.05, 0.014, 0.12), arr.starts([b // 2 + 0.25 + 0.5 * (b % 2) for b in range(ghosts)]), s.pan_gains([-0.3] * ghosts))
		claps = [b for b in range(beats) if b % 4 in (1, 3)]
		clap_buf = s.zeros(2, arr.length)
		s.place_notes(clap_buf, self.claps(len(claps)), arr.starts(claps))
		clap_buf = s.space(clap_buf, s.reverb_ir(1.2, 0.9, 6000.0), 0.35)
		x = buf + 0.55 * clap_buf
		return s.space(x, s.reverb_ir(0.6, 0.4, 5000.0), 0.1)

	def kick(self):
		s = self.s
		n = s.samples(0.42)
		t = s.time(n)
		ph = torch.remainder(45.0 * t + 120.0 * 0.035 * (1.0 - torch.exp(-t / 0.035)), 1.0)
		body = s.sin(ph) * torch.exp(-t / 0.22) * s.ramp(t, 0.0, 0.002)
		click = s.filter(s.randn(n), Synth.bandpass(1500.0, 6000.0, 1)) * torch.exp(-t / 0.003) * 0.15
		return s.fade(body + click, 0.0, 0.03)

	def hats(self, count, seconds, tau, level):
		s = self.s
		n = s.samples(seconds)
		t = s.time(n)
		noise = s.unit(s.filter(s.randn(count, n), Synth.highpass(7000.0, 3)))
		return noise * torch.exp(-t / tau) * s.ramp(t, 0.0, 0.0008) * level * (0.85 + 0.3 * s.rand(count, 1))

	def claps(self, count):
		s = self.s
		n = s.samples(0.35)
		t = s.time(n)
		noise = s.unit(s.filter(s.randn(count, n), Synth.bandpass(900.0, 5000.0, 2)))
		env = s.decay(t, 0.0, 0.006) + s.decay(t, 0.011, 0.006) + s.decay(t, 0.022, 0.006) + 0.6 * s.decay(t, 0.03, 0.12)
		return s.fade(noise * env * 0.6, 0.0, 0.03)

	def lead(self):
		s, arr = self.s, self.arr
		rel = 0.9
		n = s.samples(arr.seconds(4) + rel + 0.1)
		t = s.time(n, True)
		f = s.tensor([midi_hz(m) for _, _, m in LEAD], True).unsqueeze(-1)
		dur = s.tensor([arr.seconds(d) for _, d, _ in LEAD]).unsqueeze(-1)
		rate = 5.2
		depth = 0.0045 * s.ramp(t.float(), 0.25, 0.4).double()
		base = f * t - f * depth * torch.cos(2.0 * math.pi * rate * t) / (2.0 * math.pi * rate)
		tf = t.float()
		env = s.ramp(tf, 0.0, 0.07) * (1.0 - s.ramp(tf.unsqueeze(0), dur, rel))
		buf = s.zeros(2, arr.length)
		starts = arr.starts([b for b, _, _ in LEAD])
		for det, pan in ((-0.07, -0.6), (0.07, 0.6)):
			ph = torch.remainder(base * 2.0 ** (det / 12.0), 1.0).float()
			tone = s.sin(ph) + 0.22 * s.sin(ph * 2.0) + 0.08 * s.sin(ph * 3.0) + 0.1 * s.sin(ph * 4.0)
			s.place_notes(buf, tone * env, starts, s.pan_gains([pan] * len(LEAD)))
		x = s.echo(buf, arr.beat * 0.75, 0.45, 6, 0.45)
		x = s.space(x, s.reverb_ir(4.0, 3.5, 6000.0, 0.02), 0.6)
		return s.filter(x, Synth.highpass(300.0, 2))


class MusicLayers:
	def __init__(self, synth):
		self.s = synth
		self.arr = Arrangement(synth.sr)

	def build(self):
		arr = self.arr
		return {
			'music_pad': PadLayer(self.s, arr).render(),
			'music_bass': BassLayer(self.s, arr).render(),
			'music_arp': ArpLayer(self.s, arr).render(),
			'music_drive': DriveLayer(self.s, arr).render(),
		}
