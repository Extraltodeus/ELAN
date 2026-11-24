import math

import torch

from elan.config import RulesCfg

from .music import MusicLayers, midi_hz
from .synth import Synth

LOOP_NAMES = ('wind_low', 'wind_mid', 'wind_high', 'engine', 'boost', 'graze', 'tether', 'skim')
MUSIC_NAMES = ('music_pad', 'music_bass', 'music_arp', 'music_drive')
SHOT_NAMES = ('tether_fire', 'tether_release', 'crash', 'skim_hit', 'gate_open', 'warp', 'ribbon', 'flux_low', 'flux_dry', 'game_over', 'ui_move', 'ui_select', 'start')
PENTATONIC = (0, 3, 5, 7, 10)


class LoopFactory:
	def __init__(self, synth):
		self.s = synth

	def build(self):
		return {name: getattr(self, name)() for name in LOOP_NAMES}

	def wobble(self, n, lo, hi, depth):
		return (1.0 + depth * self.s.noise(n, Synth.bandpass(lo, hi, 2), 1)[0]).clamp_min(0.05)

	def airy(self, n, shape, common, wide):
		s = self.s
		return common * s.noise(n, shape, 1) + wide * s.noise(n, shape, 2)

	def wind_low(self):
		n = self.s.samples(6.0)
		x = self.airy(n, Synth.product(Synth.bandpass(40.0, 300.0, 2), Synth.tilt(-0.5, 100.0)), 0.7, 0.5)
		return x * self.wobble(n, 0.15, 1.2, 0.35)

	def wind_mid(self):
		n = self.s.samples(5.0)
		x = self.airy(n, Synth.bandpass(300.0, 1800.0, 2), 0.5, 0.6)
		return x * self.wobble(n, 0.3, 2.5, 0.35)

	def wind_high(self):
		n = self.s.samples(4.0)
		shape = Synth.total(
			Synth.product(Synth.bandpass(2000.0, 9000.0, 2), lambda f: 0.6 * torch.ones_like(f)),
			Synth.peak(2600.0, 25.0, 3.0), Synth.peak(3900.0, 35.0, 2.0), Synth.peak(5200.0, 50.0, 1.2))
		x = self.airy(n, shape, 0.4, 0.7)
		return x * self.wobble(n, 0.5, 3.0, 0.4)

	def engine(self):
		s = self.s
		n = s.samples(4.0)
		f0 = s.quantize(55.0, n)
		fl = s.quantize(55.25, n)
		fr = s.quantize(54.75, n)
		p0 = s.phase(f0, n, 0.37)
		base = 0.5 * s.saw(p0, f0) + 0.8 * s.sin(p0) + 0.25 * s.sin(s.phase(2.0 * f0, n, 0.11))
		left = base + 0.5 * s.saw(s.phase(fl, n, 0.61), fl)
		right = base + 0.5 * s.saw(s.phase(fr, n, 0.83), fr)
		x = s.filter(torch.stack([left, right]), Synth.lowpass(380.0, 2))
		x = s.unit(x) + 0.1 * s.noise(n, Synth.bandpass(60.0, 400.0, 2), 2)
		x = x * (1.0 + 0.12 * s.lfo(n, 2))
		return s.softclip(x * 0.4, 1.5)

	def boost(self):
		s = self.s
		n = s.samples(3.0)
		roar = s.noise(n, Synth.product(Synth.bandpass(70.0, 4500.0, 1), Synth.tilt(-0.6, 200.0)), 2)
		f0 = s.quantize(82.5, n)
		harm = s.unit(s.filter(s.saw(s.phase(f0, n), f0), Synth.lowpass(1500.0, 2)))
		flutter = (1.0 + 0.3 * s.lfo(n, round(23 * n / s.sr))) * self.wobble(n, 8.0, 40.0, 0.35)
		x = (roar + 0.5 * harm) * flutter
		return s.softclip(x * 0.35, 2.0)

	def graze(self):
		s = self.s
		n = s.samples(3.0)
		count = int(500 * n / s.sr)
		density = self.wobble(n, 2.0, 12.0, 0.8).pow(2)
		buf = s.zeros(2, n)
		for c in range(2):
			pos = (s.rand(count) * n).long().clamp(0, n - 1)
			amp = (0.3 + s.rand(count)) * torch.sign(s.rand(count) - 0.5) * density[pos]
			buf[c].index_add_(0, pos, amp)
		clicks = s.softclip(s.unit(s.filter(buf, Synth.bandpass(1500.0, 6000.0, 2))) * 0.3, 1.0)
		fz = s.quantize(100.0, n)
		buzz = 0.6 + 0.4 * s.saw(s.phase(fz, n), fz)
		sizzle = s.noise(n, Synth.bandpass(3000.0, 7000.0, 2), 2) * buzz
		return clicks + 0.08 * sizzle

	def tether(self):
		s = self.s
		n = s.samples(2.0)
		chans = []
		for f0 in (s.quantize(110.0, n), s.quantize(110.5, n)):
			ks = list(range(1, 15))
			f = s.tensor([f0 * k for k in ks])
			a = s.tensor([1.0 / k ** 1.1 for k in ks]).unsqueeze(-1)
			chans.append((s.sin(s.phase(f, n, s.rand(len(ks)))) * a).sum(0))
		x = s.unit(torch.stack([0.75 * chans[0] + 0.25 * chans[1], 0.25 * chans[0] + 0.75 * chans[1]]))
		trem = 1.0 - 0.45 * (0.5 + 0.5 * s.lfo(n, round(11 * n / s.sr)))
		fizz = s.noise(n, Synth.bandpass(2000.0, 5000.0, 2), 2) * 0.08
		return (x + fizz) * trem

	def skim(self):
		s = self.s
		n = s.samples(3.0)
		shape = Synth.product(Synth.bandpass(800.0, 9000.0, 2), Synth.total(lambda f: torch.ones_like(f), Synth.peak(3500.0, 1200.0, 1.5)))
		spray = self.airy(n, shape, 0.4, 0.7) * self.wobble(n, 4.0, 25.0, 0.5)
		splash = s.noise(n, Synth.bandpass(150.0, 600.0, 2), 2) * self.wobble(n, 1.0, 5.0, 0.6) * 0.3
		return spray + splash


class OneShotFactory:
	def __init__(self, synth):
		self.s = synth

	def build(self):
		out = {'orb_%d' % i: self.orb(i) for i in range(16)}
		for name in SHOT_NAMES:
			out[name] = getattr(self, 'make_' + name)()
		return out

	def partials(self, freqs, amps, taus, n, attack=0.002):
		s = self.s
		keep = [(f, a, tau) for f, a, tau in zip(freqs, amps, taus) if 20.0 < f < 0.45 * s.sr]
		if not keep:
			return s.zeros(n)
		t = s.time(n)
		f = s.tensor([k[0] for k in keep])
		a = s.tensor([k[1] for k in keep]).unsqueeze(-1)
		tau = s.tensor([k[2] for k in keep]).unsqueeze(-1)
		ph = s.phase(f, n, s.rand(len(keep)))
		return (s.sin(ph) * a * torch.exp(-t / tau)).sum(0) * s.ramp(t, 0.0, attack)

	def chirp(self, t, start, f0, f1, tau):
		d = (t - start).clamp_min(0.0)
		ph = torch.remainder(f1 * d + (f0 - f1) * tau * (1.0 - torch.exp(-d / tau)), 1.0)
		return self.s.sin(ph) * (t >= start).float()

	def noise(self, n, shape, channels=2):
		s = self.s
		return s.unit(s.filter(s.randn(channels, n), shape))

	def orb(self, i):
		s = self.s
		n = s.samples(0.5)
		f = midi_hz(74 + 12 * (i // 5) + PENTATONIC[i % 5])
		ratios = (1.0, 2.0, 3.0, 4.16, 5.43, 6.8)
		amps = (1.0, 0.5, 0.3, 0.16, 0.1, 0.06)
		taus = (0.34, 0.2, 0.13, 0.08, 0.05, 0.035)
		left = self.partials([f * r * 0.9985 for r in ratios], amps, taus, n)
		right = self.partials([f * r * 1.0015 for r in ratios], amps, taus, n)
		t = s.time(n)
		click = self.noise(n, Synth.highpass(4000.0, 2), 1)[0] * torch.exp(-t / 0.004) * 0.15
		x = torch.stack([left + click, right + click])
		return s.space(x, s.reverb_ir(0.8, 0.7, 7000.0), 0.25, False)

	def make_tether_fire(self):
		s = self.s
		n = s.samples(0.5)
		t = s.time(n)
		ph = torch.remainder(180.0 * t + 2400.0 * 0.03 * (1.0 - torch.exp(-t / 0.03)), 1.0)
		zap = s.filter(torch.tanh(3.0 * s.sin(ph)) * torch.exp(-t / 0.09) * s.ramp(t, 0.0, 0.002), Synth.lowpass(6000.0, 1))
		crack = self.noise(n, Synth.highpass(2500.0, 2), 1)[0] * (0.9 * s.decay(t, 0.055, 0.01) + 0.25 * s.decay(t, 0.11, 0.02))
		x = s.pan(zap * 0.8, -0.1) + s.pan(crack * 0.6, 0.15)
		return s.space(x, s.reverb_ir(0.6, 0.5, 6000.0), 0.2, False)

	def make_tether_release(self):
		s = self.s
		n = s.samples(0.6)
		m = s.samples(0.38)
		f = midi_hz(57)
		ks = range(1, 7)
		pluck = self.partials([f * k for k in ks], [1.0 / k for k in ks], [0.3 / k ** 0.7 for k in ks], m)
		tm = s.time(m)
		rev = pluck.flip(-1) * s.ramp(tm, 0.0, 0.05) * (1.0 - s.ramp(tm, m / s.sr - 0.025, 0.025))
		x = s.pan(s.shift(rev, 0.0, n), -0.2)
		return s.space(x, s.reverb_ir(0.7, 0.6, 5000.0), 0.35, False)

	def make_crash(self):
		s = self.s
		n = s.samples(1.2)
		t = s.time(n)
		burst = self.noise(n, Synth.lowpass(3500.0, 1)) * torch.exp(-t / 0.12)
		rumble = self.noise(n, Synth.lowpass(300.0, 2)) * torch.exp(-t / 0.45)
		ph = torch.remainder(45.0 * t + 60.0 * 0.06 * (1.0 - torch.exp(-t / 0.06)), 1.0)
		thump = s.sin(ph) * torch.exp(-t / 0.3) * s.ramp(t, 0.0, 0.002)
		ring = self.partials((311.0, 523.0, 787.0, 1129.0, 1663.0, 2371.0, 3307.0), (1.0, 0.8, 0.7, 0.5, 0.35, 0.25, 0.15), (0.7, 0.55, 0.45, 0.35, 0.28, 0.2, 0.15), n)
		x = burst * 0.35 + rumble * 0.35 + s.stereo(thump + ring * 0.12)
		x = s.softclip(s.normalize(x, 1.0) * 1.5, 1.5)
		return s.space(x, s.reverb_ir(1.2, 1.0, 4000.0), 0.25, False)

	def make_skim_hit(self):
		s = self.s
		n = s.samples(0.6)
		t = s.time(n)
		slap = self.noise(n, Synth.bandpass(400.0, 5000.0, 2)) * torch.exp(-t / 0.045) * s.ramp(t, 0.0, 0.002)
		spray = self.noise(n, Synth.highpass(2500.0, 2)) * s.decay(t, 0.02, 0.16) * s.ramp(t, 0.02, 0.04) * 0.5
		thud = self.chirp(t, 0.0, 140.0, 70.0, 0.03) * torch.exp(-t / 0.07) * 0.7
		k = 6
		st = (0.06 + 0.3 * s.rand(k, 1))
		f0 = 1400.0 + 1600.0 * s.rand(k, 1)
		d = (t - st).clamp_min(0.0)
		ph = torch.remainder(f0 * d + f0 * 0.5 * d * d / 0.02, 1.0)
		drops = (s.sin(ph) * torch.where(t >= st, torch.exp(-d / 0.012) * s.ramp(d, 0.0, 0.002), torch.zeros_like(d))).sum(0) * 0.12
		x = slap + spray + s.stereo(thud + drops)
		return s.space(x, s.reverb_ir(0.5, 0.4, 6000.0), 0.2, False)

	def swell(self, midis, detune, n, freq_mul, saw=False):
		s = self.s
		freqs, pans = [], []
		for m in midis:
			for d, p in ((-detune, -0.6), (detune, 0.6)):
				freqs.append(midi_hz(m + d))
				pans.append(p)
		f = s.tensor(freqs).unsqueeze(-1) * freq_mul
		ph = s.phase_curve(f)
		voices = s.saw(ph, f) if saw else s.sin(ph) + 0.25 * s.sin(ph * 2.0)
		return voices, s.pan_gains(pans)

	def make_gate_open(self):
		s = self.s
		n = s.samples(2.5)
		t = s.time(n)
		rise = 2.0 ** (-5.0 * (1.0 - s.ramp(t, 0.0, 1.6)) / 12.0)
		voices, gains = self.swell((62, 69, 74, 76, 77, 81), 0.06, n, rise)
		rates = 6.0 + 0.7 * torch.arange(voices.shape[0], device=s.device).float().unsqueeze(-1)
		trem = 1.0 - 0.3 * (0.5 + 0.5 * torch.sin(2.0 * math.pi * rates * t + 6.28 * s.rand(voices.shape[0], 1)))
		chord = gains @ (voices * trem)
		sparkle = self.noise(n, Synth.bandpass(6000.0, 12000.0, 2)) * 0.06 * s.ramp(t, 0.3, 1.5)
		env = s.ramp(t, 0.0, 1.7) * (1.0 - s.ramp(t, 1.8, 0.5))
		x = (chord / voices.shape[0] * 2.0 + sparkle) * env
		return s.space(x, s.reverb_ir(2.0, 1.8, 6000.0), 0.4, False)

	def make_warp(self):
		s = self.s
		n = s.samples(3.5)
		t = s.time(n)
		hit = 2.4
		k = 24
		prog = (t / hit).clamp(0.0, 1.0)
		base = 150.0 + 750.0 * s.rand(k, 1)
		ph = s.phase_curve(base * 4.0 ** prog)
		pans = (s.rand(k) * 2.0 - 1.0) * 0.8
		swarm = s.pan_gains(pans.tolist()) @ s.sin(ph) / k
		env_r = prog.pow(2) * (1.0 - s.ramp(t, hit, 0.05))
		riser = (self.noise(n, Synth.bandpass(300.0, 1500.0, 2)) * (1.0 - prog) + self.noise(n, Synth.bandpass(2000.0, 9000.0, 2)) * prog * 0.8) * env_r
		tb = (t - hit).clamp_min(0.0)
		bph = torch.remainder(32.0 * tb + 50.0 * 0.08 * (1.0 - torch.exp(-tb / 0.08)), 1.0)
		boom = s.sin(bph) * s.decay(t, hit, 0.5) * s.ramp(tb, 0.0, 0.003)
		boom_noise = self.noise(n, Synth.lowpass(700.0, 2)) * s.decay(t, hit, 0.2)
		air = self.noise(n, Synth.bandpass(3000.0, 10000.0, 2)) * s.decay(t, hit, 0.9) * s.ramp(t, hit, 0.05)
		x = swarm * 2.0 * env_r + riser * 0.3 + s.stereo(boom) + boom_noise * 0.35 + air * 0.2
		x = s.softclip(s.normalize(x, 1.0) * 1.3, 1.3)
		return s.space(x, s.reverb_ir(2.0, 1.6, 6000.0), 0.3, False)

	def make_ribbon(self):
		s = self.s
		n = s.samples(0.9)
		x = s.zeros(2, n)
		notes = (77, 81, 84, 89, 93)
		ks = range(1, 9)
		for j, m in enumerate(notes):
			start = 0.075 * j
			last = j == len(notes) - 1
			f = midi_hz(m)
			pl = self.partials([f * k for k in ks], [1.0 / k ** 0.9 for k in ks], [(0.5 if last else 0.25) / k ** 0.5 for k in ks], n - s.samples(start))
			x = x + s.pan(s.shift(pl, start, n), -0.4 + 0.2 * j)
		x = s.echo(x, s.samples(0.12), 0.35, 4, 0.3, False)
		return s.space(x, s.reverb_ir(1.2, 1.0, 7000.0), 0.25, False)

	def make_flux_low(self):
		s = self.s
		n = s.samples(0.75)
		t = s.time(n)
		x = s.zeros(n)
		for start, f in ((0.0, 196.0), (0.2, 164.8)):
			d = (t - start).clamp_min(0.0)
			env = s.ramp(t - start, 0.0, 0.03) * (1.0 - s.ramp(t - start, 0.12, 0.14))
			voice = s.sin(torch.remainder(f * d, 1.0)) + 0.5 * s.sin(torch.remainder(f * 0.5 * d, 1.0)) + 0.12 * s.sin(torch.remainder(f * 2.0 * d, 1.0))
			x = x + voice * env
		x = s.filter(x, Synth.lowpass(900.0, 1))
		return s.space(x, s.reverb_ir(0.6, 0.5, 3000.0), 0.2, False)

	def make_flux_dry(self):
		s = self.s
		n = s.samples(1.7)
		t = s.time(n)
		rise = (t / RulesCfg.BREATH_SWELL).clamp(0.0, 1.0)
		swell = torch.sin(rise * math.pi).pow(2)
		turns = torch.cumsum((52.0 - 5.0 * rise).double() / s.sr, -1)
		tone = s.zeros(n)
		for order, level in ((1, 1.0), (2, 0.8), (3, 0.6), (4, 0.42), (5, 0.25), (6, 0.12)):
			tone = tone + level * s.sin(torch.remainder(turns * order, 1.0).float())
		tone = torch.tanh(tone * 0.6) * swell
		breath = self.noise(n, Synth.bandpass(160.0, 620.0, 2)) * swell.pow(2) * 0.14
		air = self.noise(n, Synth.bandpass(1800.0, 5000.0, 1)) * swell.pow(3) * 0.012
		return s.space(s.stereo(tone) + breath + air, s.reverb_ir(1.2, 1.0, 2500.0), 0.22, False)

	def make_game_over(self):
		s = self.s
		n = s.samples(3.0)
		t = s.time(n)
		fall = 2.0 ** (-7.0 * s.ramp(t, 0.2, 2.6) / 12.0)
		voices, gains = self.swell((38, 50, 53, 57, 62), 0.15, n, fall, True)
		x = gains @ voices / voices.shape[0]
		bright = s.filter(x, Synth.lowpass(2500.0, 2))
		dark = s.filter(x, Synth.lowpass(400.0, 2))
		x = dark + (bright - dark) * torch.exp(-t / 0.8)
		x = x * s.ramp(t, 0.0, 0.04) * torch.exp(-t / 1.1) * (1.0 - s.ramp(t, 2.5, 0.5))
		return s.space(x, s.reverb_ir(2.5, 2.2, 4000.0), 0.4, False)

	def make_ui_move(self):
		s = self.s
		n = s.samples(0.06)
		t = s.time(n)
		tick = s.sin(s.phase(2200.0, n)) * torch.exp(-t / 0.008) * s.ramp(t, 0.0, 0.0005)
		click = self.noise(n, Synth.highpass(3000.0, 2), 1)[0] * torch.exp(-t / 0.0015) * 0.3
		return s.stereo(tick + click)

	def make_ui_select(self):
		s = self.s
		n = s.samples(0.22)
		t = s.time(n)
		x = s.zeros(n)
		for start, length, f in ((0.0, 0.06, 880.0), (0.06, 0.1, 1320.0)):
			d = t - start
			env = s.ramp(d, 0.0, 0.003) * (1.0 - s.ramp(d, length - 0.015, 0.015))
			x = x + torch.tanh(2.0 * s.sin(torch.remainder(f * d.clamp_min(0.0), 1.0))) * env
		return s.space(s.stereo(x * 0.6), s.reverb_ir(0.4, 0.3, 7000.0), 0.15, False)

	def make_start(self):
		s = self.s
		n = s.samples(1.5)
		t = s.time(n)
		prog = (t / 1.0).clamp(0.0, 1.0)
		low = self.noise(n, Synth.bandpass(200.0, 1500.0, 2), 1)[0]
		high = self.noise(n, Synth.bandpass(1500.0, 8000.0, 2), 1)[0]
		env = s.ramp(t, 0.0, 0.45) * (1.0 - s.ramp(t, 0.45, 0.7))
		whoosh = s.pan((low * (1.0 - prog) + high * prog * 0.7) * env, -0.8 + 1.6 * (t / 0.9).clamp(0.0, 1.0))
		chord = s.zeros(2, n)
		ks = range(1, 6)
		notes = (62, 65, 69, 72, 76)
		for j, m in enumerate(notes):
			f = midi_hz(m)
			pl = self.partials([f * k for k in ks], [1.0 / k for k in ks], [0.9 / k ** 0.6 for k in ks], n - s.samples(0.35), 0.01)
			chord = chord + s.pan(s.shift(pl, 0.35, n), -0.6 + 0.3 * j)
		x = whoosh * 0.6 + chord * 0.35
		return s.space(x, s.reverb_ir(2.0, 1.6, 6000.0), 0.35, False)


class SoundBank:
	def __init__(self, device, sr, seed=0):
		self.synth = Synth(device, sr, seed)
		self.sounds = {}
		self.peaks = {}
		self.loops = LOOP_NAMES + MUSIC_NAMES

	def build(self):
		with torch.no_grad():
			raw = LoopFactory(self.synth).build()
			raw.update(MusicLayers(self.synth).build())
			for name, x in raw.items():
				self.sounds[name] = self.finish(name, x, True)
			raw = OneShotFactory(self.synth).build()
			for name, x in raw.items():
				self.sounds[name] = self.finish(name, x, False)
		raw = None
		if self.synth.device.type == 'cuda':
			torch.cuda.empty_cache()
		return self.sounds

	def finish(self, name, x, loop):
		s = self.synth
		x = torch.nan_to_num(s.stereo(x).float(), 0.0, 0.0, 0.0)
		if not loop:
			x = s.fade(x, 0.001, 0.015)
		x = s.normalize(x, 0.9 if loop else 0.95)
		self.peaks[name] = float(x.abs().max())
		return s.to_int16(x)
