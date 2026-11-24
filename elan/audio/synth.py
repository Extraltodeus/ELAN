import math

import torch


class Synth:
	def __init__(self, device, sr, seed=0):
		self.device = torch.device(device)
		self.sr = int(sr)
		self.gen = torch.Generator(device=self.device)
		self.gen.manual_seed(int(seed))

	def samples(self, seconds):
		return max(1, int(round(seconds * self.sr)))

	def time(self, n, double=False):
		t = torch.arange(n, device=self.device, dtype=torch.float64) / self.sr
		return t if double else t.float()

	def tensor(self, values, double=False):
		return torch.as_tensor(values, device=self.device, dtype=torch.float64 if double else torch.float32)

	def zeros(self, *shape):
		return torch.zeros(shape, device=self.device)

	def quantize(self, f, n):
		return max(1, round(f * n / self.sr)) * self.sr / n

	def rand(self, *shape):
		return torch.rand(shape, generator=self.gen, device=self.device)

	def randn(self, *shape):
		return torch.randn(shape, generator=self.gen, device=self.device)

	def phase(self, f, n, offset=0.0):
		f = self.tensor(f, True).unsqueeze(-1)
		offset = self.tensor(offset, True).unsqueeze(-1)
		return torch.remainder(f * self.time(n, True) + offset, 1.0).float()

	def phase_curve(self, freq):
		return torch.remainder(torch.cumsum(freq.double() / self.sr, -1), 1.0).float()

	def sin(self, p):
		return torch.sin(p * (2.0 * math.pi))

	def increment(self, f, p):
		f = f.float() if torch.is_tensor(f) else self.tensor(f)
		dt = (f / self.sr).clamp(1e-7, 0.5)
		if dt.dim() == 0 or dt.shape == p.shape:
			return dt
		return dt.unsqueeze(-1)

	def saw(self, p, f):
		dt = self.increment(f, p)
		a = p / dt
		b = (p - 1.0) / dt
		blep = torch.where(p < dt, 2.0 * a - a * a - 1.0, torch.where(p > 1.0 - dt, b * b + 2.0 * b + 1.0, torch.zeros_like(p)))
		return 2.0 * p - 1.0 - blep

	def triangle(self, p):
		return 1.0 - 4.0 * (p - 0.5).abs()

	def freqs(self, n):
		return torch.fft.rfftfreq(n, 1.0 / self.sr, device=self.device)

	def noise(self, n, shape, channels=1):
		f = self.freqs(n)
		mag = shape(f).float().expand(channels, -1).contiguous()
		ph = self.rand(channels, f.numel()) * (2.0 * math.pi)
		spec = torch.polar(mag, ph)
		spec[:, 0] = 0
		return self.unit(torch.fft.irfft(spec, n=n))

	def unit(self, x):
		return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)

	@staticmethod
	def lowpass(fc, order=2):
		return lambda f: 1.0 / torch.sqrt(1.0 + (f / fc).pow(2 * order))

	@staticmethod
	def highpass(fc, order=2):
		return lambda f: 1.0 / torch.sqrt(1.0 + (fc / f.clamp_min(1e-3)).pow(2 * order))

	@staticmethod
	def bandpass(lo, hi, order=2):
		lp = Synth.lowpass(hi, order)
		hp = Synth.highpass(lo, order)
		return lambda f: lp(f) * hp(f)

	@staticmethod
	def peak(fc, width, gain=1.0):
		return lambda f: gain * torch.exp(-0.5 * ((f - fc) / width).pow(2))

	@staticmethod
	def tilt(exponent, ref=1000.0):
		return lambda f: (f.clamp_min(10.0) / ref).pow(exponent)

	@staticmethod
	def product(*shapes):
		def shape(f):
			out = shapes[0](f)
			for s in shapes[1:]:
				out = out * s(f)
			return out
		return shape

	@staticmethod
	def total(*shapes):
		def shape(f):
			out = shapes[0](f)
			for s in shapes[1:]:
				out = out + s(f)
			return out
		return shape

	def filter(self, x, shape):
		n = x.shape[-1]
		return torch.fft.irfft(torch.fft.rfft(x) * shape(self.freqs(n)), n=n)

	def circular(self, x, ir):
		n = x.shape[-1]
		m = min(ir.shape[-1], n)
		return torch.fft.irfft(torch.fft.rfft(x) * torch.fft.rfft(ir[..., :m], n=n), n=n)

	def linear(self, x, ir):
		n = x.shape[-1]
		size = n + ir.shape[-1]
		size += size % 2
		return torch.fft.irfft(torch.fft.rfft(x, n=size) * torch.fft.rfft(ir, n=size), n=size)[..., :n]

	def reverb_ir(self, seconds, t60, damp=6000.0, predelay=0.012, channels=2):
		n = self.samples(seconds)
		t = self.time(n)
		ir = self.filter(self.randn(channels, n), self.lowpass(damp, 1)) * torch.exp(-6.9078 * t / t60)
		ir[:, :self.samples(predelay)] = 0.0
		return ir / ir.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)

	def pingpong_ir(self, delay, feedback, taps):
		d = max(1, int(delay))
		ir = self.zeros(2, d * taps + 1)
		for k in range(1, taps + 1):
			ir[(k - 1) % 2, k * d] = feedback ** (k - 1)
		return ir

	def space(self, x, ir, amount, circular=True):
		x = self.stereo(x)
		y = self.circular(x, ir) if circular else self.linear(x, ir)
		return x + amount * y

	def echo(self, x, delay, feedback, taps, amount, circular=True):
		x = self.stereo(x)
		mono = x.mean(0, keepdim=True).expand(2, -1)
		ir = self.pingpong_ir(delay, feedback, taps)
		y = self.circular(mono, ir) if circular else self.linear(mono, ir)
		return x + amount * y

	def stereo(self, x):
		return torch.stack([x, x]) if x.dim() == 1 else x

	def pan(self, x, p):
		a = (torch.as_tensor(p, device=self.device, dtype=torch.float32) + 1.0) * (math.pi / 4.0)
		return torch.stack([x * torch.cos(a), x * torch.sin(a)]) * math.sqrt(2.0)

	def pan_gains(self, pans, weights=None):
		p = self.tensor(pans)
		w = torch.ones_like(p) if weights is None else self.tensor(weights)
		a = (p + 1.0) * (math.pi / 4.0)
		return torch.stack([torch.cos(a) * w, torch.sin(a) * w]) * math.sqrt(2.0)

	def ramp(self, t, start, length):
		x = ((t - start) / max(length, 1e-6)).clamp(0.0, 1.0)
		return torch.sin(x * (math.pi / 2.0)).pow(2)

	def decay(self, t, start, tau):
		d = t - start
		return torch.where(d >= 0, torch.exp(-d.clamp_min(0.0) / tau), torch.zeros_like(d))

	def lfo(self, n, cycles, phase=0.0):
		return torch.sin(self.time(n) * (2.0 * math.pi * cycles * self.sr / n) + phase)

	def shift(self, x, seconds, n):
		out = torch.zeros(x.shape[:-1] + (n,), device=self.device)
		o = min(self.samples(seconds) if seconds > 0 else 0, n)
		m = min(x.shape[-1], n - o)
		out[..., o:o + m] = x[..., :m]
		return out

	def place_stereo(self, buf, seg, start):
		n = buf.shape[-1]
		idx = (torch.arange(seg.shape[-1], device=self.device) + int(start)) % n
		buf.index_add_(1, idx, seg)
		return buf

	def place_notes(self, buf, notes, starts, gains=None):
		n = buf.shape[-1]
		k, m = notes.shape
		s = torch.as_tensor(starts, device=self.device, dtype=torch.long).reshape(-1, 1)
		idx = ((s + torch.arange(m, device=self.device)) % n).reshape(-1)
		src = notes.unsqueeze(0).expand(buf.shape[0], k, m) if gains is None else gains.unsqueeze(-1) * notes.unsqueeze(0)
		buf.index_add_(1, idx, src.reshape(buf.shape[0], -1))
		return buf

	def softclip(self, x, drive=1.0):
		return torch.tanh(x * drive) / math.tanh(drive)

	def normalize(self, x, peak=0.95):
		return x * (peak / x.abs().max().clamp_min(1e-9))

	def fade(self, x, fin=0.0, fout=0.0):
		x = x.clone()
		n = x.shape[-1]
		a = min(self.samples(fin), n) if fin > 0 else 0
		b = min(self.samples(fout), n) if fout > 0 else 0
		if a:
			x[..., :a] *= torch.sin(torch.linspace(0.0, math.pi / 2.0, a, device=self.device)).pow(2)
		if b:
			x[..., n - b:] *= torch.cos(torch.linspace(0.0, math.pi / 2.0, b, device=self.device)).pow(2)
		return x

	def to_int16(self, x):
		x = self.stereo(x)
		return (x.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).t().contiguous().cpu().numpy()
