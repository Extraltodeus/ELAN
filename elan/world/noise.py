import torch


class SpectralNoise:
	def __init__(self, device, seed):
		self.device = device
		self.gen = torch.Generator(device=device)
		self.gen.manual_seed(int(seed))

	def _radial(self, shape, voxel, aniso):
		dims = len(shape)
		k2 = None
		for i, size in enumerate(shape):
			if i == dims - 1:
				freq = torch.fft.rfftfreq(size, d=voxel, device=self.device)
			else:
				freq = torch.fft.fftfreq(size, d=voxel, device=self.device)
			view = [1] * dims
			view[i] = -1
			term = (freq * aniso[i]).view(view) ** 2
			k2 = term if k2 is None else k2 + term
		return torch.sqrt(k2)

	def _synthesize(self, shape, envelope):
		spec_shape = list(shape[:-1]) + [shape[-1] // 2 + 1, 2]
		noise = torch.randn(spec_shape, generator=self.gen, device=self.device)
		spec = torch.view_as_complex(noise) * envelope
		del noise
		field = torch.fft.irfftn(spec, s=tuple(shape))
		del spec
		field -= field.mean()
		field /= field.std().clamp_min(1e-8)
		return field

	def band(self, shape, voxel, wavelength, width=0.35, aniso=None):
		aniso = aniso or [1.0] * len(shape)
		k = self._radial(shape, voxel, aniso)
		k0 = 1.0 / wavelength
		env = torch.exp(-0.5 * ((k - k0) / (width * k0)) ** 2)
		env *= (k > 0)
		return self._synthesize(shape, env)

	def fractal(self, shape, voxel, wave_min, wave_max, beta=1.7, aniso=None):
		aniso = aniso or [1.0] * len(shape)
		k = self._radial(shape, voxel, aniso)
		k_lo = 1.0 / wave_max
		k_hi = 1.0 / wave_min
		env = k.clamp_min(k_lo) ** (-beta)
		env *= torch.exp(-(k / k_hi) ** 4)
		env *= (k >= k_lo * 0.6)
		return self._synthesize(shape, env)

	def uniform(self, shape):
		return torch.rand(shape, generator=self.gen, device=self.device)

	def normal(self, shape):
		return torch.randn(shape, generator=self.gen, device=self.device)
