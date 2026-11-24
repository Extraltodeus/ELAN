import numpy as np
import torch


class LowPass:
	def __init__(self, rate):
		self.rate = rate

	def apply(self, data, cutoff, gain):
		frames = data.shape[0]
		size = 1 << max(frames - 1, 1).bit_length()
		signal = torch.from_numpy(np.ascontiguousarray(data.T, dtype=np.float32))
		spectrum = torch.fft.rfft(signal, n=size, dim=1)
		freqs = torch.fft.rfftfreq(size, d=1.0 / self.rate)
		spectrum.mul_(1.0 / (1.0 + (freqs / cutoff) ** 4))
		return torch.fft.irfft(spectrum, n=size, dim=1)[:, :frames].T.mul(gain)
