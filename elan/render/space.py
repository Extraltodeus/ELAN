import torch
import torch.nn.functional as F

from elan.config import WorldCfg


class Space:
	def __init__(self, device):
		cfg = WorldCfg
		self.device = device
		self.scale = torch.tensor((1.0 / cfg.SIZE_X, 2.0 / cfg.SIZE_Y, 1.0 / cfg.SIZE_Z), device=device)
		detail = torch.tensor((cfg.SIZE_X / cfg.DETAIL_HALF, cfg.SIZE_Y * 0.5 / cfg.DETAIL_HALF, cfg.SIZE_Z / cfg.DETAIL_HALF), device=device)
		self.detail_scale = detail
		self.detail_offset = detail + torch.tensor((0.0, 1.0, 0.0), device=device)
		self.period = torch.tensor((cfg.SIZE_X, 0.0, cfg.SIZE_Z), device=device)
		self.inv_period = torch.tensor((1.0 / cfg.SIZE_X, 0.0, 1.0 / cfg.SIZE_Z), device=device)

	def origin(self, pos):
		cfg = WorldCfg
		x = (pos[0] - cfg.SIZE_X * 0.5) % cfg.SIZE_X + cfg.SIZE_X * 0.5
		z = (pos[2] - cfg.SIZE_Z * 0.5) % cfg.SIZE_Z + cfg.SIZE_Z * 0.5
		return (x / cfg.SIZE_X - 1.0, pos[1] * 2.0 / cfg.SIZE_Y - 1.0, z / cfg.SIZE_Z - 1.0)


class Resampler:
	fast = None

	@staticmethod
	def resize(image, size):
		if Resampler.fast is None:
			try:
				torch._C._nn.upsample_bilinear2d(torch.zeros((1, 1, 2, 2), device=image.device), (4, 4), False, None)
				Resampler.fast = True
			except Exception:
				Resampler.fast = False
		if Resampler.fast:
			return torch._C._nn.upsample_bilinear2d(image, size, False, None)
		return F.interpolate(image, size=size, mode='bilinear', align_corners=False)

	@staticmethod
	def sharp(image, size):
		return F.interpolate(image, size=size, mode='bicubic', align_corners=False)
