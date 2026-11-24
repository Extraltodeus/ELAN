import math
import torch
import torch.nn.functional as F

from elan.config import WorldCfg
from elan.world.fieldops import FieldOps
from elan.world.forms import Forms

LABELS = ('raising spires', 'weaving arches', 'lifting islands')


class TerrainBuilder:
	def __init__(self, device):
		self.device = device

	def build(self, noise, shape, tick, breathe):
		cfg = WorldCfg
		v = cfg.VOXEL
		dims = (cfg.NZ, cfg.GEN_NY, cfg.NX)
		rows = torch.arange(cfg.GEN_NY, device=self.device, dtype=torch.float32)
		ys = ((rows - cfg.SEA_ROW + 0.5) * v).view(1, -1, 1)
		warp_x = noise.band(dims, v, 150.0, 0.45)
		warp_z = noise.band(dims, v, 150.0, 0.45)
		forms = Forms(self.device, noise, shape, ys, warp_x, warp_z)
		dist = None
		for index, (name, blend) in enumerate(shape['layers']):
			tick(LABELS[min(index, len(LABELS) - 1)])
			layer = forms.layer(name)
			dist = layer if dist is None else FieldOps.smooth_min(dist, layer, blend)
			del layer
			breathe()
		for index in range(len(shape['layers']), len(LABELS)):
			tick(LABELS[index])
		if shape['tori'] > 0:
			dist = forms.tori(dist)
		tick('eroding rock')
		flute = noise.fractal(dims, v, 7.0, 44.0, 1.5, aniso=[1.0, 3.2, 1.0])
		dist.add_(flute, alpha=shape['rough'])
		phase = ys * (math.tau / shape['strata_period']) + warp_z * 1.2
		phase.add_(flute, alpha=0.5)
		del flute
		strata = torch.asin(torch.sin(phase)) * (2.0 / math.pi)
		del phase
		dist.add_(strata, alpha=shape['strata_amp'])
		del strata, warp_x, warp_z, forms.warp_x, forms.warp_z
		dist = forms.fold(dist)
		dist = torch.maximum(dist, ys - cfg.CEILING)
		dist = dist[:, cfg.SEA_ROW:cfg.SEA_ROW + cfg.NY].contiguous()
		breathe()
		tick('measuring distances')
		return self._redistance(dist, breathe)

	def _redistance(self, dist, breathe):
		cfg = WorldCfg
		v = cfg.VOXEL
		norm = FieldOps.grad_norm(dist, v, wrap_y=False)
		dist = dist / norm.clamp_min_(1.0)
		del norm
		dist.clamp_min_(-10.0)
		solid = (dist < 0.0)
		solid_fraction = float(solid.float().mean())
		coarse = F.max_pool3d(solid.float()[None, None], 2)[0, 0] > 0.5
		del solid
		reach = FieldOps.distance_to_solid(coarse, cfg.EDT_RADIUS, breathe) * (2.0 * v)
		del coarse
		reach = FieldOps.upsample_periodic(reach, 2)
		low = (reach - 7.0).clamp_min_(0.0)
		high = reach + 6.0
		outside = torch.maximum(torch.minimum(dist, high), low)
		del low, high, reach
		dist = torch.where(dist < 0.0, dist, outside)
		return dist, solid_fraction
