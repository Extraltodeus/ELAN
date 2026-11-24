import math
import numpy as np

from elan.config import RenderCfg


class ProximityRamp:
	STEPS = 64
	NEAR = 22.0
	FAR = 420.0
	SWEEP = 150.0
	CHROMA = 0.3
	LIGHT_NEAR = 0.8
	LIGHT_FAR = 0.74

	def __init__(self, palette):
		start = self._hue(palette['mote'])
		avoid = self._hue(palette['horizon'])
		direction = max((-1.0, 1.0), key=lambda d: min(self._apart(start + d * self.SWEEP * k / 8.0, avoid) for k in range(2, 9)))
		colors = np.zeros((self.STEPS, 3), dtype=np.float32)
		for step in range(self.STEPS):
			t = step / (self.STEPS - 1.0)
			light = self.LIGHT_NEAR + (self.LIGHT_FAR - self.LIGHT_NEAR) * t ** 0.8
			colors[step] = self._fit(light, math.radians(start + direction * self.SWEEP * t))
		bgr = colors[:, ::-1]
		self.glow = np.ascontiguousarray(bgr / bgr.max(1, keepdims=True))
		self.mark = np.ascontiguousarray(np.concatenate((bgr * RenderCfg.MARK_GAIN, np.ones((self.STEPS, 1), dtype=np.float32)), 1).T)
		span = math.log(self.FAR * self.FAR) - math.log(self.NEAR * self.NEAR)
		self.scale = (self.STEPS - 1.0) / span
		self.offset = -math.log(self.NEAR * self.NEAR) * self.scale

	def _apart(self, a, b):
		return abs((a - b + 180.0) % 360.0 - 180.0)

	def _hue(self, rgb):
		r, g, b = (max(float(c), 1.0e-4) for c in rgb)
		l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1.0 / 3.0)
		m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1.0 / 3.0)
		s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1.0 / 3.0)
		a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
		c = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
		return math.degrees(math.atan2(c, a))

	def _linear(self, light, chroma, hue):
		a = chroma * math.cos(hue)
		b = chroma * math.sin(hue)
		l = (light + 0.3963377774 * a + 0.2158037573 * b) ** 3
		m = (light - 0.1055613458 * a - 0.0638541728 * b) ** 3
		s = (light - 0.0894841775 * a - 1.2914855480 * b) ** 3
		return (
			4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
			-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
			-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
		)

	def _fit(self, light, hue):
		lo, hi = 0.0, self.CHROMA
		for _ in range(18):
			mid = (lo + hi) * 0.5
			rgb = self._linear(light, mid, hue)
			if min(rgb) >= 0.0 and max(rgb) <= 1.0:
				lo = mid
			else:
				hi = mid
		return np.clip(np.asarray(self._linear(light, lo, hue), dtype=np.float32), 0.0, 1.0)
