WHITE = (255, 255, 255)
PRIMARY = (220, 247, 255)
PRIMARY_DIM = (150, 185, 198)
CYAN = (125, 228, 255)
CYAN_DEEP = (60, 170, 215)
ACCENT = (255, 158, 58)
ACCENT_HOT = (255, 222, 160)
DANGER = (255, 72, 62)
GOOD = (150, 255, 205)
DARK = (4, 10, 16)
DEEP = (140, 200, 255)
DEEP_DIM = (105, 165, 235)
TIER = ((150, 185, 198), (125, 228, 255), (255, 158, 58), (255, 222, 160))
TRACK = (6, 14, 22, 150)
TRACK_EDGE = (190, 230, 245, 70)

FONT_CHAIN = ("bahnschrift", "segoeui", "consolas")
MONO_CHAIN = ("consolas", "lucidaconsole", "couriernew")
SYMBOL_CHAIN = ("segoeuisymbol", "segoeui", "arial")
THIN_CHAIN = ("segoeuisemilight", "segoeui", "bahnschrift")

REF_HEIGHT = 1080.0
REF_MIN_WIDTH = 1440.0

SIZES = {
	"tiny": 16,
	"small": 18,
	"label": 20,
	"body": 24,
	"mid": 30,
	"combo": 40,
	"score": 46,
	"speed": 84,
	"big": 60,
	"huge": 104,
	"title": 210,
	"mono": 16,
}


class Metrics:
	def __init__(self, size):
		self.resize(size)

	def resize(self, size):
		self.w = max(1, int(size[0]))
		self.h = max(1, int(size[1]))
		self.s = min(self.h / REF_HEIGHT, self.w / REF_MIN_WIDTH)
		self.cx = self.w // 2
		self.cy = self.h // 2

	def px(self, v):
		return max(1, int(round(v * self.s)))

	def font(self, name):
		return max(11, int(round(SIZES[name] * self.s)))


def lerp(a, b, t):
	return a + (b - a) * t


def mix(c1, c2, t):
	t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
	return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def clamp01(v):
	return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
