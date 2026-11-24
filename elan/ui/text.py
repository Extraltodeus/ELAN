from collections import OrderedDict

import pygame

from . import theme

ANCHORS = {
	"topleft": (0.0, 0.0),
	"midtop": (0.5, 0.0),
	"topright": (1.0, 0.0),
	"midleft": (0.0, 0.5),
	"center": (0.5, 0.5),
	"midright": (1.0, 0.5),
	"bottomleft": (0.0, 1.0),
	"midbottom": (0.5, 1.0),
	"bottomright": (1.0, 1.0),
}

NOTDEF = ""


def anchor_origin(w, h, x, y, anchor):
	fx, fy = ANCHORS[anchor]
	return int(x - w * fx), int(y - h * fy)


def soft_shadow(mask, w, h, pad, offset, blur, alpha, crisp=True):
	W, H = w + 2 * pad, h + 2 * pad
	sh = pygame.Surface((W, H), pygame.SRCALPHA)
	sh.blit(mask, (pad, pad + offset))
	f = max(2, int(blur))
	small = pygame.transform.smoothscale(sh, (max(1, W // f), max(1, H // f)))
	sh = pygame.transform.smoothscale(small, (W, H))
	sh.blit(sh.copy(), (0, 0))
	if crisp:
		sh.blit(mask, (pad, pad + offset))
	sh.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
	return sh


def rle(surf):
	if surf.get_flags() & pygame.SRCALPHA:
		surf.set_alpha(255, pygame.RLEACCEL)
	return surf


class Label:
	LEVELS = 24

	def __init__(self, surf, pad=0, w=None, h=None):
		self.surf = rle(surf)
		self.pad = pad
		self.w = surf.get_width() - 2 * pad if w is None else w
		self.h = surf.get_height() - 2 * pad if h is None else h
		self.levels = {}

	def faded(self, a):
		if a >= 250:
			return self.surf
		q = int(a * self.LEVELS / 255.0 + 0.5)
		if q <= 0:
			return None
		s = self.levels.get(q)
		if s is not None:
			return s
		s = self.surf.copy()
		s.fill((255, 255, 255, int(255 * q / self.LEVELS)), special_flags=pygame.BLEND_RGBA_MULT)
		self.levels[q] = rle(s)
		return s

	def pos(self, x, y, anchor="topleft"):
		ox, oy = anchor_origin(self.w, self.h, x, y, anchor)
		return ox - self.pad, oy - self.pad

	def draw(self, screen, x, y, anchor="topleft", alpha=255):
		s = self.faded(alpha)
		if s is None:
			return
		screen.blit(s, self.pos(x, y, anchor))

	def blit(self, screen, pos, alpha=255, area=None):
		s = self.faded(alpha)
		if s is None:
			return
		if area is None:
			screen.blit(s, pos)
			return
		screen.blit(s, pos, area)


class Fonts:
	def __init__(self):
		if not pygame.font.get_init():
			pygame.font.init()
		self.cache = {}
		self.names = {}

	def resolve(self, chain):
		key = tuple(chain)
		if key in self.names:
			return self.names[key]
		found = [n for n in chain if pygame.font.match_font(n)]
		self.names[key] = found or [None]
		return self.names[key]

	def font(self, name, size, bold=False):
		key = (name, size, bold)
		f = self.cache.get(key)
		if f is not None:
			return f
		f = pygame.font.SysFont(name, size, bold=bold) if name else pygame.font.Font(None, size)
		self.cache[key] = f
		return f

	def supports(self, f, text):
		bad = f.metrics(NOTDEF)[0]
		for c, m in zip(text, f.metrics(text)):
			if c.isspace():
				continue
			if m is None or (ord(c) > 0x2000 and m == bad):
				return False
		return True

	def chain(self, kind):
		if kind == "mono":
			return theme.MONO_CHAIN
		if kind == "thin":
			return theme.THIN_CHAIN
		return theme.FONT_CHAIN

	def get(self, size, kind="main", bold=False, text=""):
		names = self.resolve(self.chain(kind))
		first = self.font(names[0], size, bold)
		if not text or self.supports(first, text):
			return first
		for n in names[1:] + self.resolve(theme.FONT_CHAIN) + self.resolve(theme.SYMBOL_CHAIN):
			if not n:
				continue
			f = self.font(n, size, bold)
			if self.supports(f, text):
				return f
		return first

	def runs(self, size, kind, bold, text):
		first = self.get(size, kind, bold)
		if self.supports(first, text):
			return [(first, text)]
		out = []
		for c in text:
			f = first if self.supports(first, c) else self.get(size, kind, bold, c)
			if out and out[-1][0] is f:
				out[-1] = (f, out[-1][1] + c)
				continue
			out.append((f, c))
		return out


def render_line(font, text, color, spacing):
	if spacing <= 0 or len(text) < 2:
		return font.render(text, True, color)
	glyphs = [font.render(c, True, color) for c in text]
	advs = [font.size(c)[0] for c in text]
	w = sum(advs) + spacing * (len(text) - 1)
	surf = pygame.Surface((max(1, w), font.get_height()), pygame.SRCALPHA)
	x = 0
	for g, a in zip(glyphs, advs):
		surf.blit(g, (x, 0))
		x += a + spacing
	return surf


def render_runs(runs, color, spacing):
	if len(runs) == 1:
		return render_line(runs[0][0], runs[0][1], color, spacing)
	parts = [(f, render_line(f, t, color, spacing)) for f, t in runs]
	asc = max(f.get_ascent() for f, _ in runs)
	desc = max(f.get_height() - f.get_ascent() for f, _ in runs)
	w = sum(p.get_width() for _, p in parts) + spacing * (len(parts) - 1)
	surf = pygame.Surface((max(1, w), asc + desc), pygame.SRCALPHA)
	x = 0
	for f, p in parts:
		surf.blit(p, (x, asc - f.get_ascent()))
		x += p.get_width() + spacing
	return surf


def make_label(runs, color, spacing=0, shadow=1.0):
	fg = render_runs(runs, color, spacing)
	w, h = fg.get_size()
	if shadow <= 0:
		return Label(fg, 0, w, h)
	size = runs[0][0].get_height()
	blur = max(2, int(size * 0.1))
	pad = blur + 2
	mask = render_runs(runs, theme.DARK, spacing)
	sh = soft_shadow(mask, w, h, pad, max(1, size // 28), blur, int(235 * shadow), size < 48)
	sh.blit(fg, (pad, pad))
	return Label(sh, pad, w, h)


class TextCache:
	def __init__(self, fonts, limit=600):
		self.fonts = fonts
		self.limit = limit
		self.items = OrderedDict()

	def clear(self):
		self.items.clear()

	def get(self, text, size, color, kind="main", spacing=0, shadow=1.0, bold=False):
		key = (text, size, color, kind, spacing, shadow, bold)
		lab = self.items.get(key)
		if lab is not None:
			self.items.move_to_end(key)
			return lab
		lab = make_label(self.fonts.runs(size, kind, bold, text), color, spacing, shadow)
		self.items[key] = lab
		if len(self.items) > self.limit:
			self.items.popitem(last=False)
		return lab


class GlyphStrip:
	DEFAULT = "0123456789,.:/-+ m×%skx<>"

	def __init__(self, fonts, size, color, chars=DEFAULT, kind="main", bold=False, spacing=0, shadow=1.0):
		font = fonts.get(size, kind, bold, chars)
		self.glyphs = {}
		self.adv = {}
		self.table = {}
		digit = max(font.size(c)[0] for c in "0123456789")
		for c in chars:
			g = make_label([(font, c)], color, 0, shadow) if not c.isspace() else None
			a = (digit if c.isdigit() else font.size(c)[0]) + spacing
			self.glyphs[c] = g
			self.adv[c] = a
			self.table[c] = (g.surf, (a - g.w) // 2 - g.pad, -g.pad, a) if g else (None, 0, 0, a)
		self.spacing = spacing
		self.h = font.get_height()

	def width(self, s):
		adv = self.adv
		return sum([adv.get(c, 0) for c in s]) - (self.spacing if s else 0)

	def draw(self, screen, s, x, y, anchor="topleft", alpha=255):
		if alpha < 6:
			return 0
		w = self.width(s)
		x0, y0 = anchor_origin(w, self.h, x, y, anchor)
		seq = []
		add = seq.append
		if alpha >= 250:
			table = self.table
			for c in s:
				e = table.get(c)
				if e is None:
					continue
				if e[0] is not None:
					add((e[0], (x0 + e[1], y0 + e[2])))
				x0 += e[3]
			screen.blits(seq, False)
			return w
		glyphs = self.glyphs
		adv = self.adv
		for c in s:
			g = glyphs.get(c)
			a = adv.get(c, 0)
			if g:
				add((g.faded(alpha), (x0 + (a - g.w) // 2 - g.pad, y0 - g.pad)))
			x0 += a
		screen.blits(seq, False)
		return w
