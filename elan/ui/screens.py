import math

import numpy as np
import pygame

from elan.gamepad import pad_rows
from elan.input import binding_name
from elan.settings import DEFAULT_BINDINGS, DEFAULT_PAD

from . import theme, shapes
from .hud import Kit, fmt_clock, fmt_int
from .text import Label
from .theme import clamp01

NONE_LABEL = "—"

ACTION_NAMES = (
	("thrust", "THRUST"),
	("brake", "AIR BRAKE"),
	("strafe_left", "STRAFE LEFT"),
	("strafe_right", "STRAFE RIGHT"),
	("boost", "BOOST"),
	("tether", "TETHER"),
	("drift", "DRIFT"),
	("camera", "CAMERA"),
	("restart", "RESTART"),
	("pause", "PAUSE / MENU"),
	("gyro", "GYRO ON / OFF"),
	("gyro_reset", "GYRO RECENTER"),
)


class DefaultBindings:
	def binding_label(self, action, slot):
		return binding_name(DEFAULT_BINDINGS[action][slot])

	def pad_rows(self):
		return pad_rows('right', DEFAULT_PAD)


def slot_labels(provider, action):
	return [l for l in (provider.binding_label(action, s) for s in (0, 1)) if l and l != NONE_LABEL]


def keys_text(provider, action, joiner=" / "):
	labels = slot_labels(provider, action)
	return joiner.join(labels) if labels else NONE_LABEL


def first_key(provider, action):
	labels = slot_labels(provider, action)
	return labels[0] if labels else NONE_LABEL


def keyboard_rows(provider):
	return (
		("Mouse", "steer"),
		(keys_text(provider, "thrust"), "thrust"),
		(keys_text(provider, "brake"), "air brake"),
		(first_key(provider, "strafe_left") + " / " + first_key(provider, "strafe_right"), "strafe"),
		(keys_text(provider, "boost"), "boost (burns flux)"),
		(keys_text(provider, "tether"), "tether (hold)"),
		(keys_text(provider, "drift"), "drift, cut lift (hold)"),
		(keys_text(provider, "camera"), "camera"),
		("ESC", "pause"),
		(keys_text(provider, "restart"), "restart"),
		("F2 · F3 · F11 · M", "AI pilot · stats · fullscreen · mute"),
	)


def bindings_footer(provider):
	parts = (
		(keys_text(provider, "thrust", "/"), "thrust"),
		(keys_text(provider, "brake", "/"), "brake"),
		(first_key(provider, "strafe_left") + "/" + first_key(provider, "strafe_right"), "strafe"),
		(first_key(provider, "boost"), "boost"),
		(first_key(provider, "tether"), "tether"),
		(first_key(provider, "drift"), "drift"),
		(first_key(provider, "camera"), "camera"),
		(first_key(provider, "restart"), "restart"),
	)
	return "   ".join("%s %s" % p for p in parts)


TITLE_HINT = "UP / DOWN mode or pilot  ·  LEFT / RIGHT to choose  ·  SPACE, ENTER or CLICK to start  ·  AUTO-PLAY lets the AI fly"
TITLE_HINT_PAD = "UP / DOWN mode or pilot  ·  LEFT / RIGHT to choose  ·  SPACE, CLICK or A to start  ·  AUTO-PLAY lets the AI fly"

RULES = (
	"Flux drains. Motes, grazing rock and skimming the sea refill it.",
	"Out of flux you can only glide. Stall down at sea level and the run ends.",
	"Speed multiplies everything. Hard turns bleed it — tether around rock to turn for free.",
	"Skip on the sea at shallow angles. Dive steep: dart form charges SURGE for a launch.",
	"Fill the quota, find the gate, warp.",
)

PAUSE_FOOTER = "W/S or ↑/↓ select · A/D or ←/→ change · ENTER confirm · ESC resume"
CONTROLS_FOOTER = "↑/↓ select · ←/→ slot · ENTER or A rebind · DEL clear · ESC or B back"
CAPTURE_FOOTER = "ESC cancel · BACKSPACE / DEL clear the slot"


def ease(t):
	t = clamp01(t)
	return 1.0 - (1.0 - t) ** 3


def fade_at(t, delay, dur=0.45):
	return ease((t - delay) / dur)


def fmt_time(sec):
	sec = max(0, int(sec))
	if sec >= 3600:
		return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
	return f"{sec // 60}:{sec % 60:02d}"


class Shade:
	LEVELS = 10

	def __init__(self, rows, cols=None):
		self.rows = rows
		self.cols = cols
		self.size = None
		self.level = -1
		self.surf = None

	def build(self, size):
		self.size = size
		w, h = max(2, size[0] // 8), max(2, size[1] // 8)
		self.dark = shapes.shade_map(w, h, self.rows, self.cols)
		self.small = pygame.Surface((w, h))
		self.level = -1

	def draw(self, screen, fade):
		if screen.get_size() != self.size:
			self.build(screen.get_size())
		lvl = int(round(clamp01(fade) * self.LEVELS))
		if lvl <= 0:
			return
		if lvl != self.level:
			self.level = lvl
			g = (255.0 * (1.0 - self.dark * lvl / self.LEVELS)).astype(np.uint8)
			pygame.surfarray.blit_array(self.small, np.repeat(g[:, :, None], 3, axis=2))
			self.surf = pygame.transform.smoothscale(self.small, self.size)
		screen.blit(self.surf, (0, 0), special_flags=pygame.BLEND_RGB_MULT)


class BaseScreen:
	def __init__(self, kit):
		self.kit = kit
		self.m = kit.m

	def lab(self, text, size, color, spacing=0, shadow=1.0, bold=False, kind="main"):
		return self.kit.label(text, size, color, spacing, kind, shadow, bold)

	def hline(self, screen, cx, y, w, color, alpha):
		if alpha <= 0 or w <= 0:
			return
		key = (w, color)
		s = self.lines.get(key) if hasattr(self, "lines") else None
		if s is None:
			if not hasattr(self, "lines"):
				self.lines = {}
			prof = [abs(math.sin(math.pi * i / max(1, w - 1))) ** 0.7 for i in range(w)]
			s = Label(shapes.alpha_strip(w, max(1, self.m.px(2)), color, prof))
			self.lines[key] = s
		s.blit(screen, (int(cx - w // 2), int(y)), alpha)

	def highlight(self, w, h):
		cache = self.__dict__.setdefault("highlights", {})
		s = cache.get((w, h))
		if s:
			return s
		e = max(1, w // 6)
		prof = [min(1.0, i / e, (w - 1 - i) / e) * 0.22 for i in range(w)]
		s = Label(shapes.alpha_strip(w, h, theme.CYAN, prof))
		cache[(w, h)] = s
		return s

	def frame(self, w, h, color, lw=1.6):
		cache = self.__dict__.setdefault("frames", {})
		key = (w, h, color, lw)
		s = cache.get(key)
		if s:
			return s
		s = shapes.vector_sprite(w, h, shapes.frame_painter(self.m.s * lw, self.m.s * 4), color, 0.8)
		cache[key] = s
		return s

	def resize(self):
		self.lines = {}
		self.highlights = {}
		self.frames = {}


class LoadingScreen(BaseScreen):
	def resize(self):
		BaseScreen.resize(self)
		m = self.m
		self.bg = shapes.gradient_strip(m.h, 1, [(0.0, (8, 14, 24)), (0.58, (14, 20, 32)), (0.64, (38, 26, 24)), (0.70, (10, 14, 22)), (1.0, (3, 5, 9))])
		self.bg = pygame.transform.scale(pygame.transform.rotate(self.bg, -90), (m.w, m.h))
		self.bar_w = m.px(520)
		self.bar_h = max(2, m.px(3))
		self.track, self.tpad, _ = shapes.track(self.bar_w, self.bar_h, edge=False)
		self.fill = shapes.gradient_strip(self.bar_w, self.bar_h, [(0.0, theme.CYAN_DEEP), (0.8, theme.CYAN), (1.0, theme.WHITE)])
		self.pct = self.kit.strip("tiny", theme.PRIMARY_DIM, "0123456789%")
		if getattr(self, "source", None) is not None:
			self.apply_backdrop()

	def set_backdrop(self, image):
		self.source = image
		self.apply_backdrop()

	def apply_backdrop(self):
		m = self.m
		iw, ih = self.source.get_size()
		scale = max(m.w / iw, m.h / ih)
		scaled = pygame.transform.smoothscale(self.source, (int(iw * scale) + 1, int(ih * scale) + 1))
		canvas = pygame.Surface((m.w, m.h))
		canvas.blit(scaled, ((m.w - scaled.get_width()) // 2, (m.h - scaled.get_height()) // 2))
		veil = pygame.Surface((m.w, m.h))
		veil.fill((165, 165, 170))
		canvas.blit(veil, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
		self.bg = canvas

	def draw(self, screen, progress, label):
		m = self.m
		screen.blit(self.bg, (0, 0))
		title = self.lab("ÉLAN", "huge", theme.PRIMARY, spacing=40, kind="thin")
		ty = int(m.h * 0.42)
		title.draw(screen, m.cx, ty, "center")
		self.hline(screen, m.cx, ty + title.h // 2 + m.px(6), m.px(90), theme.ACCENT, 230)
		by = int(m.h * 0.56)
		x0 = m.cx - self.bar_w // 2
		screen.blit(self.track, (x0 - self.tpad, by - self.tpad))
		p = clamp01(progress or 0.0)
		fw = int(self.bar_w * p)
		if fw > 0:
			screen.blit(self.fill, (x0, by), (self.bar_w - fw, 0, fw, self.bar_h))
		self.pct.draw(screen, str(int(p * 100)) + "%", x0 + self.bar_w + m.px(14), by + self.bar_h // 2, "midleft")
		if label:
			self.lab(str(label), "small", theme.PRIMARY_DIM, spacing=2).draw(screen, m.cx, by + m.px(22), "midtop")


class ControlsSheet:
	def __init__(self, screen):
		self.s = screen
		self.provider = DefaultBindings()
		self.keyboard = None
		self.gamepad = None
		self.cols = None

	def set_provider(self, provider):
		self.provider = provider or DefaultBindings()
		self.keyboard = None

	def refresh(self):
		rows = keyboard_rows(self.provider)
		pad = self.provider.pad_rows()
		if rows == self.keyboard and pad == self.gamepad and self.cols is not None:
			return
		self.keyboard = rows
		self.layout()

	def layout(self):
		s = self.s
		m = s.m
		self.keyboard = keyboard_rows(self.provider)
		self.gamepad = self.provider.pad_rows()
		self.cols = []
		gap = m.px(18)
		for head, rows in (("KEYBOARD & MOUSE", self.keyboard), ("GAMEPAD", self.gamepad)):
			keys = [s.lab(k, "label", theme.ACCENT_HOT, 1) for k, _ in rows]
			acts = [s.lab(a, "label", theme.PRIMARY, 0) for _, a in rows]
			kw = max(l.w for l in keys)
			aw = max(l.w for l in acts)
			self.cols.append((s.lab(head, "small", theme.PRIMARY_DIM, 5), keys, acts, kw, aw, gap))
		self.col_gap = m.px(150)
		self.widths = [c[3] + c[5] + c[4] for c in self.cols]
		self.total = sum(self.widths) + self.col_gap
		self.row_h = int(self.cols[0][1][0].h * 1.12)

	def height(self):
		c = self.cols[0]
		return c[0].h + self.s.m.px(14) + self.row_h * max(len(self.keyboard), len(self.gamepad))

	def draw(self, screen, y, alpha):
		m = self.s.m
		x = m.cx - self.total // 2
		for (head, keys, acts, kw, aw, gap), w in zip(self.cols, self.widths):
			mid = x + kw + gap // 2
			head.draw(screen, mid, y, "midtop", alpha)
			self.s.hline(screen, mid, y + head.h + m.px(4), w, theme.PRIMARY_DIM, alpha * 0.5)
			ry = y + head.h + m.px(14)
			for k, a in zip(keys, acts):
				k.draw(screen, x + kw, ry, "topright", alpha)
				a.draw(screen, x + kw + gap, ry, "topleft", alpha * 0.92)
				ry += self.row_h
			x += w + self.col_gap


class TitleScreen(BaseScreen):
	def __init__(self, kit):
		BaseScreen.__init__(self, kit)
		self.shade = Shade([(0.0, 0.45), (0.12, 0.35), (0.42, 0.18), (0.52, 0.45), (0.62, 0.62), (1.0, 0.72)], [(0.0, 0.25), (0.25, 0.0), (0.75, 0.0), (1.0, 0.25)])
		self.sheet = ControlsSheet(self)

	def resize(self):
		BaseScreen.resize(self)
		self.sheet.layout()

	def draw_rows(self, screen, t, view, y, fade):
		m = self.m
		pulse = 0.8 + 0.2 * math.sin(t * 3.4)
		spread = m.px(70)
		pad = m.px(14)
		hits = []
		for r, (caption, names, picked) in enumerate(view.rows):
			focused = r == view.focus
			labels = [self.lab(name, "mid", theme.WHITE if k == picked else theme.PRIMARY_DIM, spacing=6) for k, name in enumerate(names)]
			x = m.cx - (sum(l.w for l in labels) + spread * (len(labels) - 1)) // 2
			cap = self.lab(caption, "tiny", theme.ACCENT if focused else theme.PRIMARY_DIM, spacing=5)
			cap.draw(screen, x - m.px(36), y + labels[0].h // 2, "midright", 230 * fade)
			for k, label in enumerate(labels):
				chosen = k == picked
				glow = (pulse if focused else 0.9) if chosen else 0.62
				label.draw(screen, x, y, "topleft", 255 * fade * glow)
				if chosen:
					self.hline(screen, x + label.w // 2, y + label.h + m.px(4), label.w, theme.ACCENT if focused else theme.PRIMARY_DIM, 240 * fade)
				hits.append((pygame.Rect(x - pad, y - pad, label.w + 2 * pad, label.h + 2 * pad), r, k))
				x += label.w + spread
			y += labels[0].h + m.px(10)
			if r == 0 and view.blurb:
				self.lab(view.blurb, "tiny", theme.PRIMARY, spacing=3).draw(screen, m.cx, y, "midtop", 220 * fade)
			y += m.px(32)
		return hits, y

	def draw_records(self, screen, view, y, fade):
		m = self.m
		gap = m.px(14)
		parts = []
		for caption, value in view.records:
			parts.append((self.lab(caption, "small", theme.PRIMARY_DIM, spacing=5), 230, gap))
			parts.append((self.lab(value, "body", theme.ACCENT, spacing=1), 255, gap * 3))
		if view.tail:
			parts.append((self.lab(view.tail, "small", theme.PRIMARY_DIM, spacing=4), 230, 0))
		if not parts:
			return
		width = sum(lab.w + after for lab, _, after in parts) - parts[-1][2]
		height = max(lab.h for lab, _, _ in parts)
		x = m.cx - width // 2
		for lab, alpha, after in parts:
			lab.draw(screen, x, y + height // 2, "midleft", alpha * fade)
			x += lab.w + after

	def draw(self, screen, t, view, pad_connected):
		m = self.m
		self.shade.draw(screen, fade_at(t, 0.0, 0.6))
		f0 = fade_at(t, 0.05, 0.7)
		title = self.lab("ÉLAN", "title", theme.PRIMARY, spacing=70, kind="thin")
		ty = m.px(192) + int((1.0 - f0) * m.px(24))
		title.draw(screen, m.cx, ty, "center", 255 * f0)
		ly = ty + int(title.h * 0.42)
		self.hline(screen, m.cx, ly, int(m.px(160) * (0.3 + 0.7 * f0)), theme.ACCENT, 240 * f0)
		f1 = fade_at(t, 0.35)
		tag = self.lab("momentum is everything", "body", theme.PRIMARY, spacing=6)
		tag.draw(screen, m.cx, ly + m.px(18), "midtop", 230 * f1)
		f2 = fade_at(t, 0.7)
		hits, y = self.draw_rows(screen, t, view, ly + m.px(58), f2)
		hint = TITLE_HINT_PAD if pad_connected else TITLE_HINT
		self.lab(hint, "tiny", theme.PRIMARY_DIM, spacing=3).draw(screen, m.cx, y - m.px(28), "midtop", 220 * f2)
		self.draw_records(screen, view, y, f2)
		f3 = fade_at(t, 1.0)
		self.sheet.refresh()
		rh = self.lab(RULES[0], "small", theme.PRIMARY, 1).h
		rules_y = m.h - m.px(24) - rh * len(RULES)
		sheet_y = rules_y - m.px(22) - self.sheet.height()
		self.sheet.draw(screen, sheet_y, 255 * f3)
		f4 = fade_at(t, 1.25)
		ry = rules_y
		for line in RULES:
			self.lab(line, "small", theme.PRIMARY, 1).draw(screen, m.cx, ry, "midtop", 225 * f4)
			ry += int(rh * 1.12)
		return hits


def fit_label(screen, text, sizes, color, max_w, spacing=1):
	lab = None
	for size in sizes:
		lab = screen.lab(text, size, color, spacing)
		if lab.w <= max_w:
			return lab
	return lab


class PauseScreen(BaseScreen):
	def __init__(self, kit):
		BaseScreen.__init__(self, kit)
		self.shade = Shade([(0.0, 0.5), (0.2, 0.62), (0.5, 0.7), (0.8, 0.62), (1.0, 0.5)], [(0.0, 0.3), (0.3, 0.0), (0.7, 0.0), (1.0, 0.3)])

	def draw(self, screen, t, items, selected, title="PAUSED", footer=None, extra=None):
		m = self.m
		items = list(items or [])
		f = fade_at(t, 0.0, 0.25)
		self.shade.draw(screen, f)
		title = self.lab(title, "huge", theme.PRIMARY, spacing=30, kind="thin")
		lines = [(PAUSE_FOOTER if footer is None else footer, theme.PRIMARY_DIM), (extra, theme.PRIMARY_DIM)]
		foot_h = sum(self.lab(l, "small", c, 1).h + m.px(8) for l, c in lines if l)
		arrow_l = self.lab("<", "mid", theme.ACCENT, 0, bold=True)
		arrow_r = self.lab(">", "mid", theme.ACCENT, 0, bold=True)
		labels = [self.lab(str(l), "body", theme.WHITE if i == selected else theme.PRIMARY, 2) for i, (l, _) in enumerate(items)]
		values = [self.lab(str(v), "body", theme.WHITE if i == selected else theme.PRIMARY_DIM, 1) if v is not None else None for i, (_, v) in enumerate(items)]
		pad = m.px(28)
		agap = m.px(14)
		lw = max([l.w for l in labels] or [0])
		vw = max([v.w for v in values if v] or [0])
		width = max(m.px(640), lw + vw + 2 * arrow_l.w + 2 * agap + 2 * pad + m.px(60))
		width = min(width, m.w - m.px(40))
		avail = m.h - title.h - foot_h - m.px(150)
		row_h = min(m.px(56), max(int(labels[0].h * 1.15) if labels else 1, avail // max(1, len(items))))
		block = title.h + m.px(30) + row_h * len(items)
		y = max(m.px(16), (m.h - foot_h - m.px(30) - block) // 2)
		off = int((1.0 - f) * m.px(16))
		title.draw(screen, m.cx, y + off, "midtop", 255 * f)
		self.hline(screen, m.cx, y + title.h + m.px(4) + off, m.px(120), theme.ACCENT, 230 * f)
		ry = y + title.h + m.px(30) + off
		x0 = m.cx - width // 2
		x1 = x0 + width
		hits = []
		for i, (lab, val) in enumerate(zip(labels, values)):
			cy = ry + row_h // 2
			hits.append((pygame.Rect(x0, ry, width, row_h), i))
			if i == selected:
				self.row_mark(screen, x0, cy, width, row_h, 255 * f)
			lab.draw(screen, x0 + pad, cy, "midleft", 255 * f)
			if val:
				vx = x1 - pad - arrow_r.w - agap
				val.draw(screen, vx, cy, "midright", 255 * f)
				if i == selected:
					pulse = 0.65 + 0.35 * math.sin(t * 6.0)
					arrow_r.draw(screen, x1 - pad, cy, "midright", 255 * f * pulse)
					arrow_l.draw(screen, vx - val.w - agap, cy, "midright", 255 * f * pulse)
			ry += row_h
		self.footer(screen, lines, 230 * f)
		return hits

	def row_mark(self, screen, x0, cy, w, row_h, alpha):
		hl = self.highlight(w, row_h - self.m.px(6))
		hl.blit(screen, (x0, cy - hl.h // 2), alpha)
		screen.fill(theme.ACCENT, (x0, cy - hl.h // 2, max(2, self.m.px(3)), hl.h))

	def footer(self, screen, lines, alpha):
		m = self.m
		y = m.h - m.px(30)
		for text, color in reversed([l for l in lines if l[0]]):
			lab = fit_label(self, text, ("small", "tiny"), color, m.w - m.px(40))
			lab.draw(screen, m.cx, y, "midbottom", alpha)
			y -= lab.h + m.px(8)


class ControlsView:
	def __init__(self):
		self.rows = []
		self.extras = []
		self.selected = 0
		self.slot = 0
		self.capturing = False
		self.status = ""
		self.warn = False


class ControlsScreen(PauseScreen):
	def draw(self, screen, t, view):
		m = self.m
		f = fade_at(t, 0.0, 0.25)
		a = 255 * f
		self.shade.draw(screen, f)
		title = self.lab("CONTROLS", "huge", theme.PRIMARY, spacing=30, kind="thin")
		foot = CAPTURE_FOOTER if view.capturing else CONTROLS_FOOTER
		foot_h = self.lab(foot, "small", theme.PRIMARY_DIM, 1).h + m.px(8)
		sel = view.selected
		names = [self.lab(n, "body", theme.WHITE if i == sel else theme.PRIMARY, 2) for i, (n, _) in enumerate(view.rows)]
		extras = [self.lab(e, "body", theme.WHITE if len(view.rows) + i == sel else theme.PRIMARY, 2) for i, e in enumerate(view.extras)]
		cells = [[self.lab(c, "label", theme.ACCENT_HOT if c != NONE_LABEL else theme.PRIMARY_DIM, 1) for c in slots] for _, slots in view.rows]
		heads = [self.lab(h, "tiny", theme.PRIMARY_DIM, 4) for h in ("ACTION", "PRIMARY", "SECONDARY", "GAMEPAD")]
		wait = self.lab("PRESS…", "label", theme.ACCENT, 2, bold=True)
		pad = m.px(28)
		gap = m.px(16)
		nw = max([l.w for l in names + extras] or [0])
		cw = max([m.px(190), wait.w + m.px(30)] + [c.w + m.px(30) for row in cells for c in row])
		table_w = pad + nw + m.px(40) + 3 * cw + 2 * gap + pad
		total = table_w
		status_h = self.lab("X", "body", theme.ACCENT, 2).h
		nrows = len(names) + len(extras)
		sep = m.px(16)
		avail = m.h - title.h - foot_h - status_h - heads[0].h - sep - m.px(150)
		row_h = min(m.px(50), max(int(names[0].h * 1.2), avail // max(1, nrows)))
		block = title.h + m.px(30) + heads[0].h + m.px(10) + row_h * nrows + sep + m.px(16) + status_h
		y = max(m.px(16), (m.h - foot_h - m.px(30) - block) // 2)
		off = int((1.0 - f) * m.px(16))
		title.draw(screen, m.cx, y + off, "midtop", a)
		self.hline(screen, m.cx, y + title.h + m.px(4) + off, m.px(120), theme.ACCENT, 230 * f)
		x0 = m.cx - total // 2
		x1 = x0 + table_w
		c0 = x0 + pad + nw + m.px(40)
		c1 = c0 + cw + gap
		c2 = c1 + cw + gap
		hy = y + title.h + m.px(30) + off
		heads[0].draw(screen, x0 + pad, hy, "topleft", a)
		heads[1].draw(screen, c0 + cw // 2, hy, "midtop", a)
		heads[2].draw(screen, c1 + cw // 2, hy, "midtop", a)
		heads[3].draw(screen, c2 + cw // 2, hy, "midtop", a)
		top = hy + heads[0].h + m.px(10)
		ry = top
		hits = []
		pulse = 0.55 + 0.45 * math.sin(t * 7.0)
		ch = max(4, row_h - m.px(10))
		for i, (name, row) in enumerate(zip(names, cells)):
			cy = ry + row_h // 2
			hits.append((pygame.Rect(x0, ry, c0 - x0, row_h), i, None))
			if i == sel:
				self.row_mark(screen, x0, cy, table_w, row_h, a)
			name.draw(screen, x0 + pad, cy, "midleft", a)
			for s, (cx, lab) in enumerate(zip((c0, c1, c2), row)):
				hits.append((pygame.Rect(cx, ry, cw, row_h), i, s))
				active = i == sel and s == view.slot
				if active and view.capturing:
					self.frame(cw, ch, theme.ACCENT, 2.2).draw(screen, cx, cy - ch // 2, "topleft", a * pulse)
					wait.draw(screen, cx + cw // 2, cy, "center", a * pulse)
					continue
				fr = self.frame(cw, ch, theme.ACCENT, 2.0) if active else self.frame(cw, ch, theme.PRIMARY_DIM, 1.2)
				fr.draw(screen, cx, cy - ch // 2, "topleft", a if active else a * 0.6)
				lab.draw(screen, cx + cw // 2, cy, "center", a)
			ry += row_h
		ry += sep
		for j, lab in enumerate(extras):
			i = len(names) + j
			cy = ry + row_h // 2
			hits.append((pygame.Rect(x0, ry, table_w, row_h), i, None))
			if i == sel:
				self.row_mark(screen, x0, cy, table_w, row_h, a)
			lab.draw(screen, x0 + pad, cy, "midleft", a)
			ry += row_h
		if view.status:
			col = theme.ACCENT if view.capturing or view.warn else theme.PRIMARY
			st = fit_label(self, view.status, ("body", "label", "small"), col, table_w)
			st.draw(screen, (x0 + x1) // 2, ry + m.px(16), "midtop", a * (pulse if view.capturing else 1.0))
		self.footer(screen, [(foot, theme.PRIMARY_DIM)], 230 * f)
		return hits


class GameOverScreen(BaseScreen):
	def __init__(self, kit):
		BaseScreen.__init__(self, kit)
		self.shade = Shade([(0.0, 0.5), (0.25, 0.66), (0.5, 0.72), (0.8, 0.66), (1.0, 0.55)], [(0.0, 0.3), (0.3, 0.0), (0.7, 0.0), (1.0, 0.3)])

	def resize(self):
		BaseScreen.resize(self)
		self.score_digits = self.kit.strip("huge", theme.WHITE, "0123456789,")
		self.frames = {}

	def frame(self, w, h, color):
		key = (w, h, color)
		s = self.frames.get(key)
		if s:
			return s
		s = shapes.vector_sprite(w, h, shapes.frame_painter(self.m.s * 1.6, self.m.s * 4), color, 0.8)
		self.frames[key] = s
		return s

	def draw(self, screen, t, summary):
		m = self.m
		g = lambda n, d=0: getattr(summary, n, d)
		done = bool(g("completed", False))
		timed = bool(g("timed", False))
		unit = str(g("unit", "SECTOR") or "SECTOR")
		levels = int(g("levels", 0) or 0)
		self.shade.draw(screen, fade_at(t, 0.0, 0.5) * (0.75 if done else 1.0))
		f0 = fade_at(t, 0.0, 0.5)
		y = int(m.h * (0.11 if done else 0.2))
		tone = theme.WHITE if done else theme.DANGER
		head = fit_label(self, "WELCOME TO THE SINGULARITY" if done else "FLUX DEPLETED", ("big", "mid"), tone, m.w - m.px(60), 12)
		head.draw(screen, m.cx, y + int((1.0 - f0) * m.px(20)), "midtop", 255 * f0)
		y += head.h + m.px(10)
		self.hline(screen, m.cx, y, m.px(260 if done else 200), tone, 200 * f0)
		if done:
			sub = self.lab("STORY COMPLETE  -  %d LEVELS BEHIND YOU" % levels, "small", theme.ACCENT_HOT, spacing=6)
			sub.draw(screen, m.cx, y + m.px(10), "midtop", 240 * f0)
			y += sub.h + m.px(8)
		y += m.px(34)
		f1 = fade_at(t, 0.3)
		cap = self.lab("FINAL SCORE", "small", theme.PRIMARY_DIM, spacing=5)
		cap.draw(screen, m.cx, y, "midtop", 230 * f1)
		y += cap.h
		score = int(g("score", 0) or 0)
		shown = int(score * ease((t - 0.3) / 1.1))
		self.score_digits.draw(screen, fmt_int(shown), m.cx, y, "midtop", 255 * f1)
		y += self.score_digits.h + m.px(10)
		f2 = fade_at(t, 1.3, 0.35)
		if g("new_best", False):
			nb = self.lab("NEW BEST", "body", theme.ACCENT, spacing=8, bold=True)
			fw, fh = nb.w + m.px(40), nb.h + m.px(14)
			pulse = 0.7 + 0.3 * math.sin(t * 5.0)
			fr = self.frame(fw, fh, theme.ACCENT)
			fr.draw(screen, m.cx, y + fh // 2, "center", 255 * f2 * pulse)
			nb.draw(screen, m.cx, y + fh // 2, "center", 255 * f2)
			y += fh + m.px(12)
		bc = self.lab("BEST", "small", theme.PRIMARY_DIM, spacing=5)
		bv = self.lab(fmt_int(g("best", 0) or 0), "body", theme.ACCENT_HOT, spacing=1)
		gap = m.px(14)
		bx = m.cx - (bc.w + gap + bv.w) // 2
		bc.draw(screen, bx, y + bv.h // 2, "midleft", 230 * f1)
		bv.draw(screen, bx + bc.w + gap, y + bv.h // 2, "midleft", 255 * f1)
		y += bv.h + m.px(14)
		rank = g("rank", "")
		if rank and not done:
			note = "NEW FRONTIER   -   " if g("frontier", False) else ""
			line = self.lab("%sDEEPEST %s %d   -   %s" % (note, unit, int(g("deepest", 0) or 0), rank), "small", theme.ACCENT if note else theme.PRIMARY_DIM, spacing=4)
			line.draw(screen, m.cx, y, "midtop", 240 * f2)
		if done and timed:
			y = self.draw_time(screen, t, summary, y + m.px(6), f2)
		y += m.px(56)
		duration = g("duration", 0) or 0
		stats = [
			(unit if not done else unit + "S", str(int(g("sector", 1) or 0)) + (" / %d" % levels if levels else "")),
			("MOTES", fmt_int(g("motes", 0) or 0)),
			("TOP SPEED", f"{int(g('top_speed', 0) or 0)} m/s"),
			("FLIGHT TIME", fmt_clock(duration, 2) if timed else fmt_time(duration)),
		]
		if done and timed:
			stats.pop()
		vals = [self.lab(v, "mid", theme.PRIMARY, 1) for _, v in stats]
		caps = [self.lab(c, "tiny", theme.PRIMARY_DIM, 4) for c, _ in stats]
		cw = max(m.px(220), max(max(v.w, c.w) for v, c in zip(vals, caps)) + m.px(40))
		x = m.cx - cw * len(stats) // 2
		sep_h = vals[0].h + caps[0].h + m.px(6)
		for i, (v, c) in enumerate(zip(vals, caps)):
			fi = fade_at(t, 0.6 + 0.12 * i)
			mx = x + cw // 2 + i * cw
			oy = int((1.0 - fi) * m.px(12))
			v.draw(screen, mx, y + oy, "midtop", 255 * fi)
			c.draw(screen, mx, y + v.h + m.px(4) + oy, "midtop", 230 * fi)
			if i:
				screen.fill(theme.PRIMARY_DIM, (x + i * cw, y + m.px(6), 1, sep_h - m.px(8)))
		y += sep_h + m.px(90)
		f3 = fade_at(t, 1.6)
		pulse = 0.8 + 0.2 * math.sin(t * 3.4)
		self.lab("PRESS SPACE TO FLY AGAIN", "mid", theme.WHITE, spacing=6).draw(screen, m.cx, y, "midtop", 255 * f3 * pulse)


	def draw_time(self, screen, t, summary, y, fade):
		m = self.m
		cap = self.lab("FINAL TIME", "small", theme.PRIMARY_DIM, spacing=5)
		cap.draw(screen, m.cx, y, "midtop", 230 * fade)
		y += cap.h
		val = self.lab(fmt_clock(getattr(summary, "duration", 0.0) or 0.0, 2), "big", theme.WHITE, spacing=3)
		val.draw(screen, m.cx, y, "midtop", 255 * fade)
		y += val.h + m.px(6)
		if getattr(summary, "new_time", False):
			tag = self.lab("NEW BEST TIME", "body", theme.ACCENT, spacing=8, bold=True)
			fw, fh = tag.w + m.px(40), tag.h + m.px(14)
			pulse = 0.7 + 0.3 * math.sin(t * 5.0)
			self.frame(fw, fh, theme.ACCENT).draw(screen, m.cx, y + fh // 2, "center", 255 * fade * pulse)
			tag.draw(screen, m.cx, y + fh // 2, "center", 255 * fade)
			return y + fh
		best = float(getattr(summary, "best_time", 0.0) or 0.0)
		if best <= 0.0:
			return y
		line = self.lab("BEST TIME   %s" % fmt_clock(best, 2), "small", theme.PRIMARY_DIM, spacing=4)
		line.draw(screen, m.cx, y, "midtop", 230 * fade)
		return y + line.h


class Screens:
	def __init__(self, size):
		self.kit = Kit(size)
		self.loading = LoadingScreen(self.kit)
		self.title = TitleScreen(self.kit)
		self.pause = PauseScreen(self.kit)
		self.controls = ControlsScreen(self.kit)
		self.gameover = GameOverScreen(self.kit)
		self.provider = DefaultBindings()
		self.size = None
		self.resize(size)

	def set_bindings(self, provider):
		self.provider = provider or DefaultBindings()
		self.title.sheet.set_provider(self.provider)

	def bindings_line(self):
		return bindings_footer(self.provider)

	def resize(self, size):
		size = (int(size[0]), int(size[1]))
		if size == self.size:
			return
		self.size = size
		self.kit.resize(size)
		for s in (self.loading, self.title, self.pause, self.controls, self.gameover):
			s.resize()

	def check(self, screen):
		if screen.get_size() != self.size:
			self.resize(screen.get_size())

	def set_loading_backdrop(self, path):
		try:
			self.loading.set_backdrop(pygame.image.load(path).convert())
		except (pygame.error, OSError):
			return

	def draw_loading(self, screen, progress, label):
		self.check(screen)
		self.loading.draw(screen, progress, label)

	def draw_title(self, screen, t, view, pad_connected):
		self.check(screen)
		return self.title.draw(screen, t, view, pad_connected)

	def draw_pause(self, screen, t, items, selected, title="PAUSED", footer=None, extra=None):
		self.check(screen)
		return self.pause.draw(screen, t, items, selected, title, footer, self.bindings_line() if extra is None else extra)

	def draw_controls(self, screen, t, view):
		self.check(screen)
		return self.controls.draw(screen, t, view)

	def draw_gameover(self, screen, t, summary):
		self.check(screen)
		self.gameover.draw(screen, t, summary)
