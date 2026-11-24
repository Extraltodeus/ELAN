import math

import pygame

from . import theme, shapes
from .text import Fonts, TextCache, GlyphStrip, Label, rle
from .theme import Metrics, mix, clamp01
from elan.layout import minimap_box


class Kit:
	def __init__(self, size):
		self.m = Metrics(size)
		self.fonts = Fonts()
		self.text = TextCache(self.fonts)

	def resize(self, size):
		self.m.resize(size)
		self.text.clear()

	def strip(self, size, color, chars=GlyphStrip.DEFAULT, kind="main", bold=False, spacing=0, shadow=1.0):
		return GlyphStrip(self.fonts, self.m.font(size), color, chars, kind, bold, spacing, shadow)

	def label(self, text, size, color, spacing=0, kind="main", shadow=1.0, bold=False):
		return self.text.get(text, self.m.font(size), color, kind, self.m.px(spacing) if spacing else 0, shadow, bold)


def fmt_int(v):
	return f"{int(v):,}"


def fmt_clock(sec, places=1):
	sec = max(0.0, float(sec))
	scale = 10 ** places
	ticks = int(sec * scale + 1.0e-6)
	whole = ticks // scale
	tail = "%0*d" % (places, ticks % scale)
	if whole >= 3600:
		return "%d:%02d:%02d.%s" % (whole // 3600, whole % 3600 // 60, whole % 60, tail)
	return "%d:%02d.%s" % (whole // 60, whole % 60, tail)


def fmt_dist(d):
	d = max(0.0, float(d))
	if d >= 10000:
		return f"{d / 1000.0:.1f} km"
	return f"{int(d):,} m"


class Crosshair:
	def build(self, kit):
		m = kit.m
		self.m = m
		sz = m.px(44)
		self.sprites = [
			shapes.vector_sprite(sz, sz, shapes.dot_painter(m.s * 2.2), theme.PRIMARY_DIM, 0.7),
			shapes.vector_sprite(sz, sz, shapes.bracket_painter(m.s * 14, m.s * 2.4, 0.42), theme.PRIMARY, 1.0),
			shapes.vector_sprite(sz, sz, shapes.locked_painter(m.s * 10, m.s * 2.4, m.s * 2.4), theme.ACCENT, 0.9),
		]

	def draw(self, screen, hs):
		st = getattr(hs, "tether_state", 0) or 0
		st = 0 if st < 0 else 2 if st > 2 else int(st)
		alpha = 255
		if st == 1:
			alpha = 190 + int(65 * math.sin(getattr(hs, "time", 0.0) * 9.0))
		self.sprites[st].draw(screen, self.m.cx, self.m.cy, "center", alpha)


class Prograde:
	def build(self, kit):
		m = kit.m
		sz = m.px(40)
		self.sprite = shapes.vector_sprite(sz, sz, shapes.prograde_painter(m.s * 8.5, m.s * 2.0, m.s * 7), theme.GOOD, 0.9)

	def draw(self, screen, hs):
		v = getattr(hs, "vel_marker", None)
		if v is None:
			return
		self.sprite.draw(screen, v[0], v[1], "center", 230)


class ScorePanel:
	def build(self, kit):
		m = kit.m
		self.m = m
		self.x = m.px(56)
		self.y = m.px(40)
		self.caption = kit.label("SCORE", "tiny", theme.PRIMARY_DIM, spacing=4)
		self.digits = kit.strip("score", theme.PRIMARY, "0123456789,")
		self.unscored = kit.label("AUTO-PLAY", "mid", theme.PRIMARY_DIM, spacing=5)
		self.combo = kit.strip("combo", theme.ACCENT, "0123456789×")
		self.combo_hot = kit.strip("combo", theme.ACCENT_HOT, "0123456789×")
		self.score_y = self.y + self.caption.h
		self.combo_y = self.score_y + self.digits.h - m.px(6)
		self.bar_w = m.px(150)
		self.bar_h = max(2, m.px(3))
		self.bar_y = self.combo_y + self.combo.h + m.px(2)
		self.track, self.tpad, _ = shapes.track(self.bar_w, self.bar_h, edge=False)
		self.last_combo = 1
		self.flash_t = -10.0
		self.clock_cap = kit.label("TIME", "tiny", theme.PRIMARY_DIM, spacing=4)
		self.clock = kit.strip("mid", theme.PRIMARY, "0123456789:.")
		self.clock_done = kit.strip("mid", theme.ACCENT_HOT, "0123456789:.")
		self.clock_y = self.bar_y + self.bar_h + m.px(14)

	def draw_clock(self, screen, hs):
		clock = getattr(hs, "clock", None)
		if clock is None:
			return
		self.clock_cap.draw(screen, self.x, self.clock_y)
		strip = self.clock_done if getattr(hs, "finale", False) else self.clock
		strip.draw(screen, fmt_clock(clock), self.x, self.clock_y + self.clock_cap.h)

	def draw(self, screen, hs):
		t = getattr(hs, "time", 0.0)
		self.caption.draw(screen, self.x, self.y)
		score = getattr(hs, "score", 0)
		if score is None:
			self.unscored.draw(screen, self.x, self.score_y + (self.digits.h - self.unscored.h) // 2)
		else:
			self.digits.draw(screen, fmt_int(score), self.x, self.score_y)
		self.draw_clock(screen, hs)
		combo = int(getattr(hs, "combo", 1) or 1)
		if combo > self.last_combo:
			self.flash_t = t
		self.last_combo = combo
		if combo < 2:
			return
		strip = self.combo_hot if t - self.flash_t < 0.18 else self.combo
		strip.draw(screen, "×" + str(combo), self.x, self.combo_y)
		screen.blit(self.track, (self.x - self.tpad, self.bar_y - self.tpad))
		w = int(self.bar_w * clamp01(getattr(hs, "combo_time", 0.0)))
		if w > 0:
			screen.fill(theme.ACCENT, (self.x, self.bar_y, w, self.bar_h))


class SectorPanel:
	def build(self, kit):
		m = kit.m
		self.kit = kit
		self.m = m
		self.y = m.px(36)
		self.cx = m.cx
		self.digits = kit.strip("mid", theme.PRIMARY, "0123456789/ ")
		self.qy = self.y + kit.label("SECTOR 0", "small", theme.PRIMARY_DIM, spacing=5).h
		self.bar_w = m.px(380)
		self.bar_h = max(2, m.px(4))
		self.bar_y = self.qy + self.digits.h + m.px(4)
		self.gate = kit.label("GATE OPEN — ENTER THE WORMHOLE", "body", theme.ACCENT, spacing=3, bold=True)
		self.gate_bar = Label(shapes.alpha_strip(self.bar_w, self.bar_h, theme.ACCENT, [abs(math.sin(math.pi * i / max(1, self.bar_w - 1))) ** 0.6 for i in range(self.bar_w)]))
		self.bars = {}
		self.rush_idle = kit.label("RUSH  -  FACE THE GATE", "tiny", theme.PRIMARY_DIM, spacing=4)
		self.rush_ready = kit.label("RUSH READY  -  THRUST", "tiny", theme.ACCENT_HOT, spacing=4, bold=True)
		self.rush_live = kit.label("GATE RUSH", "tiny", theme.WHITE, spacing=6, bold=True)
		self.rush_w = m.px(240)
		self.rush_track = mix(theme.ACCENT, theme.DARK, 0.8)
		self.warp = kit.label("WARP TUNNEL", "body", theme.ACCENT, spacing=5, bold=True)
		self.hyper = kit.label("HYPER TUNNEL  -  INFINITE FLUX", "body", theme.WHITE, spacing=4, bold=True)
		self.plunge = kit.label("EVENT HORIZON  -  NO WAY BACK", "body", theme.WHITE, spacing=4, bold=True)
		self.portal = kit.label("THE WHITE PORTAL IS OPEN", "body", theme.WHITE, spacing=5, bold=True)
		self.portal_bar = Label(shapes.alpha_strip(self.bar_w, self.bar_h, theme.WHITE, [abs(math.sin(math.pi * i / max(1, self.bar_w - 1))) ** 0.6 for i in range(self.bar_w)]))

	def draw_rush(self, screen, hs, cx):
		charge = clamp01(getattr(hs, "rush", 0.0) or 0.0)
		if charge <= 0.0:
			return
		m = self.m
		live = bool(getattr(hs, "rush_on", False))
		ready = bool(getattr(hs, "rush_ready", False))
		label = self.rush_live if live else (self.rush_ready if ready else self.rush_idle)
		y = self.bar_y + self.bar_h + m.px(8)
		label.draw(screen, cx, y, "midtop", 255 if live or ready else 180)
		x = cx - self.rush_w // 2
		by = y + label.h + m.px(4)
		blink = live and int(getattr(hs, "time", 0.0) * 10.0) % 2 == 0
		screen.fill(self.rush_track, (x, by, self.rush_w, self.bar_h))
		screen.fill(theme.WHITE if blink else (theme.ACCENT_HOT if live or ready else theme.PRIMARY_DIM), (x, by, int(self.rush_w * charge), self.bar_h))

	def segments(self, need):
		b = self.bars.get(need)
		if b:
			return b
		n = max(1, min(need, 60))
		gap = max(1, self.m.px(3)) if n > 1 else 0
		seg = (self.bar_w - gap * (n - 1)) / n
		w, h = self.bar_w, self.bar_h
		empty = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
		full = pygame.Surface((w, h), pygame.SRCALPHA)
		for i in range(n):
			x0 = int(round(i * (seg + gap)))
			x1 = int(round(i * (seg + gap) + seg))
			pygame.draw.rect(empty, theme.TRACK, pygame.Rect(x0, 0, max(1, x1 - x0) + 4, h + 4))
			pygame.draw.rect(empty, (200, 235, 250, 80), pygame.Rect(x0 + 2, 2, max(1, x1 - x0), h))
			pygame.draw.rect(full, theme.CYAN + (255,), pygame.Rect(x0, 0, max(1, x1 - x0), h))
		b = (rle(empty), rle(full), n, seg, gap)
		self.bars[need] = b
		return b

	def draw(self, screen, hs):
		m = self.m
		sector = int(getattr(hs, "sector", 1) or 0)
		cx = self.cx
		tunnel = getattr(hs, "tunnel", None)
		title = "SECTOR " + str(sector) if tunnel is None else "SECTOR " + str(sector) + "  >  " + str(sector + 1)
		title = str(getattr(hs, "stage", "") or title)
		self.kit.label(title, "small", theme.PRIMARY_DIM, spacing=5).draw(screen, cx, self.y, "midtop")
		x0 = cx - self.bar_w // 2
		if tunnel is not None:
			hyper = bool(getattr(hs, "hyper", False))
			plunge = bool(getattr(hs, "plunge", False))
			a = 200 + int(55 * math.sin(getattr(hs, "time", 0.0) * 6.0))
			(self.plunge if plunge else (self.hyper if hyper else self.warp)).draw(screen, cx, self.qy, "midtop", a)
			screen.fill(self.rush_track, (x0, self.bar_y, self.bar_w, self.bar_h))
			screen.fill(theme.WHITE if hyper or plunge else theme.ACCENT_HOT, (x0, self.bar_y, int(self.bar_w * clamp01(tunnel)), self.bar_h))
			return
		if getattr(hs, "finale", False):
			a = 190 + int(65 * math.sin(getattr(hs, "time", 0.0) * 2.2))
			self.portal.draw(screen, cx, self.qy, "midtop", a)
			self.portal_bar.blit(screen, (x0, self.bar_y), a)
			return
		if getattr(hs, "gate_open", False):
			a = 150 + int(105 * math.sin(getattr(hs, "time", 0.0) * 6.0))
			self.gate.draw(screen, cx, self.qy, "midtop", a)
			self.gate_bar.blit(screen, (x0, self.bar_y), a)
			self.draw_rush(screen, hs, cx)
			return
		need = max(1, int(getattr(hs, "quota_need", 1) or 1))
		q = max(0, int(getattr(hs, "quota", 0) or 0))
		self.digits.draw(screen, f"{q} / {need}", cx, self.qy, "midtop")
		empty, full, n, seg, gap = self.segments(need)
		screen.blit(empty, (x0 - 2, self.bar_y - 2))
		k = min(n, int(n * min(q, need) / need))
		w = int(round(k * (seg + gap) - gap)) if k > 0 else 0
		if w > 0:
			screen.blit(full, (x0, self.bar_y), (0, 0, w, self.bar_h))


class MarkerLayer:
	STEPS = 72

	def build(self, kit):
		m = kit.m
		self.m = m
		d = m.px(28)
		g = m.px(46)
		a = m.px(40)
		self.icons = {
			"ribbon": shapes.vector_sprite(d, d, shapes.diamond_painter(m.s * 10, m.s * 2.0), theme.CYAN, 0.9),
			"gate": shapes.vector_sprite(g, g, shapes.gate_painter(m.s * 19, m.s * 2.6), theme.ACCENT, 0.9),
			"next": shapes.vector_sprite(d, d, shapes.diamond_painter(m.s * 6.5, m.s * 1.6), theme.WHITE, 0.7),
		}
		self.arrow_base = {
			"ribbon": shapes.vector_sprite(a, a, shapes.arrow_painter(m.s * 14, 0), theme.CYAN, 0.9),
			"gate": shapes.vector_sprite(a, a, shapes.arrow_painter(m.s * 17, 0), theme.ACCENT, 0.9),
			"next": shapes.vector_sprite(a, a, shapes.arrow_painter(m.s * 10, 0), theme.WHITE, 0.7),
		}
		self.arrows = {}
		self.text = {
			"ribbon": kit.strip("tiny", theme.CYAN, "0123456789,. km"),
			"gate": kit.strip("small", theme.ACCENT, "0123456789,. km"),
			"next": kit.strip("tiny", theme.PRIMARY_DIM, "0123456789,. km"),
		}
		self.gap = m.px(4)
		self.push = m.px(44)

	def arrow(self, kind, angle):
		i = int(round(angle / (2 * math.pi) * self.STEPS)) % self.STEPS
		key = (kind, i)
		s = self.arrows.get(key)
		if s:
			return s
		base = self.arrow_base[kind].surf
		s = rle(pygame.transform.rotozoom(base, -i * 360.0 / self.STEPS, 1.0))
		self.arrows[key] = s
		return s

	def draw(self, screen, hs):
		ms = getattr(hs, "markers", None)
		if not ms:
			return
		for mk in sorted(ms, key=lambda o: getattr(o, "kind", "") == "gate"):
			kind = getattr(mk, "kind", "ribbon")
			kind = kind if kind in self.icons else "ribbon"
			x, y = int(mk.x), int(mk.y)
			txt = fmt_dist(getattr(mk, "dist", 0.0))
			strip = self.text[kind]
			if getattr(mk, "onscreen", True):
				icon = self.icons[kind]
				icon.draw(screen, x, y, "center")
				strip.draw(screen, txt, x, y + icon.h // 2 + self.gap, "midtop")
				continue
			a = getattr(mk, "angle", 0.0)
			s = self.arrow(kind, a)
			screen.blit(s, (x - s.get_width() // 2, y - s.get_height() // 2))
			tx = x - math.cos(a) * self.push
			ty = y - math.sin(a) * self.push
			w = strip.width(txt)
			tx = min(max(tx, w / 2 + 2), self.m.w - w / 2 - 2)
			ty = min(max(ty, strip.h / 2 + 2), self.m.h - strip.h / 2 - 2)
			strip.draw(screen, txt, tx, ty, "center")


class MessageStack:
	COLORS = {"info": theme.PRIMARY, "good": theme.GOOD, "bad": theme.DANGER, "big": theme.WHITE}

	def build(self, kit):
		m = kit.m
		self.kit = kit
		self.m = m
		self.y = m.px(150)
		self.big_y = int(m.h * 0.27)
		self.line = int(kit.fonts.get(m.font("body")).get_height() * 1.08)
		self.rise = m.px(10)

	def draw(self, screen, hs):
		msgs = getattr(hs, "messages", None)
		if not msgs:
			return
		m = self.m
		y = self.y
		by = self.big_y
		for msg in msgs:
			age = float(getattr(msg, "age", 0.0))
			ttl = max(0.01, float(getattr(msg, "ttl", 1.0)))
			fin = clamp01(age / 0.18)
			fout = clamp01((ttl - age) / 0.45)
			f = min(fin, fout)
			if f <= 0:
				continue
			kind = getattr(msg, "kind", "info")
			col = self.COLORS.get(kind, theme.PRIMARY)
			off = int((1.0 - fin) * self.rise)
			if kind == "big":
				lab = self.kit.label(str(msg.text), "big", col, spacing=6)
				lab.draw(screen, m.cx, by + off, "center", 255 * f)
				by += int(lab.h * 0.95)
				continue
			lab = self.kit.label(str(msg.text), "body", col, spacing=2)
			lab.draw(screen, m.cx, y + off, "midtop", 255 * f)
			y += int(self.line * (0.4 + 0.6 * min(1.0, fout * 2.0)))


class FluxBar:
	def build(self, kit, y):
		m = kit.m
		self.m = m
		self.w = m.px(520)
		self.h = max(3, m.px(6))
		self.x = m.cx - self.w // 2
		self.y = y
		ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
		self.track, self.tpad, self.ttop = shapes.track(self.w, self.h, ticks, max(2, m.px(3)), (0.0, 1.0), max(2, m.px(5)))
		spread = m.px(10)
		self.spread = spread
		self.glow = Label(shapes.glow_strip(self.w, self.h, theme.CYAN, spread))
		self.red_glow = Label(shapes.glow_strip(self.w, self.h, theme.DANGER, spread))
		self.caption = kit.label("FLUX", "tiny", theme.PRIMARY_DIM, spacing=4)
		self.caption_red = kit.label("FLUX", "tiny", theme.DANGER, spacing=4)
		self.pct = kit.strip("tiny", theme.PRIMARY_DIM, "0123456789%")
		self.lab_gap = m.px(14)

	def top(self):
		return self.y - self.ttop - max(2, self.m.px(5))

	def draw(self, screen, hs):
		t = getattr(hs, "time", 0.0)
		flux = clamp01(getattr(hs, "flux", 0.0))
		danger = clamp01(getattr(hs, "danger", 0.0))
		gain = clamp01(getattr(hs, "flux_gain", 0.0))
		pulse = 0.5 + 0.5 * math.sin(t * (5.0 + 9.0 * danger))
		hot = danger * (0.35 + 0.65 * pulse)
		cy = self.y + self.h // 2
		if danger > 0.05:
			self.red_glow.blit(screen, (self.x - self.spread, self.y - self.spread), 255 * hot)
		fw = int(self.w * flux)
		if fw > 0 and gain > 0.02:
			self.glow.blit(screen, (self.x - self.spread, self.y - self.spread), 200 * gain, (0, 0, fw + 2 * self.spread, self.h + 2 * self.spread))
		screen.blit(self.track, (self.x - self.tpad, self.y - self.ttop - self.tpad))
		if fw > 0:
			base = mix(theme.CYAN, theme.DANGER, danger * 1.6)
			col = mix(base, theme.ACCENT_HOT, 0.45 * pulse * danger)
			screen.fill(mix(col, theme.WHITE, 0.35 * gain * (1.0 - 0.6 * danger)), (self.x, self.y, fw, self.h))
			if gain > 0.02:
				hw = max(2, self.m.px(3))
				screen.fill(theme.WHITE, (self.x + max(0, fw - hw), self.y, min(hw, fw), self.h))
		self.caption.draw(screen, self.x - self.lab_gap, cy, "midright")
		if hot > 0.05:
			self.caption_red.draw(screen, self.x - self.lab_gap, cy, "midright", 255 * min(1.0, hot * 1.5))
		self.pct.draw(screen, str(int(round(flux * 100))) + "%", self.x + self.w + self.lab_gap, cy, "midleft")


class SpeedCluster:
	def build(self, kit):
		m = kit.m
		self.m = m
		self.w = m.px(640)
		self.h = max(2, m.px(4))
		self.x = m.cx - self.w // 2
		self.y = m.h - m.px(62)
		ticks = tuple(i / 20.0 for i in range(21))
		major = tuple(i / 4.0 for i in range(5))
		self.tick_h = max(2, m.px(4))
		self.major_h = max(3, m.px(9))
		self.track, self.tpad, self.ttop = shapes.track(self.w, self.h, ticks, self.tick_h, major, self.major_h)
		self.fill = shapes.gradient_strip(self.w, self.h, [(0.0, theme.CYAN_DEEP), (0.55, theme.CYAN), (0.82, theme.ACCENT), (0.97, theme.ACCENT_HOT), (1.0, theme.WHITE)])
		self.head_w = max(2, m.px(3))
		self.head_h = m.px(14)
		self.digits = kit.strip("speed", theme.PRIMARY, "0123456789")
		self.digits_hot = kit.strip("speed", theme.ACCENT_HOT, "0123456789")
		self.unit = kit.label("m/s", "body", theme.PRIMARY_DIM, spacing=1)
		self.num_bottom = self.y - self.ttop - m.px(2)
		desc = kit.fonts.get(m.font("speed")).get_descent()
		self.num_y = self.num_bottom - self.digits.h - desc // 2
		self.num_w = self.digits.width("000")
		self.num_gap = m.px(10)
		self.unit_x = m.cx + self.num_w // 2 + self.num_gap
		self.unit_y = self.num_y + self.digits.h + desc - m.px(4)
		self.graze_y = self.y + self.h + m.px(5)
		self.graze_h = max(2, m.px(2))
		half = self.w // 2
		prof = [1.0 - abs(i - half) / half for i in range(self.w)]
		self.graze = Label(shapes.alpha_strip(self.w, self.graze_h, theme.WHITE, [p ** 0.5 for p in prof]))
		self.graze_glow = Label(shapes.glow_strip(self.w, self.graze_h, theme.CYAN, m.px(6)))
		self.tags = self.build_tags(kit)
		self.tag_dx = self.num_w // 2 + m.px(120)
		self.tag_y = self.num_y + self.digits.h // 2 + desc // 4

	def build_tags(self, kit):
		m = kit.m
		out = {}
		for name, col in (("BOOST", theme.ACCENT), ("DRIFT", theme.CYAN)):
			lab = kit.text.get(name, m.font("tiny"), col, "main", m.px(3), 0.0, True)
			w = lab.w + m.px(18)
			h = lab.h + m.px(6)
			on = shapes.vector_sprite(w, h, shapes.frame_painter(m.s * 1.6, m.s * 3), col, 0.9)
			off = shapes.vector_sprite(w, h, shapes.frame_painter(m.s * 1.2, m.s * 3), theme.PRIMARY_DIM, 0.6)
			on.surf.blit(lab.surf, (on.pad + (w - lab.w) // 2, on.pad + (h - lab.h) // 2))
			dim = kit.text.get(name, m.font("tiny"), theme.PRIMARY_DIM, "main", m.px(3), 0.0, True)
			off.surf.blit(dim.surf, (off.pad + (w - dim.w) // 2, off.pad + (h - dim.h) // 2))
			out[name] = (on, off)
		return out

	def top(self):
		return self.num_y + self.digits.h // 6

	def draw(self, screen, hs):
		m = self.m
		t = getattr(hs, "time", 0.0)
		speed = max(0.0, float(getattr(hs, "speed", 0.0)))
		ref = max(1.0, float(getattr(hs, "speed_ref", 160.0) or 160.0))
		f = speed / ref
		boost = bool(getattr(hs, "boost", False))
		drift = bool(getattr(hs, "drift", False))
		graze = clamp01(getattr(hs, "graze", 0.0))
		strip = self.digits_hot if boost or f >= 1.0 else self.digits
		strip.draw(screen, str(int(speed)), self.unit_x - self.num_gap, self.num_y, "topright")
		self.unit.draw(screen, self.unit_x, self.unit_y, "bottomleft")
		on, off = self.tags["BOOST"]
		(on if boost else off).draw(screen, m.cx - self.tag_dx, self.tag_y, "center", 255 if boost else 150)
		on, off = self.tags["DRIFT"]
		(on if drift else off).draw(screen, m.cx + self.tag_dx, self.tag_y, "center", 255 if drift else 150)
		screen.blit(self.track, (self.x - self.tpad, self.y - self.ttop - self.tpad))
		fw = int(self.w * min(1.0, f))
		if fw > 0:
			screen.blit(self.fill, (self.x, self.y), (0, 0, fw, self.h))
			hc = theme.WHITE if f < 1.0 or int(t * 12) % 2 else theme.ACCENT
			screen.fill(hc, (self.x + fw - self.head_w, self.y + self.h // 2 - self.head_h // 2, self.head_w, self.head_h))
		if graze <= 0.02:
			return
		gw = int(self.w * graze)
		sh = 0.55 + 0.45 * math.sin(t * 37.0) * math.sin(t * 23.0 + 1.3)
		gx = (self.w - gw) // 2
		sp = (self.graze_glow.surf.get_width() - self.w) // 2
		self.graze_glow.blit(screen, (self.x + gx - sp, self.graze_y - sp), 200 * graze * sh, (gx, 0, gw + 2 * sp, self.graze_h + 2 * sp))
		self.graze.blit(screen, (self.x + gx, self.graze_y), 150 + 105 * sh, (gx, 0, gw, self.graze_h))


class TierTag:
	def build(self, kit, speed):
		m = kit.m
		self.kit = kit
		self.m = m
		boost = speed.tags["BOOST"][0]
		self.right = m.cx - speed.tag_dx - boost.w // 2 - m.px(26)
		self.y = speed.tag_y
		self.sprites = {}

	def sprite(self, name, tier):
		key = (name, tier)
		s = self.sprites.get(key)
		if s:
			return s
		m = self.m
		col = theme.TIER[tier]
		lab = self.kit.text.get(name, m.font("tiny"), col if tier else theme.PRIMARY_DIM, "main", m.px(3), 0.0, True)
		w = lab.w + m.px(18)
		h = lab.h + m.px(6)
		s = shapes.vector_sprite(w, h, shapes.frame_painter(m.s * (1.2 + 0.4 * tier), m.s * 3), col, 0.6 + 0.1 * tier)
		s.surf.blit(lab.surf, (s.pad + (w - lab.w) // 2, s.pad + (h - lab.h) // 2))
		self.sprites[key] = s
		return s

	def draw(self, screen, hs):
		tier = getattr(hs, "tier", 0) or 0
		tier = 0 if tier < 0 else 3 if tier > 3 else int(tier)
		name = str(getattr(hs, "tier_name", "") or "FLOW")
		alpha = 150
		if tier:
			alpha = 255 if tier < 3 else 205 + int(50 * math.sin(getattr(hs, "time", 0.0) * 8.0))
		self.sprite(name, tier).draw(screen, self.right, self.y, "midright", alpha)


class DepthGauge:
	def build(self, kit, speed):
		m = kit.m
		drift = speed.tags["DRIFT"][0]
		self.x = m.cx + speed.tag_dx + drift.w // 2 + m.px(26)
		self.y = speed.tag_y
		self.caption = kit.label("DEPTH", "tiny", theme.DEEP_DIM, spacing=4)
		self.digits = kit.strip("label", theme.DEEP, "0123456789 m", bold=True)
		self.gap = m.px(10)
		self.surge = kit.label("SURGE", "tiny", theme.DEEP_DIM, spacing=4)
		self.surge_hot = kit.label("SURGE", "tiny", theme.WHITE, spacing=4)
		self.bar_x = self.x + self.caption.w + self.gap + self.digits.width("000 m") + m.px(24)
		self.bar_w = m.px(110)
		self.bar_h = max(3, m.px(6))
		self.track = mix(theme.DEEP_DIM, theme.DARK, 0.78)

	def draw(self, screen, hs):
		depth = getattr(hs, "depth", 0.0) or 0.0
		surge = clamp01(getattr(hs, "surge", 0.0) or 0.0)
		if depth <= 0.0 and surge <= 0.02:
			return
		if depth > 0.0:
			self.caption.draw(screen, self.x, self.y, "midleft")
			self.digits.draw(screen, "%d m" % int(depth), self.x + self.caption.w + self.gap, self.y, "midleft")
		full = surge > 0.85
		blink = full and int(getattr(hs, "time", 0.0) * 9.0) % 2 == 0
		(self.surge_hot if full else self.surge).draw(screen, self.bar_x, self.y, "midleft")
		x = self.bar_x + self.surge.w + self.gap
		y = int(self.y) - self.bar_h // 2
		screen.fill(self.track, (x, y, self.bar_w, self.bar_h))
		fw = int(self.bar_w * surge)
		if fw > 0:
			screen.fill(theme.WHITE if blink else mix(theme.CYAN_DEEP, theme.ACCENT_HOT, surge * surge), (x, y, fw, self.bar_h))


class DryNotice:
	def build(self, kit, y):
		m = kit.m
		self.m = m
		self.y = y
		self.title = kit.label("FLUX DEPLETED  -  LOSING GRIP", "label", theme.DANGER, spacing=4, bold=True)
		self.tip = kit.label("grip fades with speed  -  tether and sling, grab motes, graze, skim or breach the sea to relight", "tiny", theme.PRIMARY_DIM, spacing=2)
		self.tip_y = y + self.title.h + m.px(2)
		self.stalled = kit.label("STALLED", "big", theme.DANGER, spacing=8)
		self.bar_w = m.px(300)
		self.bar_h = max(3, m.px(6))
		self.track = mix(theme.DANGER, theme.DARK, 0.75)
		self.place(int(m.h * 0.36))

	def place(self, y):
		self.stall_y = y
		self.bar_y = y + self.stalled.h // 2 + self.m.px(12)

	def draw(self, screen, hs):
		if not getattr(hs, "dry", False):
			return
		t = getattr(hs, "time", 0.0)
		m = self.m
		self.title.draw(screen, m.cx, self.y, "midtop", 190 + int(65 * math.sin(t * 6.0)))
		self.tip.draw(screen, m.cx, self.tip_y, "midtop", 220)
		stall = clamp01(getattr(hs, "stall", 0.0) or 0.0)
		if stall <= 0.0:
			return
		self.stalled.draw(screen, m.cx, self.stall_y, "center", 255 if int(t * 8.0) % 2 else 150)
		x = m.cx - self.bar_w // 2
		screen.fill(self.track, (x, self.bar_y, self.bar_w, self.bar_h))
		screen.fill(theme.DANGER, (x, self.bar_y, int(self.bar_w * (1.0 - stall)), self.bar_h))


class PerfPanel:
	def __init__(self, source="perf", left=False):
		self.source = source
		self.left = left

	def build(self, kit):
		m = kit.m
		self.m = m
		self.cache = TextCache(kit.fonts, 96)
		self.size = m.font("mono")
		self.margin = m.px(18)
		self.bpad = m.px(6)
		self.backs = {}

	def draw(self, screen, hs):
		lines = getattr(hs, self.source, None)
		if not lines:
			return
		labs = [self.cache.get(str(s), self.size, theme.PRIMARY, "mono", 0, 0.0) for s in lines]
		w = max(l.w for l in labs)
		h = sum(l.h for l in labs)
		x = self.margin if self.left else self.m.w - self.margin - w
		y = (self.m.h - h) // 2 if self.left else self.margin
		screen.blit(self.backing(w, h), (x - self.bpad, y - self.bpad))
		screen.blits([(l.surf, (x, y + i * l.h)) for i, l in enumerate(labs)], False)

	def backing(self, w, h):
		q = self.m.px(24)
		key = ((w + q - 1) // q * q, h)
		b = self.backs.get(key)
		if b:
			return b
		b = pygame.Surface((key[0] + 2 * self.bpad, h + 2 * self.bpad), pygame.SRCALPHA)
		b.fill((2, 6, 10, 165))
		self.backs[key] = b
		return b


class HintLine:
	def __init__(self):
		self.provider = None

	def build(self, kit):
		self.kit = kit
		self.m = kit.m

	def draw(self, screen, hs):
		h = getattr(hs, "hint", None)
		if not h:
			return
		if self.provider is not None:
			h = self.provider.hint_text()
		self.kit.label(str(h), "small", theme.PRIMARY_DIM, spacing=1).draw(screen, self.m.cx, self.m.h - self.m.px(8), "midbottom")


class Hud:
	def __init__(self, size):
		self.kit = Kit(size)
		self.crosshair = Crosshair()
		self.prograde = Prograde()
		self.score = ScorePanel()
		self.sector = SectorPanel()
		self.markers = MarkerLayer()
		self.messages = MessageStack()
		self.speed = SpeedCluster()
		self.flux = FluxBar()
		self.perf = PerfPanel()
		self.dev = PerfPanel("dev", True)
		self.hint = HintLine()
		self.tier = TierTag()
		self.depth = DepthGauge()
		self.dry = DryNotice()
		self.minimap = 'bottom_right'
		self.size = None
		self.resize(size)

	def resize(self, size):
		size = (int(size[0]), int(size[1]))
		if size == self.size:
			return
		self.size = size
		self.kit.resize(size)
		for w in (self.crosshair, self.prograde, self.score, self.sector, self.markers, self.messages, self.speed, self.perf, self.dev, self.hint):
			w.build(self.kit)
		self.flux.build(self.kit, self.speed.top() - self.kit.m.px(18))
		self.tier.build(self.kit, self.speed)
		self.depth.build(self.kit, self.speed)
		self.dry.build(self.kit, int(self.kit.m.h * 0.6))
		self.arrange()

	def set_minimap(self, mode):
		if mode == self.minimap:
			return
		self.minimap = mode
		if self.size is not None:
			self.arrange()

	def arrange(self):
		m = self.kit.m
		sector = self.sector
		messages = self.messages
		dry = self.dry
		sector.cx = m.cx
		messages.y = m.px(150)
		messages.big_y = int(m.h * 0.27)
		dry.place(int(m.h * 0.36))
		if self.minimap != 'top_center':
			return
		y0, _, size, _ = minimap_box(m.h, m.w, self.minimap)
		sector.cx = m.w - m.px(56) - max(sector.bar_w, sector.gate.w) // 2
		messages.y = y0 + size + m.px(8)
		messages.big_y = messages.y + int(messages.line * 3.2)
		dry.place(int(m.h * 0.44))

	def set_bindings(self, provider):
		self.hint.provider = provider

	def draw(self, screen, hs):
		if screen.get_size() != self.size:
			self.resize(screen.get_size())
		self.markers.draw(screen, hs)
		self.prograde.draw(screen, hs)
		self.crosshair.draw(screen, hs)
		self.score.draw(screen, hs)
		self.sector.draw(screen, hs)
		self.messages.draw(screen, hs)
		self.flux.draw(screen, hs)
		self.speed.draw(screen, hs)
		self.tier.draw(screen, hs)
		self.depth.draw(screen, hs)
		self.dry.draw(screen, hs)
		self.perf.draw(screen, hs)
		self.dev.draw(screen, hs)
		self.hint.draw(screen, hs)
