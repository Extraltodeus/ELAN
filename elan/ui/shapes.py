import math

import numpy as np
import pygame

from . import theme
from .text import Label, soft_shadow, rle

SS = 4


def vector_sprite(w, h, painter, color, shadow=0.85):
	w, h = max(1, int(w)), max(1, int(h))
	big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
	painter(big, SS, color)
	fg = pygame.transform.smoothscale(big, (w, h))
	if shadow <= 0:
		return Label(fg, 0, w, h)
	mbig = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
	painter(mbig, SS, theme.DARK)
	mask = pygame.transform.smoothscale(mbig, (w, h))
	blur = max(2, min(w, h) // 8)
	pad = blur + 2
	sh = soft_shadow(mask, w, h, pad, 0, blur, int(220 * shadow))
	sh.blit(fg, (pad, pad))
	return Label(sh, pad, w, h)


def ring_painter(r, lw):
	def paint(s, k, c):
		W, H = s.get_size()
		pygame.draw.circle(s, c, (W // 2, H // 2), int(r * k), max(1, int(lw * k)))
	return paint


def dot_painter(r):
	def paint(s, k, c):
		W, H = s.get_size()
		pygame.draw.circle(s, c, (W // 2, H // 2), max(1, int(r * k)))
	return paint


def bracket_painter(r, lw, span):
	def paint(s, k, c):
		W, H = s.get_size()
		R = int(r * k)
		rect = pygame.Rect(W // 2 - R, H // 2 - R, 2 * R, 2 * R)
		for i in range(4):
			a = math.pi / 4 + i * math.pi / 2
			pygame.draw.arc(s, c, rect, a - span, a + span, max(1, int(lw * k)))
	return paint


def locked_painter(r, lw, dot):
	def paint(s, k, c):
		W, H = s.get_size()
		cx, cy = W // 2, H // 2
		pygame.draw.circle(s, c, (cx, cy), int(r * k), max(1, int(lw * k)))
		pygame.draw.circle(s, c, (cx, cy), max(1, int(dot * k)))
		for i in range(4):
			a = i * math.pi / 2
			x0, y0 = cx + math.cos(a) * (r + 2) * k, cy + math.sin(a) * (r + 2) * k
			x1, y1 = cx + math.cos(a) * (r + 6) * k, cy + math.sin(a) * (r + 6) * k
			pygame.draw.line(s, c, (x0, y0), (x1, y1), max(1, int(lw * k)))
	return paint


def prograde_painter(r, lw, tick):
	def paint(s, k, c):
		W, H = s.get_size()
		cx, cy = W // 2, H // 2
		pygame.draw.circle(s, c, (cx, cy), int(r * k), max(1, int(lw * k)))
		pygame.draw.circle(s, c, (cx, cy), max(1, int(lw * 0.9 * k)))
		for a in (-math.pi / 2, 0.0, math.pi):
			x0, y0 = cx + math.cos(a) * r * k, cy + math.sin(a) * r * k
			x1, y1 = cx + math.cos(a) * (r + tick) * k, cy + math.sin(a) * (r + tick) * k
			pygame.draw.line(s, c, (x0, y0), (x1, y1), max(1, int(lw * k)))
	return paint


def diamond_painter(r, lw):
	def paint(s, k, c):
		W, H = s.get_size()
		cx, cy = W // 2, H // 2
		R = r * k
		pts = [(cx, cy - R), (cx + R, cy), (cx, cy + R), (cx - R, cy)]
		pygame.draw.polygon(s, c, pts, max(1, int(lw * k)))
		q = R * 0.32
		pygame.draw.polygon(s, c, [(cx, cy - q), (cx + q, cy), (cx, cy + q), (cx - q, cy)])
	return paint


def gate_painter(r, lw):
	def paint(s, k, c):
		W, H = s.get_size()
		cx, cy = W // 2, H // 2
		pygame.draw.circle(s, c, (cx, cy), int(r * k), max(1, int(lw * k)))
		pygame.draw.circle(s, c, (cx, cy), int(r * 0.62 * k), max(1, int(lw * 0.6 * k)))
	return paint


def arrow_painter(r, lw):
	def paint(s, k, c):
		W, H = s.get_size()
		cx, cy = W // 2, H // 2
		R = r * k
		pts = [(cx + R, cy), (cx - R * 0.55, cy - R * 0.75), (cx - R * 0.15, cy), (cx - R * 0.55, cy + R * 0.75)]
		pygame.draw.polygon(s, c, pts)
	return paint


def frame_painter(lw, radius):
	def paint(s, k, c):
		W, H = s.get_size()
		m = int(lw * k)
		pygame.draw.rect(s, c, pygame.Rect(m // 2, m // 2, W - m, H - m), max(1, m), int(radius * k))
	return paint


def gradient_strip(w, h, stops):
	w, h = max(1, int(w)), max(1, int(h))
	xs = np.linspace(0.0, 1.0, w)
	pos = [p for p, _ in stops]
	rgb = np.stack([np.interp(xs, pos, [c[i] for _, c in stops]) for i in range(3)], axis=1)
	arr = np.repeat(rgb[:, None, :], h, axis=1).astype(np.uint8)
	surf = pygame.Surface((w, h))
	pygame.surfarray.blit_array(surf, arr)
	return surf


def alpha_strip(w, h, color, xprof, yprof=None):
	w, h = max(1, int(w)), max(1, int(h))
	surf = pygame.Surface((w, h), pygame.SRCALPHA)
	surf.fill(color[:3] + (255,))
	a = np.asarray(xprof, dtype=np.float32)[:, None]
	if yprof is not None:
		a = a * np.asarray(yprof, dtype=np.float32)[None, :]
	else:
		a = np.repeat(a, h, axis=1)
	alpha = pygame.surfarray.pixels_alpha(surf)
	alpha[:] = np.clip(a * 255.0, 0, 255).astype(np.uint8)
	del alpha
	return surf


def glow_strip(w, h, color, spread):
	W, H = int(w) + 2 * spread, int(h) + 2 * spread
	core = pygame.Surface((W, H), pygame.SRCALPHA)
	pygame.draw.rect(core, color[:3] + (255,), pygame.Rect(spread, spread, int(w), int(h)))
	f = max(2, spread // 2)
	small = pygame.transform.smoothscale(core, (max(1, W // f), max(1, H // f)))
	glow = pygame.transform.smoothscale(small, (W, H))
	glow.blit(glow.copy(), (0, 0))
	return glow


def track(w, h, ticks=(), tick_h=0, tick_major=(), major_h=0, edge=True):
	w, h = max(1, int(w)), max(1, int(h))
	top = max(tick_h, major_h)
	pad = 2
	surf = pygame.Surface((w + 2 * pad, h + top + 2 * pad), pygame.SRCALPHA)
	pygame.draw.rect(surf, theme.TRACK, pygame.Rect(0, top, w + 2 * pad, h + 2 * pad))
	if edge:
		pygame.draw.rect(surf, theme.TRACK_EDGE, pygame.Rect(pad, top + pad, w, h), 1)
	for f in ticks:
		x = pad + int(round(f * (w - 1)))
		big = f in tick_major
		th = major_h if big else tick_h
		col = (225, 245, 255, 200) if big else (200, 235, 250, 110)
		pygame.draw.line(surf, col, (x, top + pad - th), (x, top + pad - 1))
	return rle(surf), pad, top


def shade_map(w, h, rows, cols=None):
	ys = np.linspace(0.0, 1.0, h)
	v = np.interp(ys, [p for p, _ in rows], [d for _, d in rows])
	m = np.repeat(v[None, :], w, axis=0)
	if cols is not None:
		xs = np.linspace(0.0, 1.0, w)
		hv = np.interp(xs, [p for p, _ in cols], [d for _, d in cols])
		m = 1.0 - (1.0 - m) * (1.0 - hv[:, None])
	return m
