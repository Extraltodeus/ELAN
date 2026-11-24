import io
import os
import struct

import numpy as np
import pygame

SIZE = 512
SIZES = (16, 24, 32, 48, 64, 128, 256)
KITE = {'nose': (0.0, 0.0, 1.55), 'tail': (0.0, 0.05, -0.7), 'keel': (0.0, -0.42, 0.2), 'fin': (0.0, 0.62, -1.0), 'left': (-1.75, 0.42, -1.05), 'right': (1.75, 0.42, -1.05)}
EDGES = (('nose', 'left', 1.0), ('nose', 'right', 1.0), ('left', 'tail', 1.0), ('right', 'tail', 1.0), ('nose', 'tail', 0.6), ('tail', 'fin', 0.8), ('fin', 'nose', 0.35), ('nose', 'keel', 0.5), ('keel', 'tail', 0.5))
CYAN = np.array((0.49, 0.89, 1.0))
WHITE = np.array((0.86, 0.97, 1.0))
ACCENT = np.array((1.0, 0.62, 0.23))
HOT = np.array((1.0, 0.87, 0.63))
DEEP = np.array((0.016, 0.04, 0.07))
NIGHT = np.array((0.05, 0.12, 0.22))


class Canvas:
	def __init__(self, size):
		self.size = size
		axis = (np.arange(size) + 0.5) / size * 2.0 - 1.0
		self.x, self.y = np.meshgrid(axis, -axis)
		self.color = np.zeros((size, size, 3))

	def segment(self, a, b):
		ax, ay = a
		bx, by = b
		dx, dy = bx - ax, by - ay
		t = np.clip(((self.x - ax) * dx + (self.y - ay) * dy) / max(dx * dx + dy * dy, 1.0e-9), 0.0, 1.0)
		return np.hypot(self.x - ax - t * dx, self.y - ay - t * dy), t

	def add(self, weight, color):
		self.color += weight[..., None] * color

	def line(self, a, b, width, glow, color, strength=1.0):
		d, _ = self.segment(a, b)
		self.add(strength * (np.clip(1.0 - (d - width) / (2.0 / self.size), 0.0, 1.0) + 0.55 * np.exp(-d / glow)), color)

	def trail(self, a, b, width, color):
		d, t = self.segment(a, b)
		fade = (1.0 - t) ** 1.6
		self.add(fade * (np.clip(1.0 - (d - width * fade) / (2.0 / self.size), 0.0, 1.0) + 0.6 * np.exp(-d / (width * 4.0))), color)


def project(point):
	x, y, z = point
	pitch = 0.42
	cy, sy = np.cos(pitch), np.sin(pitch)
	y, z = y * cy + z * sy, z * cy - y * sy
	depth = 5.2 - z
	return (x / depth * 2.25, y / depth * 2.25 - 0.08)


def backdrop(canvas):
	r = np.hypot(canvas.x, canvas.y)
	glow = np.exp(-((canvas.y + 0.55) ** 2) / 0.09) * np.exp(-(canvas.x ** 2) / 0.9)
	sky = np.clip(1.0 - r * 0.75, 0.0, 1.0)
	canvas.color = DEEP + sky[..., None] * NIGHT + glow[..., None] * ACCENT * 0.55
	arc = np.abs(np.hypot(canvas.x, canvas.y + 2.35) - 1.62)
	canvas.add(np.exp(-arc / 0.012) * np.clip(1.0 - np.abs(canvas.x) * 0.9, 0.0, 1.0), HOT * 0.9)
	rng = np.random.default_rng(7)
	for _ in range(26):
		sx, sy = rng.uniform(-0.85, 0.85), rng.uniform(-0.1, 0.85)
		d = np.hypot(canvas.x - sx, canvas.y - sy)
		canvas.add(np.exp(-d / 0.006) * rng.uniform(0.3, 1.0), WHITE)


def craft(canvas):
	points = {name: project(p) for name, p in KITE.items()}
	for side, tip in (('left', -1.0), ('right', 1.0)):
		a = points[side]
		canvas.trail(a, (a[0] * 0.45 + tip * 0.02, a[1] - 0.62), 0.012, ACCENT)
	for a, b, weight in EDGES:
		canvas.line(points[a], points[b], 0.007 + 0.009 * weight, 0.03 + 0.02 * weight, CYAN if weight < 0.9 else WHITE, 0.55 + 0.6 * weight)
	nose = points['nose']
	d = np.hypot(canvas.x - nose[0], canvas.y - nose[1])
	canvas.add(np.exp(-d / 0.025), WHITE)


def mask(canvas):
	edge = 0.94
	radius = 0.3
	qx = np.maximum(np.abs(canvas.x) - edge + radius, 0.0)
	qy = np.maximum(np.abs(canvas.y) - edge + radius, 0.0)
	d = np.hypot(qx, qy) - radius
	return np.clip(0.5 - d * canvas.size / 2.0, 0.0, 1.0)


def tonemap(color):
	color = 1.0 - np.exp(-color * 1.25)
	return np.clip(color ** (1.0 / 1.1), 0.0, 1.0)


def render():
	canvas = Canvas(SIZE)
	backdrop(canvas)
	craft(canvas)
	rgb = (tonemap(canvas.color) * 255.0 + 0.5).astype(np.uint8)
	alpha = (mask(canvas) * 255.0 + 0.5).astype(np.uint8)
	surface = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA, 32)
	pygame.surfarray.pixels3d(surface)[...] = rgb.transpose(1, 0, 2)
	pygame.surfarray.pixels_alpha(surface)[...] = alpha.T
	return surface


def png_bytes(surface):
	stream = io.BytesIO()
	pygame.image.save(surface, stream, 'icon.png')
	return stream.getvalue()


def write_ico(path, surface):
	images = [png_bytes(pygame.transform.smoothscale(surface, (s, s))) for s in SIZES]
	offset = 6 + 16 * len(images)
	header = struct.pack('<HHH', 0, 1, len(images))
	entries = b''
	for size, data in zip(SIZES, images):
		entries += struct.pack('<BBBBHHII', size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
		offset += len(data)
	with open(path, 'wb') as handle:
		handle.write(header + entries + b''.join(images))


def main():
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	assets = os.path.join(root, 'assets')
	pygame.init()
	surface = render()
	pygame.image.save(pygame.transform.smoothscale(surface, (256, 256)), os.path.join(assets, 'icon.png'))
	write_ico(os.path.join(assets, 'icon.ico'), surface)


if __name__ == '__main__':
	main()
