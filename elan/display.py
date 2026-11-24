import ctypes
import os
import time
import numpy as np
import pygame
import torch


class Display:
	def __init__(self, fullscreen, window_size=(1280, 720), want_vsync=False):
		self.want_vsync = want_vsync
		os.environ.setdefault('SDL_RENDER_SCALE_QUALITY', '1')
		try:
			ctypes.windll.user32.SetProcessDPIAware()
		except Exception:
			pass
		pygame.display.init()
		info = pygame.display.Info()
		self.desktop = (info.current_w, info.current_h)
		self.window_size = window_size
		self.fullscreen = fullscreen
		self.surface = None
		self.size = (0, 0)
		self.vsync = False
		self.order = (0, 1, 2)
		self.direct = True
		self.staging = None
		self.refresh = self._refresh_rate()
		self.open(fullscreen)

	def _refresh_rate(self):
		try:
			user32 = ctypes.windll.user32
			gdi32 = ctypes.windll.gdi32
			dc = user32.GetDC(0)
			rate = gdi32.GetDeviceCaps(dc, 116)
			user32.ReleaseDC(0, dc)
			return float(rate) if rate >= 30 else 60.0
		except Exception:
			return 60.0

	def open(self, fullscreen):
		self.fullscreen = fullscreen
		attempts = []
		if fullscreen:
			logical = self.desktop
			if logical[0] > 2600 and logical[0] % 2 == 0 and logical[1] % 2 == 0:
				logical = (logical[0] // 2, logical[1] // 2)
			if self.want_vsync:
				attempts.append((logical, pygame.FULLSCREEN | pygame.SCALED, 1))
			attempts.append((logical, pygame.FULLSCREEN | pygame.SCALED, 0))
			attempts.append((self.desktop, pygame.FULLSCREEN, 0))
		attempts.append((self.window_size, 0, 0))
		for size, flags, vsync in attempts:
			try:
				self.surface = pygame.display.set_mode(size, flags, vsync=vsync)
			except pygame.error:
				continue
			self.vsync = bool(vsync)
			break
		self.size = self.surface.get_size()
		pygame.display.set_caption('ÉLAN')
		shifts = self.surface.get_shifts()
		slots = [0, 1, 2]
		for channel, shift in ((2, shifts[0]), (1, shifts[1]), (0, shifts[2])):
			if shift // 8 < 3:
				slots[shift // 8] = channel
		self.order = tuple(slots)
		self.direct = self.surface.get_bytesize() == 4 and self.surface.get_pitch() == self.size[0] * 4
		self.staging = None

	def toggle_fullscreen(self):
		self.open(not self.fullscreen)

	def needs_reorder(self):
		return self.order != (0, 1, 2)

	def present(self, frame):
		if not self.direct:
			self._present_slow(frame)
			return
		if self.needs_reorder():
			frame = frame[..., [self.order[0], self.order[1], self.order[2], 3]]
		buffer = self.surface.get_buffer()
		try:
			target = torch.frombuffer(buffer, dtype=torch.uint8)
			target.copy_(frame.reshape(-1))
			del target
		except Exception:
			self.direct = False
			del buffer
			self._present_slow(frame)
			return
		del buffer

	def _present_slow(self, frame):
		rgb = frame[..., [2, 1, 0]].cpu().numpy()
		image = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), 'RGB')
		self.surface.blit(image, (0, 0))

	def flip(self):
		pygame.display.flip()

	def save(self, path):
		pygame.image.save(self.surface, path)


class FramePacer:
	def __init__(self, display, cap):
		self.display = display
		self.cap = cap
		self.last = time.perf_counter()

	def wait(self):
		if self.display.vsync and self.cap <= 0:
			self.last = time.perf_counter()
			return
		limit = self.cap if self.cap > 0 else 240.0
		target = self.last + 1.0 / limit
		now = time.perf_counter()
		remaining = target - now
		if remaining > 0.002:
			time.sleep(remaining - 0.001)
		while time.perf_counter() < target:
			pass
		self.last = max(target, time.perf_counter() - 0.05)


class ResolutionGovernor:
	def __init__(self, renderer, scales, target_time):
		self.renderer = renderer
		self.scales = scales
		self.target = target_time
		self.index = len(scales) - 1
		self.work = target_time * 0.7
		self.gpu = 0.0
		self.cooldown = 1.5
		self.locked = False

	def lock(self, scale):
		self.locked = True
		self.renderer.set_scale(scale)

	def unlock(self):
		self.locked = False
		self.renderer.set_scale(self.scales[self.index])

	def update(self, dt, work_time, gpu_wait):
		self.work += (work_time - self.work) * 0.08
		self.gpu += (gpu_wait - self.gpu) * 0.08
		if self.locked:
			return
		self.cooldown -= dt
		if self.cooldown > 0.0:
			return
		if self.work > self.target * 0.97 and self.gpu > self.work * 0.22 and self.index > 0:
			self.index -= 1
			self.renderer.set_scale(self.scales[self.index])
			self.cooldown = 0.7
		elif (self.work < self.target * 0.8 or self.gpu < self.target * 0.2) and self.index < len(self.scales) - 1:
			self.index += 1
			self.renderer.set_scale(self.scales[self.index])
			self.cooldown = 2.5
