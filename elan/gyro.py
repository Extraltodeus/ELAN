import ctypes
import math
import os

import pygame


class MotionSensor:
	GYRO = 2

	def __init__(self):
		self.sdl = None
		self.handle = None
		self.ready = False
		self.buffer = (ctypes.c_float * 3)()
		try:
			sdl = ctypes.CDLL(os.path.join(os.path.dirname(pygame.__file__), 'SDL2.dll'))
			sdl.SDL_GameControllerOpen.restype = ctypes.c_void_p
			sdl.SDL_GameControllerOpen.argtypes = (ctypes.c_int,)
			sdl.SDL_GameControllerHasSensor.restype = ctypes.c_int
			sdl.SDL_GameControllerHasSensor.argtypes = (ctypes.c_void_p, ctypes.c_int)
			sdl.SDL_GameControllerSetSensorEnabled.restype = ctypes.c_int
			sdl.SDL_GameControllerSetSensorEnabled.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
			sdl.SDL_GameControllerGetSensorData.restype = ctypes.c_int
			sdl.SDL_GameControllerGetSensorData.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int)
			self.sdl = sdl
		except (OSError, AttributeError):
			self.sdl = None

	def attach(self, index):
		self.handle = None
		self.ready = False
		if self.sdl is None or index is None:
			return
		try:
			handle = self.sdl.SDL_GameControllerOpen(index)
			if not handle or not self.sdl.SDL_GameControllerHasSensor(handle, self.GYRO):
				return
			if self.sdl.SDL_GameControllerSetSensorEnabled(handle, self.GYRO, 1) != 0:
				return
			self.handle = handle
			self.ready = True
		except (OSError, ctypes.ArgumentError):
			self.ready = False

	def rates(self):
		if not self.ready:
			return None
		try:
			if self.sdl.SDL_GameControllerGetSensorData(self.handle, self.GYRO, self.buffer, 3) != 0:
				return None
		except (OSError, ctypes.ArgumentError):
			self.ready = False
			return None
		return self.buffer[0], self.buffer[1], self.buffer[2]


class TiltSteer:
	LIMIT = 0.5
	DEAD = 0.035
	TAU = 0.016

	def __init__(self):
		self.turn = 0.0
		self.pitch = 0.0
		self.out = [0.0, 0.0]

	def reset(self):
		self.turn = 0.0
		self.pitch = 0.0
		self.out = [0.0, 0.0]

	def _shape(self, angle):
		size = abs(angle)
		if size <= self.DEAD:
			return 0.0
		amount = min((size - self.DEAD) / (self.LIMIT - self.DEAD), 1.0)
		return amount if angle > 0.0 else -amount

	def step(self, rates, dt, smoothing, axis):
		pitch_rate, yaw_rate, roll_rate = rates
		bound = self.LIMIT * 1.25
		self.turn = max(-bound, min(bound, self.turn - (roll_rate if axis == 'roll' else yaw_rate) * dt))
		self.pitch = max(-bound, min(bound, self.pitch + pitch_rate * dt))
		x = self._shape(self.turn)
		y = self._shape(self.pitch)
		if smoothing <= 0:
			self.out = [x, y]
			return x, y
		k = 1.0 - math.exp(-max(dt, 0.0) / (self.TAU * smoothing))
		self.out[0] += (x - self.out[0]) * k
		self.out[1] += (y - self.out[1]) * k
		return self.out[0], self.out[1]
