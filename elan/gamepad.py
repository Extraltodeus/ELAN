import pygame

from elan.gyro import MotionSensor

AXIS_BASE = 100
STICK_CODES = {'left': (100, 101, 102, 103), 'right': (104, 105, 106, 107)}
NAV_UP = (11, 102)
NAV_DOWN = (12, 103)
NAV_LEFT = (13, 100)
NAV_RIGHT = (14, 101)
CONFIRM = 0
BACK = 1

NAMES = {
	0: 'A', 1: 'B', 2: 'X', 3: 'Y', 4: 'BACK', 5: 'GUIDE', 6: 'START', 7: 'L3', 8: 'R3', 9: 'LB', 10: 'RB',
	11: 'D-UP', 12: 'D-DOWN', 13: 'D-LEFT', 14: 'D-RIGHT',
	100: 'LS LEFT', 101: 'LS RIGHT', 102: 'LS UP', 103: 'LS DOWN',
	104: 'RS LEFT', 105: 'RS RIGHT', 106: 'RS UP', 107: 'RS DOWN',
	109: 'LT', 111: 'RT',
}

RAW_BUTTONS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 9, 5: 10, 6: 4, 7: 6, 8: 7, 9: 8, 10: 5}


def pad_name(code):
	return NAMES.get(code, 'PAD %d' % code)


def pad_code(value):
	if not isinstance(value, str) or not value.startswith('pad:') or not value[4:].isdigit():
		return None
	return int(value[4:])


def pad_label(value):
	code = pad_code(value)
	return '—' if code is None else pad_name(code)


def pad_rows(look, table):
	rows = [('Right stick' if look == 'right' else 'Left stick', 'steer')]
	for action, word in (('thrust', 'thrust'), ('brake', 'air brake')):
		rows.append((pad_label(table.get(action)), word))
	rows.append((pad_label(table.get('strafe_left')) + ' / ' + pad_label(table.get('strafe_right')), 'strafe'))
	for action, word in (('boost', 'boost'), ('tether', 'tether'), ('drift', 'drift'), ('camera', 'camera'), ('pause', 'pause'), ('restart', 'restart')):
		rows.append((pad_label(table.get(action)), word))
	rows.append((pad_label(table.get('gyro')) + ' / ' + pad_label(table.get('gyro_reset')), 'gyro on-off / recenter'))
	rows.append(('D-pad · A · B', 'menus'))
	return tuple(rows)


class Gamepad:
	DEADZONE = 0.16
	TRIGGER_DEADZONE = 0.06
	PRESS = 0.5
	BUTTONS = 15

	def __init__(self):
		self.api = None
		self.device = None
		self.raw = None
		self.name = ''
		self.axes = [0.0] * 6
		self.held = set()
		self.edges = set()
		self.trigger_seen = [False, False]
		self.motion = MotionSensor()
		pygame.joystick.init()
		try:
			from pygame._sdl2 import controller
			controller.init()
			self.api = controller
		except Exception:
			self.api = None
		self.open()

	def connected(self):
		return self.device is not None or self.raw is not None

	def status(self):
		if not self.connected():
			return 'NO GAMEPAD DETECTED'
		note = '  (GENERIC LAYOUT)' if self.device is None else ('  ·  GYRO READY' if self.motion.ready else '  ·  NO GYRO')
		return 'GAMEPAD: %s%s' % (self.name.upper(), note)

	def open(self):
		self.device = None
		self.raw = None
		self.name = ''
		self.axes = [0.0] * 6
		self.held = set()
		self.edges = set()
		self.trigger_seen = [False, False]
		self.motion.attach(None)
		try:
			count = pygame.joystick.get_count()
		except pygame.error:
			return
		for index in range(count):
			if self.api is not None and self._open_controller(index):
				return
		for index in range(count):
			if self._open_raw(index):
				return

	def _open_controller(self, index):
		try:
			if not self.api.is_controller(index):
				return False
			self.device = self.api.Controller(index)
			self.name = str(self.api.name_forindex(index) or 'controller')
			self.motion.attach(index)
			return True
		except Exception:
			self.device = None
			return False

	def _open_raw(self, index):
		try:
			self.raw = pygame.joystick.Joystick(index)
			self.raw.init()
			self.name = str(self.raw.get_name())
			return True
		except pygame.error:
			self.raw = None
			return False

	def _stick(self, value):
		size = abs(value)
		if size < self.DEADZONE:
			return 0.0
		return (1.0 if value > 0.0 else -1.0) * min((size - self.DEADZONE) / (1.0 - self.DEADZONE), 1.0)

	def _trigger(self, value):
		if value < self.TRIGGER_DEADZONE:
			return 0.0
		return min((value - self.TRIGGER_DEADZONE) / (1.0 - self.TRIGGER_DEADZONE), 1.0)

	def _read_controller(self, held):
		device = self.device
		for button in range(self.BUTTONS):
			if device.get_button(button):
				held.add(button)
		for axis in range(4):
			self.axes[axis] = self._stick(device.get_axis(axis) / 32767.0)
		for axis in (4, 5):
			self.axes[axis] = self._trigger(max(device.get_axis(axis), 0) / 32767.0)

	def _read_raw(self, held):
		raw = self.raw
		for button in range(min(raw.get_numbuttons(), 16)):
			if raw.get_button(button) and button in RAW_BUTTONS:
				held.add(RAW_BUTTONS[button])
		if raw.get_numhats() > 0:
			x, y = raw.get_hat(0)
			if y > 0:
				held.add(11)
			if y < 0:
				held.add(12)
			if x < 0:
				held.add(13)
			if x > 0:
				held.add(14)
		count = raw.get_numaxes()
		for axis in range(4):
			self.axes[axis] = self._stick(raw.get_axis(axis)) if axis < count else 0.0
		for slot, axis in enumerate((4, 5)):
			value = raw.get_axis(axis) if axis < count else 0.0
			if value != 0.0:
				self.trigger_seen[slot] = True
			self.axes[axis] = self._trigger((value + 1.0) * 0.5) if self.trigger_seen[slot] else 0.0

	def poll(self):
		held = set()
		try:
			if self.device is not None:
				self._read_controller(held)
			elif self.raw is not None:
				self._read_raw(held)
		except Exception:
			if self.device is not None:
				self.api = None
			self.open()
			return
		for axis, value in enumerate(self.axes):
			if value > self.PRESS:
				held.add(AXIS_BASE + axis * 2 + 1)
			elif value < -self.PRESS:
				held.add(AXIS_BASE + axis * 2)
		self.edges = held - self.held
		self.held = held

	def value(self, code):
		if code is None:
			return 0.0
		if code < AXIS_BASE:
			return 1.0 if code in self.held else 0.0
		axis, positive = divmod(code - AXIS_BASE, 2)
		if axis >= len(self.axes):
			return 0.0
		amount = self.axes[axis]
		return max(amount, 0.0) if positive else max(-amount, 0.0)

	def stick(self, side):
		base = 0 if side == 'left' else 2
		return self.axes[base], self.axes[base + 1]
