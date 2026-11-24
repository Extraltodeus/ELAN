import copy
import math

import pygame

from elan.config import FlightCfg
from elan.gamepad import BACK, CONFIRM, NAV_DOWN, NAV_LEFT, NAV_RIGHT, NAV_UP, Gamepad, pad_code, pad_label, pad_rows
from elan.gyro import TiltSteer
from elan.physics import Controls
from elan.settings import ACTIONS, CAMERA_MODES, DEFAULT_BINDINGS, DEFAULT_PAD

RESERVED_KEYS = (pygame.K_ESCAPE, pygame.K_F2, pygame.K_F3, pygame.K_F11, pygame.K_F12, pygame.K_m, pygame.K_BACKSPACE, pygame.K_DELETE)
WHEEL_BUTTONS = (4, 5)
SMOOTH_TAU = 0.006
STICK_TAU = 0.02

KEY_NAMES = {
	'left shift': 'L-SHIFT',
	'right shift': 'R-SHIFT',
	'left ctrl': 'L-CTRL',
	'right ctrl': 'R-CTRL',
	'left alt': 'L-ALT',
	'right alt': 'R-ALT',
	'left meta': 'L-META',
	'right meta': 'R-META',
	'left gui': 'L-WIN',
	'right gui': 'R-WIN',
	'return': 'ENTER',
	'keypad enter': 'KP ENTER',
	'caps lock': 'CAPS',
	'page up': 'PG UP',
	'page down': 'PG DN',
}

MOUSE_NAMES = {1: 'MOUSE 1', 2: 'MOUSE 3', 3: 'MOUSE 2'}

HINT_ACTIONS = (('thrust', 'thrust'), ('boost', 'boost'), ('tether', 'tether'), ('drift', 'drift'), ('brake', 'brake'), ('camera', 'camera'))


def parse_binding(value):
	if not value or ':' not in value:
		return None
	kind, _, code = value.partition(':')
	if not code.isdigit() or kind not in ('key', 'mouse'):
		return None
	return kind, int(code)


def key_binding(key):
	return 'key:%d' % key


def mouse_binding(button):
	return 'mouse:%d' % button


def pad_binding(code):
	return 'pad:%d' % code


def bindable_key(key):
	return key not in RESERVED_KEYS


def bindable_button(button):
	return button not in WHEEL_BUTTONS


def binding_name(value):
	if pad_code(value) is not None:
		return pad_label(value)
	parsed = parse_binding(value)
	if parsed is None:
		return '—'
	kind, code = parsed
	if kind == 'mouse':
		return MOUSE_NAMES.get(code, 'MOUSE %d' % (code - 2))
	name = pygame.key.name(code) if pygame.display.get_init() else ''
	if not name:
		return 'KEY %d' % code
	return KEY_NAMES.get(name, name.upper())


class MenuKeys:
	def __init__(self):
		self.up = False
		self.down = False
		self.left = False
		self.right = False
		self.confirm = False
		self.key_confirm = False
		self.click = False
		self.back = False
		self.pause = False
		self.escape = False
		self.restart = False
		self.camera = False
		self.autopilot = False
		self.perf = False
		self.fullscreen = False
		self.mute = False
		self.shot = False
		self.quit = False
		self.focus_lost = False


class MouseSmoother:
	def __init__(self):
		self.pending = [0.0, 0.0]

	def reset(self):
		self.pending = [0.0, 0.0]

	def step(self, dx, dy, dt, degree):
		p = self.pending
		p[0] += dx
		p[1] += dy
		if degree <= 0:
			out = (p[0], p[1])
			self.reset()
			return out
		k = 1.0 - math.exp(-max(dt, 0.0) / (SMOOTH_TAU * degree))
		ox, oy = p[0] * k, p[1] * k
		p[0] -= ox
		p[1] -= oy
		return ox, oy


class Bindings:
	def __init__(self, settings):
		self.settings = settings
		self.version = 0
		self.rebuild()

	def table(self):
		return self.settings.get('bindings')

	def rebuild(self):
		self.lookup = {}
		for action in ACTIONS:
			for value in self.table()[action]:
				parsed = parse_binding(value)
				if parsed:
					self.lookup.setdefault(action, []).append(parsed)
		self.version += 1

	def of(self, action):
		return self.lookup.get(action, ())

	def get(self, action, slot):
		return self.table()[action][slot]

	def owners(self, binding):
		return [(a, s) for a in ACTIONS for s in (0, 1) if self.table()[a][s] == binding]

	def rebind(self, action, slot, binding):
		table = self.table()
		cleared = []
		if binding:
			for a, s in self.owners(binding):
				if (a, s) == (action, slot):
					continue
				table[a][s] = ''
				cleared.append((a, s))
		table[action][slot] = binding or ''
		self.rebuild()
		return cleared

	def reset(self):
		self.settings.set('bindings', copy.deepcopy(DEFAULT_BINDINGS))
		self.rebuild()


class InputManager:
	NAV_DELAY = 380
	NAV_REPEAT = 90

	def __init__(self, settings):
		self.settings = settings
		self.bindings = Bindings(settings)
		self.smoother = MouseSmoother()
		self.gamepad = Gamepad()
		self.tilt = TiltSteer()
		self.stick = [0.0, 0.0]
		self.notice = None
		self.nav_held = None
		self.nav_next = 0
		self.captured = False
		self.tether_edge = False
		self.edges = set()
		self.mouse_held = set()
		self.hint_cache = (None, '')

	def pad_connected(self):
		return self.gamepad.connected()

	def capture(self, state):
		self.captured = state
		pygame.event.set_grab(state)
		pygame.mouse.set_visible(not state)
		pygame.mouse.get_rel()
		self.smoother.reset()
		self.tilt.reset()
		self.stick = [0.0, 0.0]

	def feed(self, events):
		menu = MenuKeys()
		self.notice = None
		self.tether_edge = False
		self.edges = set()
		for event in events:
			kind = event.type
			if kind == pygame.QUIT:
				menu.quit = True
			elif kind == pygame.KEYDOWN:
				self._key(event.key, menu)
				self.edges.add(('key', event.key))
			elif kind == pygame.MOUSEBUTTONDOWN:
				if event.button == 1:
					menu.confirm = True
					menu.click = True
				if bindable_button(event.button):
					self.mouse_held.add(event.button)
					self.edges.add(('mouse', event.button))
			elif kind == pygame.MOUSEBUTTONUP:
				self.mouse_held.discard(event.button)
			elif kind in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
				self.gamepad.open()
			elif kind == pygame.WINDOWFOCUSLOST:
				menu.focus_lost = True
				self.mouse_held.clear()
		self._pad_menu(menu)
		self.tether_edge = self.tether_edge or self.edge('tether') or self.pad_edge('tether')
		menu.restart = self.edge('restart') or self.pad_edge('restart')
		menu.camera = self.edge('camera') or self.pad_edge('camera')
		return menu

	def pad_table(self):
		return self.settings.get('pad_bindings')

	def pad_value(self, action):
		return self.gamepad.value(pad_code(self.pad_table().get(action)))

	def pad_edge(self, action):
		code = pad_code(self.pad_table().get(action))
		return code is not None and code in self.gamepad.edges

	def _pad_menu(self, menu):
		pad = self.gamepad
		pad.poll()
		if not pad.connected():
			return
		direction = None
		for name, codes in (('up', NAV_UP), ('down', NAV_DOWN), ('left', NAV_LEFT), ('right', NAV_RIGHT)):
			if codes[0] in pad.held or codes[1] in pad.held:
				direction = name
				break
		now = pygame.time.get_ticks()
		fire = False
		if direction != self.nav_held:
			self.nav_held = direction
			self.nav_next = now + self.NAV_DELAY
			fire = direction is not None
		elif direction is not None and now >= self.nav_next:
			self.nav_next = now + self.NAV_REPEAT
			fire = True
		if fire:
			setattr(menu, direction, True)
		if CONFIRM in pad.edges:
			menu.confirm = True
			menu.key_confirm = True
		if BACK in pad.edges:
			menu.back = True
		if self.pad_edge('pause'):
			menu.pause = True
			menu.back = True
		if self.pad_edge('gyro'):
			self.toggle_gyro()
		if self.pad_edge('gyro_reset') and self.settings.get('gyro_enabled'):
			self.tilt.reset()
			self.notice = 'GYRO CENTERED'

	def toggle_gyro(self):
		if not self.gamepad.motion.ready:
			self.notice = 'NO GYRO ON THIS GAMEPAD'
			return
		state = not self.settings.get('gyro_enabled')
		self.settings.set('gyro_enabled', state)
		self.settings.save()
		self.tilt.reset()
		self.notice = 'GYRO STEERING ON' if state else 'GYRO STEERING OFF'

	def edge(self, action):
		return any(b in self.edges for b in self.bindings.of(action))

	def _key(self, key, menu):
		if key == pygame.K_ESCAPE:
			menu.pause = True
			menu.back = True
			menu.escape = True
		elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
			menu.confirm = True
			menu.key_confirm = True
		elif key in (pygame.K_UP, pygame.K_w, pygame.K_z):
			menu.up = True
		elif key in (pygame.K_DOWN, pygame.K_s):
			menu.down = True
		elif key in (pygame.K_LEFT, pygame.K_a, pygame.K_q):
			menu.left = True
		elif key in (pygame.K_RIGHT, pygame.K_d):
			menu.right = True
		elif key == pygame.K_F2:
			menu.autopilot = True
		elif key == pygame.K_F3:
			menu.perf = True
		elif key == pygame.K_F11:
			menu.fullscreen = True
		elif key == pygame.K_m:
			menu.mute = True
		elif key == pygame.K_F12:
			menu.shot = True

	def held(self, action, keys):
		for kind, code in self.bindings.of(action):
			if kind == 'mouse':
				if self.captured and code in self.mouse_held:
					return True
				continue
			if self.key_down(keys, code):
				return True
		return False

	def key_down(self, keys, code):
		try:
			return bool(keys[code])
		except (IndexError, KeyError, TypeError):
			return False

	def sample(self, dt):
		controls = Controls()
		keys = pygame.key.get_pressed()
		sensitivity = FlightCfg.MOUSE_SENS * self.settings.get('mouse_sensitivity')
		flip = -1.0 if self.settings.get('invert_y') else 1.0
		if self.captured:
			mx, my = pygame.mouse.get_rel()
			mx, my = self.smoother.step(mx, my, dt, self.settings.get('mouse_smoothing'))
			controls.look_x = mx * sensitivity
			controls.look_y = my * sensitivity * flip
		controls.thrust = 1.0 if self.held('thrust', keys) else 0.0
		controls.brake = 1.0 if self.held('brake', keys) else 0.0
		controls.strafe = (1.0 if self.held('strafe_right', keys) else 0.0) - (1.0 if self.held('strafe_left', keys) else 0.0)
		controls.boost = self.held('boost', keys)
		controls.tether = self.held('tether', keys)
		controls.drift = self.held('drift', keys)
		controls.tether_pressed = self.tether_edge
		pad = self.gamepad
		if not pad.connected():
			return controls
		look_x, look_y = pad.stick(self.settings.get('pad_look'))
		degree = self.settings.get('pad_smoothing')
		if degree > 0:
			k = 1.0 - math.exp(-max(dt, 0.0) / (STICK_TAU * degree))
			self.stick[0] += (look_x - self.stick[0]) * k
			self.stick[1] += (look_y - self.stick[1]) * k
			look_x, look_y = self.stick
		rate = FlightCfg.PAD_LOOK * self.settings.get('pad_sensitivity') * dt
		controls.look_x += look_x * abs(look_x) * rate
		controls.look_y += look_y * abs(look_y) * rate * (-1.0 if self.settings.get('pad_invert_y') else 1.0)
		rates = pad.motion.rates() if self.settings.get('gyro_enabled') else None
		if rates is not None:
			turn, lift = self.tilt.step(rates, dt, self.settings.get('gyro_smoothing'), self.settings.get('gyro_axis'))
			rate = FlightCfg.PAD_LOOK * self.settings.get('gyro_sensitivity') * dt
			controls.look_x += turn * (0.35 + 0.65 * abs(turn)) * rate
			controls.look_y -= lift * (0.35 + 0.65 * abs(lift)) * rate * (-1.0 if self.settings.get('gyro_invert_y') else 1.0)
		controls.strafe = max(-1.0, min(1.0, controls.strafe + self.pad_value('strafe_right') - self.pad_value('strafe_left')))
		controls.thrust = max(controls.thrust, self.pad_value('thrust'))
		controls.brake = max(controls.brake, self.pad_value('brake'))
		controls.boost = controls.boost or self.pad_value('boost') > 0.5
		controls.tether = controls.tether or self.pad_value('tether') > 0.5
		controls.drift = controls.drift or self.pad_value('drift') > 0.5
		return controls

	def binding_label(self, action, slot):
		if slot == 2:
			return pad_label(self.pad_table().get(action))
		if action not in ACTIONS:
			return 'ESC' if action == 'pause' and slot == 0 else '—'
		return binding_name(self.bindings.get(action, slot))

	def pad_rows(self):
		return pad_rows(self.settings.get('pad_look'), self.pad_table())

	def rebind_pad(self, action, binding):
		table = self.pad_table()
		cleared = []
		if binding:
			for other in table:
				if other != action and table[other] == binding:
					table[other] = ''
					cleared.append((other, 2))
		table[action] = binding or ''
		self.bindings.version += 1
		return cleared

	def swap_sticks(self):
		table = self.pad_table()
		for action in table:
			code = pad_code(table[action])
			if code is not None and 100 <= code <= 107:
				table[action] = pad_binding(code + 4 if code < 104 else code - 4)
		self.bindings.version += 1

	def action_label(self, action, joiner=' / '):
		names = [self.binding_label(action, s) for s in (0, 1) if self.bindings.get(action, s)]
		return joiner.join(names) if names else '—'

	def primary_label(self, action):
		for s in (0, 1):
			if self.bindings.get(action, s):
				return self.binding_label(action, s)
		return '—'

	def hint_text(self):
		if self.hint_cache[0] == self.bindings.version:
			return self.hint_cache[1]
		parts = ['MOUSE steer']
		for action, word in HINT_ACTIONS:
			label = self.primary_label(action)
			if label != '—':
				parts.append('%s %s' % (label, word))
		text = '   '.join(parts)
		self.hint_cache = (self.bindings.version, text)
		return text

	def rebind(self, action, slot, binding):
		if slot == 2:
			return self.rebind_pad(action, binding)
		return self.bindings.rebind(action, slot, binding)

	def reset_bindings(self):
		self.bindings.reset()
		self.settings.set('pad_bindings', dict(DEFAULT_PAD))
		self.settings.set('pad_look', 'right')

	def binding_version(self):
		return self.bindings.version

	def camera_mode(self):
		return self.settings.get('camera')

	def toggle_camera(self):
		mode = CAMERA_MODES[(CAMERA_MODES.index(self.camera_mode()) + 1) % len(CAMERA_MODES)]
		self.settings.set('camera', mode)
		return mode
