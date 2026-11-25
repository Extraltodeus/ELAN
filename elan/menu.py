import pygame

from elan.input import RESERVED_KEYS, binding_name, bindable_button, key_binding, mouse_binding, pad_binding
from elan.settings import ACTIONS, GYRO_AXES, MINIMAP_MODES, PAD_ACTIONS, PAD_LOOKS
from elan.ui import ACTION_NAMES, OPTIONS_FOOTER, ControlsView

SCALE_CHOICES = ('auto', 0.5, 0.67, 0.75, 0.83, 1.0)
BACK_GUARD = 0.15
CLEAR_KEYS = (pygame.K_BACKSPACE, pygame.K_DELETE)
NAMES = dict(ACTION_NAMES)
SLOT_NAMES = ('PRIMARY', 'SECONDARY', 'GAMEPAD')
CAPTURE_TIMEOUT = 7000


def sign(v):
	return (v > 0) - (v < 0)


class Option:
	WHEEL = 1

	def __init__(self, label):
		self.label = label

	def value(self):
		return None

	def step(self, d):
		return None

	def confirm(self):
		return None


class ActionOption(Option):
	def __init__(self, label, result):
		Option.__init__(self, label)
		self.result = result

	def confirm(self):
		return self.result


class RangeOption(Option):
	def __init__(self, label, settings, key, lo, hi, inc, fmt):
		Option.__init__(self, label)
		self.settings = settings
		self.key = key
		self.lo = lo
		self.hi = hi
		self.inc = inc
		self.fmt = fmt

	def value(self):
		return self.fmt(self.settings.get(self.key))

	def step(self, d):
		old = self.settings.get(self.key)
		new = round(max(self.lo, min(self.hi, old + self.inc * d)), 2)
		if isinstance(self.inc, int):
			new = int(new)
		if new == old:
			return None
		self.settings.set(self.key, new)
		return 'changed'


class CycleOption(Option):
	def __init__(self, label, settings, key, choices, names):
		Option.__init__(self, label)
		self.settings = settings
		self.key = key
		self.choices = choices
		self.names = names

	def index(self):
		current = self.settings.get(self.key)
		return self.choices.index(current) if current in self.choices else 0

	def value(self):
		return self.names[self.index()]

	def step(self, d):
		self.settings.set(self.key, self.choices[(self.index() + d) % len(self.choices)])
		return 'changed'

	def confirm(self):
		return self.step(1)


class StickOption(CycleOption):
	def __init__(self, label, settings, input_manager):
		CycleOption.__init__(self, label, settings, 'pad_look', PAD_LOOKS, ('RIGHT', 'LEFT'))
		self.input = input_manager

	def step(self, d):
		CycleOption.step(self, d)
		self.input.swap_sticks()
		return 'changed'


class FullscreenOption(Option):
	def __init__(self, menu):
		Option.__init__(self, 'FULLSCREEN')
		self.menu = menu

	def value(self):
		return 'ON' if self.menu.fullscreen else 'OFF'

	def step(self, d):
		return 'fullscreen'

	def confirm(self):
		return 'fullscreen'


class ScreenOption(Option):
	def __init__(self, settings):
		Option.__init__(self, 'SCREEN')
		self.settings = settings

	def count(self):
		try:
			return max(pygame.display.get_num_displays(), 1)
		except pygame.error:
			return 1

	def current(self):
		return min(self.settings.get('screen'), self.count() - 1)

	def value(self):
		return '%d / %d' % (self.current() + 1, self.count())

	def step(self, d):
		count = self.count()
		if count < 2:
			return None
		self.settings.set('screen', (self.current() + d) % count)
		return 'screen'

	def confirm(self):
		return self.step(1)


class LevelOption(Option):
	WHEEL = -1

	def __init__(self, menu):
		Option.__init__(self, 'JUMP TO')
		self.menu = menu

	def value(self):
		target = self.menu.target
		return target.label() if target is not None else '-'

	def step(self, d):
		target = self.menu.target
		if target is None or not target.move(d):
			return None
		return 'level'

	def confirm(self):
		return 'jump' if self.menu.target is not None else None


class PageOption(Option):
	def __init__(self, label, menu, page):
		Option.__init__(self, label)
		self.menu = menu
		self.page = page

	def confirm(self):
		self.menu.show(self.page)
		return None


def percent(v):
	return '%d%%' % round(float(v) * 100)


def scale_name(v):
	return 'AUTO' if v == 'auto' else percent(v)


class HoverTracker:
	def __init__(self):
		self.hits = []
		self.pos = None

	def at(self, pos):
		if pos is None:
			return None
		for hit in self.hits:
			if hit[0].collidepoint(pos):
				return hit
		return None

	def current(self):
		return self.at(self.pos)


class MainPage:
	TITLE = 'PAUSED'
	PARENT = None
	FOOTER = None

	def __init__(self, menu, settings):
		self.menu = menu
		self.index = 0
		self.hover = HoverTracker()
		self.options = self.build(menu, settings)

	def note(self):
		return None

	def build(self, menu, settings):
		return [ActionOption('RESUME', 'resume'), ActionOption('RESTART RUN', 'restart')] + self.jumps(menu) + self.tuning(menu, settings) + [ActionOption('MAIN MENU', 'title'), ActionOption('QUIT TO DESKTOP', 'quit')]

	def jumps(self, menu):
		return [LevelOption(menu)] if menu.dev else []

	def tuning(self, menu, settings):
		return [
			PageOption('CONTROLS…', menu, 'controls'),
			PageOption('GAMEPAD…', menu, 'gamepad'),
			RangeOption('MOUSE SENSITIVITY', settings, 'mouse_sensitivity', 0.2, 3.0, 0.1, lambda v: '%.1f' % v),
			RangeOption('MOUSE SMOOTHING', settings, 'mouse_smoothing', 0, 10, 1, lambda v: 'OFF' if not v else str(int(v))),
			CycleOption('MOUSE INVERT Y', settings, 'invert_y', (False, True), ('OFF', 'ON')),
			CycleOption('CAMERA', settings, 'camera', ('chase', 'cockpit'), ('CHASE', 'COCKPIT')),
			RangeOption('SCREEN SHAKE', settings, 'screen_shake', 0.0, 2.0, 0.1, percent),
			CycleOption('WARP EFFECT', settings, 'warp_effect', (True, False), ('ON', 'OFF')),
			CycleOption('AI PILOT TYPE', settings, 'ai_pilot', (True, False), ('LEARNED', 'CLASSIC')),
			CycleOption('MINIMAP', settings, 'minimap', MINIMAP_MODES, ('BOTTOM RIGHT', 'BOTTOM LEFT', 'TOP RIGHT', 'TOP CENTER', 'OFF')),
			CycleOption('RENDER SCALE', settings, 'render_scale', SCALE_CHOICES, tuple(scale_name(c) for c in SCALE_CHOICES)),
			RangeOption('MASTER VOLUME', settings, 'master_volume', 0.0, 1.0, 0.05, percent),
			RangeOption('MUSIC VOLUME', settings, 'music_volume', 0.0, 1.0, 0.05, percent),
			RangeOption('EFFECTS VOLUME', settings, 'sfx_volume', 0.0, 1.0, 0.05, percent),
			RangeOption('PICKUP VOLUME', settings, 'pickup_volume', 0.0, 1.0, 0.05, percent),
			CycleOption('MUSIC', settings, 'music_mode', ('dj', 'linear', 'synth'), ('DJ STEMS', 'LINEAR', 'SYNTH')),
			FullscreenOption(menu),
			ScreenOption(settings),
		]

	def reset(self):
		self.index = 0

	def items(self):
		return [(o.label, o.value()) for o in self.options]

	def select(self, index):
		if index == self.index:
			return
		self.index = index
		self.menu.sound = 'move'

	def activate(self):
		self.menu.sound = 'select'
		return self.options[self.index].confirm()

	def change(self, d):
		result = self.options[self.index].step(d)
		if result:
			self.menu.sound = 'move'
		return result

	def mouse(self, events):
		result = None
		for event in events:
			kind = event.type
			if kind == pygame.MOUSEMOTION:
				self.hover.pos = event.pos
				hit = self.hover.current()
				if hit:
					self.select(hit[1])
			elif kind == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
				self.hover.pos = event.pos
				hit = self.hover.current()
				if not hit:
					continue
				self.select(hit[1])
				result = self.activate() if event.button == 1 else self.change(-1)
			elif kind == pygame.MOUSEWHEEL and event.y:
				hit = self.hover.current()
				if hit and hit[1] == self.index:
					result = self.change(sign(event.y) * self.options[self.index].WHEEL)
			if result:
				return result
		return result

	def update(self, keys, events):
		count = len(self.options)
		if keys.up:
			self.select((self.index - 1) % count)
		if keys.down:
			self.select((self.index + 1) % count)
		step = (1 if keys.right else 0) - (1 if keys.left else 0)
		if step:
			result = self.change(step)
			if result:
				return result
		if keys.back and self.menu.age > BACK_GUARD:
			self.menu.sound = 'select'
			if self.PARENT is None:
				return 'resume'
			self.menu.show(self.PARENT)
			return None
		result = self.mouse(events)
		if result or self.menu.page is not self:
			return result
		if getattr(keys, 'key_confirm', keys.confirm and not getattr(keys, 'click', False)):
			return self.activate()
		return None

	def draw(self, surface, t):
		screens = self.menu.screens
		self.hover.hits = screens.draw_pause(surface, t, self.items(), self.index, self.TITLE, self.FOOTER, self.note()) or []


class OptionsPage(MainPage):
	TITLE = 'OPTIONS'
	FOOTER = OPTIONS_FOOTER

	def build(self, menu, settings):
		return self.jumps(menu) + self.tuning(menu, settings) + [ActionOption('BACK', 'resume')]


class GamepadPage(MainPage):
	TITLE = 'GAMEPAD'
	PARENT = 'main'

	def note(self):
		return self.menu.input.gamepad.status()

	def build(self, menu, settings):
		degree = lambda v: 'OFF' if not v else str(int(v))
		gain = lambda v: '%.1f' % v
		switch = ((False, True), ('OFF', 'ON'))
		return [
			RangeOption('STICK SENSITIVITY', settings, 'pad_sensitivity', 0.3, 3.0, 0.1, gain),
			RangeOption('STICK SMOOTHING', settings, 'pad_smoothing', 0, 10, 1, degree),
			CycleOption('STICK INVERT Y', settings, 'pad_invert_y', *switch),
			StickOption('STEERING STICK', settings, menu.input),
			CycleOption('GYRO STEERING', settings, 'gyro_enabled', *switch),
			RangeOption('GYRO SENSITIVITY', settings, 'gyro_sensitivity', 0.3, 3.0, 0.1, gain),
			RangeOption('GYRO SMOOTHING', settings, 'gyro_smoothing', 0, 10, 1, degree),
			CycleOption('GYRO INVERT Y', settings, 'gyro_invert_y', *switch),
			CycleOption('GYRO TURN AXIS', settings, 'gyro_axis', GYRO_AXES, ('YAW (POINT)', 'ROLL (WHEEL)')),
			PageOption('BACK', menu, 'main'),
		]


class BindingCapture:
	def __init__(self, page):
		self.page = page
		self.active = False
		self.started = 0

	def start(self):
		self.active = True
		self.started = pygame.time.get_ticks()

	def stop(self):
		self.active = False

	def feed(self, events):
		page = self.page
		pad = page.slot == 2
		for event in events:
			kind = event.type
			if kind == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					return page.cancel_capture()
				if event.key in CLEAR_KEYS:
					return page.assign('')
				if pad:
					continue
				if event.key in RESERVED_KEYS:
					page.notify('%s IS RESERVED' % binding_name(key_binding(event.key)), True)
					continue
				return page.assign(key_binding(event.key))
			if not pad and kind == pygame.MOUSEBUTTONDOWN and bindable_button(event.button):
				return page.assign(mouse_binding(event.button))
		edges = page.input.gamepad.edges
		if pad and edges:
			return page.assign(pad_binding(min(edges)))
		if pygame.time.get_ticks() - self.started > CAPTURE_TIMEOUT:
			return page.cancel_capture()
		return None


class ControlsPage:
	EXTRAS = ('RESET TO DEFAULTS', 'BACK')

	def __init__(self, menu, input_manager):
		self.menu = menu
		self.input = input_manager
		self.capture = BindingCapture(self)
		self.hover = HoverTracker()
		self.view = ControlsView()
		self.reset()

	def reset(self):
		self.row = 0
		self.slot = 0
		self.status = ''
		self.warn = False
		self.capture.stop()

	def count(self):
		return len(PAD_ACTIONS) + len(self.EXTRAS)

	def action(self):
		return PAD_ACTIONS[self.row] if self.row < len(PAD_ACTIONS) else None

	def notify(self, text, warn=False):
		self.status = text
		self.warn = warn

	def select(self, row, slot=None):
		if row == self.row and (slot is None or slot == self.slot):
			return
		self.row = row
		if slot is not None:
			self.slot = slot
		self.menu.sound = 'move'

	def begin_capture(self):
		if self.action() not in ACTIONS and self.slot < 2:
			self.notify('ESC IS FIXED ON THE KEYBOARD' if self.action() == 'pause' else 'GAMEPAD ONLY', True)
			self.menu.sound = 'move'
			return
		self.capture.start()
		what = 'A GAMEPAD BUTTON, TRIGGER OR STICK' if self.slot == 2 else 'A KEY OR MOUSE BUTTON'
		self.notify('PRESS %s FOR %s (%s)…' % (what, NAMES[self.action()], SLOT_NAMES[self.slot]))
		self.menu.sound = 'select'

	def cancel_capture(self):
		self.capture.stop()
		self.notify('')
		self.menu.sound = 'move'
		return None

	def assign(self, binding):
		action = self.action()
		self.capture.stop()
		cleared = self.input.rebind(action, self.slot, binding)
		self.menu.sound = 'select'
		if cleared:
			a, s = cleared[0]
			self.notify('%s REMOVED FROM %s (%s)' % (binding_name(binding), NAMES[a], SLOT_NAMES[s]), True)
		elif binding:
			self.notify('%s BOUND TO %s' % (binding_name(binding), NAMES[action]))
		else:
			self.notify('%s %s CLEARED' % (NAMES[action], SLOT_NAMES[self.slot]))
		self.menu.save()
		return 'changed'

	def activate(self):
		action = self.action()
		if action:
			self.begin_capture()
			return None
		self.menu.sound = 'select'
		if self.row == len(PAD_ACTIONS):
			self.input.reset_bindings()
			self.notify('DEFAULT CONTROLS RESTORED')
			self.menu.save()
			return 'changed'
		self.menu.show('main')
		return None

	def clear_requested(self, events):
		return any(e.type == pygame.KEYDOWN and e.key in CLEAR_KEYS for e in events)

	def mouse(self, events):
		for event in events:
			kind = event.type
			if kind == pygame.MOUSEMOTION:
				self.hover.pos = event.pos
				hit = self.hover.current()
				if hit:
					self.select(hit[1], hit[2])
			elif kind == pygame.MOUSEBUTTONDOWN and event.button == 1:
				self.hover.pos = event.pos
				hit = self.hover.current()
				if not hit:
					continue
				self.select(hit[1], hit[2])
				if hit[2] is not None or self.action() is None:
					return True, self.activate()
		return False, None

	def update(self, keys, events):
		if self.capture.active:
			return self.capture.feed(events)
		count = self.count()
		if keys.up:
			self.select((self.row - 1) % count)
		if keys.down:
			self.select((self.row + 1) % count)
		step = (1 if keys.right else 0) - (1 if keys.left else 0)
		if step and self.action():
			self.select(self.row, (self.slot + step) % 3)
		if keys.back and self.menu.age > BACK_GUARD:
			self.menu.sound = 'select'
			self.menu.show('main')
			return None
		if self.action() and self.clear_requested(events):
			return self.assign('')
		done, result = self.mouse(events)
		if done:
			return result
		if getattr(keys, 'key_confirm', keys.confirm and not getattr(keys, 'click', False)):
			return self.activate()
		return None

	def build_view(self):
		v = self.view
		v.rows = [(NAMES[a], [self.input.binding_label(a, s) for s in (0, 1, 2)]) for a in PAD_ACTIONS]
		v.extras = list(self.EXTRAS)
		v.selected = self.row
		v.slot = self.slot
		v.capturing = self.capture.active
		v.status = self.status or self.input.gamepad.status()
		v.warn = self.warn
		return v

	def draw(self, surface, t):
		self.hover.hits = self.menu.screens.draw_controls(surface, t, self.build_view()) or []


class PauseMenu:
	EXITS = ('resume', 'restart', 'title', 'quit')

	def __init__(self, settings, input_manager, screens, dev=False):
		self.dev = bool(dev)
		self.target = None
		self.settings = settings
		self.input = input_manager
		self.screens = screens
		self.fullscreen = False
		self.sound = None
		self.age = 0.0
		self.main = MainPage(self, settings)
		self.options = OptionsPage(self, settings)
		self.controls = ControlsPage(self, input_manager)
		self.gamepad = GamepadPage(self, settings)
		self.pages = {'main': self.main, 'controls': self.controls, 'gamepad': self.gamepad}
		self.page = self.main
		screens.set_bindings(input_manager)

	def open(self, standalone=False):
		root = self.options if standalone else self.main
		self.age = 0.0
		self.sound = None
		root.reset()
		self.gamepad.reset()
		self.controls.reset()
		self.pages['main'] = root
		self.page = root

	def show(self, name):
		page = self.pages[name]
		if page is self.controls:
			self.controls.reset()
		self.page = page
		self.age = 0.0

	def capturing(self):
		return self.page is self.controls and self.controls.capture.active

	def save(self):
		self.settings.save()

	def update(self, keys, events, dt):
		self.sound = None
		self.age += dt
		result = self.page.update(keys, events or [])
		if result in self.EXITS:
			self.save()
		return result

	def draw(self, surface, t):
		self.page.draw(surface, t)
