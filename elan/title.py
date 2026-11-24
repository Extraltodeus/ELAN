import pygame


class TitleView:
	def __init__(self):
		self.rows = []
		self.focus = 0
		self.blurb = ''
		self.records = []
		self.tail = None
		self.links = []


class TitleMenu:
	PILOTS = ('FLY', 'AUTO-PLAY')
	LINKS = (('OPTIONS', 'options'), ('SCORES', 'scores'), ('CREDITS', 'credits'))
	MODE_ROW = 0
	PILOT_ROW = 1
	LINK_ROW = 2
	ROWS = LINK_ROW + len(LINKS)

	def __init__(self, settings, modes, mode, autoplay):
		self.settings = settings
		self.modes = modes
		self.mode = modes.index(mode)
		self.pilot = 1 if autoplay else 0
		self.row = self.PILOT_ROW
		self.hits = []
		self.hover = None
		self.sound = None

	def current(self):
		return self.modes[self.mode]

	def autoplay(self):
		return self.pilot == 1

	def _link(self, row):
		return self.LINKS[row - self.LINK_ROW][1] if row >= self.LINK_ROW else None

	def _count(self, row):
		if row >= self.LINK_ROW:
			return 1
		return len(self.modes) if row == self.MODE_ROW else len(self.PILOTS)

	def _value(self, row):
		if row >= self.LINK_ROW:
			return 0
		return self.mode if row == self.MODE_ROW else self.pilot

	def _pick(self, row, index):
		if index == self._value(row):
			return False
		self.sound = 'move'
		if row == self.PILOT_ROW:
			self.pilot = index
			return True
		self.mode = index
		self.settings.set('game_mode', self.current().KEY)
		self.settings.save()
		return True

	def _focus(self, row):
		if row == self.row:
			return
		self.row = row
		self.sound = 'move'

	def _at(self, pos):
		for rect, row, index in self.hits:
			if rect.collidepoint(pos):
				return row, index
		return None

	def _mouse(self, events):
		clicked = None
		for event in events:
			kind = event.type
			if kind == pygame.MOUSEMOTION:
				self.hover = self._at(event.pos)
				if self.hover is None:
					continue
				self._focus(self.hover[0])
				if self.hover[0] == self.PILOT_ROW:
					self._pick(*self.hover)
			elif kind == pygame.MOUSEBUTTONDOWN and event.button == 1:
				self.hover = self._at(event.pos)
				clicked = self.hover
			elif kind == pygame.MOUSEWHEEL and event.y and self.hover is not None and self.hover[0] == self.row:
				step = 1 if event.y < 0 else -1
				self._pick(self.row, (self._value(self.row) + step) % self._count(self.row))
		return clicked

	def update(self, menu, events, armed):
		self.sound = None
		clicked = self._mouse(events)
		if menu.up:
			self._focus((self.row - 1) % self.ROWS)
		if menu.down:
			self._focus((self.row + 1) % self.ROWS)
		step = (1 if menu.right else 0) - (1 if menu.left else 0)
		if step:
			self._pick(self.row, (self._value(self.row) + step) % self._count(self.row))
		if not menu.confirm or not armed:
			return None
		if menu.click:
			if clicked is None:
				return 'start'
			self._focus(clicked[0])
			self._pick(*clicked)
			if clicked[0] == self.MODE_ROW:
				return None
			return self._link(clicked[0]) or 'start'
		return self._link(self.row) or 'start'

	def view(self, career, clock):
		mode = self.current()
		settings = self.settings
		out = TitleView()
		out.rows = [('MODE', tuple(m.NAME for m in self.modes), self.mode), ('PILOT', self.PILOTS, self.pilot)]
		out.focus = self.row
		out.blurb = mode.BLURB
		out.links = [(label, self.LINK_ROW + k, self.row == self.LINK_ROW + k) for k, (label, _) in enumerate(self.LINKS)]
		best = int(settings.get(mode.SCORE_KEY))
		if best:
			out.records.append(('BEST', '{:,}'.format(best)))
		if mode.TIME_KEY and float(settings.get(mode.TIME_KEY)) > 0.0:
			out.records.append(('BEST TIME', clock(float(settings.get(mode.TIME_KEY)))))
		out.tail = career.line(mode)
		return out
