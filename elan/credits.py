import pygame

STUDIO = 'Claude'

CREDITS = (
	('title', 'ÉLAN'),
	('tag', 'momentum is everything'),
	('gap', 3),
	('lead', 'DIRECTION', 'Extraltodeus'),
	('gap', 2),
	('head', 'DESIGN'),
	('role', 'GAME DESIGN', STUDIO),
	('role', 'LEVEL DESIGN', STUDIO),
	('role', 'SYSTEMS AND BALANCE', STUDIO),
	('role', 'NARRATIVE', STUDIO),
	('role', 'USER EXPERIENCE', STUDIO),
	('gap', 1),
	('head', 'PROGRAMMING'),
	('role', 'LEAD PROGRAMMING', STUDIO),
	('role', 'ENGINE', STUDIO),
	('role', 'RENDERING', STUDIO),
	('role', 'FLIGHT PHYSICS', STUDIO),
	('role', 'GAMEPLAY', STUDIO),
	('role', 'WORLD GENERATION', STUDIO),
	('role', 'AUDIO PROGRAMMING', STUDIO),
	('role', 'TOOLS', STUDIO),
	('gap', 1),
	('head', 'ART'),
	('role', 'ART DIRECTION', STUDIO),
	('role', 'ENVIRONMENT ART', STUDIO),
	('role', 'VISUAL EFFECTS', STUDIO),
	('role', 'LIGHTING', STUDIO),
	('role', 'USER INTERFACE', STUDIO),
	('role', 'TECHNICAL ART', STUDIO),
	('gap', 1),
	('head', 'AUDIO'),
	('role', 'MUSIC', STUDIO),
	('role', 'SOUND DESIGN', STUDIO),
	('role', 'MIXING', STUDIO),
	('gap', 1),
	('head', 'ARTIFICIAL PILOT'),
	('role', 'FLIGHT INSTRUCTOR', STUDIO),
	('role', 'STUDENT', STUDIO),
	('gap', 1),
	('head', 'PRODUCTION'),
	('role', 'PRODUCTION', STUDIO),
	('role', 'QUALITY ASSURANCE', STUDIO),
	('role', 'PLAYTESTING', STUDIO),
	('role', 'CATERING', STUDIO),
	('gap', 2),
	('head', 'MADE WITH'),
	('line', 'PyTorch  ·  Pygame  ·  YuE  ·  Demucs  ·  ComfyUI'),
	('gap', 2),
	('head', 'SPECIAL THANKS'),
	('line', 'You, for flying'),
	('gap', 3),
	('line', 'No GPUs were harmed. Two were kept very warm.'),
	('gap', 3),
	('tag', 'WELCOME TO THE SINGULARITY'),
)


class CreditsView:
	def __init__(self, entries, offset):
		self.entries = entries
		self.offset = offset


class CreditsRoll:
	SPEED = 54.0
	NUDGE = 110.0
	HOLD = 1.4
	START = 380.0
	EASE = 90.0
	CREEP = 0.2

	def __init__(self):
		self.offset = self.START
		self.limit = 0.0
		self.hold = 0.0

	def open(self):
		self.offset = self.START
		self.hold = 0.0

	def _clamp(self, offset):
		offset = max(offset, 0.0)
		return min(offset, self.limit) if self.limit > 0.0 else offset

	def _pace(self):
		if self.limit <= 0.0:
			return self.SPEED
		return self.SPEED * min(max((self.limit - self.offset) / self.EASE, self.CREEP), 1.0)

	def _nudge(self, amount):
		self.offset = self._clamp(self.offset + amount)
		self.hold = self.HOLD

	def update(self, dt, menu, events):
		if menu.back or menu.escape:
			return 'back'
		for event in events:
			if event.type == pygame.MOUSEWHEEL and event.y:
				self._nudge(-event.y * self.NUDGE)
		if menu.up:
			self._nudge(-self.NUDGE)
		if menu.down:
			self._nudge(self.NUDGE)
		self.hold = max(self.hold - dt, 0.0)
		if self.hold <= 0.0:
			self.offset += self._pace() * dt
		self.offset = self._clamp(self.offset)
		return None

	def view(self):
		return CreditsView(CREDITS, self.offset)
