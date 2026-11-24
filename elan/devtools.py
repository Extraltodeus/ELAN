import pygame

from elan.world.palette import PaletteBook


class DevTools:
	KEYS = (
		(pygame.K_F1, 'legend', 'F1    hide / show this panel'),
		(pygame.K_F4, 'dry', 'F4    flux to zero'),
		(pygame.K_F5, 'refill', 'F5    flux to full'),
		(pygame.K_F6, 'gate', 'F6    open the gate'),
		(pygame.K_F7, 'warp', 'F7    warp tunnel now'),
		(pygame.K_F8, 'hyper', 'F8    hyper tunnel now'),
		(pygame.K_F9, 'exit', 'F9    skip to the tunnel exit'),
		(pygame.K_PAGEDOWN, 'next', 'PGDN  next sector'),
		(pygame.K_PAGEUP, 'previous', 'PGUP  previous sector'),
	)
	EXIT = 0.72

	def __init__(self, enabled):
		self.enabled = bool(enabled)
		self.legend = True
		self.book = PaletteBook()
		self.table = {key: name for key, name, _ in self.KEYS}
		self.text = ['DEVELOPER MODE  -  NO RECORDS'] + [label for _, _, label in self.KEYS]

	def lines(self):
		if not self.enabled or not self.legend:
			return None
		return self.text

	def commands(self, events):
		out = []
		for event in events:
			if event.type != pygame.KEYDOWN:
				continue
			name = self.table.get(event.key)
			if name == 'legend':
				self.legend = not self.legend
			elif name:
				out.append(name)
		return out

	def run(self, app):
		if not self.enabled:
			return
		for name in self.commands(app.events):
			getattr(self, '_' + name)(app, app.game)

	def _dry(self, app, game):
		if game.phase != 'sector':
			game.say('DEV  -  NO DRY FLUX IN A TUNNEL', 'info', 1.6)
			return
		game.flux = 0.0
		game.say('DEV  -  FLUX ZERO', 'info', 1.6)

	def _refill(self, app, game):
		game.flux = 1.0
		game.flux_gain = 1.0
		game.say('DEV  -  FLUX FULL', 'info', 1.6)

	def _gate(self, app, game):
		if game.phase != 'sector' or game.gate_open:
			return
		game.quota = max(game.quota, game.quota_need)
		game.open_gate()

	def _warp(self, app, game):
		if game.phase != 'sector':
			return
		game.warp_requested = True

	def _hyper(self, app, game):
		if game.phase != 'sector':
			return
		sector = game.sector
		while not self.book.hyper(sector):
			sector += 1
		game.sector = sector
		game.warp_requested = True

	def _exit(self, app, game):
		if game.phase != 'tunnel':
			game.say('DEV  -  ONLY IN A TUNNEL', 'info', 1.6)
			return
		game.travel = max(game.travel, game.travel_need * self.EXIT)

	def _next(self, app, game):
		app.jump(game.sector + 1)

	def _previous(self, app, game):
		app.jump(game.sector - 1)
