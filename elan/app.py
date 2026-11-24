import os
import random
import time
import pygame
import torch

from elan.config import AppCfg, RenderCfg, StoryCfg
from elan.settings import Settings
from elan.career import Career
from elan.display import Display, FramePacer, ResolutionGovernor
from elan.input import InputManager
from elan.menu import PauseMenu
from elan.physics import Controls
from elan.autopilot import Autopilot
from elan.devtools import DevTools
from elan.game import Game
from elan.modes import MODES, mode_of
from elan.scores import ScoreBoard
from elan.title import TitleMenu
from elan.credits import CreditsRoll
from elan.render import Renderer
from elan.world import BackgroundBaker
from elan.audio import AudioDirector
from elan.ui import Hud, Screens
from elan.ui.hud import fmt_clock


class App:
	MENUS = ('title', 'credits', 'options', 'scores')

	def __init__(self, options):
		torch.set_grad_enabled(False)
		self.options = options
		self.root = options.root
		self.settings = Settings(self.root)
		self.career = Career(self.settings)
		self.board = ScoreBoard(self.root, self.settings)
		self.assisted = False
		self.piloted = bool(options.autoplay)
		self.banked = False
		self.render_device, self.bake_device = self._devices(options)
		fullscreen = self.settings.get('fullscreen') and not options.windowed
		self.display = Display(fullscreen, options.window, options.vsync)
		pygame.font.init()
		self._icon()
		self.screens = Screens(self.display.size)
		splash = os.path.join(self.root, 'assets', 'splash.png')
		if os.path.isfile(splash):
			self.screens.set_loading_backdrop(splash)
		self.hud = Hud(self.display.size)
		self.input = InputManager(self.settings)
		self.pause_menu = PauseMenu(self.settings, self.input, self.screens)
		self.hud.set_bindings(self.input)
		self.screens.set_bindings(self.input)
		self.events = []
		self.audio = None
		self.renderer = None
		self.game = None
		self.world = None
		self.autopilot = Autopilot()
		self.pilot = None
		self.dev = DevTools(getattr(options, 'developer', False))
		self.preview = bool(getattr(options, 'finale', False))
		self.mode = mode_of('story' if self.preview else (getattr(options, 'mode', None) or self.settings.get('game_mode')))
		self.title = TitleMenu(self.settings, MODES, self.mode, options.autoplay)
		self.credits = CreditsRoll()
		self.crowned = None
		self.first = max(int(options.sector) - 1, 0)
		self.baker = BackgroundBaker(self.bake_device, self.render_device, 0.0)
		self.seed = options.seed if options.seed is not None else random.randrange(1, 1000000)
		cap = options.fps if options.fps > 0 else (0 if self.display.vsync else self.display.refresh)
		self.pacer = FramePacer(self.display, cap)
		self.governor = None
		self.state = 'loading'
		self.closed = False
		self.state_time = 0.0
		self.running = True
		self.clock = 0.0
		self.frames = 0
		self.show_perf = options.perf
		self.perf_lines = None
		self.perf_timer = 0.0
		self.fps = 0.0
		self.summary = None
		self.shot_index = 0
		self.frame_times = []

	def _devices(self, options):
		if not torch.cuda.is_available():
			return torch.device('cpu'), torch.device('cpu')
		count = torch.cuda.device_count()
		render = options.gpu if options.gpu is not None else 0
		render = max(0, min(render, count - 1))
		bake = options.bake_gpu
		if bake is None:
			bake = next((i for i in range(count) if i != render), render)
		bake = max(0, min(bake, count - 1))
		return torch.device('cuda:%d' % render), torch.device('cuda:%d' % bake)

	def _icon(self):
		path = os.path.join(self.root, 'assets', 'icon.png')
		if not os.path.isfile(path):
			return
		try:
			pygame.display.set_icon(pygame.image.load(path))
		except pygame.error:
			return

	def _opening(self, mode):
		if self.preview and mode.LEVELS:
			return mode.sector_kind(mode.LEVELS), mode.LEVELS
		return 'sector', mode.clamp(self.first)

	def _pump(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
				self.running = False

	def _loading(self, progress, label):
		self._pump()
		if self.render_device.type != 'cuda':
			label = '%s  -  no CUDA GPU found, the CPU fallback is very slow' % label
		self.screens.draw_loading(self.display.surface, progress, label)
		self.display.flip()

	def _obtain(self, kind, sector, start, span):
		key = (kind, self.seed, sector)
		self.baker.throttle = 0.0
		while self.running:
			if self.baker.error:
				raise RuntimeError(self.baker.error)
			if self.baker.ready(*key):
				return self.baker.take(*key)
			self.baker.request(*key, urgent=True)
			self._loading(start + span * self.baker.fraction, self.baker.label)
			time.sleep(0.02)
		return None

	def _load(self):
		self._loading(0.02, 'waking the lambos')
		kind, first = self._opening(self.mode)
		self.baker.request(kind, self.seed, first)
		self._loading(0.05, 'tuning instruments')
		self.audio = AudioDirector(str(self.bake_device), self.settings.get('master_volume'), self.settings.get('music_volume'), self.settings.get('sfx_volume'))
		self.audio.music_folder = os.path.join(self.root, 'assets', 'music')
		self.audio.finale_path = getattr(self.options, 'finale_track', None)
		if self.options.mute:
			self.audio.enabled = False
		self.audio.set_music_mode(self.settings.get('music_mode'))
		self.audio.build(self.seed)
		self._loading(0.2, 'building the eye')
		self.renderer = Renderer(self.render_device, self.display.size)
		world = self._obtain(kind, first, 0.25, 0.6)
		if world is None:
			return
		self._loading(0.9, 'first light')
		self.renderer.set_world(world)
		self.world = world
		self.game = Game(self.renderer.scene, self.audio)
		self.game.attract = True
		self.game.camera.mode = self.input.camera_mode()
		self.game.hint_text = self.input.hint_text()
		self.game.shake_scale = self.settings.get('screen_shake')
		self.game.warp_enabled = bool(self.settings.get('warp_effect'))
		self.game.quota_limit = self.options.quota
		self._apply_minimap()
		self.game.start_run(first)
		self.game.enter_world(world, True)
		for _ in range(3):
			self.renderer.render(self.game.build_view(1.0 / 60.0, 0.0))
		target = 1.0 / (self.options.fps if self.options.fps > 0 else self.display.refresh)
		self.governor = ResolutionGovernor(self.renderer, RenderCfg.SCALES, target)
		self._apply_scale()
		self.audio.start()
		self._queue_next()
		self._enter('title')
		if self.options.autoplay:
			self._begin_run()

	def _queue_next(self):
		game = self.game
		self.baker.throttle = AppCfg.BAKE_THROTTLE
		self.baker.expect(() if game.finale else game.mode.keys(self.seed, game.sector))

	def _apply_scale(self):
		choice = self.options.scale if self.options.scale is not None else self.settings.get('render_scale')
		if choice == 'auto' or choice is None:
			self.governor.unlock()
			return
		self.governor.lock(float(choice))

	def _enter(self, state):
		self.state = state
		self.state_time = 0.0
		self.input.capture(state == 'play' and not self.options.autoplay)
		if state != 'play':
			self.baker.trim()

	def _begin_run(self):
		self._bank()
		game = self.game
		mode = self.mode
		kind, first = self._opening(mode)
		game.attract = False
		if self.pilot is not None:
			self.pilot.world = None
		final = kind != 'sector'
		self.assisted = self.piloted or bool(self.options.autoplay) or first > 0 or self.options.quota > 0 or self.dev.enabled or final
		self.crowned = None
		self.banked = False
		game.career = None if self.assisted else self.career
		self.career.begin(mode)
		game.start_run(first, mode, not self.piloted)
		if self.world.sector != first or self.world.kind != 'sector' or bool(self.world.palette.get('final')) != final:
			self.seed = random.randrange(1, 1000000)
			self.baker.forget(self.seed)
			world = self._obtain(kind, first, 0.0, 1.0)
			if world is None:
				return
			self.world = world
		self.renderer.set_world(self.world)
		game.enter_world(self.world, True)
		self.audio.play('start', 0.9)
		self.audio.music_event('start')
		self._queue_next()
		self._enter('play')

	def _dive(self):
		game = self.game
		if game.finale:
			self._ascend()
			return
		self.audio.play('warp', 1.0)
		keys = game.mode.keys(self.seed, game.sector)
		tunnel = self._obtain(keys[0][0], game.sector, 0.0, 1.0)
		if tunnel is None:
			return
		self.audio.music_event('warp')
		self.world = tunnel
		self.renderer.set_world(tunnel)
		game.enter_tunnel(tunnel)
		self.baker.throttle = AppCfg.BAKE_THROTTLE
		self.baker.expect((keys[1],))
		self.pacer.last = time.perf_counter()

	def _ascend(self):
		game = self.game
		game.warp_requested = False
		if self.state != 'play':
			return
		game.flash = 1.0
		self.audio.play('warp', 0.9)
		self._finish_run()
		game.attract = True

	def jump(self, sector):
		game = self.game
		sector = game.mode.clamp(sector)
		world = self._obtain('sector', sector, 0.0, 1.0)
		if world is None:
			return
		game.kite.release()
		game.sealed = None
		self.crowned = None
		self.banked = False
		game.sector = sector
		self.world = world
		self.renderer.set_world(world)
		game.enter_world(world, False)
		self.audio.play('start', 0.8)
		self._queue_next()
		self.input.mouse_held.clear()
		self.input.capture(not self.options.autoplay)
		self.pacer.last = time.perf_counter()

	def _arrive(self):
		game = self.game
		key = game.mode.keys(self.seed, game.sector)[1]
		if self.baker.error:
			raise RuntimeError(self.baker.error)
		if not self.baker.ready(*key):
			self.baker.request(*key)
			return
		world = self.baker.take(*key)
		game.complete_warp()
		self.world = world
		self.renderer.set_world(world)
		game.enter_world(world, False)
		self.audio.play('start', 0.8)
		if game.finale and not game.attract:
			self.crowned = self._record()
		self._queue_next()
		self.pacer.last = time.perf_counter()

	def _travel(self):
		game = self.game
		if game.warp_requested:
			self._dive()
		elif game.arrival:
			self._arrive()

	def _drive(self, game, dt):
		if self.pilot is None:
			from elan.ai.pilot import LearnedPilot
			self.pilot = LearnedPilot(self.options.pilot or os.path.join(self.root, 'assets', 'ai', 'pilot.pt'))
		if self.pilot.ready and self.settings.get('ai_pilot'):
			return self.pilot.drive(game, dt)
		return self.autopilot.drive(game, dt)

	def describe(self):
		game = self.game
		renderer = getattr(self, 'renderer', None)
		parts = ['state %s' % self.state, 'frame %d' % self.frames, self.input.gamepad.status().lower()]
		if renderer is not None:
			parts.append('render %dx%d on %s' % (renderer.w, renderer.h, renderer.device))
		if game is not None and game.world is not None:
			kite = game.kite
			parts.append('sector %d  seed %s  speed %.0f  pos %.0f %.0f %.0f  flux %.2f' % (game.sector + 1, getattr(game.world, 'seed', '?'), kite.speed, kite.pos[0], kite.pos[1], kite.pos[2], game.flux))
		return '  '.join(parts)

	def _toggle_fullscreen(self):
		self.display.toggle_fullscreen()
		self.settings.set('fullscreen', self.display.fullscreen)
		self.renderer.set_output(self.display.size)
		self.hud.resize(self.display.size)
		self.screens.resize(self.display.size)
		self.input.capture(self.state == 'play')

	def _apply_settings(self):
		settings = self.settings
		self._apply_scale()
		self.audio.set_music_mode(settings.get('music_mode'))
		self.audio.set_volumes(settings.get('master_volume'), settings.get('music_volume'), settings.get('sfx_volume'), settings.get('pickup_volume'))
		self.game.camera.mode = self.input.camera_mode()
		self.game.hint_text = self.input.hint_text()
		self.game.shake_scale = settings.get('screen_shake')
		self.game.warp_enabled = bool(settings.get('warp_effect'))
		self._apply_minimap()

	def _apply_minimap(self):
		mode = self.settings.get('minimap')
		minimap = self.renderer.minimap
		minimap.enabled = mode != 'off'
		if minimap.enabled:
			minimap.set_position(mode)
		self.hud.set_minimap(mode)

	def _toggle_camera(self):
		mode = self.input.toggle_camera()
		self.game.camera.mode = mode
		self.game.camera.trail = None
		self.game.say('CAMERA  %s' % mode.upper(), 'info', 1.4)

	def _to_title(self):
		game = self.game
		self._bank()
		self.settings.save()
		game.kite.release()
		game.messages = []
		game.attract = True
		game.over = False
		game.mode = MODES[0]
		self.autopilot.ribbon = -1
		self._enter('title')

	def _pause_menu(self, menu, dt):
		action = self.pause_menu.update(menu, self.events, dt)
		sound = self.pause_menu.sound
		if sound:
			self.audio.play('ui_move' if sound == 'move' else 'ui_select', 0.5)
		if action == 'resume':
			self._apply_settings()
			self._enter('title' if self.state == 'options' else 'play')
		elif action == 'restart':
			self._apply_settings()
			self._begin_run()
		elif action == 'title':
			self._apply_settings()
			self._to_title()
		elif action == 'quit':
			self.running = False
		elif action == 'fullscreen':
			self._toggle_fullscreen()
		elif action == 'changed':
			self._apply_settings()

	def _live(self):
		return self.game is not None and not self.game.attract and self.state in ('play', 'pause')

	def _bank(self):
		if self._live() and not self.banked:
			self._record()

	def _boards(self):
		return [(mode.NAME, mode.UNIT, mode.TIMED, self.board.top(mode.KEY)) for mode in MODES]

	def _toggle_pilot(self):
		game = self.game
		if not self.piloted and not self.dev.enabled:
			game.say('AI PILOT LOCKED  -  PICK AUTO-PLAY ON THE MAIN MENU', 'info', 2.4)
			return
		self.options.autoplay = not self.options.autoplay
		self.assisted = True
		game.career = None
		self.input.capture(not self.options.autoplay)
		game.say('AI PILOT ON' if self.options.autoplay else 'AI PILOT OFF', 'info', 1.8)

	def _record(self):
		settings = self.settings
		mode = self.game.mode
		best = settings.get(mode.SCORE_KEY)
		summary = self.game.summary(best)
		self.banked = True
		if mode.TIME_KEY:
			summary.best_time = float(settings.get(mode.TIME_KEY))
		if self.assisted:
			summary.new_best = False
			summary.best = int(best)
			summary.board = self.board.top(mode.KEY)
			return summary
		if summary.new_best:
			settings.set(mode.SCORE_KEY, summary.score)
		if summary.completed and mode.TIME_KEY:
			summary.new_time = summary.best_time <= 0.0 or summary.duration < summary.best_time
			if summary.new_time:
				settings.set(mode.TIME_KEY, summary.duration)
			settings.set('story_clears', int(settings.get('story_clears')) + 1)
		settings.save()
		summary.place = self.board.add(mode.KEY, summary.score, summary.sector, summary.duration, summary.completed)
		summary.board = self.board.top(mode.KEY)
		return summary

	def _finish_run(self):
		self.summary = self.crowned or self._record()
		self.crowned = None
		self._enter('gameover')

	def _step(self, dt, menu):
		game = self.game
		self.state_time += dt
		if menu.quit:
			self.running = False
		if menu.perf:
			self.show_perf = not self.show_perf
		if menu.mute:
			self.audio.toggle_mute()
		if menu.fullscreen:
			self._toggle_fullscreen()
		if self.state == 'title':
			if menu.escape:
				self.running = False
				return 0.0
			action = self.title.update(menu, self.events, self.state_time > 0.4)
			if self.title.sound:
				self.audio.play('ui_move', 0.5)
			if action == 'start':
				self.mode = self.title.current()
				self.options.autoplay = self.title.autoplay()
				self.piloted = self.options.autoplay
				self._begin_run()
				return 0.0
			if action in self.MENUS:
				self.audio.play('ui_select', 0.5)
				self.credits.open()
				self.pause_menu.open(True)
				self._enter(action)
				return 0.0
			game.update(dt, self._drive(game, dt))
			self._travel()
			return dt
		if self.state == 'options':
			self._pause_menu(menu, dt)
			if not self.running:
				return 0.0
			game.update(dt, self._drive(game, dt))
			self._travel()
			return dt
		if self.state == 'scores':
			if (menu.back or menu.confirm) and self.state_time > 0.3:
				self.audio.play('ui_select', 0.5)
				self._enter('title')
				return 0.0
			game.update(dt, self._drive(game, dt))
			self._travel()
			return dt
		if self.state == 'credits':
			if self.credits.update(dt, menu, self.events) == 'back':
				self.audio.play('ui_select', 0.5)
				self._enter('title')
				return 0.0
			game.update(dt, self._drive(game, dt))
			self._travel()
			return dt
		if self.state == 'play':
			if menu.pause or menu.focus_lost:
				self.pause_menu.open()
				self._enter('pause')
				return 0.0
			if menu.camera:
				self._toggle_camera()
			if menu.restart:
				self._begin_run()
				return 0.0
			if menu.autopilot:
				self._toggle_pilot()
			self.dev.run(self)
			if not self.running:
				return 0.0
			controls =self._drive(game, dt) if self.options.autoplay else self.input.sample(dt)
			game.update(dt, controls)
			self._travel()
			if game.over:
				self._finish_run()
			return dt
		if self.state == 'pause':
			self._pause_menu(menu, dt)
			return 0.0
		if self.state == 'gameover':
			done = self.summary is not None and self.summary.completed
			game.update(dt, self._drive(game, dt) if done else Controls())
			if menu.back and self.state_time > 0.5:
				self._to_title()
			elif menu.confirm and done and self.state_time > 1.0:
				self._to_title()
			elif (menu.confirm or menu.restart) and self.state_time > 1.0:
				self._begin_run()
			elif self.options.autoplay and self.state_time > (StoryCfg.REPLAY_DELAY if done else 7.0):
				self._begin_run()
			return dt
		return dt

	def _overlay(self):
		surface = self.display.surface
		game = self.game
		if self.state == 'title':
			view = self.title.view(self.career, lambda seconds: fmt_clock(seconds, 2))
			self.title.hits = self.screens.draw_title(surface, self.state_time, view, self.input.pad_connected()) or []
			if self.show_perf:
				self.hud.draw(surface, self._perf_only())
			return
		if self.state == 'gameover':
			self.screens.draw_gameover(surface, self.state_time, self.summary)
			return
		if self.state == 'credits':
			self.credits.limit = self.screens.draw_credits(surface, self.state_time, self.credits.view())
			return
		if self.state == 'scores':
			self.screens.draw_scores(surface, self.state_time, self._boards(), self.board.limit)
			return
		if self.state == 'options':
			self.pause_menu.fullscreen = self.display.fullscreen
			self.pause_menu.draw(surface, self.state_time)
			return
		state = game.hud_state(self.display.size, self.perf_lines if self.show_perf else None)
		state.dev = self.dev.lines()
		self.hud.draw(surface, state)
		if self.state == 'pause':
			self.pause_menu.fullscreen = self.display.fullscreen
			self.pause_menu.draw(surface, self.state_time)

	def _perf_only(self):
		state = self.game.hud_state(self.display.size, self.perf_lines)
		state.markers = []
		state.messages = []
		state.vel_marker = None
		state.hint = None
		return state

	def _perf(self, dt, work, gpu_wait, issue):
		self.fps += (1.0 / max(dt, 1.0e-4) - self.fps) * 0.05
		self.perf_timer -= dt
		if not self.show_perf or self.perf_timer > 0.0:
			return
		self.perf_timer = 0.25
		renderer = self.renderer
		baking = '%s %d%%' % (self.baker.label, round(self.baker.fraction * 100)) if self.baker.busy() else 'idle'
		self.perf_lines = [
			'%5.1f fps   frame %4.1f ms' % (self.fps, self.governor.work * 1000.0),
			'issue %4.1f ms   gpu wait %4.1f ms' % (issue * 1000.0, self.governor.gpu * 1000.0),
			'render %dx%d  (%d%%)  out %dx%d' % (renderer.w, renderer.h, round(renderer.scale * 100), self.display.size[0], self.display.size[1]),
			'draw %s   bake %s (%s)' % (self.render_device, self.bake_device, baking),
			'seed %d  sector %d  rock %.1f%%' % (self.seed, self.world.sector + 1, self.world.solid_fraction * 100.0),
		]

	def _shot(self, folder, name):
		os.makedirs(folder, exist_ok=True)
		self.display.save(os.path.join(folder, name))

	def run(self):
		self._load()
		last = time.perf_counter()
		options = self.options
		while self.running:
			now = time.perf_counter()
			dt = min(now - last, AppCfg.MAX_DT)
			last = now
			self.events = pygame.event.get()
			menu = self.input.feed(self.events)
			if self.input.notice and self.game is not None and self.state == 'play':
				self.game.say(self.input.notice, 'info', 1.6)
			game_dt = self._step(dt, menu)
			if not self.running:
				break
			self.clock += dt
			view = self.game.build_view(max(game_dt, 1.0e-4), self.clock)
			self.renderer.minimap.shown = self.state in ('play', 'pause')
			issue_start = time.perf_counter()
			frame = self.renderer.render(view)
			issue = time.perf_counter() - issue_start
			wait_start = time.perf_counter()
			self.display.present(frame)
			gpu_wait = time.perf_counter() - wait_start
			self.game.set_probe(self.renderer.read_probe())
			self._overlay()
			work = time.perf_counter() - now
			self.display.flip()
			flip = time.perf_counter() - now - work
			self.pacer.wait()
			mix = self.game.mix(self.state in self.MENUS, self.state == 'pause')
			mix.credits = 1.0 if self.state == 'credits' else 0.0
			self.audio.update(dt, mix)
			self.governor.update(dt, work, gpu_wait)
			self._perf(dt, work, gpu_wait, issue)
			self.frames += 1
			if menu.shot:
				self._shot(os.path.join(self.root, 'screenshots'), 'elan_%s.png' % time.strftime('%Y%m%d_%H%M%S'))
			if options.shots and self.frames % options.shot_every == 0:
				self._shot(options.shots, 'frame_%05d.png' % self.frames)
			if options.frames:
				self.frame_times.append((work, issue, gpu_wait, flip, dt))
				if self.frames >= options.frames:
					self.running = False
		self._bank()
		self.shutdown()

	def report(self):
		if not self.frame_times:
			return
		tail = self.frame_times[len(self.frame_times) // 3:]
		count = len(tail)
		work = sum(t[0] for t in tail) / count * 1000.0
		issue = sum(t[1] for t in tail) / count * 1000.0
		wait = sum(t[2] for t in tail) / count * 1000.0
		game = self.game
		flip = sorted(t[3] for t in tail)
		interval = sum(t[4] for t in tail) / count * 1000.0
		print('frames %d  work %.2f ms  issue %.2f ms  gpu wait %.2f ms  render %dx%d' % (self.frames, work, issue, wait, self.renderer.w, self.renderer.h))
		print('flip min %.2f  median %.2f  max %.2f ms   frame interval %.2f ms (%.1f fps)  vsync %s' % (flip[0] * 1000.0, flip[count // 2] * 1000.0, flip[-1] * 1000.0, interval, 1000.0 / interval, self.display.vsync))
		print('mode %s  score %d  motes %d  sector %d  flux %.2f  top speed %.0f  over %s  finale %s' % (game.mode.KEY, int(game.score), game.motes_total, game.sector + 1, game.flux, game.top_speed, game.over, game.finale))
		print('average speed %.0f  impacts %d  flight time %.0f s  motes/min %.0f' % (game.odometer / max(game.duration, 1.0), game.impacts, game.duration, game.motes_total * 60.0 / max(game.duration, 1.0)))

	def shutdown(self):
		if self.closed:
			return
		self.closed = True
		self.report()
		self.settings.save()
		if self.audio is not None:
			self.audio.shutdown()
		pygame.quit()
