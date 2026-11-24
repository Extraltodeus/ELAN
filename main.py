import argparse
import os
import sys

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_JOYSTICK_HIDAPI_PS4_RUMBLE', '1')
os.environ.setdefault('SDL_JOYSTICK_HIDAPI_PS5_RUMBLE', '1')


def parse():
	parser = argparse.ArgumentParser(prog='elan')
	parser.add_argument('--windowed', action='store_true')
	parser.add_argument('--window', default='1280x720')
	parser.add_argument('--screen', type=int, default=0)
	parser.add_argument('--gpu', type=int, default=None)
	parser.add_argument('--bake-gpu', type=int, default=None)
	parser.add_argument('--scale', default=None)
	parser.add_argument('--fps', type=int, default=0)
	parser.add_argument('--seed', type=int, default=None)
	parser.add_argument('--mute', action='store_true')
	parser.add_argument('--vsync', action='store_true')
	parser.add_argument('--perf', action='store_true')
	parser.add_argument('--autoplay', action='store_true')
	parser.add_argument('--frames', type=int, default=0)
	parser.add_argument('--shots', default=None)
	parser.add_argument('--shot-every', type=int, default=60)
	parser.add_argument('--sector', type=int, default=1)
	parser.add_argument('--quota', type=int, default=0)
	parser.add_argument('--pilot', default=None)
	parser.add_argument('--mode', choices=('chill', 'story'), default=None)
	parser.add_argument('--finale', action='store_true')
	parser.add_argument('--finale-track', default=None)
	parser.add_argument('--developper', '--developer', '--dev', dest='developer', action='store_true')
	options = parser.parse_args()
	width, _, height = options.window.lower().partition('x')
	options.window = (int(width), int(height or 720))
	if options.scale is not None and options.scale != 'auto':
		options.scale = float(options.scale)
	options.root = os.path.dirname(os.path.abspath(__file__))
	return options


def main():
	options = parse()
	sys.path.insert(0, options.root)
	from elan.crashlog import CrashLog
	log = CrashLog(options.root).install()
	from elan.app import App
	app = App(options)
	log.probe = app.describe
	try:
		app.run()
	finally:
		app.shutdown()


if __name__ == '__main__':
	main()
