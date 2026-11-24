import faulthandler
import os
import sys
import threading
import time
import traceback


class CrashLog:
	NAME = 'elan_error.log'

	def __init__(self, root):
		self.path = os.path.join(root, self.NAME)
		self.handle = None
		self.probe = None

	def install(self):
		try:
			self.handle = open(self.path, 'a', encoding='utf-8')
			faulthandler.enable(self.handle)
		except OSError:
			self.handle = None
		sys.excepthook = self._main
		threading.excepthook = self._thread
		return self

	def _context(self):
		lines = ['python %s  args: %s' % (sys.version.split()[0], ' '.join(sys.argv[1:]) or '-')]
		torch = sys.modules.get('torch')
		pygame = sys.modules.get('pygame')
		try:
			if torch is not None:
				names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
				lines.append('torch %s  gpus: %s' % (torch.__version__, ', '.join(names) or 'none'))
			if pygame is not None:
				lines.append('pygame %s' % pygame.version.ver)
			if self.probe is not None:
				lines.append(str(self.probe()))
		except Exception as error:
			lines.append('context unavailable: %r' % (error,))
		return lines

	def write(self, title, kind, value, trace):
		if self.handle is None:
			return
		try:
			stamp = time.strftime('%Y-%m-%d %H:%M:%S')
			body = ''.join(traceback.format_exception(kind, value, trace))
			self.handle.write('\n=== %s  %s ===\n%s\n%s' % (stamp, title, '\n'.join(self._context()), body))
			self.handle.flush()
			sys.stderr.write('error written to %s\n' % self.path)
		except (OSError, ValueError):
			return

	def _main(self, kind, value, trace):
		if not issubclass(kind, KeyboardInterrupt):
			self.write('main thread', kind, value, trace)
		sys.__excepthook__(kind, value, trace)

	def _thread(self, args):
		if args.exc_type is SystemExit:
			return
		name = args.thread.name if args.thread is not None else 'unknown'
		self.write('thread %s' % name, args.exc_type, args.exc_value, args.exc_traceback)
		sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)
