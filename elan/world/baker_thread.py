import threading
import traceback
import torch

from elan.world.bake import WorldBaker


class BackgroundBaker:
	def __init__(self, bake_device, render_device, throttle):
		self.bake_device = torch.device(bake_device)
		self.render_device = torch.device(render_device)
		self.throttle = throttle
		self.lock = threading.Lock()
		self.thread = None
		self.running = False
		self.queue = []
		self.results = {}
		self.current = None
		self.error = None
		self.keep = None
		self.wanted = None
		self.fraction = 0.0
		self.label = ''

	def busy(self):
		return self.running

	def request(self, kind, seed, sector, urgent=False):
		key = (kind, seed, sector)
		with self.lock:
			if self.wanted is not None:
				self.wanted.add(key)
			if key in self.results or key == self.current:
				return
			if key in self.queue:
				if urgent:
					self.queue.remove(key)
					self.queue.insert(0, key)
				return
			self.queue.insert(0 if urgent else len(self.queue), key)
			if self.running:
				return
			self.running = True
			self.thread = threading.Thread(target=self._run, daemon=True)
		self.thread.start()

	def expect(self, keys):
		keys = [tuple(key) for key in keys]
		with self.lock:
			self.wanted = set(keys)
			self.queue = [key for key in self.queue if key in self.wanted]
			for key in [key for key in self.results if key not in self.wanted]:
				del self.results[key]
		for key in keys:
			self.request(*key)

	def forget(self, seed):
		with self.lock:
			self.keep = seed
			self.queue = [key for key in self.queue if key[1] == seed]
			for key in [key for key in self.results if key[1] != seed]:
				del self.results[key]

	def _progress(self, fraction, label):
		self.fraction = fraction
		self.label = label

	def _run(self):
		while True:
			with self.lock:
				if not self.queue:
					self.current = None
					self.running = False
					return
				key = self.queue.pop(0)
				self.current = key
				self.fraction = 0.0
				self.label = 'queued'
			try:
				baker = WorldBaker(self.bake_device, self.throttle, self)
				world = baker.bake_and_upload(key[0], key[1], key[2], self.render_device, self._progress)
				with self.lock:
					if (self.keep is None or key[1] == self.keep) and (self.wanted is None or key in self.wanted):
						self.results[key] = world
					self.current = None
				world = None
			except Exception:
				self.error = traceback.format_exc()
				with self.lock:
					self.current = None
					self.running = False
				return

	def trim(self):
		if self.running or self.bake_device.type != 'cuda':
			return
		torch.cuda.empty_cache()

	def ready(self, kind, seed, sector):
		return (kind, seed, sector) in self.results

	def take(self, kind, seed, sector):
		with self.lock:
			return self.results.pop((kind, seed, sector), None)

	def wait(self, timeout=None):
		if self.thread is not None:
			self.thread.join(timeout)
