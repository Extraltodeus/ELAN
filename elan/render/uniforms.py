import torch


class Uniforms:
	def __init__(self, device):
		self.device = torch.device(device)
		self.layout = {}
		self.size = 0
		self.cpu = None
		self.gpu = None
		self.array = None

	def slot(self, name, count):
		self.layout[name] = (self.size, count)
		self.size += count

	def build(self):
		pinned = self.device.type == 'cuda'
		self.cpu = torch.zeros(self.size, dtype=torch.float32, pin_memory=pinned)
		self.array = self.cpu.numpy()
		self.gpu = torch.zeros(self.size, dtype=torch.float32, device=self.device)

	def view(self, name, *shape):
		start, count = self.layout[name]
		chunk = self.gpu[start:start + count]
		return chunk.view(*shape) if shape else chunk

	def set(self, name, values):
		start, count = self.layout[name]
		self.array[start:start + count] = values

	def upload(self):
		self.gpu.copy_(self.cpu, non_blocking=True)
