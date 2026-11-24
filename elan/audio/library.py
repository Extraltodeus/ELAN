import os

import numpy as np
import pygame

KINDS = ('.wav', '.ogg')
FOLDERS = ('music', 'music_ogg')


def music_folder(root):
	assets = os.path.join(root, 'assets')
	for folder in FOLDERS:
		path = os.path.join(assets, folder)
		if any(os.path.isfile(os.path.join(path, 'flight' + ext)) for ext in KINDS):
			return path
	return os.path.join(assets, FOLDERS[0])


def locate(folder, name):
	for ext in KINDS:
		path = os.path.join(folder, name + ext)
		if os.path.isfile(path):
			return path
	return os.path.join(folder, name + KINDS[0])


def present(folder, name):
	return os.path.isfile(locate(folder, name))


def is_ogg(path):
	return path.lower().endswith('.ogg')


def _int16(data, size):
	if size == 16:
		return data.astype(np.int32) - 32768
	if size == -32:
		return data.astype(np.int64) >> 16
	if size == 32:
		return np.clip(data.astype(np.float32), -1.0, 1.0) * 32767.0
	if size == 8:
		return (data.astype(np.int32) - 128) << 8
	if size == -8:
		return data.astype(np.int32) << 8
	return data


def read_ogg(path):
	rate, size, channels = pygame.mixer.get_init()
	data = pygame.sndarray.array(pygame.mixer.Sound(file=path))
	data = _int16(data, size)
	if data.ndim == 1:
		data = data[:, None]
	if data.shape[1] == 1:
		data = np.repeat(data, 2, axis=1)
	return np.ascontiguousarray(np.clip(data[:, :2], -32768, 32767).astype(np.int16)), rate
