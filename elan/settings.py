import copy
import json
import os

import pygame

from elan.config import AppCfg
from elan.modes import MODE_KEYS

ACTIONS = ('thrust', 'brake', 'strafe_left', 'strafe_right', 'boost', 'tether', 'drift', 'camera', 'restart')

DEFAULT_BINDINGS = {
	'thrust': ['key:%d' % pygame.K_w, 'key:%d' % pygame.K_z],
	'brake': ['key:%d' % pygame.K_s, 'key:%d' % pygame.K_DOWN],
	'strafe_left': ['key:%d' % pygame.K_a, 'key:%d' % pygame.K_q],
	'strafe_right': ['key:%d' % pygame.K_d, 'key:%d' % pygame.K_RIGHT],
	'boost': ['', 'key:%d' % pygame.K_SPACE],
	'tether': ['mouse:1', 'key:%d' % pygame.K_e],
	'drift': ['key:%d' % pygame.K_LSHIFT, ''],
	'camera': ['key:%d' % pygame.K_v, ''],
	'restart': ['key:%d' % pygame.K_p, ''],
}

PAD_ACTIONS = ACTIONS + ('pause', 'gyro', 'gyro_reset')

DEFAULT_PAD_LOOK = 'left'

DEFAULT_PAD = {
	'thrust': 'pad:111',
	'brake': 'pad:109',
	'strafe_left': 'pad:104',
	'strafe_right': 'pad:105',
	'boost': 'pad:10',
	'tether': 'pad:0',
	'drift': 'pad:9',
	'camera': 'pad:3',
	'restart': '',
	'pause': 'pad:6',
	'gyro': 'pad:2',
	'gyro_reset': 'pad:1',
}

CAMERA_MODES = ('chase', 'cockpit')
PAD_LOOKS = ('right', 'left')
GYRO_AXES = ('yaw', 'roll')
MUSIC_MODES = ('dj', 'linear', 'synth')
MINIMAP_MODES = ('bottom_right', 'bottom_left', 'top_right', 'top_center', 'off')
VOLUMES = ('master_volume', 'music_volume', 'sfx_volume', 'pickup_volume')

DEFAULTS = {
	'best_score': 0,
	'best_sector': 0,
	'game_mode': 'chill',
	'story_best_score': 0,
	'story_best_level': 0,
	'story_best_time': 0.0,
	'story_clears': 0,
	'worlds_seen': [],
	'mouse_sensitivity': 0.8,
	'mouse_smoothing': 6,
	'invert_y': False,
	'camera': 'chase',
	'master_volume': 0.7,
	'music_volume': 1.0,
	'music_mode': 'dj',
	'screen_shake': 0.2,
	'minimap': 'top_center',
	'sfx_volume': 0.6,
	'pickup_volume': 0.4,
	'warp_effect': True,
	'ai_pilot': True,
	'render_scale': 0.5,
	'fullscreen': True,
	'screen': 0,
	'bindings': DEFAULT_BINDINGS,
	'pad_bindings': DEFAULT_PAD,
	'pad_look': DEFAULT_PAD_LOOK,
	'pad_sensitivity': 1.0,
	'pad_smoothing': 4,
	'pad_invert_y': True,
	'gyro_enabled': True,
	'gyro_sensitivity': 1.2,
	'gyro_smoothing': 0,
	'gyro_invert_y': False,
	'gyro_axis': 'yaw',
}


def valid_binding(value):
	if value == '':
		return True
	if not isinstance(value, str) or ':' not in value:
		return False
	kind, _, code = value.partition(':')
	return kind in ('key', 'mouse') and code.isdigit()


def clean_bindings(data):
	out = copy.deepcopy(DEFAULT_BINDINGS)
	if not isinstance(data, dict):
		return out
	for action in ACTIONS:
		slots = data.get(action)
		if not isinstance(slots, list) or len(slots) != 2 or not all(valid_binding(s) for s in slots):
			continue
		out[action] = [str(s) for s in slots]
	return out


def clean_pad(data):
	out = dict(DEFAULT_PAD)
	if not isinstance(data, dict):
		return out
	for action in PAD_ACTIONS:
		value = data.get(action)
		if value == '' or (isinstance(value, str) and value.startswith('pad:') and value[4:].isdigit()):
			out[action] = value
	return out


class Settings:
	def __init__(self, root):
		self.path = os.path.join(root, AppCfg.SETTINGS_FILE)
		self.values = copy.deepcopy(DEFAULTS)
		self.load()

	def load(self):
		if not os.path.isfile(self.path):
			return
		try:
			with open(self.path, 'r', encoding='utf-8') as handle:
				data = json.load(handle)
		except (OSError, ValueError):
			return
		if not isinstance(data, dict):
			return
		for key in DEFAULTS:
			if key in data:
				self.values[key] = data[key]
		self.sanitize()

	def sanitize(self):
		v = self.values
		for key in ('best_sector', 'story_best_score', 'story_best_level', 'story_clears'):
			try:
				v[key] = max(0, int(v[key]))
			except (TypeError, ValueError):
				v[key] = 0
		try:
			v['story_best_time'] = max(0.0, float(v['story_best_time']))
		except (TypeError, ValueError):
			v['story_best_time'] = 0.0
		if v['game_mode'] not in MODE_KEYS:
			v['game_mode'] = DEFAULTS['game_mode']
		if not isinstance(v['worlds_seen'], list):
			v['worlds_seen'] = []
		v['worlds_seen'] = [str(key) for key in v['worlds_seen']]
		try:
			v['mouse_smoothing'] = max(0, min(10, int(v['mouse_smoothing'])))
		except (TypeError, ValueError):
			v['mouse_smoothing'] = DEFAULTS['mouse_smoothing']
		if v['camera'] not in CAMERA_MODES:
			v['camera'] = DEFAULTS['camera']
		if v['music_mode'] not in MUSIC_MODES:
			v['music_mode'] = DEFAULTS['music_mode']
		try:
			v['screen_shake'] = max(0.0, min(2.0, float(v['screen_shake'])))
		except (TypeError, ValueError):
			v['screen_shake'] = DEFAULTS['screen_shake']
		try:
			v['screen'] = max(0, int(v['screen']))
		except (TypeError, ValueError):
			v['screen'] = DEFAULTS['screen']
		v['warp_effect'] = bool(v['warp_effect'])
		v['ai_pilot'] = bool(v['ai_pilot'])
		if isinstance(v['minimap'], bool):
			v['minimap'] = 'bottom_right' if v['minimap'] else 'off'
		if v['minimap'] not in MINIMAP_MODES:
			v['minimap'] = DEFAULTS['minimap']
		for key in VOLUMES:
			try:
				v[key] = max(0.0, min(1.0, float(v[key])))
			except (TypeError, ValueError):
				v[key] = DEFAULTS[key]
		v['bindings'] = clean_bindings(v['bindings'])
		v['pad_bindings'] = clean_pad(v['pad_bindings'])
		if v['pad_look'] not in PAD_LOOKS:
			v['pad_look'] = DEFAULTS['pad_look']
		for key in ('pad_sensitivity', 'gyro_sensitivity'):
			try:
				v[key] = max(0.3, min(3.0, float(v[key])))
			except (TypeError, ValueError):
				v[key] = DEFAULTS[key]
		for key in ('pad_smoothing', 'gyro_smoothing'):
			try:
				v[key] = max(0, min(10, int(v[key])))
			except (TypeError, ValueError):
				v[key] = DEFAULTS[key]
		for key in ('pad_invert_y', 'gyro_enabled', 'gyro_invert_y'):
			v[key] = bool(v[key])
		if v['gyro_axis'] not in GYRO_AXES:
			v['gyro_axis'] = DEFAULTS['gyro_axis']

	def save(self):
		try:
			with open(self.path, 'w', encoding='utf-8') as handle:
				json.dump(self.values, handle, indent='\t')
		except OSError:
			return

	def get(self, key):
		return self.values[key]

	def set(self, key, value):
		self.values[key] = value
