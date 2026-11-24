import json
import os
import time

from elan.config import AppCfg
from elan.modes import MODES


class ScoreBoard:
	def __init__(self, root, settings):
		self.path = os.path.join(root, AppCfg.SCORES_FILE)
		self.limit = AppCfg.SCORES_KEPT
		self.tables = {mode.KEY: [] for mode in MODES}
		if os.path.isfile(self.path):
			self.load()
			return
		self.inherit(settings)

	def entry(self, score, stage, seconds, cleared, date):
		return {'score': max(int(score), 0), 'stage': max(int(stage), 0), 'time': max(float(seconds), 0.0), 'cleared': bool(cleared), 'date': str(date)[:10]}

	def clean(self, rows):
		out = []
		if not isinstance(rows, list):
			return out
		for row in rows:
			if not isinstance(row, dict):
				continue
			try:
				out.append(self.entry(row.get('score', 0), row.get('stage', 0), row.get('time', 0.0), row.get('cleared', False), row.get('date', '')))
			except (TypeError, ValueError):
				continue
		out.sort(key=lambda row: -row['score'])
		return [row for row in out if row['score'] > 0][:self.limit]

	def inherit(self, settings):
		for mode in MODES:
			best = int(settings.get(mode.SCORE_KEY))
			if best > 0:
				self.tables[mode.KEY].append(self.entry(best, 0, 0.0, False, ''))
		if any(self.tables.values()):
			self.save()

	def load(self):
		try:
			with open(self.path, 'r', encoding='utf-8') as handle:
				data = json.load(handle)
		except (OSError, ValueError):
			return
		if not isinstance(data, dict):
			return
		for key in self.tables:
			self.tables[key] = self.clean(data.get(key))

	def save(self):
		try:
			with open(self.path, 'w', encoding='utf-8') as handle:
				json.dump(self.tables, handle, indent='\t')
		except OSError:
			return

	def top(self, key):
		return list(self.tables.get(key, ()))

	def add(self, key, score, stage, seconds, cleared):
		rows = self.tables.get(key)
		if rows is None or int(score) <= 0:
			return 0
		row = self.entry(score, stage, seconds, cleared, time.strftime('%Y-%m-%d'))
		place = sum(1 for other in rows if other['score'] >= row['score'])
		if place >= self.limit:
			return 0
		rows.insert(place, row)
		del rows[self.limit:]
		self.save()
		return place + 1
