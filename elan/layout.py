from elan.config import RenderCfg


def minimap_box(h, w, position):
	radius = max(int(h * RenderCfg.MINIMAP_SIZE), 24)
	size = radius * 2 + 8
	margin = max(int(h * 0.03), 4)
	y0 = h - size - int(h * 0.05)
	if position.startswith('top'):
		y0 = max(int(h * 0.015), 2) if position.endswith('center') else margin
	x0 = w - size - margin
	if position.endswith('left'):
		x0 = margin
	elif position.endswith('center'):
		x0 = (w - size) // 2
	return y0, x0, size, radius
