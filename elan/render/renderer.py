import math
import time
import torch
import torch.nn.functional as F

from elan.config import RenderCfg
from elan.render.uniforms import Uniforms
from elan.render.space import Space
from elan.render.march import RayMarcher
from elan.render.shade import SurfaceShader
from elan.render.atmosphere import Atmosphere
from elan.render.points import PointSplatter
from elan.render.scene import PointScene
from elan.render.post import PostFX
from elan.render.lens import Wormhole
from elan.render.minimap import Minimap
from elan.render.lightfield import LightField
from elan.render.flow import TunnelFlow


class View:
	def __init__(self):
		self.pos = (0.0, 100.0, 0.0)
		self.right = (1.0, 0.0, 0.0)
		self.up = (0.0, 1.0, 0.0)
		self.forward = (0.0, 0.0, 1.0)
		self.velocity = (0.0, 0.0, 0.0)
		self.fov = RenderCfg.FOV_BASE
		self.time = 0.0
		self.dt = 1.0 / 60.0
		self.rush = 0.0
		self.flash = 0.0
		self.damage = 0.0
		self.persistence = 0.0
		self.tether = None
		self.craft = None
		self.submerged = False
		self.tint = (1.0, 1.0, 1.0)
		self.warp = 0.0
		self.surge = 0.0
		self.wormhole = None
		self.tunnel = None
		self.route = None


class Renderer:
	JITTER = ((0.0, 0.0), (0.25, -0.25), (-0.25, 0.25), (0.375, 0.375), (-0.375, -0.375), (0.125, -0.375), (-0.125, 0.375), (0.375, -0.125))

	def __init__(self, device, out_size):
		self.device = torch.device(device)
		self.out_w, self.out_h = out_size
		uniforms = Uniforms(self.device)
		for name, count in (
			('origin', 3), ('raymat', 9), ('cam', 3), ('viewmat', 9), ('focal', 2), ('center', 2), ('pix', 1), ('hitk', 1),
			('ripple', 2), ('dust_base', 3), ('dust_shift', 3), ('streak', 9), ('hand', 3), ('rope', 3),
			('emit_pos', 3), ('emit_vel', 3), ('emit_color', 3), ('exposure', 3),
			('craft_pos', 3), ('craft_basis', 9), ('tips', 12), ('fresh_from', 6), ('fresh_step', 6),
			('mini_slice', 6), ('mini_turn', 6), ('mini_route', 12), ('mini_state', 4), ('mini_vel', 3),
			('flow_ring', 3), ('flow_streak', 3), ('flow_core', 3), ('flow_phase', 2), ('iris_phase', 3),
		):
			uniforms.slot(name, count)
		uniforms.build()
		self.uniforms = uniforms
		self.space = Space(self.device)
		self.marcher = RayMarcher(self.device, self.space)
		self.shader = SurfaceShader(self.device, uniforms, self.space)
		self.atmosphere = Atmosphere(self.device, uniforms, self.space)
		self.splatter = PointSplatter(self.device, uniforms, self.space)
		self.scene = PointScene(self.device, uniforms)
		self.post = PostFX(self.device, uniforms)
		self.minimap = Minimap(self.device, uniforms, self.space)
		self.lens = Wormhole(self.device)
		self.lights = LightField(self.device)
		self.flow = TunnelFlow()
		self.origin = uniforms.view('origin', 1, 1, 1, 1, 3)
		self.far = torch.tensor(1.0e9, device=self.device)
		self.raymat = uniforms.view('raymat', 3, 3)
		self.world = None
		self.relief = None
		self.exposure_tint = (1.0, 1.0, 1.0)
		self.scale = 0.0
		self.w = 0
		self.h = 0
		self.n = 0
		self.base = None
		self.center_index = 0
		self.probe = None
		self.frame = 0
		self.pixel = 0.001
		self.profile = False
		self.timings = {}
		self.set_scale(1.0)

	def set_output(self, out_size):
		self.out_w, self.out_h = out_size
		scale = self.scale
		self.scale = 0.0
		self.set_scale(scale)

	def set_scale(self, scale):
		if scale == self.scale:
			return
		self.scale = scale
		w = max(int(self.out_w * scale) // 4 * 4, 64)
		h = max(int(self.out_h * scale) // 4 * 4, 36)
		self.w = w
		self.h = h
		self.n = w * h
		xs = (torch.arange(w, device=self.device, dtype=torch.float32) + 0.5) / w * 2.0 - 1.0
		ys = 1.0 - (torch.arange(h, device=self.device, dtype=torch.float32) + 0.5) / h * 2.0
		grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
		self.base = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), -1).view(self.n, 3).contiguous()
		self.center_index = (h // 2) * w + w // 2
		self.marcher.allocate(h, w)
		self.shader.allocate(h, w)
		self.atmosphere.allocate(h, w)
		self.splatter.allocate(h, w)
		self.post.allocate(h, w, self.out_h, self.out_w)
		self.minimap.allocate(self.out_h, self.out_w)
		self.lens.allocate(h, w)

	def set_world(self, world):
		self.world = world
		self.relief = world.detail[:, 0:1]
		self.shader.set_world(world)
		self.atmosphere.set_world(world)
		self.scene.set_world(world)
		self.lights.set_world(world, self.scene.layout)
		self.flow.set_world(world)
		self.minimap.set_world(world, self.scene.layout)
		self.post.reset()

	def _set_uniforms(self, view):
		u = self.uniforms
		w, h = self.w, self.h
		tan_h = math.tan(math.radians(view.fov) * 0.5)
		tan_v = tan_h * h / w
		jitter = self.JITTER[self.frame & 7]
		jx = jitter[0] * 2.0 / w * tan_h
		jy = jitter[1] * 2.0 / h * tan_v
		r, up, f = view.right, view.up, view.forward
		u.set('origin', self.space.origin(view.pos))
		u.set('raymat', (
			r[0] * tan_h, r[1] * tan_h, r[2] * tan_h,
			up[0] * tan_v, up[1] * tan_v, up[2] * tan_v,
			f[0] + r[0] * jx + up[0] * jy, f[1] + r[1] * jx + up[1] * jy, f[2] + r[2] * jx + up[2] * jy,
		))
		u.set('cam', view.pos)
		u.set('viewmat', (r[0], up[0], f[0], r[1], up[1], f[1], r[2], up[2], f[2]))
		u.set('focal', (0.5 * w / tan_h, -0.5 * h / tan_v))
		u.set('center', (0.5 * w - 0.5, 0.5 * h - 0.5))
		pixel = 2.0 * tan_h / w
		self.pixel = pixel
		u.set('pix', pixel * 1.1)
		u.set('hitk', pixel * 3.0)
		flow = view.time * 0.035
		u.set('ripple', (view.pos[0] / 16.0 + flow, view.pos[2] / 16.0 + flow * 0.6))
		gain = RenderCfg.EXPOSURE * (1.0 + view.flash * 7.0)
		tint = view.tint
		u.set('exposure', (gain * tint[2], gain * tint[1], gain * tint[0]))
		self.flow.prepare(view, u)

	def _mark(self, name, start):
		if not self.profile:
			return start
		torch.cuda.synchronize(self.device)
		now = time.perf_counter()
		self.timings[name] = self.timings.get(name, 0.0) * 0.9 + (now - start) * 100.0
		return now

	def render(self, view):
		world = self.world
		h, w, n = self.h, self.w, self.n
		clock = time.perf_counter()
		self._set_uniforms(view)
		self.scene.prepare(view, self.uniforms)
		if world.coords is None:
			self.minimap.prepare(view, self.uniforms)
		self.uniforms.upload()
		rays = F.normalize(torch.matmul(self.base, self.raymat), dim=1)
		rays_world = rays.view(1, 1, h, w, 3)
		rays_n = rays_world * self.space.scale
		t, pos = self.marcher.march(world.sdf, self.relief, self.origin, rays_n, self.space.detail_offset, self.space.detail_scale, self.pixel)
		clock = self._mark('march', clock)
		self.shader.glow_field = self.lights.update(self.scene, view.dt)
		color, cover, sea = self.shader.shade(world, pos, t, rays_world, view.submerged, view.time)
		clock = self._mark('shade', clock)
		if world.coords is not None:
			color = self.atmosphere.tunnel(color, t)
		else:
			shaft = self.atmosphere.shafts(self.origin, rays_n, t, self.frame)
			color = self.atmosphere.compose(world, color, cover, sea, t, rays_world, shaft, view.pos[1], view.submerged)
		clock = self._mark('atmosphere', clock)
		self.scene.simulate(view.dt)
		points, tints = self.scene.gather(view.time)
		depth = t.view(n)
		blocker = torch.where(cover > 0.5, depth, self.far)
		self.splatter.splat(points, tints, self.scene.far_points, self.scene.far_colors, blocker, self.scene.motes)
		self.splatter.compose(color)
		if view.wormhole is not None:
			self.lens.apply(color, blocker, view, view.wormhole[0], view.wormhole[1], view.wormhole[2])
		index = self.center_index
		self.probe = torch.stack((depth[index], cover[index], sea[index]))
		clock = self._mark('points', clock)
		overlay = None
		if self.minimap.enabled and self.minimap.shown and world.coords is None:
			overlay = self.minimap.draw(self.scene)
		out = self.post.process(color.view(1, 3, h, w), view, self.frame, overlay)
		self._mark('post', clock)
		self.frame += 1
		return out

	def read_probe(self):
		if self.probe is None:
			return (1.0e9, 0.0, 0.0)
		return tuple(self.probe.tolist())
