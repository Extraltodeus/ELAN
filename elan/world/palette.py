import colorsys
import math
import random

BASE_SHAPE = {
	'layers': (('towers', 8.0), ('strands', 12.0), ('islands', 8.0)),
	'tower_wave': 270.0,
	'tower_thr': 1.28,
	'tower_taper': 0.55,
	'tower_flare': 0.3,
	'tower_base': 135.0,
	'tower_rise': 105.0,
	'warp_amp': 16.0,
	'strand_wave': 340.0,
	'strand_r': 0.21,
	'strand_keep': 0.1,
	'strand_floor': 0.0,
	'island_wave': 190.0,
	'island_thr': 2.2,
	'island_alt': 225.0,
	'island_band': 70.0,
	'island_flat': 0.45,
	'canyon_wave': 640.0,
	'canyon_width': 48.0,
	'canyon_bank': 74.0,
	'canyon_steps': 5.0,
	'canyon_riser': 0.22,
	'canyon_floor': -16.0,
	'mesa_base': 160.0,
	'mesa_rise': 36.0,
	'cells': 22,
	'cell_jitter': 0.62,
	'organ_wave': 400.0,
	'organ_thr': 0.6,
	'organ_gain': 150.0,
	'organ_jitter': 30.0,
	'organ_step': 10.0,
	'groove': 5.0,
	'hang_wave': 330.0,
	'hang_thr': 1.3,
	'hang_gain': 130.0,
	'hang_top': 300.0,
	'lattice_n': 3.0,
	'lattice_y': 300.0,
	'lattice_warp': 18.0,
	'lattice_thr': 1.0,
	'lattice_alt': 150.0,
	'lattice_band': 90.0,
	'tori': 0,
	'tori_alt': (130.0, 250.0),
	'tori_major': (95.0, 150.0),
	'tori_minor': (9.0, 14.0),
	'fold': 0,
	'rough': 2.0,
	'strata_amp': 2.4,
	'strata_period': 13.0,
}

BASE_STYLE = {
	'relief': 1.0,
	'vein_spread': 0.3,
	'moss': 1.0,
	'neon': 0.0,
	'neon_color': (0.1, 0.9, 1.0),
	'neon_period': 18.0,
	'iridescence': 0.0,
	'iris_flow': 0.0,
	'headlight': 0.0,
	'glow': 0.8,
	'sunk': 0.25,
	'corridor': 8,
	'shallow': 0.33,
	'aurora': 0.0,
	'sky_fold': 0,
	'sun_disc': 1.0,
	'cruise': None,
	'gate_alt': (70.0, 250.0),
	'cloud_deck': None,
	'far_fade': 0.0,
}

BIOMES = {
	'ember': {
		'name': 'EMBER DUSK',
		'sun_az': 35.0, 'sun_el': 13.0,
		'sun_color': (1.0, 0.55, 0.26), 'sun_power': 5.2,
		'zenith': (0.035, 0.06, 0.2), 'horizon': (0.95, 0.42, 0.24), 'glow': (1.0, 0.5, 0.18),
		'fog_density': 1.0,
		'rock_a': (0.17, 0.135, 0.13), 'rock_b': (0.36, 0.24, 0.175), 'top': (0.33, 0.3, 0.19),
		'vein': (0.1, 0.85, 1.0), 'vein_power': 3.2,
		'sea': (0.012, 0.04, 0.065),
		'amb_sky': (0.2, 0.24, 0.42), 'amb_ground': (0.11, 0.065, 0.06), 'rim': (0.75, 0.42, 0.34),
		'mote': (1.0, 0.95, 0.72), 'gate': (1.0, 0.45, 0.12), 'dust': (1.0, 0.75, 0.5),
		'planet_a': (0.78, 0.5, 0.36), 'planet_b': (0.42, 0.27, 0.33), 'ring': (0.9, 0.75, 0.6),
		'planet_az': -72.0, 'planet_el': 34.0, 'planet_size': 9.5, 'ring_tilt': 22.0,
		'stars': 1.0, 'nebula': (0.3, 0.16, 0.5), 'cloud': 0.8,
		'shape': {}, 'style': {},
	},
	'glacier': {
		'name': 'GLACIER DAWN',
		'sun_az': -140.0, 'sun_el': 17.0,
		'sun_color': (1.0, 0.78, 0.6), 'sun_power': 4.6,
		'zenith': (0.07, 0.17, 0.42), 'horizon': (0.95, 0.62, 0.66), 'glow': (1.0, 0.8, 0.55),
		'fog_density': 0.85,
		'rock_a': (0.2, 0.24, 0.31), 'rock_b': (0.44, 0.48, 0.56), 'top': (0.8, 0.84, 0.9),
		'vein': (1.0, 0.25, 0.75), 'vein_power': 2.8,
		'sea': (0.02, 0.07, 0.12),
		'amb_sky': (0.3, 0.38, 0.6), 'amb_ground': (0.12, 0.12, 0.16), 'rim': (0.8, 0.7, 0.85),
		'mote': (1.0, 0.82, 0.35), 'gate': (1.0, 0.35, 0.8), 'dust': (0.85, 0.92, 1.0),
		'planet_a': (0.55, 0.7, 0.85), 'planet_b': (0.85, 0.85, 0.95), 'ring': (0.75, 0.85, 1.0),
		'planet_az': 100.0, 'planet_el': 47.0, 'planet_size': 6.5, 'ring_tilt': -14.0,
		'stars': 0.45, 'nebula': (0.2, 0.3, 0.6), 'cloud': 1.1,
		'shape': {'tower_thr': 1.2, 'tower_rise': 100.0, 'strand_r': 0.19, 'strata_period': 16.0, 'rough': 1.4},
		'style': {},
	},
	'toxic': {
		'name': 'TOXIC BLOOM',
		'sun_az': 80.0, 'sun_el': 21.0,
		'sun_color': (0.85, 1.0, 0.45), 'sun_power': 4.2,
		'zenith': (0.015, 0.09, 0.11), 'horizon': (0.5, 0.82, 0.38), 'glow': (0.9, 1.0, 0.4),
		'fog_density': 1.25,
		'rock_a': (0.13, 0.09, 0.17), 'rock_b': (0.27, 0.19, 0.3), 'top': (0.1, 0.36, 0.3),
		'vein': (0.7, 1.0, 0.15), 'vein_power': 3.4,
		'sea': (0.01, 0.06, 0.05),
		'amb_sky': (0.16, 0.34, 0.32), 'amb_ground': (0.07, 0.05, 0.1), 'rim': (0.45, 0.8, 0.5),
		'mote': (1.0, 0.45, 0.8), 'gate': (0.6, 1.0, 0.3), 'dust': (0.7, 1.0, 0.6),
		'planet_a': (0.5, 0.75, 0.45), 'planet_b': (0.2, 0.4, 0.4), 'ring': (0.8, 0.9, 0.55),
		'planet_az': -150.0, 'planet_el': 28.0, 'planet_size': 13.0, 'ring_tilt': 35.0,
		'stars': 0.7, 'nebula': (0.1, 0.4, 0.3), 'cloud': 0.6,
		'shape': {'strand_r': 0.25, 'strand_wave': 290.0, 'island_thr': 2.05, 'warp_amp': 24.0, 'rough': 2.0},
		'style': {},
	},
	'blood': {
		'name': 'BLOOD MOON',
		'sun_az': -20.0, 'sun_el': 9.0,
		'sun_color': (1.0, 0.25, 0.14), 'sun_power': 6.0,
		'zenith': (0.03, 0.008, 0.05), 'horizon': (0.78, 0.15, 0.1), 'glow': (1.0, 0.3, 0.12),
		'fog_density': 1.15,
		'rock_a': (0.07, 0.065, 0.075), 'rock_b': (0.19, 0.12, 0.11), 'top': (0.16, 0.13, 0.13),
		'vein': (1.0, 0.42, 0.08), 'vein_power': 4.0,
		'sea': (0.03, 0.008, 0.012),
		'amb_sky': (0.22, 0.1, 0.16), 'amb_ground': (0.1, 0.03, 0.03), 'rim': (0.9, 0.3, 0.22),
		'mote': (0.6, 0.85, 1.0), 'gate': (1.0, 0.9, 0.5), 'dust': (1.0, 0.5, 0.3),
		'planet_a': (0.6, 0.2, 0.16), 'planet_b': (0.25, 0.08, 0.1), 'ring': (0.8, 0.45, 0.35),
		'planet_az': 55.0, 'planet_el': 52.0, 'planet_size': 16.0, 'ring_tilt': 8.0,
		'stars': 1.4, 'nebula': (0.5, 0.08, 0.15), 'cloud': 0.5,
		'shape': {'tower_thr': 1.2, 'tower_base': 125.0, 'tower_taper': 0.7, 'strand_r': 0.23, 'island_thr': 2.0, 'rough': 2.1, 'strata_amp': 1.3},
		'style': {'glow': 1.4},
	},
	'violet': {
		'name': 'VIOLET HOUR',
		'sun_az': 160.0, 'sun_el': 11.0,
		'sun_color': (1.0, 0.5, 0.75), 'sun_power': 4.8,
		'zenith': (0.06, 0.03, 0.22), 'horizon': (0.82, 0.38, 0.72), 'glow': (1.0, 0.55, 0.6),
		'fog_density': 0.95,
		'rock_a': (0.12, 0.12, 0.2), 'rock_b': (0.3, 0.24, 0.38), 'top': (0.42, 0.36, 0.5),
		'vein': (0.3, 1.0, 0.75), 'vein_power': 3.0,
		'sea': (0.02, 0.02, 0.08),
		'amb_sky': (0.27, 0.22, 0.5), 'amb_ground': (0.09, 0.06, 0.12), 'rim': (0.8, 0.5, 0.9),
		'mote': (1.0, 0.95, 0.6), 'gate': (0.4, 1.0, 0.8), 'dust': (0.9, 0.7, 1.0),
		'planet_a': (0.7, 0.55, 0.85), 'planet_b': (0.35, 0.3, 0.6), 'ring': (0.95, 0.8, 0.95),
		'planet_az': -40.0, 'planet_el': 40.0, 'planet_size': 11.0, 'ring_tilt': -28.0,
		'stars': 1.1, 'nebula': (0.35, 0.2, 0.6), 'cloud': 0.9,
		'shape': {'strand_wave': 300.0, 'strand_r': 0.23, 'tower_wave': 240.0, 'island_alt': 200.0},
		'style': {},
	},
	'neon': {
		'name': 'NEON CANYON',
		'sun_az': -60.0, 'sun_el': 24.0,
		'sun_color': (0.5, 0.62, 1.0), 'sun_power': 1.5,
		'zenith': (0.004, 0.006, 0.022), 'horizon': (0.07, 0.04, 0.16), 'glow': (0.35, 0.18, 0.7),
		'fog_density': 0.75,
		'rock_a': (0.05, 0.05, 0.075), 'rock_b': (0.13, 0.11, 0.18), 'top': (0.11, 0.1, 0.16),
		'vein': (1.0, 0.12, 0.62), 'vein_power': 4.6,
		'sea': (0.012, 0.01, 0.035),
		'amb_sky': (0.1, 0.12, 0.26), 'amb_ground': (0.04, 0.03, 0.07), 'rim': (0.35, 0.3, 0.9),
		'mote': (0.35, 1.0, 0.85), 'gate': (1.0, 0.3, 0.75), 'dust': (0.6, 0.8, 1.0),
		'planet_a': (0.5, 0.3, 0.8), 'planet_b': (0.15, 0.1, 0.35), 'ring': (0.6, 0.9, 1.0),
		'planet_az': 120.0, 'planet_el': 38.0, 'planet_size': 14.0, 'ring_tilt': 18.0,
		'stars': 2.0, 'nebula': (0.28, 0.1, 0.55), 'cloud': 0.25,
		'shape': {
			'layers': (('canyon', 8.0), ('strands', 10.0), ('islands', 8.0)),
			'strand_r': 0.17, 'strand_keep': 0.45, 'strand_floor': 60.0, 'island_thr': 2.45, 'island_alt': 290.0, 'island_band': 45.0,
			'warp_amp': 12.0, 'rough': 1.5, 'strata_amp': 1.2, 'strata_period': 18.0,
		},
		'style': {'corridor': 5, 'vein_spread': 1.3, 'neon': 1.6, 'neon_color': (0.1, 0.85, 1.0), 'headlight': 0.6, 'glow': 5.0, 'moss': 0.3, 'sun_disc': 3.0},
	},
	'reef': {
		'name': 'ABYSSAL REEF',
		'sun_az': 110.0, 'sun_el': 36.0,
		'sun_color': (1.0, 0.94, 0.78), 'sun_power': 4.4,
		'zenith': (0.04, 0.22, 0.48), 'horizon': (0.5, 0.86, 0.84), 'glow': (1.0, 0.95, 0.7),
		'fog_density': 0.9,
		'rock_a': (0.32, 0.13, 0.17), 'rock_b': (0.62, 0.34, 0.27), 'top': (0.78, 0.62, 0.44),
		'vein': (0.15, 1.0, 0.8), 'vein_power': 2.6,
		'sea': (0.008, 0.085, 0.1),
		'amb_sky': (0.24, 0.42, 0.55), 'amb_ground': (0.1, 0.1, 0.12), 'rim': (0.5, 0.9, 0.85),
		'mote': (1.0, 0.9, 0.45), 'gate': (1.0, 0.4, 0.62), 'dust': (0.7, 1.0, 0.95),
		'planet_a': (0.6, 0.8, 0.85), 'planet_b': (0.3, 0.55, 0.7), 'ring': (0.9, 0.95, 1.0),
		'planet_az': -30.0, 'planet_el': 55.0, 'planet_size': 7.5, 'ring_tilt': 30.0,
		'stars': 0.2, 'nebula': (0.1, 0.3, 0.4), 'cloud': 0.9,
		'shape': {
			'layers': (('lattice', 8.0), ('towers', 10.0)),
			'tower_thr': 1.75, 'tower_base': 150.0, 'tower_rise': 90.0, 'warp_amp': 22.0,
			'rough': 2.2, 'strata_amp': 1.0, 'strata_period': 9.0,
		},
		'style': {'sunk': 0.5, 'shallow': 0.7, 'corridor': 5, 'glow': 0.8, 'moss': 0.7},
	},
	'basalt': {
		'name': 'BASALT ORGAN',
		'sun_az': -105.0, 'sun_el': 23.0,
		'sun_color': (1.0, 0.88, 0.7), 'sun_power': 4.6,
		'zenith': (0.09, 0.13, 0.2), 'horizon': (0.56, 0.62, 0.64), 'glow': (0.95, 0.85, 0.65),
		'fog_density': 1.05,
		'rock_a': (0.085, 0.09, 0.105), 'rock_b': (0.21, 0.21, 0.23), 'top': (0.3, 0.38, 0.2),
		'vein': (1.0, 0.45, 0.08), 'vein_power': 3.6,
		'sea': (0.012, 0.04, 0.048),
		'amb_sky': (0.26, 0.31, 0.38), 'amb_ground': (0.08, 0.08, 0.085), 'rim': (0.7, 0.75, 0.8),
		'mote': (1.0, 0.8, 0.35), 'gate': (0.3, 0.9, 1.0), 'dust': (0.9, 0.95, 1.0),
		'planet_a': (0.7, 0.7, 0.65), 'planet_b': (0.4, 0.42, 0.5), 'ring': (0.85, 0.85, 0.8),
		'planet_az': 60.0, 'planet_el': 30.0, 'planet_size': 12.0, 'ring_tilt': -20.0,
		'stars': 0.3, 'nebula': (0.2, 0.2, 0.3), 'cloud': 1.5,
		'shape': {'layers': (('basalt', 8.0),), 'rough': 0.35, 'strata_amp': 0.5, 'strata_period': 10.0},
		'style': {'relief': 0.45, 'glow': 1.0},
	},
	'aurora': {
		'name': 'MIDNIGHT AURORA',
		'sun_az': 150.0, 'sun_el': 32.0,
		'sun_color': (0.6, 0.76, 1.0), 'sun_power': 1.9,
		'zenith': (0.003, 0.008, 0.022), 'horizon': (0.02, 0.07, 0.085), 'glow': (0.12, 0.55, 0.42),
		'fog_density': 0.7,
		'rock_a': (0.06, 0.075, 0.1), 'rock_b': (0.15, 0.18, 0.23), 'top': (0.5, 0.56, 0.68),
		'vein': (0.2, 1.0, 0.62), 'vein_power': 4.2,
		'sea': (0.006, 0.02, 0.03),
		'amb_sky': (0.09, 0.15, 0.24), 'amb_ground': (0.03, 0.045, 0.06), 'rim': (0.3, 0.8, 0.65),
		'mote': (1.0, 0.6, 0.85), 'gate': (0.35, 1.0, 0.6), 'dust': (0.7, 1.0, 0.9),
		'planet_a': (0.55, 0.65, 0.8), 'planet_b': (0.2, 0.25, 0.4), 'ring': (0.7, 0.9, 0.9),
		'planet_az': -100.0, 'planet_el': 44.0, 'planet_size': 8.0, 'ring_tilt': 12.0,
		'stars': 2.4, 'nebula': (0.08, 0.3, 0.35), 'cloud': 0.15,
		'shape': {'tower_thr': 1.22, 'tower_rise': 115.0, 'strand_r': 0.2, 'island_thr': 2.1},
		'style': {'aurora': 1.0, 'headlight': 0.6, 'glow': 5.0, 'sun_disc': 3.4},
	},
	'void': {
		'name': 'MIRROR VOID',
		'sun_az': 20.0, 'sun_el': 19.0,
		'sun_color': (1.0, 0.96, 0.9), 'sun_power': 6.6,
		'zenith': (0.0, 0.0, 0.004), 'horizon': (0.012, 0.014, 0.035), 'glow': (0.22, 0.26, 0.5),
		'fog_density': 0.3,
		'rock_a': (0.13, 0.12, 0.125), 'rock_b': (0.31, 0.28, 0.26), 'top': (0.34, 0.31, 0.29),
		'vein': (0.2, 0.6, 1.0), 'vein_power': 3.2,
		'sea': (0.004, 0.006, 0.014),
		'amb_sky': (0.05, 0.06, 0.1), 'amb_ground': (0.03, 0.03, 0.045), 'rim': (0.5, 0.6, 0.95),
		'mote': (1.0, 0.72, 0.3), 'gate': (0.45, 0.8, 1.0), 'dust': (0.8, 0.85, 1.0),
		'planet_a': (0.85, 0.6, 0.4), 'planet_b': (0.45, 0.3, 0.3), 'ring': (0.95, 0.85, 0.7),
		'planet_az': -55.0, 'planet_el': 31.0, 'planet_size': 24.0, 'ring_tilt': 16.0,
		'stars': 3.0, 'nebula': (0.3, 0.14, 0.5), 'cloud': 0.0,
		'shape': {
			'layers': (('islands', 8.0),),
			'island_wave': 150.0, 'island_thr': 1.95, 'island_alt': 190.0, 'island_band': 150.0, 'island_flat': 0.9,
			'tori': 6, 'rough': 2.6, 'strata_amp': 0.8,
		},
		'style': {'glow': 3.0, 'headlight': 0.35, 'moss': 0.0},
	},
	'mandala': {
		'name': 'PRISM MANDALA',
		'sun_az': 45.0, 'sun_el': 27.0,
		'sun_color': (1.0, 0.82, 0.95), 'sun_power': 4.4,
		'zenith': (0.03, 0.012, 0.11), 'horizon': (0.52, 0.3, 0.72), 'glow': (0.9, 0.6, 1.0),
		'fog_density': 0.8,
		'rock_a': (0.15, 0.15, 0.22), 'rock_b': (0.4, 0.37, 0.5), 'top': (0.62, 0.6, 0.78),
		'vein': (0.4, 0.9, 1.0), 'vein_power': 3.0,
		'sea': (0.02, 0.015, 0.06),
		'amb_sky': (0.26, 0.22, 0.46), 'amb_ground': (0.09, 0.07, 0.13), 'rim': (0.7, 0.6, 1.0),
		'mote': (1.0, 0.93, 0.65), 'gate': (0.3, 1.0, 0.88), 'dust': (0.95, 0.8, 1.0),
		'planet_a': (0.8, 0.6, 0.9), 'planet_b': (0.4, 0.3, 0.65), 'ring': (1.0, 0.85, 1.0),
		'planet_az': 0.0, 'planet_el': 62.0, 'planet_size': 10.0, 'ring_tilt': 0.0,
		'stars': 1.7, 'nebula': (0.4, 0.2, 0.65), 'cloud': 0.0,
		'shape': {
			'tower_wave': 250.0, 'tower_thr': 1.22, 'tower_rise': 110.0, 'strand_r': 0.19, 'strand_wave': 300.0, 'island_thr': 2.35,
			'fold': 256, 'rough': 1.0, 'strata_amp': 1.6, 'strata_period': 11.0,
		},
		'style': {'iridescence': 0.75, 'sky_fold': 8, 'relief': 0.6, 'glow': 1.0, 'moss': 0.5},
	},
	'dream': {
		'name': 'LIQUID DREAM',
		'sun_az': -125.0, 'sun_el': 29.0,
		'sun_color': (1.0, 0.9, 0.84), 'sun_power': 3.9,
		'zenith': (0.2, 0.26, 0.56), 'horizon': (1.0, 0.74, 0.8), 'glow': (1.0, 0.9, 0.8),
		'fog_density': 0.9,
		'rock_a': (0.42, 0.38, 0.5), 'rock_b': (0.66, 0.56, 0.68), 'top': (0.84, 0.8, 0.9),
		'vein': (1.0, 0.6, 0.9), 'vein_power': 1.2,
		'sea': (0.05, 0.055, 0.12),
		'amb_sky': (0.36, 0.38, 0.6), 'amb_ground': (0.16, 0.12, 0.18), 'rim': (1.0, 0.8, 0.9),
		'mote': (1.0, 0.48, 0.28), 'gate': (0.3, 0.6, 1.0), 'dust': (1.0, 0.9, 0.95),
		'planet_a': (1.0, 0.8, 0.85), 'planet_b': (0.7, 0.6, 0.9), 'ring': (1.0, 0.95, 0.9),
		'planet_az': 70.0, 'planet_el': 42.0, 'planet_size': 15.0, 'ring_tilt': -32.0,
		'stars': 0.3, 'nebula': (0.3, 0.2, 0.5), 'cloud': 0.45,
		'shape': {
			'tower_wave': 300.0, 'tower_thr': 1.4, 'tower_taper': 0.2, 'tower_flare': 0.55, 'strand_r': 0.3, 'strand_wave': 380.0, 'island_thr': 1.9,
			'island_wave': 170.0, 'island_flat': 0.7, 'warp_amp': 34.0, 'rough': 0.0, 'strata_amp': 0.0,
		},
		'style': {'relief': 0.0, 'iridescence': 0.75, 'iris_flow': 0.35, 'glow': 0.8, 'moss': 0.4},
	},
}

ORDER = ('ember', 'glacier', 'neon', 'toxic', 'reef', 'blood', 'basalt', 'aurora', 'void', 'violet', 'mandala', 'dream')

TUNNEL = {
	'name': 'WARP TUNNEL',
	'sun_color': (1.0, 1.0, 1.0), 'sun_power': 0.0,
	'fog_density': 0.55,
	'rock_a': (0.03, 0.03, 0.05), 'rock_b': (0.07, 0.07, 0.11), 'top': (0.07, 0.07, 0.11),
	'vein': (0.4, 0.8, 1.0), 'vein_power': 0.0,
	'sea': (0.0, 0.0, 0.0),
	'amb_sky': (0.1, 0.11, 0.16), 'amb_ground': (0.05, 0.05, 0.08), 'rim': (0.5, 0.6, 1.0),
	'horizon': (0.5, 0.5, 0.5),
	'style': {'relief': 0.5, 'glow': 6.0, 'headlight': 0.5, 'moss': 0.0},
}


SINGULARITY = {
	'name': 'THE SINGULARITY',
	'sun_az': 30.0, 'sun_el': 10.0,
	'sun_color': (1.0, 0.74, 0.5), 'sun_power': 4.2,
	'zenith': (0.07, 0.15, 0.45), 'horizon': (1.0, 0.62, 0.5), 'glow': (1.0, 0.74, 0.46),
	'fog_density': 0.85,
	'rock_a': (0.7, 0.72, 0.77), 'rock_b': (0.86, 0.86, 0.9), 'top': (0.93, 0.94, 0.96),
	'vein': (0.5, 0.9, 1.0), 'vein_power': 1.8,
	'sea': (0.03, 0.065, 0.11),
	'amb_sky': (0.36, 0.42, 0.72), 'amb_ground': (0.3, 0.24, 0.24), 'rim': (1.0, 0.84, 0.74),
	'mote': (1.0, 1.0, 1.0), 'gate': (1.0, 1.0, 1.0), 'dust': (1.0, 0.95, 0.88),
	'planet_a': (1.0, 0.88, 0.78), 'planet_b': (0.72, 0.66, 0.86), 'ring': (1.0, 0.94, 0.86),
	'planet_az': -75.0, 'planet_el': 40.0, 'planet_size': 11.0, 'ring_tilt': 18.0,
	'stars': 0.5, 'nebula': (0.3, 0.25, 0.5), 'cloud': 0.5,
	'shape': {
		'layers': (('isles', 6.0), ('citadel', 3.0), ('spires', 4.0)),
		'isle_wave': 560.0, 'isle_thr': 0.05, 'isle_gain': 26.0, 'isle_top': 9.0,
		'city_cells': 26, 'city_jitter': 0.55, 'city_core': 1.3, 'city_shore': 0.08, 'city_fill': 0.9,
		'city_height': (14.0, 96.0), 'city_peak': 1.3, 'city_girth': (5.0, 9.5),
		'city_taper': 0.8, 'city_tip': 0.4, 'city_flare': 0.5, 'city_foot': 8.0,
		'city_domes': 0.14, 'dome_scale': 1.9,
		'spire_cells': 3, 'spire_jitter': 0.6, 'spire_shore': 0.2, 'spire_fill': 0.6,
		'spire_height': (225.0, 300.0), 'spire_girth': 12.0,
		'spire_decks': ((0.74, 2.2, 2.6), (0.9, 1.5, 2.2)),
		'tori': 3, 'tori_alt': (215.0, 280.0), 'tori_major': (60.0, 105.0), 'tori_minor': (5.0, 8.0),
		'rough': 0.0, 'strata_amp': 0.0, 'warp_amp': 0.0,
	},
	'style': {
		'relief': 0.0, 'moss': 0.0, 'vein_spread': 0.25, 'neon': 2.2, 'neon_color': (0.55, 0.9, 1.0), 'neon_period': 14.0,
		'iridescence': 0.1, 'iris_flow': 0.1, 'glow': 1.2, 'sunk': 0.0, 'shallow': 0.0, 'corridor': 6,
		'cruise': (185.0, 275.0), 'gate_alt': (210.0, 265.0), 'far_fade': 170.0,
		'cloud_deck': {
			'band': (104.0, 192.0), 'wave': 400.0, 'detail': (26.0, 170.0), 'thr': 0.3, 'gain': 2.6, 'crown': 3.2,
			'density': 0.045, 'sun': 1.55, 'sky': 1.0, 'wind': (2.6, 1.1),
		},
	},
}

NUMERALS = ('', ' II', ' III', ' IV', ' V', ' VI', ' VII', ' VIII', ' IX', ' X')


class PaletteBook:
	def _turn(self, color, shift):
		peak = max(max(color), 1.0e-6)
		scale = max(peak, 1.0)
		h, l, s = colorsys.rgb_to_hls(color[0] / scale, color[1] / scale, color[2] / scale)
		r, g, b = colorsys.hls_to_rgb((h + shift) % 1.0, l, s)
		return (r * scale, g * scale, b * scale)

	def _recolor(self, values, shift):
		for name, value in values.items():
			if isinstance(value, tuple) and len(value) == 3 and all(isinstance(v, float) for v in value) and name not in ('tori_alt', 'tori_major', 'tori_minor'):
				values[name] = self._turn(value, shift)

	def order(self, seed, cycle):
		keys = list(ORDER)
		if cycle <= 0:
			return keys
		random.Random(seed * 31 + cycle * 977).shuffle(keys)
		previous = self.order(seed, cycle - 1)
		if keys[0] == previous[-1]:
			keys[0], keys[-1] = keys[-1], keys[0]
		return keys

	def key(self, sector, seed):
		count = len(ORDER)
		return self.order(seed, sector // count)[sector % count]

	def pick(self, sector, seed, key=None):
		key = key or self.key(sector, seed)
		base = BIOMES[key]
		rng = random.Random(seed * 7919 + sector)
		palette = dict(base)
		palette['key'] = key
		shape = dict(BASE_SHAPE)
		shape.update(base['shape'])
		tighten = min(sector // len(ORDER), 3) * 0.02
		shape['strand_r'] = shape['strand_r'] + tighten
		shape['island_thr'] = shape['island_thr'] - tighten * 1.5
		palette['shape'] = shape
		style = dict(BASE_STYLE)
		style.update(base['style'])
		palette['style'] = style
		palette['sun_az'] = base['sun_az'] + rng.uniform(-25.0, 25.0)
		palette['sun_el'] = base['sun_el'] + rng.uniform(-2.0, 3.0)
		palette['planet_az'] = base['planet_az'] + rng.uniform(-30.0, 30.0)
		cycle = sector // len(ORDER)
		if cycle > 0:
			shift = rng.uniform(0.05, 0.16) * rng.choice((-1.0, 1.0))
			self._recolor(palette, shift)
			self._recolor(style, shift)
			palette['name'] = base['name'] + NUMERALS[min(cycle, len(NUMERALS) - 1)]
		return palette

	def finale(self, seed):
		rng = random.Random(seed * 7919 + 4242)
		palette = dict(SINGULARITY)
		palette['key'] = 'singularity'
		palette['final'] = True
		shape = dict(BASE_SHAPE)
		shape.update(SINGULARITY['shape'])
		palette['shape'] = shape
		style = dict(BASE_STYLE)
		style.update(SINGULARITY['style'])
		palette['style'] = style
		palette['sun_az'] = SINGULARITY['sun_az'] + rng.uniform(-25.0, 25.0)
		palette['planet_az'] = SINGULARITY['planet_az'] + rng.uniform(-30.0, 30.0)
		return palette

	def plunge(self, sector, seed):
		palette = self.tunnel(sector, seed)
		palette['hyper'] = False
		palette['plunge'] = True
		palette['name'] = 'EVENT HORIZON'
		palette['mote'] = (1.0, 1.0, 1.0)
		palette['dust'] = (1.0, 1.0, 1.0)
		palette['flow_to'] = (1.0, 1.0, 1.0)
		palette['core_to'] = (0.0, 0.0, 0.0)
		return palette

	def hyper(self, sector):
		return (sector + 1) % 10 == 0

	def tunnel(self, sector, seed):
		source = self.pick(sector, seed)
		target = self.pick(sector + 1, seed)
		palette = dict(TUNNEL)
		style = dict(BASE_STYLE)
		style.update(TUNNEL['style'])
		palette['style'] = style
		palette['key'] = 'tunnel'
		palette['hyper'] = self.hyper(sector)
		if palette['hyper']:
			palette['name'] = 'HYPER TUNNEL'
		palette['mote'] = target['mote']
		palette['gate'] = source['gate']
		palette['dust'] = target['dust']
		palette['flow_from'] = source['gate']
		palette['flow_to'] = target['gate']
		palette['core_from'] = source['glow']
		palette['core_to'] = target['horizon']
		return palette

	def sun_direction(self, palette):
		az = math.radians(palette['sun_az'])
		el = math.radians(palette['sun_el'])
		return (math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))
