#############################
# NVIDIA  All Rights Reserved
# Haoyu Yang 
# Design Automation Research
# Last Update: Sep 05 2025
#############################

import numpy as np


def hann_window_1d(n: int) -> np.ndarray:
	if n <= 1:
		return np.ones((n,), dtype=np.float32)
	return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float32) / (n - 1))


def hann_window_2d(h: int, w: int) -> np.ndarray:
	wy = hann_window_1d(h)
	wx = hann_window_1d(w)
	return np.outer(wy, wx).astype(np.float32)


def blend_tile_into(canvas: np.ndarray, weight: np.ndarray, tile: np.ndarray, y0: int, x0: int) -> None:
	"""
	Overlap-add with weights.
	canvas, weight: full-size arrays (H, W)
	tile: (h, w)
	"""
	h, w = tile.shape
	canvas[y0:y0+h, x0:x0+w] += tile * weight[:h, :w]


def finalize_canvas(canvas: np.ndarray, weight: np.ndarray) -> np.ndarray:
	mask = weight > 0
	canvas[mask] = canvas[mask] / weight[mask]
	return canvas


