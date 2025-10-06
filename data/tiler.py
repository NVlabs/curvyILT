#############################
# NVIDIA  All Rights Reserved
# Haoyu Yang 
# Design Automation Research
# Last Update: Sep 05 2025
#############################

import math
from typing import Iterator, List, Tuple, Dict

import numpy as np


def _next_multiple(value: int, multiple: int) -> int:

	if value % multiple == 0:
		return value
	return ((value // multiple) + 1) * multiple


def pad_to_grid(array: np.ndarray, step_h: int, step_w: int, mode: str = "constant") -> Tuple[np.ndarray, Tuple[int, int]]:

	assert array.ndim == 2, "pad_to_grid expects HxW array"
	h, w = array.shape
	new_h = _next_multiple(h, step_h)
	new_w = _next_multiple(w, step_w)
	pad_h = new_h - h
	pad_w = new_w - w
	if pad_h == 0 and pad_w == 0:
		return array, (0, 0)
	if mode == "constant":
		padded = np.pad(array, ((0, pad_h), (0, pad_w)), mode="constant")
	else:
		padded = np.pad(array, ((0, pad_h), (0, pad_w)), mode=mode)
	return padded, (pad_h, pad_w)


class TileSpec:
	__slots__ = ("y0", "y1", "x0", "x1", "core_y0", "core_y1", "core_x0", "core_x1")

	def __init__(self, y0: int, y1: int, x0: int, x1: int, core_y0: int, core_y1: int, core_x0: int, core_x1: int) -> None:
		self.y0 = y0
		self.y1 = y1
		self.x0 = x0
		self.x1 = x1
		self.core_y0 = core_y0
		self.core_y1 = core_y1
		self.core_x0 = core_x0
		self.core_x1 = core_x1


class TileIndexer:
	"""
	Generate overlapped 2048x2048 tiles with 1024x1024 trusted cores on the downsampled (mask_s) grid.

	Inputs are in the mask_s grid (already divided by scale_factor).

	- tile_px_full = 2048 (original grid)
	- core_px_full = 1024 (original grid)
	- step = core size (50% overlap)
	- halo = (tile - core) / 2
	"""

	def __init__(self, height_s: int, width_s: int, scale_factor: int, offset_y_s: int = 0, offset_x_s: int = 0) -> None:
		self.scale_factor = scale_factor
		self.tile_size_s = 2048 // scale_factor
		self.core_size_s = 1024 // scale_factor
		self.halo_s = (self.tile_size_s - self.core_size_s) // 2
		self.step_s = self.core_size_s
		self.height_s = height_s
		self.width_s = width_s
		self.offset_y_s = max(0, int(offset_y_s))
		self.offset_x_s = max(0, int(offset_x_s))

	def tiles(self) -> Iterator[TileSpec]:
		# Ensure cores cover edges, allow offset starts
		y_starts = list(range(self.offset_y_s, self.height_s, self.step_s))
		y_last = max(0, self.height_s - self.core_size_s)
		if len(y_starts) == 0 or y_starts[-1] != y_last:
			y_starts.append(y_last)
		x_starts = list(range(self.offset_x_s, self.width_s, self.step_s))
		x_last = max(0, self.width_s - self.core_size_s)
		if len(x_starts) == 0 or x_starts[-1] != x_last:
			x_starts.append(x_last)

		for y_core in y_starts:
			for x_core in x_starts:
				t_y0 = y_core - self.halo_s
				t_x0 = x_core - self.halo_s
				t_y1 = t_y0 + self.tile_size_s
				t_x1 = t_x0 + self.tile_size_s

				# clamp to image bounds
				off_y = 0
				off_x = 0
				if t_y0 < 0:
					off_y = -t_y0
					t_y0 = 0
					t_y1 = min(self.tile_size_s - off_y, self.height_s)
				else:
					t_y1 = min(t_y1, self.height_s)

				if t_x0 < 0:
					off_x = -t_x0
					t_x0 = 0
					t_x1 = min(self.tile_size_s - off_x, self.width_s)
				else:
					t_x1 = min(t_x1, self.width_s)

				# ensure only full-size tiles are yielded to keep core shapes uniform across a batch
				if (t_y1 - t_y0) != self.tile_size_s or (t_x1 - t_x0) != self.tile_size_s:
					continue

				# core within tile (relative coordinates)
				c_y0 = self.halo_s - off_y
				c_x0 = self.halo_s - off_x
				c_y1 = c_y0 + self.core_size_s
				c_x1 = c_x0 + self.core_size_s

				# clip core to the actual tile if truncated at borders
				h = t_y1 - t_y0
				w = t_x1 - t_x0
				c_y0 = max(0, c_y0)
				c_x0 = max(0, c_x0)
				c_y1 = min(h, c_y1)
				c_x1 = min(w, c_x1)

				yield TileSpec(t_y0, t_y1, t_x0, t_x1, c_y0, c_y1, c_x0, c_x1)


