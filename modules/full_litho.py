#############################
# NVIDIA  All Rights Reserved
# Haoyu Yang 
# Design Automation Research
# Last Update: Sep 05 2025
#############################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from curvyilt import get_kernels 


class FullChipLitho(nn.Module):
	"""
	Global full-chip lithography simulator with a single trainable mask_s.
	Forward operates on overlapped tiles and returns core-region predictions.
	"""

	def __init__(self, target_full: torch.Tensor, scale_factor: int = 8, avepool_kernel: int = 3, mask_steepness: float = 4.0, resist_th: float = 0.225, resist_steepness: float = 50.0, mask_shift: float = 0.5, max_dose: float = 1.02, min_dose: float = 0.98) -> None:
		super().__init__()

		assert target_full.ndim == 4 and target_full.shape[1] == 1, "target_full should be NCHW with C=1"
		# Keep target_full as a registered buffer so it moves with .to(device)/.cuda(rank)
		self.register_buffer("target_full", target_full, persistent=False)
		self.scale_factor = scale_factor
		self.avepool = nn.AvgPool2d(kernel_size=avepool_kernel, stride=1, padding=avepool_kernel//2)

		# parameters for mask and resist
		self.mask_steepness = mask_steepness
		self.resist_th = resist_th
		self.resist_steepness = resist_steepness
		self.mask_shift = mask_shift
		self.max_dose = max_dose
		self.min_dose = min_dose

		# downsampled grid
		_, _, H, W = self.target_full.shape
		self.Hs = H // self.scale_factor
		self.Ws = W // self.scale_factor

		# Initialize trainable mask on downsampled grid from downsampled target
		target_ds = F.avg_pool2d(self.target_full, self.scale_factor)
		self.mask_s = nn.Parameter(target_ds.detach().clone())

		# Load optics kernels (shared across tiles) and register as buffers
		fo, defo, fo_scale, defo_scale = get_kernels()
		self.register_buffer("kernel_focus", torch.tensor(fo), persistent=False)
		self.register_buffer("kernel_defocus", torch.tensor(defo), persistent=False)
		self.register_buffer("kernel_focus_scale", torch.tensor(fo_scale), persistent=False)
		self.register_buffer("kernel_defocus_scale", torch.tensor(defo_scale), persistent=False)

		self.kernel_num, self.kernel_dim1, self.kernel_dim2 = fo.shape  # 24, 35, 35
		# Note: frequency center offset must be computed per tile size at runtime.
		# Keep a placeholder for clarity; do not use for slicing tiles.
		self.offset_s = self.Hs//2 - self.kernel_dim1//2

		self.iter = 0

	def run_litho_sim(self, mask_tile: torch.Tensor):
		# optics on fixed-size tile
		_, _, h, w = mask_tile.shape
		offset_tile = (h // 2) - (self.kernel_dim1 // 2)

		mask = self.avepool(mask_tile)
		mask = torch.sigmoid(self.mask_steepness * (mask - self.mask_shift))
		mask_fft = torch.fft.fftshift(torch.fft.fft2(mask), dim=(-2, -1))
		mask_fft_rep = torch.repeat_interleave(mask_fft, self.kernel_num, 1)
		mask_fft_max = mask_fft_rep * self.max_dose
		mask_fft_min = mask_fft_rep * self.min_dose

		x_out = mask_fft_rep[:, :, offset_tile:offset_tile + self.kernel_dim1, offset_tile:offset_tile + self.kernel_dim2] * self.kernel_focus
		x_out = torch.fft.ifft2(x_out, s=(h, w))
		x_out = x_out.real * x_out.real + x_out.imag * x_out.imag
		x_out = x_out * self.kernel_focus_scale
		x_out = torch.sum(x_out, axis=1, keepdims=True)
		nominal = torch.sigmoid(self.resist_steepness * (x_out - self.resist_th))

		x_out_max = mask_fft_max[:, :, offset_tile:offset_tile + self.kernel_dim1, offset_tile:offset_tile + self.kernel_dim2] * self.kernel_focus
		x_out_max = torch.fft.ifft2(x_out_max, s=(h, w))
		x_out_max = x_out_max.real * x_out_max.real + x_out_max.imag * x_out_max.imag
		x_out_max = x_out_max * self.kernel_focus_scale
		x_out_max = torch.sum(x_out_max, axis=1, keepdims=True)
		outer = torch.sigmoid(self.resist_steepness * (x_out_max - self.resist_th))

		x_out_min = mask_fft_min[:, :, offset_tile:offset_tile + self.kernel_dim1, offset_tile:offset_tile + self.kernel_dim2] * self.kernel_defocus
		x_out_min = torch.fft.ifft2(x_out_min, s=(h, w))
		x_out_min = x_out_min.real * x_out_min.real + x_out_min.imag * x_out_min.imag
		x_out_min = x_out_min * self.kernel_defocus_scale
		x_out_min = torch.sum(x_out_min, axis=1, keepdims=True)
		inner = torch.sigmoid(self.resist_steepness * (x_out_min - self.resist_th))

		return nominal, inner, outer, x_out, x_out_min, x_out_max

	def _mask_to_fft(self, mask: torch.Tensor) -> torch.Tensor:
		mask = self.avepool(mask)
		mask = torch.sigmoid(self.mask_steepness * (mask - self.mask_shift))
		mask_fft = torch.fft.fftshift(torch.fft.fft2(mask), dim=(-2, -1))
		return mask_fft

	@torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
	def forward_tiles(self, tiles: list) -> dict:
		"""
		Return fixed-size cores for all tiles by building fixed-size tiles (2048/scale)
		centered on each core with padding, running optics, then cropping fixed cores (1024/scale).
		"""
		outputs = {
			"nominal": [],
			"outer": [],
			"inner": [],
			"mask_core": [],
			"core_boxes": [],
			"mask_tiles": []
		}

		# fixed sizes on downsampled grid
		tile_s = 2048 // self.scale_factor
		core_s = 1024 // self.scale_factor
		halo_s = (tile_s - core_s) // 2
		offset_tile = (tile_s // 2) - (self.kernel_dim1 // 2)

		for t in tiles:
			# absolute core position in downsampled grid
			core_y_abs = t.y0 + t.core_y0
			core_x_abs = t.x0 + t.core_x0

			# fixed-size tile window around the core
			y_win0 = core_y_abs - halo_s
			x_win0 = core_x_abs - halo_s
			y_win1 = y_win0 + tile_s
			x_win1 = x_win0 + tile_s

			# clamp and extract from global mask_s
			y0 = max(0, y_win0)
			x0 = max(0, x_win0)
			y1 = min(self.Hs, y_win1)
			x1 = min(self.Ws, x_win1)
			tile_mask = self.mask_s[:, :, y0:y1, x0:x1]
			n, _, h0, w0 = tile_mask.shape

			# pad to fixed tile size
			pad_top = max(0, 0 - (y_win0 - y0))
			pad_left = max(0, 0 - (x_win0 - x0))
			pad_bottom = tile_s - (h0 + pad_top)
			pad_right = tile_s - (w0 + pad_left)
			if pad_top or pad_bottom or pad_left or pad_right:
				tile_mask = F.pad(tile_mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

			# optics on fixed-size tile
			_, _, h, w = tile_mask.shape
			mask = self.avepool(tile_mask)
			mask = torch.sigmoid(self.mask_steepness * (mask - self.mask_shift))
			mask_fft = torch.fft.fftshift(torch.fft.fft2(mask), dim=(-2, -1))
			mask_fft_rep = torch.repeat_interleave(mask_fft, self.kernel_num, 1)
			mask_fft_max = mask_fft_rep * self.max_dose
			mask_fft_min = mask_fft_rep * self.min_dose

			x_out = mask_fft_rep[:, :, offset_tile:offset_tile + self.kernel_dim1, offset_tile:offset_tile + self.kernel_dim2] * self.kernel_focus
			x_out = torch.fft.ifft2(x_out, s=(h, w))
			x_out = x_out.real * x_out.real + x_out.imag * x_out.imag
			x_out = x_out * self.kernel_focus_scale
			x_out = torch.sum(x_out, axis=1, keepdims=True)
			nominal = torch.sigmoid(self.resist_steepness * (x_out - self.resist_th))

			x_out_max = mask_fft_max[:, :, offset_tile:offset_tile + self.kernel_dim1, offset_tile:offset_tile + self.kernel_dim2] * self.kernel_focus
			x_out_max = torch.fft.ifft2(x_out_max, s=(h, w))
			x_out_max = x_out_max.real * x_out_max.real + x_out_max.imag * x_out_max.imag
			x_out_max = x_out_max * self.kernel_focus_scale
			x_out_max = torch.sum(x_out_max, axis=1, keepdims=True)
			outer = torch.sigmoid(self.resist_steepness * (x_out_max - self.resist_th))

			x_out_min = mask_fft_min[:, :, offset_tile:offset_tile + self.kernel_dim1, offset_tile:offset_tile + self.kernel_dim2] * self.kernel_defocus
			x_out_min = torch.fft.ifft2(x_out_min, s=(h, w))
			x_out_min = x_out_min.real * x_out_min.real + x_out_min.imag * x_out_min.imag
			x_out_min = x_out_min * self.kernel_defocus_scale
			x_out_min = torch.sum(x_out_min, axis=1, keepdims=True)
			inner = torch.sigmoid(self.resist_steepness * (x_out_min - self.resist_th))

			# fixed-size core crop at center of the tile
			cy0 = halo_s
			cx0 = halo_s
			cy1 = cy0 + core_s
			cx1 = cx0 + core_s
			outputs["mask_core"].append(mask[:, :, cy0:cy1, cx0:cx1])
			outputs["nominal"].append(nominal[:, :, cy0:cy1, cx0:cx1])
			outputs["outer"].append(outer[:, :, cy0:cy1, cx0:cx1])
			outputs["inner"].append(inner[:, :, cy0:cy1, cx0:cx1])
			outputs["core_boxes"].append((core_y_abs, core_x_abs, core_s, core_s))
			outputs["mask_tiles"].append(tile_mask)

		# concat batch
		for k in ("mask_core", "nominal", "outer", "inner", "mask_tiles"):
			outputs[k] = torch.cat(outputs[k], dim=0) if len(outputs[k]) > 0 else None
		return outputs

	@torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
	def forward_tiles_full(self, tiles: list) -> dict:
		"""
		Compute full-tile predictions for overlap-add stitching with Hann windows.
		Returns dict with lists per tile: outputs and tile boxes (absolute coords in mask_s grid).
		"""
		outputs = {
			"nominal": [],
			"outer": [],
			"inner": [],
			"tile_boxes": []
		}

		for t in tiles:
			tile_mask = self.mask_s[:, :, t.y0:t.y1, t.x0:t.x1]
			n, _, h, w = tile_mask.shape

			mask = self.avepool(tile_mask)
			mask = torch.sigmoid(self.mask_steepness * (mask - self.mask_shift))
			mask_fft = torch.fft.fftshift(torch.fft.fft2(mask), dim=(-2, -1))
			mask_fft_rep = torch.repeat_interleave(mask_fft, self.kernel_num, 1)

			mask_fft_max = mask_fft_rep * self.max_dose
			mask_fft_min = mask_fft_rep * self.min_dose

			offset_t = (h // 2) - (self.kernel_dim1 // 2)
			y0i = max(0, offset_t)
			x0i = max(0, offset_t)
			y1i = min(h, offset_t + self.kernel_dim1)
			x1i = min(w, offset_t + self.kernel_dim2)
			ky0 = max(0, -offset_t)
			kx0 = max(0, -offset_t)
			ky1 = ky0 + (y1i - y0i)
			kx1 = kx0 + (x1i - x0i)

			x_out = torch.view_as_complex(torch.zeros((n, self.kernel_num, h, w, 2), dtype=torch.float32, device=mask.device))
			x_out[:, :, y0i:y1i, x0i:x1i] = (
				mask_fft_rep[:, :, y0i:y1i, x0i:x1i] * self.kernel_focus[:, ky0:ky1, kx0:kx1]
			)
			x_out = torch.fft.ifft2(x_out)
			x_out = x_out.real * x_out.real + x_out.imag * x_out.imag
			x_out = x_out * self.kernel_focus_scale
			x_out = torch.sum(x_out, axis=1, keepdims=True)
			nominal = torch.sigmoid(self.resist_steepness * (x_out - self.resist_th))

			x_out_max = torch.view_as_complex(torch.zeros((n, self.kernel_num, h, w, 2), dtype=torch.float32, device=mask.device))
			x_out_max[:, :, y0i:y1i, x0i:x1i] = (
				mask_fft_max[:, :, y0i:y1i, x0i:x1i] * self.kernel_focus[:, ky0:ky1, kx0:kx1]
			)
			x_out_max = torch.fft.ifft2(x_out_max)
			x_out_max = x_out_max.real * x_out_max.real + x_out_max.imag * x_out_max.imag
			x_out_max = x_out_max * self.kernel_focus_scale
			x_out_max = torch.sum(x_out_max, axis=1, keepdims=True)
			outer = torch.sigmoid(self.resist_steepness * (x_out_max - self.resist_th))

			x_out_min = torch.view_as_complex(torch.zeros((n, self.kernel_num, h, w, 2), dtype=torch.float32, device=mask.device))
			x_out_min[:, :, y0i:y1i, x0i:x1i] = (
				mask_fft_min[:, :, y0i:y1i, x0i:x1i] * self.kernel_defocus[:, ky0:ky1, kx0:kx1]
			)
			x_out_min = torch.fft.ifft2(x_out_min)
			x_out_min = x_out_min.real * x_out_min.real + x_out_min.imag * x_out_min.imag
			x_out_min = x_out_min * self.kernel_defocus_scale
			x_out_min = torch.sum(x_out_min, axis=1, keepdims=True)
			inner = torch.sigmoid(self.resist_steepness * (x_out_min - self.resist_th))

			outputs["nominal"].append(nominal)
			outputs["outer"].append(outer)
			outputs["inner"].append(inner)
			outputs["tile_boxes"].append((t.y0, t.x0, h, w))

		return outputs


