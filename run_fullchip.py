#############################
# NVIDIA  All Rights Reserved
# Haoyu Yang 
# Design Automation Research
# Last Update: Sep 05 2025
#############################

import argparse
from datetime import datetime
import time

import cv2
import os
import numpy as np
import torch
import torch.nn as nn

from data.tiler import TileIndexer, TileSpec
from modules.full_litho import FullChipLitho
from curvyilt import l2_loss, evaluation
from utils.stitch import hann_window_2d, blend_tile_into, finalize_canvas
from kornia.morphology import opening, closing, dilation, erosion

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--image", type=str, required=True, help="Path to full-chip target (grayscale PNG)")
	parser.add_argument("--scale_factor", type=int, default=8)
	parser.add_argument("--iters", type=int, default=200)
	parser.add_argument("--lr", type=float, default=1.0)
	parser.add_argument("--save", type=str, default=None, help="Optional path to save mask image")
	parser.add_argument("--pad", type=int, default=512, help="Pad pixels (full-res) added to all sides before optimization")
	parser.add_argument("--sim_dir", type=str, default=None, help="Directory to save stitched simulated images; defaults next to input image")
	parser.add_argument("--morph", type=int, default=3, help="Morphological operations to apply to the mask")
	return parser.parse_args()


def load_target(image_path: str) -> torch.Tensor:
	im = cv2.imread(image_path, -1) / 255.0
	assert im.ndim == 2, "Expect grayscale image"
	H, W = im.shape
	im = np.expand_dims(np.expand_dims(im, 0), 0)
	return torch.from_numpy(im).float().cuda(), H, W


def main():
	args = parse_args()

	target_full, H, W = load_target(args.image)
	pad = int(args.pad)
	if pad < 0:
		pad = 0
	if pad % args.scale_factor != 0:
		raise AssertionError(f"--pad ({pad}) must be divisible by --scale_factor ({args.scale_factor})")

	# pad target on full-res grid (NCHW). torch.nn.functional.pad expects (W_left, W_right, H_top, H_bottom)
	if pad > 0:
		import torch.nn.functional as F
		target_full = F.pad(target_full, (pad, pad, pad, pad), mode="constant", value=0.0)
		H_p, W_p = H + 2 * pad, W + 2 * pad
	else:
		H_p, W_p = H, W

	# build a validity mask to ignore padded regions in loss
	import torch.nn.functional as F
	valid_full = torch.ones((1,1,H,W), device=target_full.device, dtype=target_full.dtype)
	if pad > 0:
		valid_full = F.pad(valid_full, (pad, pad, pad, pad), mode="constant", value=0.0)
	valid_ds = nn.functional.avg_pool2d(valid_full, args.scale_factor)

	model = FullChipLitho(target_full=target_full, scale_factor=args.scale_factor).cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
	scaler = torch.cuda.amp.GradScaler()

	# tile generator on downsampled grid (use padded size)
	Hs = H_p // args.scale_factor
	Ws = W_p // args.scale_factor
	# define 4-phase offsets to shuffle which overlaps co-occur
	core_s = 1024 // args.scale_factor
	offsets_s = [
		(0, 0),
		(0, core_s // 2),
		(core_s // 2, 0),
		(core_s // 2, core_s // 2),
	]
	indexer = TileIndexer(height_s=Hs, width_s=Ws, scale_factor=args.scale_factor)
	tiles = list(indexer.tiles())
	morph = args.morph
	scale_factor = args.scale_factor
	if morph>0:
		morph_kernel_opt_opening =torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph,morph)).astype(np.float32)).cuda()
		morph_kernel_opt_closing =torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph+2,morph+2)).astype(np.float32)).cuda()
		morph_kernel_opening = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph*scale_factor+1,morph*scale_factor+1)).astype(np.float32)).cuda()
		morph_kernel_closing = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((morph-2)*scale_factor+1,(morph-2)*scale_factor+1)).astype(np.float32)).cuda()
		#morph_kernel_closing = morph_kernel_opening
	start = time.time()
	for it in range(args.iters):
		loss_total = 0.0
		with torch.autocast(device_type="cuda", dtype=torch.float16):
			# process tiles in small mini-batches to limit memory, step per batch
			batch_size = 8
			#if it > 250: batch_size =1 
			# choose offset phase for this iteration
			phase =it % len(offsets_s)
			if it > 20: phase = 3 #warm up with dancing tile
			oy_s, ox_s = offsets_s[phase]
			indexer = TileIndexer(height_s=Hs, width_s=Ws, scale_factor=args.scale_factor, offset_y_s=oy_s, offset_x_s=ox_s)
			tiles = list(indexer.tiles())
			for b in range(0, len(tiles), batch_size):
				optimizer.zero_grad()
				batch_tiles = tiles[b:b+batch_size]
				out = model.forward_tiles(batch_tiles)
				# gather target cores and validity masks
				target_ds = nn.functional.avg_pool2d(model.target_full, args.scale_factor)
				core_tensors = []
				mask_tensors = []
				for (y0_core, x0_core, h_core, w_core) in out["core_boxes"]:
					core_tensors.append(target_ds[:, :, y0_core:y0_core+h_core, x0_core:x0_core+w_core])
					mask_tensors.append(valid_ds[:, :, y0_core:y0_core+h_core, x0_core:x0_core+w_core])
				target_core = torch.cat(core_tensors, dim=0)
				valid_core = torch.cat(mask_tensors, dim=0)
				#
				mask_tiles = out["mask_tiles"]
				mask_tiles_o = opening(mask_tiles, morph_kernel_opt_opening, engine="convolution")
				mask_tiles_c = closing(mask_tiles, morph_kernel_opt_closing, engine="convolution")

				# weighted losses (ignore padded areas)
				loss_l2 = torch.sum(((out["outer"] - target_core) * valid_core) ** 2)
				loss_pvb = torch.sum(((out["inner"] - out["outer"]) * valid_core) ** 2)
				loss_mrc = torch.sum(((mask_tiles_o - mask_tiles)) ** 2) + torch.sum(((mask_tiles_c - mask_tiles)) ** 2)
				loss = loss_l2 + loss_pvb #+ 0.0001*loss_mrc
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				# Blend gradients across overlapping tiles with a Hann window to encourage seamless mask updates
				if model.mask_s.grad is not None:
					g = model.mask_s.grad
					weight_accum = torch.zeros_like(g)
					for t in batch_tiles:
						h = t.y1 - t.y0
						w = t.x1 - t.x0
						win_np = hann_window_2d(h, w)
						win = torch.from_numpy(win_np).to(g.device, dtype=g.dtype).view(1,1,h,w)
						# accumulate sum of squared Hann weights
						weight_accum[:, :, t.y0:t.y1, t.x0:t.x1] += 1.0#win * win
					mask = weight_accum > 0
					g_blend = g.clone()
					# normalize by sqrt of summed squared weights; clamp only to avoid divide-by-zero
					g_blend[mask] = g[mask] / torch.sqrt(torch.clamp(weight_accum[mask], min=1e-8))
					model.mask_s.grad.copy_(g_blend)
				scaler.step(optimizer)
				scaler.update()
				loss_total += loss.item()
		if (it+1) % 10 == 0:
			print(f"{datetime.now()}: iter {it+1}, loss {loss_total:.3f}")

	end = time.time()
	print(f"Total time: {end-start:.2f}s")

	# Save final mask (upsampled + thresholded)
	with torch.no_grad():
		# 1) Save optimized mask
		mask_up = nn.functional.interpolate(input=model.avepool(model.mask_s).data, scale_factor=model.scale_factor, mode='bicubic', align_corners=False, antialias=True)
		mask_bin = (mask_up > 0.5).float()
		mask_img = (mask_bin[0,0].cpu().numpy() * 255).astype(np.uint8)
		mask_path = args.image + f".fullchip_mask_sf{args.scale_factor}.png" if args.save is None else args.save
		cv2.imwrite(mask_path, mask_img)
		print(f"Saved mask to {mask_path}")

		# 2) Clean up mask with tiled morphological operations
		print("Cleaning up mask with tiled morphological operations...")
		H_p, W_p = mask_bin.shape[2], mask_bin.shape[3]
		clean_canvas = torch.zeros_like(mask_bin)
		tile_size = 2048
		core_size = 2000
		halo = (tile_size - core_size) // 2
		
		y_starts = list(range(0, H_p, core_size))
		x_starts = list(range(0, W_p, core_size))
		clean_tiles_spec = []
		for y_core_abs in y_starts:
			for x_core_abs in x_starts:
				y0 = max(0, min(y_core_abs - halo, H_p - tile_size))
				x0 = max(0, min(x_core_abs - halo, W_p - tile_size))
				
				if y0 + tile_size > H_p: y0 = H_p - tile_size
				if x0 + tile_size > W_p: x0 = W_p - tile_size
				y0 = max(0, y0)
				x0 = max(0, x0)

				y1 = y0 + tile_size
				x1 = x0 + tile_size

				cy0 = y_core_abs - y0
				cx0 = x_core_abs - x0
				
				h_core = min(core_size, H_p - y_core_abs)
				w_core = min(core_size, W_p - x_core_abs)
				if h_core <= 0 or w_core <= 0: continue
				
				cy1 = cy0 + h_core
				cx1 = cx0 + w_core
				
				clean_tiles_spec.append((y0, y1, x0, x1, cy0, cy1, cx0, cx1, y_core_abs, x_core_abs, h_core, w_core))

		batch_size_clean = 2
		for b in range(0, len(clean_tiles_spec), batch_size_clean):
			batch = clean_tiles_spec[b:b+batch_size_clean]
			
			tile_tensors = [mask_bin[:, :, y0:y1, x0:x1] for y0, y1, x0, x1, _, _, _, _, _, _, _, _ in batch]
			input_tiles = torch.cat(tile_tensors, dim=0)

			tile_open = opening(input_tiles, morph_kernel_opening, engine="convolution")
			tile_close = closing(input_tiles, morph_kernel_closing, engine="convolution")
			cleaned_tiles = tile_open + tile_close - input_tiles
			cleaned_tiles = opening(cleaned_tiles, morph_kernel_opening, engine="convolution")
			cleaned_tiles = closing(cleaned_tiles, morph_kernel_closing, engine="convolution")
			
			for i, spec in enumerate(batch):
				_, _, _, _, cy0, cy1, cx0, cx1, y_core_abs, x_core_abs, h_core, w_core = spec
				core = cleaned_tiles[i:i+1, :, cy0:cy1, cx0:cx1]
				clean_canvas[:, :, y_core_abs:y_core_abs+h_core, x_core_abs:x_core_abs+w_core] = core
		
		mask_clean = clean_canvas
		mask_clean_img = (mask_clean[0,0].cpu().numpy() * 255).astype(np.uint8)
		if args.save is None:
			mask_clean_path = args.image + f".fullchip_mask_sf{args.scale_factor}_clean.png"
		else:
			base, ext = os.path.splitext(args.save)
			mask_clean_path = base + "_clean" + ext
		cv2.imwrite(mask_clean_path, mask_clean_img)
		print(f"Saved cleaned mask to {mask_clean_path}")
		
		# 3) Assemble simulated outputs using the cleaned mask
		print("Performing litho simulation on the cleaned mask at original resolution.")

		H_full, W_full = mask_clean.shape[2], mask_clean.shape[3]
		canvas_nom = np.zeros((H_full, W_full), dtype=np.float32)
		canvas_out = np.zeros((H_full, W_full), dtype=np.float32)
		canvas_inn = np.zeros((H_full, W_full), dtype=np.float32)
		canvas_x_out = np.zeros((H_full, W_full), dtype=np.float32)
		canvas_x_out_min = np.zeros((H_full, W_full), dtype=np.float32)
		canvas_x_out_max = np.zeros((H_full, W_full), dtype=np.float32)

		tile_size = 2048
		core_size = 1024
		halo = (tile_size - core_size) // 2

		y_starts = list(range(0, H_full, core_size))
		x_starts = list(range(0, W_full, core_size))
		sim_tiles_spec = []
		for y_core_abs in y_starts:
			for x_core_abs in x_starts:
				y0 = max(0, min(y_core_abs - halo, H_full - tile_size))
				x0 = max(0, min(x_core_abs - halo, W_full - tile_size))
				
				if y0 + tile_size > H_full: y0 = H_full - tile_size
				if x0 + tile_size > W_full: x0 = W_full - tile_size
				y0 = max(0, y0)
				x0 = max(0, x0)

				y1 = y0 + tile_size
				x1 = x0 + tile_size

				cy0 = y_core_abs - y0
				cx0 = x_core_abs - x0

				h_core = min(core_size, H_full - y_core_abs)
				w_core = min(core_size, W_full - x_core_abs)
				if h_core <= 0 or w_core <= 0: continue

				cy1 = cy0 + h_core
				cx1 = cx0 + w_core
				
				sim_tiles_spec.append((y0, y1, x0, x1, cy0, cy1, cx0, cx1, y_core_abs, x_core_abs, h_core, w_core))

		batch_size_sim = 4
		for b in range(0, len(sim_tiles_spec), batch_size_sim):
			batch = sim_tiles_spec[b:b+batch_size_sim]
			
			tile_tensors = [mask_clean[:, :, y0:y1, x0:x1] for y0, y1, x0, x1, _, _, _, _, _, _, _, _ in batch]
			input_tiles = torch.cat(tile_tensors, dim=0)
			
			nominal, inner, outer, x_out, x_out_min, x_out_max = model.run_litho_sim(input_tiles)
			
			for i, spec in enumerate(batch):
				_, _, _, _, cy0, cy1, cx0, cx1, y_core_abs, x_core_abs, h_core, w_core = spec
				
				nom_core = nominal[i:i+1, :, cy0:cy1, cx0:cx1]
				out_core = outer[i:i+1, :, cy0:cy1, cx0:cx1]
				inn_core = inner[i:i+1, :, cy0:cy1, cx0:cx1]
				x_out_core = x_out[i:i+1, :, cy0:cy1, cx0:cx1]
				x_out_min_core = x_out_min[i:i+1, :, cy0:cy1, cx0:cx1]
				x_out_max_core = x_out_max[i:i+1, :, cy0:cy1, cx0:cx1]
				
				canvas_nom[y_core_abs:y_core_abs+h_core, x_core_abs:x_core_abs+w_core] = nom_core.squeeze().cpu().numpy()
				canvas_out[y_core_abs:y_core_abs+h_core, x_core_abs:x_core_abs+w_core] = out_core.squeeze().cpu().numpy()
				canvas_inn[y_core_abs:y_core_abs+h_core, x_core_abs:x_core_abs+w_core] = inn_core.squeeze().cpu().numpy()

		
		# Crop padded regions
		if pad > 0:
			canvas_nom = canvas_nom[pad:pad+H, pad:pad+W]
			canvas_out = canvas_out[pad:pad+H, pad:pad+W]
			canvas_inn = canvas_inn[pad:pad+H, pad:pad+W]


		# Binarize outputs
		canvas_nom_bin = (canvas_nom > 0.5).astype(np.float32)
		canvas_out_bin = (canvas_out > 0.5).astype(np.float32)
		canvas_inn_bin = (canvas_inn > 0.5).astype(np.float32)

		print(np.sum(canvas_nom), np.sum(canvas_out), np.sum(canvas_inn))

		# Save stitched outputs
		def save_image(arr: np.ndarray, suffix: str):
			png = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
			if args.sim_dir is not None:
				os.makedirs(args.sim_dir, exist_ok=True)
				base = os.path.basename(args.image)
				out_path = os.path.join(args.sim_dir, f"{base}.{suffix}_sf{args.scale_factor}.png")
			else:
				out_path = args.image + f".{suffix}_sf{args.scale_factor}.png"
			cv2.imwrite(out_path, png)
			print(f"Saved {suffix} to {out_path}")

		save_image(canvas_nom_bin, "stitched_nominal_clean")
		save_image(canvas_out_bin, "stitched_outer_clean")
		save_image(canvas_inn_bin, "stitched_inner_clean")
  
		# Final evaluation
		print("Performing final evaluation...")
		target_full_cropped = target_full
		if pad > 0:
			target_full_cropped = target_full[:, :, pad:pad+H, pad:pad+W]
		
		mask_clean_cropped = mask_clean
		if pad > 0:
			mask_clean_cropped = mask_clean[:, :, pad:pad+H, pad:pad+W]

		results = evaluation(
			mask_clean_cropped,
			target_full_cropped,
			torch.from_numpy(canvas_nom_bin).unsqueeze(0).unsqueeze(0).cuda(),
			torch.from_numpy(canvas_inn_bin).unsqueeze(0).unsqueeze(0).cuda(),
			torch.from_numpy(canvas_out_bin).unsqueeze(0).unsqueeze(0).cuda()
		)
		print(f"Final evaluation results: L2={results.get_l2()}, PVB={results.get_pvb()}, EPE={results.get_epe()}")


if __name__ == "__main__":
	main()


