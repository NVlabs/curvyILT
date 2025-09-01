#############################
#NVIDIA  All Rights Reserved
#Haoyu Yang 
#Design Automation Research
#Last Update: Aug 14 2025
#############################
import curvyilt
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import time 

if __name__=="__main__":
    retarget=1
    morph=3 #3
    result = np.zeros((11,7))
    #id = 2
    iters=200
    images = []
    for id in range(1, 11):
        image_path = './benchmarks/M1_test%g/M1_test%g.png' % (id, id)
        img = cv2.imread(image_path, -1) / 255.0
        images.append(img)
    
    target_images_np = np.stack(images, axis=0)
    target_images_np = np.expand_dims(target_images_np, axis=1)
    target_images = torch.from_numpy(target_images_np).float().cuda()
    

    
    solver = curvyilt.solver_batch(target_images=target_images, morph = morph, scale_factor=8)
    mask_init = nn.functional.avg_pool2d(target_images, solver.litho.scale_factor)
    if retarget:
        outpath = None
        kernel_convex=39
        kernel_concave=39
        sm_target = curvyilt.corner_retargeting_morph_2(target_images, kernel_convex, kernel_concave, outpath)
        solver.target_s = torch.tensor(sm_target)
        solver.target_s = nn.functional.avg_pool2d(solver.target_s, solver.litho.scale_factor)

    solver.litho.mask_s.data = mask_init.detach()
    start_time = time.time()
    for iter in range(iters):
        solver.optimize()
        print("Iteration %d, loss: %.3f"%(iter, solver.loss))
    end_time = time.time()
    print("Total time: %.3f seconds"%(end_time - start_time))

    with torch.no_grad():

        masks = nn.functional.interpolate(input=solver.litho.avepool(solver.litho.mask_s).data, scale_factor=solver.litho.scale_factor, mode = 'bicubic', align_corners=False, antialias=True)
        masks = (masks > 0.5).float()
        
        n, c, h, w = masks.shape
        # Arrange images in a 2x5 grid
        grid_img = masks.cpu().numpy().squeeze(1).reshape(2, 5, h, w).transpose(0, 2, 1, 3).reshape(2 * h, 5 * w)
        
        cv2.imwrite('./benchmarks/batch_masks_retarget_%g_morph_%g.png'%(retarget,morph), grid_img * 255)
        

