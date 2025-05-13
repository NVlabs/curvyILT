#############################
#NVIDIA  All Rights Reserved
#Haoyu Yang 
#Design Automation Research
#Last Update: Aug 25 2024
#############################
import curvyilt
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn

if __name__=="__main__":
    retarget=1
    morph=3 #3
    result = np.zeros((11,7))
    #id = 2
    iters=np.array([300,200,500,200,200,200,200,200,200,200])
    for id in range(1,11):

        solver = curvyilt.solver(image_path='./benchmarks/M1_test%g/M1_test%g.png'%(id,id),morph = morph, scale_factor=8)


        if retarget:
            target = cv2.imread('./benchmarks/M1_test%g/M1_test%g.png'%(id,id), -1)/255.0
            kernel_convex=39
            kernel_concave=39
            outpath="./benchmarks/M1_test%g/M1_test2_r_x%g_a%g.png"%(id,kernel_convex,kernel_concave)
            sm_target = curvyilt.corner_retargeting_morph(target, kernel_convex, kernel_concave, outpath)
            solver.target_s = torch.tensor(sm_target)
            solver.target_s = nn.functional.avg_pool2d(solver.target_s.view(1,1,solver.mask_dim1,solver.mask_dim2).cuda(), solver.litho.scale_factor)
        
        



        s=datetime.now()

        for iter in range(iters[id-1]):
            solver.optimize()
            #print("%s: iter %g: loss: %.3f, loss_l2: %.3f, loss_pvb: %.3f + %.3f"%(datetime.now(), solver.iteration, solver.loss, solver.loss_l2, solver.loss_pvb_i, solver.loss_pvb_o))

        e=datetime.now()
   

        with torch.no_grad():
            if False:
                solver.litho.mask_s.data=solver.litho.standalone_mask_morph()
                solver.litho.mask.data = nn.functional.interpolate(input=solver.litho.avepool(solver.litho.mask_s).data, scale_factor=solver.litho.scale_factor, mode = 'bicubic', align_corners=False, antialias=True)
            else:
                solver.litho.mask.data = nn.functional.interpolate(input=solver.litho.avepool(solver.litho.mask_s).data, scale_factor=solver.litho.scale_factor, mode = 'bicubic', align_corners=False, antialias=True)

            mask, cmask, x_out, x_out_max, x_out_min = solver.litho.forward_test(use_morph=True)
            
            results = curvyilt.evaluation(mask, solver.target, x_out, x_out_min, x_out_max) 
            l2 =results.get_l2()
            pvb=results.get_pvb()
            epe=results.get_epe()
            msa=results.get_msa()
            msd=results.get_msd()
            result[id-1,0:6]=[l2,pvb,epe,msa,msd,(e-s).total_seconds()]
            print("Design: %g, runtime is: %s, final L2 is: %g, final PVB is: %g, final EPE is: %g, msa is %g, msd is %g"%(id, e-s,l2, pvb, epe, msa, msd))

            final_image = torch.cat((solver.target, mask, solver.litho.aerial, x_out), dim=3).cpu().detach().numpy()[0,0,:,:]*255

        
            if True:
                mm = mask.cpu().detach().numpy()[0,0,:,:]*255
                aa = solver.litho.aerial.cpu().detach().numpy()[0,0,:,:]*255
                zz = x_out.cpu().detach().numpy()[0,0,:,:]*255
                cv2.imwrite(solver.image_path+".mask_retarget_%g_morph_%g.png"%(retarget,morph), mm)
                cv2.imwrite(solver.image_path+".aerial_retarget_%g_morph_%g.png"%(retarget,morph), aa)
                cv2.imwrite(solver.image_path+".resist_retarget_%g_morph_%g.png"%(retarget,morph), zz)



            cv2.imwrite(solver.image_path+".final_retarget_%g_morph_%g.png"%(retarget,morph), final_image)
        

    result[-1]=np.mean(result[:-1], axis=0)
    pd.DataFrame(result).to_csv('./benchmarks/result_default_retarget_%g_morph_%g.csv'%(retarget,morph))