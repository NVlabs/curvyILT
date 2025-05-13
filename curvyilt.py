#############################
#NVIDIA  All Rights Reserved
#Haoyu Yang 
#Design Automation Research
#Last Update: May 13 2025
#############################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from eval_util import *
import os
import math  
import pdb
from kornia.morphology import opening, closing, dilation, erosion




def get_kernels():
    # Load and process kernel_head
    kernel_head = np.load("./tcc/optKernel_bc.npy")
    kernel_head = kernel_head[:, :24]

    # Load and process kernel_scale
    kernel_scale = np.load("./tcc/optKernel_scale.npy")
    kernel_scale = kernel_scale[:, :24]
    a, b = kernel_scale.shape
    kernel_scale = kernel_scale.reshape(a, b, 1, 1)

    # Extract focus and defocus components
    kernels_fft_focus = kernel_head[0]
    kernels_fft_defocus = kernel_head[1]
    kernels_scale_focus = kernel_scale[0]
    kernels_scale_defocus = kernel_scale[1]

    return kernels_fft_focus, kernels_fft_defocus, kernels_scale_focus, kernels_scale_defocus




class LpLoss(object): 
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        pseudo_y_norms = torch.ones(num_examples).type('torch.cuda.FloatTensor') *250000.0
        y_norms = torch.where(y_norms==0,pseudo_y_norms,y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class litho(nn.Module):
    def __init__(self, target_path, mask_steepness=4, resist_th =0.225, resist_steepness=50, mask_shift=0.5, pvb_coefficient=0, max_dose=1.02, min_dose=0.98, avepool_kernel=3, morph=0, scale_factor=8, pixel_size=1, max_iter=None):
        super(litho, self).__init__()
        self.target_image = torch.tensor(cv2.imread(target_path, -1))/255.0 

        self.mask_dim1, self.mask_dim2 = self.target_image.shape
        self.target_image = self.target_image.view(1,1,self.mask_dim1,self.mask_dim2).cuda()
        self.mask = nn.Parameter(self.target_image)    
        self.fo, self.defo, self.fo_scale, self.defo_scale = get_kernels()

        self.kernel_focus = torch.tensor(self.fo).cuda()
        self.kernel_focus_scale = torch.tensor(self.fo_scale).cuda()
        self.kernel_defocus = torch.tensor(self.defo).cuda()
        self.kernel_defocus_scale = torch.tensor(self.defo_scale).cuda()


        self.kernel_num, self.kernel_dim1, self.kernel_dim2 = self.fo.shape # 24 35 35
        self.offset = self.mask_dim1//2-self.kernel_dim1//2
        self.max_dose = max_dose
        self.min_dose = min_dose
        self.resist_steepness = resist_steepness
        self.mask_steepness = mask_steepness
        self.resist_th = resist_th
        self.mask_shift = mask_shift
        self.scale_factor = scale_factor
        self.mask_dim1_s = self.mask_dim1//self.scale_factor
        self.mask_dim2_s = self.mask_dim2//self.scale_factor
        self.target_image_s = nn.functional.avg_pool2d(torch.tensor(cv2.imread(target_path, -1)).view(1,1,self.mask_dim1,self.mask_dim2).cuda()/255.0, self.scale_factor)
        self.mask_s      = nn.Parameter(self.target_image_s)
        self.avepool  = nn.AvgPool2d(kernel_size=avepool_kernel, stride=1, padding = avepool_kernel//2)
        self.offset_s = self.mask_dim1_s//2-self.kernel_dim1//2

        self.max_iter=max_iter
        #morphlogic for mask simplification
        self.morph = morph
        if self.morph>0:
            self.morph_kernel_opt_opening =torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph,morph)).astype(np.float32)).cuda()
            self.morph_kernel_opt_closing =torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph+2,morph+2)).astype(np.float32)).cuda()
            self.morph_kernel_opening = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph*scale_factor+1,morph*scale_factor+1)).astype(np.float32)).cuda()
            self.morph_kernel_closing = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((morph-2)*self.scale_factor+1,(morph-2)*self.scale_factor+1)).astype(np.float32)).cuda()
        self.iter=0

    def standalone_mask_morph(self):
        mask = self.avepool(self.mask_s)
        mask = torch.sigmoid(self.mask_steepness*(mask-self.mask_shift))
        mask_o = opening(mask, self.morph_kernel_opt_opening)
        mask_c = closing(mask, self.morph_kernel_opt_closing)
        mask = mask_o+mask_c-mask
        return mask

    def forward_test(self,use_morph=False):  
        mask = torch.zeros_like(self.target_image).cuda()
        cmask = self.mask.data
        #print(mask.shape, cmask.shape)
        mask[self.mask.data>=0.5]=1.0
        mask[self.mask.data<0.5]=0.0
        if self.morph>0 and use_morph:
            mask_o = opening(mask, self.morph_kernel_opening, engine="convolution")
            mask_c = closing(mask, self.morph_kernel_closing, engine="convolution")
            mask = mask_o+mask_c-mask
            mask = opening(mask, self.morph_kernel_opening, engine="convolution")
            mask = closing(mask, self.morph_kernel_closing, engine="convolution")

        mask_fft = torch.fft.fftshift(torch.fft.fft2(mask)) 
        mask_fft = torch.repeat_interleave(mask_fft, self.kernel_num, 1) 


        mask_fft_max = torch.fft.fftshift(torch.fft.fft2(mask*self.max_dose)) 
        mask_fft_max = torch.repeat_interleave(mask_fft_max, self.kernel_num, 1) 

        mask_fft_min = torch.fft.fftshift(torch.fft.fft2(mask*self.min_dose)) 
        mask_fft_min = torch.repeat_interleave(mask_fft_min, self.kernel_num, 1) 

        x_out = torch.view_as_complex(torch.zeros((1, self.kernel_num, self.mask_dim1, self.mask_dim2, 2), dtype=torch.float32).cuda())

        x_out_max = torch.view_as_complex(torch.zeros((1, self.kernel_num, self.mask_dim1, self.mask_dim2, 2), dtype=torch.float32).cuda())

        x_out_min = torch.view_as_complex(torch.zeros((1, self.kernel_num, self.mask_dim1, self.mask_dim2, 2), dtype=torch.float32).cuda())


        x_out[:, :, self.offset: self.offset+self.kernel_dim1, self.offset: self.offset+self.kernel_dim2] = mask_fft[:, :, self.offset: self.offset+self.kernel_dim1, self.offset: self.offset+self.kernel_dim2] * self.kernel_focus
        x_out = torch.fft.ifft2(x_out)
        x_out = x_out.real*x_out.real + x_out.imag*x_out.imag
        x_out = x_out * self.kernel_focus_scale
        x_out = torch.sum(x_out, axis=1, keepdims=True)
        self.aerial = x_out
        x_out = torch.sigmoid(self.resist_steepness*(x_out-self.resist_th))


        x_out_max[:, :, self.offset: self.offset+self.kernel_dim1, self.offset: self.offset+self.kernel_dim2] = mask_fft_max[:, :, self.offset: self.offset+self.kernel_dim1, self.offset: self.offset+self.kernel_dim2] * self.kernel_focus
        x_out_max = torch.fft.ifft2(x_out_max)
        x_out_max = x_out_max.real*x_out_max.real + x_out_max.imag*x_out_max.imag
        x_out_max = x_out_max * self.kernel_focus_scale
        x_out_max = torch.sum(x_out_max, axis=1, keepdims=True)
        x_out_max = torch.sigmoid(self.resist_steepness*(x_out_max-self.resist_th))

        x_out_min[:, :, self.offset: self.offset+self.kernel_dim1, self.offset: self.offset+self.kernel_dim2] = mask_fft_min[:, :, self.offset: self.offset+self.kernel_dim1, self.offset: self.offset+self.kernel_dim2] * self.kernel_defocus
        x_out_min = torch.fft.ifft2(x_out_min)
        x_out_min = x_out_min.real*x_out_min.real + x_out_min.imag*x_out_min.imag
        x_out_min = x_out_min * self.kernel_defocus_scale
        x_out_min = torch.sum(x_out_min, axis=1, keepdims=True)
        x_out_min = torch.sigmoid(self.resist_steepness*(x_out_min-self.resist_th))

        x_out[x_out>=0.5]=1.0
        x_out[x_out<0.5]=0.0
        x_out_max[x_out_max>=0.5]=1.0
        x_out_max[x_out_max<0.5]=0.0
        x_out_min[x_out_min>=0.5]=1.0
        x_out_min[x_out_min<0.5]=0.0


        return mask, cmask, x_out, x_out_max, x_out_min
    
    
    def forward(self,): 

        mask = self.avepool(self.mask_s) 

        mask = torch.sigmoid(self.mask_steepness*(mask-self.mask_shift))

        if self.morph>0 and self.iter % 2==0 and self.iter >50:
            mask_o = opening(mask, self.morph_kernel_opt_opening, engine="convolution")
            mask_c = closing(mask, self.morph_kernel_opt_closing, engine="convolution")
            mask = mask_o+mask_c-mask


        mask_fft = torch.fft.fftshift(torch.fft.fft2(mask)) 
        self.i_mask_fft = mask_fft
        mask_fft = torch.repeat_interleave(mask_fft, self.kernel_num, 1) 
        mask_fft_max = mask_fft*self.max_dose
        mask_fft_min = mask_fft*self.min_dose


        x_out = torch.view_as_complex(torch.zeros((1, self.kernel_num, self.mask_dim1_s, self.mask_dim2_s, 2), dtype=torch.float32).cuda())
        x_out_max = torch.view_as_complex(torch.zeros((1, self.kernel_num, self.mask_dim1_s, self.mask_dim2_s, 2), dtype=torch.float32).cuda())
        x_out_min = torch.view_as_complex(torch.zeros((1, self.kernel_num, self.mask_dim1_s, self.mask_dim2_s, 2), dtype=torch.float32).cuda())


        x_out[:, :, self.offset_s: self.offset_s+self.kernel_dim1, self.offset_s: self.offset_s+self.kernel_dim2] = mask_fft[:, :, self.offset_s: self.offset_s+self.kernel_dim1, self.offset_s: self.offset_s+self.kernel_dim2] * self.kernel_focus
        x_out = torch.fft.ifft2(x_out)
        x_out = x_out.real*x_out.real + x_out.imag*x_out.imag
        x_out = x_out * self.kernel_focus_scale
        x_out = torch.sum(x_out, axis=1, keepdims=True)
        x_out = torch.sigmoid(self.resist_steepness*(x_out-self.resist_th))


        x_out_max[:, :, self.offset_s: self.offset_s+self.kernel_dim1, self.offset_s: self.offset_s+self.kernel_dim2] = mask_fft_max[:, :, self.offset_s: self.offset_s+self.kernel_dim1, self.offset_s: self.offset_s+self.kernel_dim2] * self.kernel_focus
        x_out_max = torch.fft.ifft2(x_out_max)
        x_out_max = x_out_max.real*x_out_max.real + x_out_max.imag*x_out_max.imag
        x_out_max = x_out_max * self.kernel_focus_scale
        x_out_max = torch.sum(x_out_max, axis=1, keepdims=True)
        x_out_max = torch.sigmoid(self.resist_steepness*(x_out_max-self.resist_th))

        x_out_min[:, :, self.offset_s: self.offset_s+self.kernel_dim1, self.offset_s: self.offset_s+self.kernel_dim2] = mask_fft_min[:, :, self.offset_s: self.offset_s+self.kernel_dim1, self.offset_s: self.offset_s+self.kernel_dim2] * self.kernel_defocus
        x_out_min = torch.fft.ifft2(x_out_min)
        x_out_min = x_out_min.real*x_out_min.real + x_out_min.imag*x_out_min.imag
        x_out_min = x_out_min * self.kernel_defocus_scale
        x_out_min = torch.sum(x_out_min, axis=1, keepdims=True)
        x_out_min = torch.sigmoid(self.resist_steepness*(x_out_min-self.resist_th))


        return x_out, x_out_max, x_out_min


class solver():
    def __init__(self, image_path, avepool_kernel=5,morph=0,scale_factor=1,pixel_size=1,all_tech=0):
        
        self.image_path = image_path
        self.litho = litho(target_path=image_path, avepool_kernel=avepool_kernel, morph=morph,scale_factor=scale_factor,pixel_size=pixel_size).to("cuda:0")
        self.optimizer = torch.optim.SGD(self.litho.parameters(), lr=1)
        #self.optimizer = Prodigy(self.litho.parameters(), lr=1, weight_decay=0)
        self.iteration = 0
        self.target = torch.tensor(cv2.imread(image_path, -1))/255.0 
        self.target_s = torch.tensor(cv2.imread(image_path, -1))/255.0
        self.mask_dim1, self.mask_dim2 = self.target.shape
        self.target = self.target.view(1,1,self.mask_dim1,self.mask_dim2).cuda()
        self.mask_dim1_s = self.mask_dim1//self.litho.scale_factor
        self.mask_dim2_s = self.mask_dim2//self.litho.scale_factor
        self.target_s = nn.functional.avg_pool2d(self.target_s.view(1,1,self.mask_dim1,self.mask_dim2).cuda(), self.litho.scale_factor)
        self.lowpass = 2
        self.lowpass_mask = torch.ones(1,1,self.mask_dim1_s, self.mask_dim2_s).cuda()
        self.lowpass_offset = self.mask_dim1_s//2-self.lowpass//2
        self.lowpass_mask[:,:,self.lowpass_offset:self.lowpass_offset+self.lowpass, self.lowpass_offset:self.lowpass_offset+self.lowpass]=0
        self.loss =0
        self.loss_l2=0
        self.loss_pvb=0
        self.loss_pvb_i =0
        self.loss_pvb_o =0
        self.loss_pvb = 0
        self.loss_lowpass_reg=0
        self.lowpass_reg_lambda = 1e-3

    def backward(self):

        self.loss.backward()



    def forward(self):

        self.mask = self.litho.mask_s.data
        
        self.nominal, self.outer, self.inner = self.litho.forward()
        self.i_mask_fft = self.litho.i_mask_fft
        self.iteration = self.iteration + 1
        self.litho.iter = self.iteration

    def optimize(self):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.forward()
            self.loss_l2 = l2_loss(self.outer, self.target_s)
            self.loss_pvb = l2_loss(self.inner, self.outer) 
            self.loss_lowpass_reg = torch.norm(torch.abs(self.i_mask_fft*self.lowpass_mask))
            self.loss = self.loss_l2 + self.loss_pvb + self.lowpass_reg_lambda*self.loss_lowpass_reg
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


        

def l2_loss(x, y):
    return torch.sum(torch.pow((x - y), 2))


class evaluation():
    def __init__(self, mask, target, nominal, inner, outer):
        self.mask = mask
        self.target = target
        self.nominal=nominal
        self.inner=inner
        self.outer=outer 
 

    def get_l2(self):
        return  torch.sum(torch.abs(self.nominal - self.target)).cpu().numpy()
    
    def get_pvb(self):

        pvb = torch.zeros_like(self.outer).cuda()
        pvb[self.outer==1.0]=1
        pvb[self.inner==1.0]=0
        pvb=torch.sum(pvb)

        return pvb.cpu().numpy()
    
    def get_epe(self):
        epe=0
        print("EPE Checker Not implemented, please use the utility from neuralILT or implement it by yourself")

        return epe
    
    def get_msa(self):
        mask_numpy = self.mask.detach().cpu().numpy()[0,0,:,:] 
        msa = min_area_check(mask_numpy)

        return msa
    def get_msd(self):
        mask_numpy = self.mask.detach().cpu().numpy()[0,0,:,:]
        msd = min_dist_check(mask_numpy)

        return msd







        

def corner_smooth(im, kernel=45):
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel,kernel))
    b=cv2.dilate(im,k)
    b=cv2.erode(b,k,iterations=2)
    b=cv2.dilate(b,k)
    return b


def morph_close_cv2(im, kernel=10):
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel,kernel))
    b=cv2.dilate(im,k)
    b=cv2.erode(b,k)
    return b 

def morph_open_cv2(im, kernel=20):
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel,kernel))
    b=cv2.erode(im,k)
    b=cv2.dilate(b,k)
    return b 

#corner retargeting


def _corner_index(image, kernel_convex=20, kernel_concave=None,output_filename=None):
    if kernel_concave==None:
        kernel_concave=kernel_convex
    image_open = morph_open_cv2(image,kernel_convex)
    image_close= morph_close_cv2(image,kernel_concave)
    return image_close-image_open



def corner_retargeting_morph(image, kernel_convex=20, kernel_concave=None,output_filename=None):
    if kernel_concave==None:
        kernel_concave=kernel_convex
    image_open = morph_open_cv2(image,kernel_convex)
    image_close= morph_close_cv2(image,kernel_concave)
    image= image_open + image_close - image
    if output_filename:
        cv2.imwrite(output_filename, image * 255)  # Convert to 0-255 scale for PNG
    return image

if __name__=="__main__":
    path="./benchmarks/M1_test2/M1_test2.png"
    kernel=51
    outpath="./benchmarks/M1_test2/M1_test2_r%g.png"%kernel
    aa=cv2.imread(path,-1)/255.0
    #bb=round_manhattan_corners(image=aa,kernel_size=kernel,output_filename=outpath)
    bb=corner_retargeting_morph(image=aa,kernel=kernel,output_filename=outpath)