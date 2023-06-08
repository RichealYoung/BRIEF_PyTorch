import sys
from matplotlib import pyplot as plt
import omegaconf.listconfig
import pandas as pd
from tqdm import tqdm
import math
import os
from typing import Callable, List, Tuple,Dict, Union
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from einops import rearrange,repeat
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from utils.dataset import create_flattened_coords
from utils.io import *
from utils.ssim import ssim as ssim_calc
from utils.ssim import ms_ssim as ms_ssim_calc
from copy import deepcopy
from scipy import ndimage
from utils.tool import get_type_max, range_limit
from utils.adaptive_blocking import cal_feature
import copy
import cv2

def omegaconf2list(opt,prefix='',sep = '.'):
    notation_list = []
    for k, v in opt.items():
        k = str(k)
        if isinstance(v, omegaconf.listconfig.ListConfig):
            notation_list.append("{}{}={}".format(prefix,k,v))
            # if k in ['iter_list','step_list']: # do not sparse list
            #     dot_notation_list.append("{}{}={}".format(prefix,k,v))
            # else:
            #     templist = []
            #     for v_ in v:
            #         templist.append('{}{}={}'.format(prefix,k,v_))
            #     dot_notation_list.append(templist)   
        elif isinstance(v,(float,str,int,)):
            notation_list.append("{}{}={}".format(prefix,k,v))
        elif v is None:
            notation_list.append("{}{}=~".format(prefix,k,))
        elif isinstance(v, omegaconf.dictconfig.DictConfig):
            nested_flat_list = omegaconf2list(v, prefix + k + sep,sep=sep)
            if nested_flat_list:
                notation_list.extend(nested_flat_list)
        else:
            raise NotImplementedError
    return notation_list
def omegaconf2dotlist(opt,prefix='',):
    return omegaconf2list(opt,prefix,sep='.')
def omegaconf2dict(opt,sep):
    notation_list = omegaconf2list(opt,sep=sep)
    dict = {notation.split('=', maxsplit=1)[0]:notation.split('=', maxsplit=1)[1] for notation in notation_list}
    return dict
def reconstruct_flattened(data_shape:tuple,sample_size:int,sample_nf:Callable,device:str='cpu',half:bool=False,coords_mode:str='-1,1') -> torch.Tensor:
    """decompress
    """
    *coords_shape, data_channel = data_shape
    # sample
    with torch.no_grad():
        if len(coords_shape) == 2:
            h,w = coords_shape
            pop_size = h*w
            coords = create_flattened_coords((h,w),coords_mode).to(device)
            flattened_data = torch.zeros((pop_size,data_channel),device=device)
            if half:
                flattened_data = flattened_data.half()
        elif len(coords_shape) == 3:
            d,h,w = coords_shape
            pop_size = d*h*w
            coords = create_flattened_coords((d,h,w),coords_mode).to(device)
            flattened_data = torch.zeros((pop_size,data_channel),device=device)
            if half:
                flattened_data = flattened_data.half()
        else:
            raise NotImplementedError
        for index in tqdm(range(math.ceil(pop_size/sample_size)),'Decompressing',leave=False,file=sys.stdout):
            start_idx = index*sample_size
            end_idx = min(start_idx+sample_size,pop_size)
            sampled_coords = coords[start_idx:end_idx,:]
            if half:
                sampled_coords = sampled_coords.half()
            flattened_data[start_idx:end_idx,:] = sample_nf(sampled_coords)
        if len(coords_shape) == 2:
            data = rearrange(flattened_data,'(h w) c -> h w c',h=h,w=w)   
        elif len(coords_shape) == 3:
            data = rearrange(flattened_data,'(d h w) c -> d h w c',d=d,h=h,w=w)   
    return data
def reconstruct_cropped(data_shape:tuple,sample_size:int,mods:List[torch.Tensor],sample_nf:Callable,ps_h:int,ps_w:int,ol_h:int,ol_w:int,ps_d:int=None,ol_d:int=None,device:str='cpu') -> torch.Tensor:
    """decompress
    """
    batch_size,data_channel,*coords_shape = data_shape
    # sample
    with torch.no_grad():
        if len(coords_shape) == 2:
            return NotImplementedError
        elif len(coords_shape) == 3:
            d,h,w = coords_shape
            pc_d = math.ceil((d-ol_d)/(ps_d-ol_d))
            pc_h = math.ceil((h-ol_h)/(ps_h-ol_h))
            pc_w = math.ceil((w-ol_w)/(ps_w-ol_w))
            pop_size = ps_d*ps_h*ps_w
            coords = create_flattened_coords((ps_d,ps_h,ps_w)).to(device)
            coords = repeat(coords,'pop c -> n pc_d pc_h pc_w pop c',n=batch_size,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w)
            cropped_data = torch.zeros((batch_size,pc_d,pc_h,pc_w,pop_size,data_channel),device=device)
        else:
            raise NotImplementedError
        for index in tqdm(range(math.ceil(pop_size/sample_size)),'Decompressing',leave=False,file=sys.stdout):
            start_idx = index*sample_size
            end_idx = min(start_idx+sample_size,pop_size)
            sampled_coords = coords[...,start_idx:end_idx,:]
            cropped_data[...,start_idx:end_idx,:] = sample_nf(sampled_coords,mods)
        cropped_data = cropped_data.cpu()
    # merge
    if len(coords_shape) == 2:
        return NotImplementedError
    elif len(coords_shape) == 3:
        cropped_data = rearrange(cropped_data,'n pc_d pc_h pc_w (ps_d ps_h ps_w) c -> n pc_d pc_h pc_w c ps_d ps_h ps_w',n = batch_size,ps_d = ps_d,ps_h = ps_h,ps_w = ps_w)
        # merge 
        data = torch.zeros((batch_size,data_channel,*coords_shape))
        weights = torch.zeros((batch_size,data_channel,*coords_shape))
        #FIXME
        weights_patch = torch.zeros((batch_size,data_channel,ps_d,ps_h,ps_w))
        center_idx = (ps_d//2,ps_h//2,ps_w//2)
        for d_idx in range(pc_d):
            for h_idx in range(pc_h):
                for w_idx in range(pc_w):
                    weights_patch[...,d_idx,h_idx,w_idx] = math.sqrt((d_idx-center_idx[0])**2+(h_idx-center_idx[1])**2+(w_idx-center_idx[2])**2)
        weights_patch = torch.abs(weights_patch-weights_patch.max())+1
        for pc_d_idx in range(pc_d):
            if pc_d_idx == 0:
                data_d_start = 0
            elif pc_d_idx == pc_d-1:
                data_d_start = d-ps_d
            else:
                data_d_start = pc_d_idx*(ps_d-ol_d)
            for pc_h_idx in range(pc_h):
                if pc_h_idx == 0:
                    data_h_start = 0
                elif pc_h_idx == pc_h-1:
                    data_h_start = h-ps_h
                else:
                    data_h_start = pc_h_idx*(ps_h-ol_h)
                for pc_w_idx in range(pc_w):
                    if pc_w_idx == 0:
                        data_w_start = 0
                    elif pc_w_idx == pc_w-1:
                        data_w_start = w-ps_w
                    else:
                        data_w_start = pc_w_idx*(ps_w-ol_w)
                    data[...,data_d_start:data_d_start+ps_d,data_h_start:data_h_start+ps_h,data_w_start:data_w_start+ps_w] += cropped_data[:,pc_d_idx,pc_h_idx,pc_w_idx,...]*weights_patch
                    weights[...,data_d_start:data_d_start+ps_d,data_h_start:data_h_start+ps_h,data_w_start:data_w_start+ps_w] += weights_patch
        data = data/weights
    else:
        raise NotImplementedError
    return data
def loss_bpp_func(likelihoods:torch.Tensor) -> torch.Tensor:
    """bits per pixel
    """
    if len(likelihoods.shape) == 5:
        n,c,d,h,w = likelihoods.shape
        num_pixels = d*h*w*n
    elif len(likelihoods.shape) == 4:
        n,c,h,w = likelihoods.shape
        num_pixels = h*w*n
    else:
        raise NotImplementedError
    loss_bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    return loss_bpp
def configure_optimizer(parameters,optimizer:str,lr:float) -> torch.optim.Optimizer:
    if optimizer == 'Adam':
        Optimizer = torch.optim.Adam(parameters,lr=lr)
    elif optimizer == 'Adamax':
        Optimizer = torch.optim.Adamax(parameters,lr=lr)
    elif optimizer == 'SGD':
        Optimizer = torch.optim.SGD(parameters,lr=lr)
    else:
        raise NotImplementedError
    return Optimizer
def configure_lr_scheduler(optimizer,lr_scheduler_opt):
    lr_scheduler_opt = deepcopy(lr_scheduler_opt)
    lr_scheduler_name = lr_scheduler_opt.pop('name')
    if lr_scheduler_name == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'none':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000000000])
    else:
        raise NotImplementedError
    return lr_scheduler
def gradient_descent(loss:torch.Tensor,optimizer_list:List[torch.optim.Optimizer]):
    for optimizer in optimizer_list:
        optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizer_list:
        optimizer.step()
def init_y(batch_size:int,y_channel:int,pc_shape:tuple,device:str='cpu') -> torch.Tensor:
    y = nn.Parameter(torch.empty((batch_size,y_channel,*pc_shape),device=device))
    nn.init.xavier_normal_(y,10000)
    return y
def init_z(batch_size:int,z_channel:int,pc_shape:tuple,device:str='cpu') -> torch.Tensor:
    z = nn.Parameter(torch.empty((batch_size,z_channel,*pc_shape),device=device))
    nn.init.xavier_normal_(z,10000)
    return z
def annealed_temperature(t:int, r:float, ub:float, lb:float=1e-8, scheme:str='exp', t0:int=700):
    """Return the temperature at time step t, based on a chosen annealing schedule.
    Args:
        t (int): step/iteration number
        r (float): decay strength
        ub (float): maximum/init temperature
        lb (float, optional): small const like 1e-8 to prevent numerical issue when temperature gets too close to 0
        scheme (str, optional): [description]. Defaults to 'exp'.
        t0 (int, optional): [description]. Defaults to 700.fixes temperature at ub for initial t0 iterations
    """
    if scheme == 'exp':
        tau = math.exp(-r * t)
    elif scheme == 'exp0':
        # Modified version of above that fixes temperature at ub for initial t0 iterations
        tau = ub * math.exp(-r * (t - t0))
    elif scheme == 'linear':
        # Cool temperature linearly from ub after the initial t0 iterations
        tau = -r * (t - t0) + ub
    else:
        raise NotImplementedError
    return min(max(tau, lb), ub)
def mip_ops(data:np.ndarray,save_dir:Union[None,str]=None,data_name:str='',suffix:str='') -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    assert len(data.shape) == 4
    mip_d = data.max(0)
    mip_h = data.max(1)
    mip_w = data.max(2)
    if save_dir is not None:
        save_img(opj(save_dir,data_name+'_mip_d'+suffix), mip_d)
        save_img(opj(save_dir,data_name+'_mip_h'+suffix), mip_h)
        save_img(opj(save_dir,data_name+'_mip_w'+suffix), mip_w)
    return mip_d,mip_h,mip_w

def preprocess(data:np.ndarray,denoise_level:int,denoise_close:Union[bool,List[int]],clip_range:List[int]):
    if denoise_close == False:
        data[data<=denoise_level]= 0
    else:
        if len(data.shape) == 4:
            data[ndimage.binary_opening(data<=denoise_level, structure=np.ones(tuple(list(denoise_close)+[1])),iterations=1)]=0
        elif len(data.shape) == 3:
            data[ndimage.binary_opening(data<=denoise_level, structure=np.ones(tuple(list(denoise_close[:2])+[1])),iterations=1)]=0
    clip_range = range_limit(data,clip_range)
    data = data.clip(*clip_range)
    return data
def parse_checkpoints(checkpoints:Union[str,int],max_steps:int):
    if checkpoints == 'none':
        checkpoints = [max_steps]
    elif 'every' in checkpoints:
        _,interval = checkpoints.split('_')
        interval = int(interval)
        checkpoints = list(range(interval,max_steps,interval))
        checkpoints.append(max_steps)
    elif isinstance(checkpoints,int):
        if checkpoints >= max_steps:
            checkpoints = [max_steps]
        else:
            checkpoints = [checkpoints,max_steps]
    else:
        checkpoints = [int(s) for s in checkpoints.split(",") if int(s) < max_steps]
        checkpoints.append(max_steps)
    return checkpoints
def parse_weight(data:np.ndarray,weight_type_list:List[str]):
    if not isinstance(data,np.ndarray):
        data = np.array(data)
    weight = np.ones_like(data).astype(np.float32)
    # shape = data.shape[:-1]
    # weight = np.ones(shape).astype(np.float32)
    # # rgb2gray
    # if data.shape[-1] == 3:
    #     if len(data.shape) == 4:
    #         for i in range(data.shape[0]):
    #             data[i] = cv2.cvtColor(data[i],cv2.COLOR_RGB2GRAY)
    #     elif len(data.shape) == 3:
    #         data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    for weight_type in weight_type_list:
        if 'quantile' in weight_type:
            _,ge_thres,ql,qh,scale = weight_type.split('_')
            ge_thres,ql,qh,scale = float(ge_thres),float(ql),float(qh),float(scale)
            l = np.quantile(data[data>=ge_thres],ql)
            h = np.quantile(data[data>=ge_thres],qh)
            l,h = range_limit(data,[l,h])
            weight[(data>=l) * (data<=h)] = scale
        elif 'value' in weight_type:
            _,l,h,scale = weight_type.split('_')
            l,h,scale = float(l),float(h),float(scale)
            l,h = range_limit(data,[l,h])
            weight[(data>=l) * (data<=h)] = scale
        elif 'exp' in weight_type:
            _,mid_x,mid_value = weight_type.split('_')
            mid_x,mid_value = float(mid_x),float(mid_value)
            a = -np.log(mid_value)/mid_x
            weight = np.exp(-a*data)
        elif weight_type == 'none':
            pass
        else:
            raise NotImplementedError
    return weight
def plot_conv3d_weight(name:str,weight:torch.Tensor,savedir:str):
    weight = weight.cpu().numpy()
    for fig_idx in range(weight.shape[0]):
        fig = plt.figure(figsize = (20,20))
        figname = '{}_out_channel_{}'.format(name,fig_idx)
        fig.suptitle(figname)
        for row_idx in range(weight.shape[1]):
            for col_idx in range(weight.shape[2]):
                weight_ = weight[fig_idx,row_idx,col_idx]
                fig.add_subplot(weight.shape[1],weight.shape[2],row_idx*weight.shape[2]+col_idx+1)
                im = fig.axes[-1].imshow(weight_,cmap='Greys_r',vmin=weight_.min(),vmax=weight_.max())
                fig.axes[-1].set_xticks([])
                fig.axes[-1].set_yticks([])
                fig.axes[-1].set_title('in_channel:{} d:{}'.format(row_idx,col_idx))
                for i in range(weight.shape[3]):
                    for j in range(weight.shape[4]):
                        text = fig.axes[-1].text(j, i,'{:.5f}'.format(weight_[j,i]),size=5,
                                    ha="center", va="center", color="red")
        plt.tight_layout()
        plt.savefig(opj(savedir,figname+'.png'))
    pass
def divide_data(data:np.ndarray,divide_type:str,):
    divide_img = copy.deepcopy(data)
    data_chunk_list = []
    # 3d dataset->dhwc or thwc
    if len(data.shape) == 4:
        if 'total' in divide_type:
            _,num_d,num_h,num_w = divide_type.split('_')
            num_d,num_h,num_w = int(num_d),int(num_h),int(num_w)
            # assert data.shape[0]%num_d==0 and data.shape[1]%num_h==0 and data.shape[2]%num_w==0, "{} cannot be divided by ({},{},{}) equally!".format(data.shape[:3],num_d,num_h,num_w)
            chunk_d,chunk_h,chunk_w = int(data.shape[0]/num_d),int(data.shape[1]/num_h),int(data.shape[2]/num_w)
        elif 'every' in divide_type:
            _,chunk_d,chunk_h,chunk_w = divide_type.split('_')
            chunk_d,chunk_h,chunk_w = int(chunk_d),int(chunk_h),int(chunk_w)
            # assert data.shape[0]%chunk_d==0 and data.shape[1]%chunk_h==0 and data.shape[2]%chunk_w==0, "{} cannot be divided by ({},{},{}) equally!".format(data.shape[:3],chunk_d,chunk_h,chunk_w)
        else:
            raise NotImplementedError
        dsections = [i for i in range(data.shape[0]) if i%chunk_d==0]
        hsections = [i for i in range(data.shape[1]) if i%chunk_h==0]
        wsections = [i for i in range(data.shape[2]) if i%chunk_w==0]
        dsections.append(data.shape[0])
        hsections.append(data.shape[1])
        wsections.append(data.shape[2])
        for di in range(len(dsections)-1):
            for hi in range(len(hsections)-1):
                for wi in range(len(wsections)-1):
                    data_chunk_list.append({'data':data[dsections[di]:dsections[di+1],hsections[hi]:hsections[hi+1],wsections[wi]:wsections[wi+1]],'d':[dsections[di],dsections[di+1]-1],'h':[hsections[hi],hsections[hi+1]-1],'w':[wsections[wi],wsections[wi+1]-1]})
                    x,y,z = wsections[wi],hsections[hi],dsections[di]
                    w,h,d = wsections[wi+1]-wsections[wi],hsections[hi+1]-hsections[hi],dsections[di+1]-dsections[di]
                    divide_img[z,y:y+h,x:x+w] = 2000
                    divide_img[z+d-1,y:y+h,x:x+w] = 2000
                    divide_img[z:z+d,y,x:x+w] = 2000
                    divide_img[z:z+d,y+h-1,x:x+w] = 2000
                    divide_img[z:z+d,y:y+h,x] = 2000
                    divide_img[z:z+d,y:y+h,x+w-1] = 2000
        for data_chunk in data_chunk_list:
            data_chunk['total_size'] = data.size
            data_chunk['size'] = data_chunk['data'].size
            data_chunk['name'] = 'd_{}_{}-h_{}_{}-w_{}_{}'.format(*data_chunk['d'],*data_chunk['h'],*data_chunk['w'])
    # 2d dataset->hwc
    elif len(data.shape) == 3:
        if 'total' in divide_type:
            _,_,num_h,num_w = divide_type.split('_')
            num_h,num_w = int(num_h),int(num_w)
            # assert data.shape[0]%num_h==0 and data.shape[1]%num_w==0, "{} cannot be divided by ({},{}) equally!".format(data.shape[:2],num_h,num_w)
            chunk_h,chunk_w = int(data.shape[0]/num_h),int(data.shape[1]/num_w)
        elif 'every' in divide_type:
            _,_,chunk_h,chunk_w = divide_type.split('_')
            chunk_h,chunk_w = int(chunk_h),int(chunk_w)
            # assert data.shape[0]%chunk_h==0 and data.shape[1]%chunk_w==0, "{} cannot be divided by ({},{}) equally!".format(data.shape[:2],chunk_h,chunk_w)
        else:
            raise NotImplementedError
        hsections = [i for i in range(data.shape[0]) if i%chunk_h==0]
        wsections = [i for i in range(data.shape[1]) if i%chunk_w==0]
        hsections.append(data.shape[0])
        wsections.append(data.shape[1])
        for hi in range(len(hsections)-1):
            for wi in range(len(wsections)-1):
                data_chunk_list.append({'data':data[hsections[hi]:hsections[hi+1],wsections[wi]:wsections[wi+1]],'h':[hsections[hi],hsections[hi+1]-1],'w':[wsections[wi],wsections[wi+1]-1]})
                cv2.rectangle(divide_img, (wsections[wi], hsections[hi]), (wsections[wi+1], hsections[hi+1]), (0, 0, 255), 2)
        for data_chunk in data_chunk_list:
            data_chunk['total_size'] = data.size
            data_chunk['size'] = data_chunk['data'].size
            data_chunk['name'] = 'h_{}_{}-w_{}_{}'.format(*data_chunk['h'],*data_chunk['w'])
    else:
        raise NotImplementedError
    return data_chunk_list, divide_img
def alloc_param(data_chunk_list:List[dict],param_size:float,param_alloc:str,param_size_thres:float):
    if param_alloc == 'equal':
        for data_chunk in data_chunk_list:
            data_chunk['param_size'] = param_size/len(data_chunk_list)
    elif param_alloc == 'by_size':
        for data_chunk in data_chunk_list:
            data_chunk['param_size'] = param_size*(data_chunk['size'])/data_chunk['total_size']
    elif param_alloc == 'by_var':
        var_total = 0
        for data_chunk in data_chunk_list:
            var_total += ((data_chunk['data']-data_chunk['data'].mean())**2).mean()
        for data_chunk in data_chunk_list:
            data_chunk['param_size'] = param_size*((data_chunk['data']-data_chunk['data'].mean())**2).mean()/var_total
            data_chunk['param_size'] = float(data_chunk['param_size'])
    elif param_alloc == 'by_d':
        d_total = 0
        for data_chunk in data_chunk_list:
            d_total += 1/cal_feature(data_chunk['data'])
        for data_chunk in data_chunk_list:
            data_chunk['param_size'] = param_size*(1/cal_feature(data_chunk['data']))/d_total
            data_chunk['param_size'] = float(data_chunk['param_size'])
    elif param_alloc == 'by_dv':
        d_total = 0
        for data_chunk in data_chunk_list:
            d_total += data_chunk['size']/cal_feature(data_chunk['data'])
        for data_chunk in data_chunk_list:
            data_chunk['param_size'] = param_size*(data_chunk['size']/cal_feature(data_chunk['data']))/d_total
            data_chunk['param_size'] = float(data_chunk['param_size'])

    # filter the too small param_size
    data_chunk_list_ = [data_chunk for data_chunk in data_chunk_list if data_chunk['param_size'] >= param_size_thres]
    if len(data_chunk_list_) < len(data_chunk_list):
        return alloc_param(data_chunk_list_,param_size,param_alloc,param_size_thres)
    return data_chunk_list_
    
def merge_divided_data(decompressed_data_chunk_list:List[dict],data_shape):
    max = get_type_max(decompressed_data_chunk_list[0]['data'])
    decompressed_data = np.zeros(data_shape,dtype=np.float32)
    for decompressed_data_chunk in decompressed_data_chunk_list:
        hstart,hend = decompressed_data_chunk['h']
        wstart,wend = decompressed_data_chunk['w']
        if len(data_shape) == 4:
            dstart,dend = decompressed_data_chunk['d']
            decompressed_data[dstart:dend+1,hstart:hend+1,wstart:wend+1] += decompressed_data_chunk['data']
        elif len(data_shape) == 3:
            decompressed_data[hstart:hend+1,wstart:wend+1] += decompressed_data_chunk['data']
        else:
            raise NotImplementedError
    decompressed_data = decompressed_data.clip(None,max)
    decompressed_data = decompressed_data.astype(decompressed_data_chunk_list[0]['data'].dtype)
    return decompressed_data

def cal_mse(data1:np.ndarray,data2:np.ndarray):
    mse = ((data1-data2)**2).mean()
    return mse

def cal_psnr(origin_data:np.ndarray, decompressed_data:np.ndarray,data_range):
    data1 = np.copy(origin_data)
    data2 = np.copy(decompressed_data)
    mse = np.mean(np.power(data1/data_range-data2/data_range,2))
    psnr = -10*np.log10(mse)
    return psnr

def cal_ssim(origin_data:np.ndarray, decompressed_data:np.ndarray, data_range):
    origin_data = torch.from_numpy(origin_data)
    decompressed_data = torch.from_numpy(decompressed_data)
    # transform to NCHW or NCDHW
    if len(origin_data.shape) == 3:
        data1 = rearrange(origin_data, 'h w (n c) -> n c h w', n=1)
        data2 = rearrange(decompressed_data, 'h w (n c) -> n c h w', n=1)
        ssim_value = ssim_calc(data1,data2,data_range)
    elif len(origin_data.shape) == 4:
        ssim_value_total = 0 
        for i in range(origin_data.shape[0]):
            data1 = copy.deepcopy(origin_data[i])
            data2 = copy.deepcopy(decompressed_data[i])
            data1 = rearrange(data1, 'h w (n c) -> n c h w', n=1)
            data2 = rearrange(data2, 'h w (n c) -> n c h w', n=1)
            ssim_value_total += ssim_calc(data1,data2,data_range)
        ssim_value = ssim_value_total/origin_data.shape[0]
    return float(ssim_value)

def eval_performance(steps:int, data1:np.ndarray, data2:np.ndarray, Log, mse:bool, psnr:bool, ssim:bool):
    # performance
    performance_dict = {}
    performance_dict['steps'] = steps

    # calculate indicators
    max_range = get_type_max(data1)
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)
    if mse:
        mse_value = cal_mse(data1,data2)
        Log.log_metrics({'mse':mse_value},steps)
        performance_dict['mse'] = mse_value
    if psnr:
        psnr_value = cal_psnr(data1,data2,max_range)
        Log.log_metrics({'psnr':psnr_value},steps)
        performance_dict['psnr'] = psnr_value
    if ssim:
        ssim_value = cal_ssim(data1, data2, max_range)
        Log.log_metrics({'ssim':ssim_value},steps)
        performance_dict['ssim'] = ssim_value

    return performance_dict

    