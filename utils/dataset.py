from utils.Typing import NormalizeOpt,CropOpt, TransformOpt
from tqdm import tqdm
import math
from typing import Callable, List, Tuple,Dict
import torch
import torch.optim
import random
from einops import rearrange,repeat
from utils.io import *
from utils.transform import *
def create_coords(coords_shape:Tuple,mode='n11') -> torch.Tensor:
    if mode == 'n11':
        minimum = -1
        maximum = 1
    elif mode == '0p1':
        minimum = 0
        maximum = 1
    else:
        minimum,maximum = mode.split(',')
        minimum,maximum = float(minimum),float(maximum)
    if len(coords_shape) == 2:
        coords = torch.stack(torch.meshgrid(
        torch.linspace(minimum,maximum,coords_shape[0]),
        torch.linspace(minimum,maximum,coords_shape[1])),
        axis=-1)
        return coords
    elif len(coords_shape) == 3:
        coords = torch.stack(torch.meshgrid(
        torch.linspace(minimum,maximum,coords_shape[0]),
        torch.linspace(minimum,maximum,coords_shape[1]),
        torch.linspace(minimum,maximum,coords_shape[2])),
        axis=-1)
        return coords
    else:
        raise NotImplementedError
def create_flattened_coords(coords_shape:Tuple,mode='n11') -> torch.Tensor:
    if mode == 'n11':
        minimum = -1
        maximum = 1
    elif mode == '0p1':
        minimum = 0
        maximum = 1
    else:
        minimum,maximum = mode.split(',')
        minimum,maximum = float(minimum),float(maximum)
    if len(coords_shape) == 2:
        coords = torch.stack(torch.meshgrid(
        torch.linspace(minimum,maximum,coords_shape[0]),
        torch.linspace(minimum,maximum,coords_shape[1])),
        axis=-1)
        flattened_coords = rearrange(coords,'h w c -> (h w) c')
        return flattened_coords
    elif len(coords_shape) == 3:
        coords = torch.stack(torch.meshgrid(
        torch.linspace(minimum,maximum,coords_shape[0]),
        torch.linspace(minimum,maximum,coords_shape[1]),
        torch.linspace(minimum,maximum,coords_shape[2])),
        axis=-1)
        flattened_coords = rearrange(coords,'d h w c -> (d h w) c')
        return flattened_coords
    else:
        raise NotImplementedError

def crop_data(data:torch.Tensor,ps_h:int,ps_w:int,ol_h:int,ol_w:int,ps_d:int=None,ol_d:int=None) -> Tuple[torch.Tensor,tuple]:
    batch_size,data_channel,*coords_shape = data.shape
    if len(coords_shape) == 2:
        return NotImplementedError
    elif len(coords_shape) == 3:
        d,h,w = coords_shape
        pc_d = math.ceil((d-ol_d)/(ps_d-ol_d))
        pc_h = math.ceil((h-ol_h)/(ps_h-ol_h))
        pc_w = math.ceil((w-ol_w)/(ps_w-ol_w))
        pc_shape = (pc_d,pc_h,pc_w)
        # crop
        cropped_data = torch.zeros((batch_size,pc_d,pc_h,pc_w,data_channel,ps_d,ps_h,ps_w))
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
                    cropped_data[:,pc_d_idx,pc_h_idx,pc_w_idx,...] = data[...,data_d_start:data_d_start+ps_d,data_h_start:data_h_start+ps_h,data_w_start:data_w_start+ps_w]
    else:
        raise NotImplementedError
    return cropped_data,pc_shape
class CroppedSampler:
    def __init__(self,cropped_data: torch.Tensor,sample_size:int,shuffle:bool=True) -> None:
        self.cropped_data = cropped_data
        self.sample_size = sample_size
        self.shuffle = shuffle
        if len(self.cropped_data.shape) == 6:
            batch_size,pc_h,pc_w,data_channel,ps_h,ps_w = self.cropped_data.shape
            self.coords = create_flattened_coords((ps_h,ps_w))
            self.coords = repeat(self.coords,'pop c -> n pc_h pc_w pop c',n=batch_size,pc_h=pc_h,pc_w=pc_w)
            self.data = rearrange(cropped_data,'n pc_h pc_w data_channel ps_h ps_w -> n pc_h pc_w (ps_h ps_w) data_channel')
            self.pop_size = ps_h*ps_w
        if len(self.cropped_data.shape) == 8:
            batch_size,pc_d,pc_h,pc_w,data_channel,ps_d,ps_h,ps_w = self.cropped_data.shape
            self.coords = create_flattened_coords((ps_d,ps_h,ps_w))
            self.coords = repeat(self.coords,'pop c -> n pc_d pc_h pc_w pop c',n=batch_size,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w)
            self.data = rearrange(cropped_data,'n pc_d pc_h pc_w data_channel ps_d ps_h ps_w -> n pc_d pc_h pc_w (ps_d ps_h ps_w) data_channel')
            self.pop_size = ps_d*ps_h*ps_w
    def __len__(self):
        return math.ceil(self.pop_size/self.sample_size)
    def __iter__(self,):
        self.index = 0
        if self.shuffle:
            idxs= torch.randperm(self.pop_size)
        else:
            idxs = torch.arange(0,self.pop_size)
        if len(self.cropped_data.shape) == 6:
            batch_size,pc_h,pc_w,data_channel,ps_h,ps_w = self.cropped_data.shape
            self.coords_idxs = repeat(idxs,'pop -> n pc_h pc_w pop c',n=batch_size,pc_h=pc_h,pc_w=pc_w,c=3)
            self.data_idxs = repeat(idxs,'pop -> n pc_h pc_w pop c',n=batch_size,pc_h=pc_h,pc_w=pc_w,c=data_channel)
        elif len(self.cropped_data.shape) == 8:
            batch_size,pc_d,pc_h,pc_w,data_channel,ps_d,ps_h,ps_w = self.cropped_data.shape

            self.coords_idxs = repeat(idxs,'pop -> n pc_d pc_h pc_w pop c',n=batch_size,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w,c=3)
            self.data_idxs = repeat(idxs,'pop -> n pc_d pc_h pc_w pop c',n=batch_size,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w,c=data_channel)
        else:
            raise NotImplementedError
        return self
    def __next__(self,) -> Tuple[torch.Tensor,torch.Tensor]:
        if self.index < self.__len__():
            start_idx = self.index*self.sample_size
            end_idx = min(start_idx+self.sample_size,self.pop_size)
            sampled_coords_idxs = self.coords_idxs[...,start_idx:end_idx,:]
            sampled_data_idxs = self.data_idxs[...,start_idx:end_idx,:]
            sampled_coords  = self.coords.gather(-2,sampled_coords_idxs)
            sampled_data  = self.data.gather(-2,sampled_data_idxs)
            self.index += 1
            return sampled_coords,sampled_data
        else:
            raise StopIteration
class CropDataset:
    def __init__(self,batch_size:int,sample_size:int,Normalize_opt:NormalizeOpt,Transform_opt:TransformOpt=None,crop_opt:CropOpt=None,shuffle_path:bool=True,shuffle_sampler:bool=True,data_dir:str=None,data_path_list:List[str]=None) -> None:
        if (data_dir is not None) and (data_path_list is not None):
            raise "Only one args can be used !"
        if data_dir is not None:
            self.data_path_list = gen_pathlist_fromimgdir(data_dir)
        elif data_path_list is not None:
            self.data_path_list = data_path_list
        else:
            raise "At least one args is given !"
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.Normalize_opt = Normalize_opt
        if Transform_opt is not None:
            self.transform = Transform([Resize3d,RandomResize3d,Crop3d,RandomCrop3d,FlipRoat3d],
                [Transform_opt.Resize3d,Transform_opt.RandomResize3d,Transform_opt.Crop3d,Transform_opt.RandomCrop3d,Transform_opt.FlipRoat3d])
        else:
            self.transform = lambda x:x
        self.crop_opt = crop_opt
        self.shuffle_path = shuffle_path
        self.shuffle_sampler = shuffle_sampler
    def __len__(self):
        return math.ceil(len(self.data_path_list)/self.batch_size)
    def __iter__(self,):
        if self.shuffle_path:
            random.shuffle(self.data_path_list)
        self.index = 0
        return self
    def __next__(self) -> Tuple[torch.Tensor,CroppedSampler,tuple,dict]: 
        if self.index < self.__len__():
            start_idx = self.index*self.batch_size
            end_idx = min(start_idx+self.batch_size,len(self.data_path_list))
            path_idxs = torch.arange(start_idx,end_idx)
            path_list = [self.data_path_list[idx] for idx in path_idxs]
            data = read_data_batch(path_list)
            data,sideinfos = normalize_data(data,**self.Normalize_opt)
            data = self.transform(data)
            cropped_data,pc_shape = crop_data(data,**self.crop_opt)
            cropped_sampler = CroppedSampler(cropped_data,self.sample_size,self.shuffle_sampler)
            self.index += 1
            try:
                return cropped_data,cropped_sampler,pc_shape, sideinfos | {'data_shape':list(data.shape)}
            except:
                return cropped_data,cropped_sampler,pc_shape, {**sideinfos,**{'data_shape':list(data.shape)}}
        else:
            raise StopIteration