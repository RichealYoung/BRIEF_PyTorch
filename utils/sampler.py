import torch

import numpy as np

from einops import rearrange
from typing import Any, Dict, List, Tuple, Union
from utils.dataset import create_coords, create_flattened_coords

class RandomCubeSampler:
    def __init__(self,data: torch.Tensor,weight:np.ndarray,coords_mode:str,cube_count:int,cube_len:List[int],sample_count:int,device:str='cpu',gpu_force:bool=False) -> None:
        self.sample_count = sample_count
        self.device = device
        if len(data.shape) == 5:
            batch_size,data_channel,self.d,self.h,self.w = data.shape
            assert batch_size == 1
            self.coords = create_coords((self.d,self.h,self.w),mode=coords_mode)
            self.data = rearrange(data,'n data_channel d h w -> d h w (n data_channel)')
            weight = torch.from_numpy(weight)
            self.weight = rearrange(weight,'n data_channel d h w -> d h w (n data_channel)')
            for i in range(len(cube_len)):
                cube_len[i] = min(cube_len[i],data.shape[2+i])
            self.cube_len_d,self.cube_len_h,self.cube_len_w = cube_len
            self.cube_count = cube_count
            self.gpu_force = gpu_force
            if gpu_force:
                self.coords = self.coords.to(device)
                self.weight = self.weight.to(device)
            else:
                self.data = self.data.cpu()
            self.data_cubes = self.data.unfold(0,cube_len[0],1).unfold(1,cube_len[1],1).unfold(2,cube_len[2],1)
            self.coords_cubes = self.coords.unfold(0,cube_len[0],1).unfold(1,cube_len[1],1).unfold(2,cube_len[2],1)
            self.weight_cubes = self.weight.unfold(0,cube_len[0],1).unfold(1,cube_len[1],1).unfold(2,cube_len[2],1)
            self.data_cubes = rearrange(self.data_cubes,'dc hc wc c ds hs ws -> (dc hc wc) ds hs ws c')
            self.coords_cubes = rearrange(self.coords_cubes,'dc hc wc c ds hs ws -> (dc hc wc) ds hs ws c')
            self.weight_cubes = rearrange(self.weight_cubes,'dc hc wc c ds hs ws -> (dc hc wc) ds hs ws c')
            self.pop_size = self.data_cubes.shape[0]
        else:
            raise NotImplementedError
    def __len__(self):
        return self.sample_count
    def __iter__(self,):
        self.index = 0
        return self
    def __next__(self,) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        if self.index < self.__len__():
            sampled_idxs = torch.randint(0,self.pop_size,(self.cube_count,))
            sampled_coords  = self.coords_cubes[sampled_idxs,:]
            sampled_data  = self.data_cubes[sampled_idxs,:]
            sampled_weight = self.weight_cubes[sampled_idxs,:]
            if not self.gpu_force:
                sampled_coords  = sampled_coords.to(self.device)
                sampled_data  = sampled_data.to(self.device)
                sampled_weight = sampled_weight.to(self.device)    
            self.index += 1
            return sampled_coords,sampled_data,sampled_weight
        else:
            raise StopIteration
class RandompointSampler:
    def __init__(self,data: torch.Tensor,weight:np.ndarray,coords_mode:str,sample_size:int,sample_count:int,device:str='cpu') -> None:
        self.sample_size = sample_size
        self.sample_count = sample_count
        if len(data.shape) == 5:
            batch_size,data_channel,d,h,w = data.shape
            assert batch_size == 1
            self.coords = create_flattened_coords((d,h,w),mode=coords_mode).to(device)
            self.data = rearrange(data,'n data_channel d h w -> (n d h w) data_channel')
            weight = torch.from_numpy(weight).to(device)
            self.weight = rearrange(weight,'n data_channel d h w -> (n d h w) data_channel')
            self.pop_size = d*h*w
        elif len(data.shape) == 4:
            batch_size,data_channel,h,w = data.shape
            assert batch_size == 1
            self.coords = create_flattened_coords((h,w),mode=coords_mode).to(device)
            self.data = rearrange(data,'n data_channel h w -> (n h w) data_channel')
            weight = torch.from_numpy(weight).to(device)
            self.weight = rearrange(weight,'n data_channel h w -> (n h w) data_channel')
            self.pop_size = h*w
        else:
            raise NotImplementedError
    def __len__(self):
        return self.sample_count
    def __iter__(self,):
        self.index = 0
        return self
    def __next__(self,) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        if self.index < self.__len__():
            sampled_idxs = torch.randint(0,self.pop_size,(self.sample_size,))
            sampled_coords  = self.coords[sampled_idxs,:]
            sampled_data  = self.data[sampled_idxs,:]
            sampled_weight = self.weight[sampled_idxs,:]
            self.index += 1
            return sampled_coords,sampled_data,sampled_weight
        else:
            raise StopIteration