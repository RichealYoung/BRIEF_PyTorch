from typing import List
import random
import torch
from einops import rearrange
import torch.nn.functional as F
class Transform:
    def __init__(self,operator_list:List,args_list:List):
        self.operator_instance_list=[]
        assert len(operator_list)==len(args_list)
        for operator,args in zip(operator_list,args_list):
            operator_instance=operator(**args)
            if operator_instance.check():
                self.operator_instance_list.append(operator_instance)
    def __call__(self,img:torch.Tensor) -> torch.Tensor:
        for operator_instance in self.operator_instance_list:
            img = operator_instance(img)
        return img
class Crop3d:
    def __init__(self,crop_d:int,crop_h:int,crop_w:int,crop_d_start:int,crop_h_start:int,crop_w_start:int):
        self.crop_d=crop_d
        self.crop_h=crop_h
        self.crop_w=crop_w
        self.crop_d_start=crop_d_start
        self.crop_h_start=crop_h_start
        self.crop_w_start=crop_w_start
    def check(self):
        if self.crop_d and self.crop_h and self.crop_w:
            return True
        else:
            print('not crop')
            return False
    def __call__(self,img:torch.Tensor,) -> torch.Tensor:
        N,C,D,H,W = img.shape
        if self.crop_d and self.crop_h and self.crop_w:
            crop_d = min(self.crop_d, D - self.crop_d_start)
            crop_h = min(self.crop_h, H - self.crop_h_start)
            crop_w = min(self.crop_w, W - self.crop_w_start)  
            img_cropped = img[:,:,self.crop_d_start:self.crop_d_start + crop_d,self.crop_h_start:self.crop_h_start + crop_h,self.crop_w_start:self.crop_w_start + crop_w]
        else:
            img_cropped = img
        return img_cropped
class RandomCrop3d:
    def __init__(self,randomcrop_d:int,randomcrop_h:int,randomcrop_w:int):
        self.randomcrop_d=randomcrop_d
        self.randomcrop_h=randomcrop_h
        self.randomcrop_w=randomcrop_w
    def check(self):
        if self.randomcrop_d and self.randomcrop_h and self.randomcrop_w:
            return True
        else:
            print('not randomcrop')
            return False
    def __call__(self,img:torch.Tensor,) -> torch.Tensor:
        N,C,D,H,W = img.shape
        randomcrop_d_start = random.randint(0, max(0, D - self.randomcrop_d))
        randomcrop_h_start = random.randint(0, max(0, H - self.randomcrop_h))
        randomcrop_w_start = random.randint(0, max(0, W - self.randomcrop_w))
        randomcrop3d=Crop3d(self.randomcrop_d,self.randomcrop_h,self.randomcrop_w,randomcrop_d_start,randomcrop_h_start,randomcrop_w_start)
        return randomcrop3d(img)
class Resize3d:
    def __init__(self,resize_d:int,resize_h:int,resize_w:int,mode:str='trilinear'):
        """
        Args:
            mode (str): see details in https://pytorch.org/docs/master/generated/torch.nn.functional.interpolate.html#torch-nn-functional-interpolate
        """
        self.resize_d=resize_d
        self.resize_h=resize_h
        self.resize_w=resize_w
        self.mode = mode
    def check(self):
        if self.resize_d and self.resize_h and self.resize_w:
            return True
        else:
            print('not resize')
            return False
    def __call__(self,img:torch.Tensor,) -> torch.Tensor:      
        img_resized = F.interpolate(img,(self.resize_d,self.resize_h,self.resize_w),mode = self.mode,align_corners=False)
        return img_resized
class RandomResize3d:
    def __init__(self,resize_d_ratio:list,resize_h_ratio:list,resize_w_ratio:list,mode:str='trilinear'):
        """
        Args:
            mode (str): see details in https://pytorch.org/docs/master/generated/torch.nn.functional.interpolate.html#torch-nn-functional-interpolate
        """
        self.resize_d_ratio=resize_d_ratio
        self.resize_h_ratio=resize_h_ratio
        self.resize_w_ratio=resize_w_ratio
        self.mode = mode
    def check(self):
        if self.resize_d_ratio!=[1,1] and self.resize_h_ratio!=[1,1] and self.resize_w_ratio!=[1,1]:
            return True
        else:
            print('not randomresize')
            return False
    def __call__(self,img:torch.Tensor,) -> torch.Tensor:
        N,C,D,H,W = img.shape
        randomresize_d = int((torch.rand((1))*(self.resize_d_ratio[1]-self.resize_d_ratio[0])+self.resize_d_ratio[0])*D)
        randomresize_h = int((torch.rand((1))*(self.resize_h_ratio[1]-self.resize_h_ratio[0])+self.resize_h_ratio[0])*H)
        randomresize_w = int((torch.rand((1))*(self.resize_w_ratio[1]-self.resize_w_ratio[0])+self.resize_w_ratio[0])*W)
        randomresize3d = Resize3d(randomresize_d,randomresize_h,randomresize_w,self.mode)
        return randomresize3d(img)
class FlipRoat3d:
    def __init__(self,flip:bool,drot90:bool,hrot90:bool,wrot90:bool):
        self.flip=flip
        self.drot90=drot90
        self.hrot90=hrot90
        self.wrot90=wrot90
    def check(self):
        if self.flip or self.drot90 or self.hrot90 or self.wrot90:
            return True
        else:
            print('not fliproat')
            return False
    def __call__(self,img:torch.Tensor,) -> torch.Tensor:
        dflip = (random.random() < 0.5) * self.flip
        hflip = (random.random() < 0.5) * self.flip
        wflip = (random.random() < 0.5) * self.flip
        drot90 = (random.random() < 0.5) * self.drot90
        hrot90 = (random.random() < 0.5) * self.hrot90
        wrot90 = (random.random() < 0.5) * self.wrot90
        return fliproat3d(img,dflip,hflip,wflip,drot90,hrot90,wrot90)
def fliproat3d(img:torch.Tensor,dflip:bool,hflip:bool,wflip:bool,drot90:bool,hrot90:bool,wrot90:bool) -> torch.Tensor:
    if dflip:
        img = torch.flip(img,[-3])
    if hflip:
        img = torch.flip(img,[-2])
    if wflip:
        img = torch.flip(img,[-1])
    if drot90:
        img = rearrange(img,'n c d h w -> n c d w h')
    if hrot90:
        img = rearrange(img,'n c d h w -> n c w h d')
    if wrot90:
        img = rearrange(img,'n c d h w -> n c h d w')
    return img
