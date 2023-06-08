import copy
from typing import Any, List, Optional, Tuple, Union
from utils.misc import configure_optimizer, gradient_descent
from einops import rearrange,repeat,reduce
from einops.layers.torch import Rearrange, Reduce
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import torch.optim
from compressai.entropy_models import EntropyBottleneck,GaussianConditional
def get_nnmodule_param_count(module:nn.Module):
    param_count = 0
    for param in module.state_dict().values():
        param_count += int(np.prod(param.shape))
    return param_count
########################
class PosEncodingSIREN(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.in_channel = len(T)
        self.out_channel = len(T)
        self.W = [2*math.pi/t for t in T]
    def forward(self, coords):
        for i in range(len(self.W)):
            c = coords[..., i]
            w = self.W[i]
            coords[..., i] = torch.sin(w*c)
        return coords

class SIRENPos(nn.Module):
    def __init__(self, coords_channel=3,data_channel=1,features=256,layers=5,w0=30,T=[2,2],**kwargs):
        super().__init__()
        self.net=[]
        self.fre = PosEncodingSIREN(T)
        # self.net.append(PosEncodingSIREN(T))
        self.net.append(nn.Sequential(nn.Linear(coords_channel,features),Sine(w0)))
        for i in range(layers-2):
            self.net.append(nn.Sequential(nn.Linear(features,features),Sine()))
        self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
    def forward(self,coords):
        coords = self.fre(coords)
        output = self.net(coords)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,**kwargs):
        param_count=coords_channel*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,**kwargs):
        a = layers-2
        b = coords_channel+1+layers-2+data_channel
        c = -param_count+data_channel
        if a == 0:
            features = round(-c/b)
        else:
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features

class PosEncodingNeRF(nn.Module):
    def __init__(self, in_channel,frequencies=10):
        super().__init__()

        self.in_channel = in_channel
        self.frequencies = frequencies
        self.out_channel = in_channel + 2 * in_channel * self.frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_channel)
        coords_pos_enc = coords
        for i in range(self.frequencies):
            for j in range(self.in_channel):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * math.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * math.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_channel).squeeze(1)
class NeRF(nn.Module):
    """
    B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, 
    and R. Ng, “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,” 
    arXiv:2003.08934 [cs], Aug. 2020, Accessed: May 04, 2021. [Online]. 
    Available: http://arxiv.org/abs/2003.08934
    """
    def __init__(self,coords_channel=3,data_channel=1,frequencies=10,features=256,layers=5,skip=True,**kwargs):
        super().__init__()
        self.skip = skip
        self.skip_layer = (layers-1)//2 if skip else -1
        self.positional_encoding = PosEncodingNeRF(in_channel=coords_channel,frequencies=frequencies)
        in_channel = self.positional_encoding.out_channel
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(in_channel,features),nn.ReLU(inplace=True)))
        for i in range(layers-2):
            if self.skip_layer == i+1:
                self.net.append(nn.Sequential(nn.Linear(in_channel+features,features),nn.ReLU(inplace=True)))
            else:
                self.net.append(nn.Sequential(nn.Linear(features,features),nn.ReLU(inplace=True)))
        if self.skip_layer==layers-1:
            self.net.append(nn.Sequential(nn.Linear(in_channel+features,data_channel),nn.Sigmoid()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.ModuleList(self.net)
    def forward(self,coords):
        codings = self.positional_encoding(coords)
        shape = list(coords.shape[:-1])+list(codings.shape[-1:])
        output = codings.reshape(shape)
        for idx,model in enumerate(self.net):
            if idx == self.skip_layer:
                output = torch.cat([codings,output],1)
            output = model(output)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,frequencies,layers,skip,**kwargs):
        d =  coords_channel + 2 * coords_channel * frequencies
        if skip:
            param_count=d*features+features+(layers-2)*(features**2+features)+d*features+features*data_channel+data_channel
        else:
            param_count=d*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,frequencies,layers,skip,**kwargs):
        d =  coords_channel + 2 * coords_channel * frequencies
        a = layers-2
        if skip:
            b = 2*d+1+layers-2+data_channel
        else:
            b = d+1+layers-2+data_channel
        c = -param_count+data_channel
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features
###########################
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_channel,embsize=256,scale=10):
        super().__init__()
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.in_channel = in_channel
        self.embsize = embsize
        self.out_channel = 2 * self.embsize
        self.bvals = nn.Parameter(torch.normal(0,1,size=(embsize,in_channel))*scale)
        self.bvals.requires_grad=False
    def forward(self, coords):
        # coords = coords.view(coords.shape[0], -1, self.in_channel)
        emb = torch.cat([
            torch.sin((2.*math.pi*coords)@self.bvals.T),
            torch.cos((2.*math.pi*coords)@self.bvals.T)],-1)
        return emb
class FFN(nn.Module):
    """
    M. Tancik et al., “Fourier Features Let Networks Learn High Frequency 
    Functions in Low Dimensional Domains,” arXiv:2006.10739 [cs], 
    Jun. 2020, Accessed: May 06, 2021. [Online]. Available: http://arxiv.org/abs/2006.10739
    """
    def __init__(self,coords_channel=3,data_channel=1,embsize=256,scale=10,features=256,layers=5,skip=False,**kwargs):
        super().__init__()
        self.skip = skip
        self.skip_layer = (layers-1)//2 if skip else -1
        self.fourierfeature_embedding = FourierFeatureEmbedding(in_channel=coords_channel,embsize=embsize,scale=scale)
        in_channel = self.fourierfeature_embedding.out_channel
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(in_channel,features),nn.ReLU(inplace=True)))
        for i in range(layers-2):
            if self.skip_layer == i+1:
                self.net.append(nn.Sequential(nn.Linear(in_channel+features,features),nn.ReLU(inplace=True)))
            else:
                self.net.append(nn.Sequential(nn.Linear(features,features),nn.ReLU(inplace=True)))
        if self.skip_layer==layers-1:
            self.net.append(nn.Sequential(nn.Linear(in_channel+features,data_channel),nn.Sigmoid()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.ModuleList(self.net)
        # print(self.net)
    def forward(self,coords):
        emb = self.fourierfeature_embedding(coords)
        output = emb
        for idx,model in enumerate(self.net):
            if idx == self.skip_layer:
                output = torch.cat([emb,output],1)
            output = model(output)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,embsize,layers,skip,**kwargs):
        d =  2 * embsize
        if skip:
            param_count=d*features+features+(layers-2)*(features**2+features)+d*features+features*data_channel+data_channel+coords_channel*embsize
        else:
            param_count=d*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel+coords_channel*embsize
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,embsize,layers,skip,**kwargs):
        d =  2 * embsize
        a = layers-2
        if skip:
            b = 2*d+1+layers-2+data_channel
        else:
            b = d+1+layers-2+data_channel
        c = -param_count+data_channel+coords_channel*embsize
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features
###########################
class HalfResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return 0.5*(self.fn(x, **kwargs) + x)
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
class Sine(nn.Module):
    def __init__(self,w0=30):
        super().__init__()
        self.w0=w0

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)
class SIREN(nn.Module):
    """
    V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, 
    “Implicit Neural Representations with Periodic Activation Functions,” 
    arXiv:2006.09661 [cs, eess], Jun. 2020, Accessed: May 04, 2021. [Online].
    Available: http://arxiv.org/abs/2006.09661

    Y. Lu, K. Jiang, J. A. Levine, and M. Berger,
    “Compressive Neural Representations,” 
    Computer Graphics Forum, p. 12, 2021.
    """
    def __init__(self, coords_channel=3,data_channel=1,features=256,layers=5,w0=30,res=False,output_act=False,**kwargs):
        super().__init__()
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(coords_channel,features),Sine(w0)))
        for i in range(layers-2):
            if res:
                self.net.append(
                    HalfResidual(
                        nn.Sequential(nn.Linear(features,features),
                        Sine(),
                        nn.Sequential(nn.Linear(features,features),
                        Sine(),))))
            else:
                self.net.append(nn.Sequential(nn.Linear(features,features),Sine()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel),Sine()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
        # print(self.net)
        
    def forward(self,coords):
        output = self.net(coords)
        return output
    def forward_syn(self,coords,mods):
        n,pc_d,pc_h,pc_w,pop,c = coords.shape
        coords = rearrange(coords,'n pc_d pc_h pc_w pop c -> (n pc_d pc_h pc_w) pop c')
        output = coords
        for layer, mod in zip(self.net, mods):
            output = layer(output)
            mod = rearrange(mod,'n c pc_d pc_h pc_w -> (n pc_d pc_h pc_w) () c')
            output *= mod.sigmoid()
        output = self.net[-1](output)
        output = rearrange(output,'(n pc_d pc_h pc_w) pop c -> n pc_d pc_h pc_w pop c',n=n,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w)
        return output
    def forward_syn_wocrop(self,coords,mods):
        output = coords
        for layer, mod in zip(self.net, mods):
            output = layer(output)
            mod = rearrange(mod,'n c -> n () c')
            output *= mod.sigmoid()
        output = self.net[-1](output)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,res,**kwargs):
        if res:
            param_count=coords_channel*features+features+2*(layers-2)*(features**2+features)+features*data_channel+data_channel
        else:
            param_count=coords_channel*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,res,**kwargs):
        if res:
            a = (layers-2)*2
            b = coords_channel+1+2*layers-4+data_channel
            c = -param_count+data_channel
            # features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        else:
            a = layers-2
            b = coords_channel+1+layers-2+data_channel
            c = -param_count+data_channel
            
        if a == 0:
            features = round(-c/b)
        else:
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features

class SIRENFT(nn.Module):
    def __init__(self, coords_channel=3,data_channel=1,features=256,layers=5,w0=30,res=False,output_act=False,ratio=1,**kwargs):
        super().__init__()
        self.first_feature = int(features*ratio)
        features = int(features)
        self.net=[]
        # print(self.first_feature,features)
        self.net.append(nn.Sequential(nn.Linear(coords_channel,self.first_feature),Sine(w0)))
        self.net.append(nn.Sequential(nn.Linear(self.first_feature,features),Sine(w0)))
        for i in range(layers-3):
            if res:
                self.net.append(
                    HalfResidual(
                        nn.Sequential(nn.Linear(features,features),
                        Sine(),
                        nn.Sequential(nn.Linear(features,features),
                        Sine(),))))
            else:
                self.net.append(nn.Sequential(nn.Linear(features,features),Sine()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel),Sine()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
        # print(self.net)
    def forward(self,coords):
        output = self.net(coords)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,res,ratio,**kwargs):
        first_feature = int(features*ratio)
        features = int(features)
        param_count = coords_channel*first_feature+first_feature+first_feature*features+features+(layers-3)*(features**2+features)+features*data_channel+data_channel
        # pc = c*(f+p)+f+p+(f+p)*f+f+(l-3)*(f**2+f)+f*d+d ==> (l-2)f**2+(c+p+l+d-1)f+(c*p+p+d-pc) = 0
        return int(param_count)
    @staticmethod
    def check_param_count(param_count,coords_channel,data_channel,layers,res,ratio,**kwargs):
        features = 1
        first_feature = int(features*ratio)
        limit = coords_channel*first_feature+first_feature+first_feature*features+features+(layers-3)*(features**2+features)+features*data_channel+data_channel
        if param_count >= limit:
            return True
        else:
            return False
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,res,ratio,**kwargs):
        # pc = c*f*r+f*r+f*r*f+f+(l-3)*(f**2+f)+f*d+d ==> (r+l-3)f**2+(c*r+r+1+l-3+d)f+(d-pc)=0
        a = ratio+layers-3
        b = coords_channel*ratio+ratio+1+layers-3+data_channel
        c = data_channel-param_count
        features = (-b+math.sqrt(b**2-4*a*c))/(2*a)
        return features
class SIREN_Pyramid(nn.Module):
    def __init__(self, coords_channel=3,data_channel=1,features=256,layers=5,w0=30,res=False,output_act=False,features_dis=10,**kwargs):
        super().__init__()
        # print(features)
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(coords_channel,features),Sine(w0)))
        for i in range(layers-2):
            if res: 
                self.net.append(
                    HalfResidual(
                        nn.Sequential(nn.Linear(features,features),
                        Sine(),
                        nn.Sequential(nn.Linear(features,features),
                        Sine(),))))
            else:
                self.net.append(nn.Sequential(nn.Linear(features-i*features_dis,features-(i+1)*features_dis),Sine()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features-(layers-2)*features_dis,data_channel),Sine()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features-(layers-2)*features_dis,data_channel)))
        self.net=nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
        # print(self.net)
    def forward(self,coords):
        output = self.net(coords)
        return output
    def forward_syn(self,coords,mods):
        n,pc_d,pc_h,pc_w,pop,c = coords.shape
        coords = rearrange(coords,'n pc_d pc_h pc_w pop c -> (n pc_d pc_h pc_w) pop c')
        output = coords
        for layer, mod in zip(self.net, mods):
            output = layer(output)
            mod = rearrange(mod,'n c pc_d pc_h pc_w -> (n pc_d pc_h pc_w) () c')
            output *= mod.sigmoid()
        output = self.net[-1](output)
        output = rearrange(output,'(n pc_d pc_h pc_w) pop c -> n pc_d pc_h pc_w pop c',n=n,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w)
        return output
    def forward_syn_wocrop(self,coords,mods):
        output = coords
        for layer, mod in zip(self.net, mods):
            output = layer(output)
            mod = rearrange(mod,'n c -> n () c')
            output *= mod.sigmoid()
        output = self.net[-1](output)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,res,features_dis,**kwargs):
        if res:
            param_count=coords_channel*features+features+2*(layers-2)*(features**2+features)+features*data_channel+data_channel
        else:
            # param_count=coords_channel*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel
            param_count=coords_channel*features+features
            for i in range(layers-2):
                param_count += (features-i*features_dis)*(features-(i+1)*features_dis) + (features-(i+1)*features_dis)
            param_count += (features-(layers-2)*features_dis)*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def check_param_count(param_count,coords_channel,data_channel,layers,res,features_dis,**kwargs):
        features = 1 + (layers-2)*features_dis 
        limit=coords_channel*features+features
        for i in range(layers-2):
            limit += (features-i*features_dis)*(features-(i+1)*features_dis) + (features-(i+1)*features_dis)
        limit += (features-(layers-2)*features_dis)*data_channel+data_channel
        if param_count >= limit:
            return True
        else:
            return False
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,res,features_dis,**kwargs):
        if res: 
            a = (layers-2)*2
            b = coords_channel+1+2*layers-4+data_channel
            c = -param_count+data_channel
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        else:
            # pc = (l-2)*f**2+[c+1+(1-d)*(l-2)-(l-2)*(l-3)*d+o]*f+[(l-2)*(1-d)**2/4-(l-2)*(l-3)*d+(l-2)*(l-3)*(2*l-5)*d**2/6-(l-2)*(1+d)**2/4]-(l-2)*d*o+o
            l = layers
            c = coords_channel
            d = features_dis
            o = data_channel
            a = (l-2)
            b = c+1+(1-d)*(l-2)-(l-2)*(l-3)*d+o
            c = (l-2)*(1-d)**2/4-(l-2)*(l-3)*d+(l-2)*(l-3)*(2*l-5)*d**2/6-(l-2)*(1+d)**2/4-(l-2)*d*o+o-param_count
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
            if features-(l-2)*d <= 0:
                raise ValueError
        return features
class SIRENPS(nn.Module):
    def __init__(self, coords_channel=3,data_channel=1,features=256,layers=5,w0=30,res=False,output_act=False,ratio=1,**kwargs):
        super().__init__()
        # print(features)
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(coords_channel,int(features*ratio**(layers-2))),Sine(w0)))
        for i in range(layers-2):
            if res: 
                self.net.append(
                    HalfResidual(
                        nn.Sequential(nn.Linear(int(features),int(features)),
                        Sine(),
                        nn.Sequential(nn.Linear(int(features),int(features)),
                        Sine(),))))
            else:
                l1 = int(features*ratio**(layers-2-i))
                l2 = int(features*ratio**(layers-2-i-1))
                self.net.append(nn.Sequential(nn.Linear(l1,l2),Sine()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(int(features),data_channel),Sine()))
        else:
            self.net.append(nn.Sequential(nn.Linear(int(features),data_channel)))
        self.net=nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
        # print(self.net)
    def forward(self,coords):
        output = self.net(coords)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,res,ratio,**kwargs):
        if res: 
            param_count=coords_channel*features+features+2*(layers-2)*(features**2+features)+features*data_channel+data_channel
        else:
            l1 = coords_channel
            l2 = int(features*ratio**(layers-2))
            param_count=l1*l2+l2
            for i in range(layers-2):
                l1 = int(features*ratio**(layers-2-i))
                l2 = int(features*ratio**(layers-2-i-1))
                param_count += l1*l2 + l2
            param_count += features*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def check_param_count(param_count,coords_channel,data_channel,layers,res,ratio,**kwargs):
        features = 1                         
        l1 = coords_channel
        l2 = int(features*ratio**(layers-2))
        limit=l1*l2+l2
        for i in range(layers-2):
            l1 = int(features*ratio**(layers-2-i))
            l2 = int(features*ratio**(layers-2-i-1))
            limit += l1*l2 + l2
        limit += features*data_channel+data_channel
        if param_count >= limit:
            return True
        else:
            return False
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,res,ratio,**kwargs):
        if res: 
            a = (layers-2)*2
            b = coords_channel+1+2*layers-4+data_channel
            c = -param_count+data_channel
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        else:
            # pc = r*(1-r**(l-2))/(1-r)*f**2 + ((1-r**(l-2))/(1-r)+(c+1)*r**(l-2)+o)*f + o
            l = layers
            c = coords_channel
            o = data_channel
            r = ratio
            a = r*(1-(r**2)**(l-2))/(1-r**2)
            b = (1-r**(l-2))/(1-r)+(c+1)*r**(l-2)+o
            c = o-param_count
            # print(a,b,c)
            # a = r**5+r**3+r
            # b = 4*r**3+r**2+r+2
            # c = 1 - param_count
            # print(a,b,c)
            # features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
            features = (-b+math.sqrt(b**2-4*a*c))/(2*a)
            if features <= 0:
                raise ValueError
            # 
            l1 = coords_channel
            l2 = features*ratio**(layers-2)
            limit = l1*l2+l2
            for i in range(layers-2):
                l1 = features*ratio**(layers-2-i)     # in 
                l2 = features*ratio**(layers-2-i-1)
                limit += l1*l2 + l2
            limit += features*data_channel+data_channel
            assert abs(param_count-limit) < 1, "ERROR!"
            # quit()
        return features
class SIREN_RELU(nn.Module):
    """
    V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, 
    “Implicit Neural Representations with Periodic Activation Functions,” 
    arXiv:2006.09661 [cs, eess], Jun. 2020, Accessed: May 04, 2021. [Online].
    Available: http://arxiv.org/abs/2006.09661

    Y. Lu, K. Jiang, J. A. Levine, and M. Berger,
    “Compressive Neural Representations,” 
    Computer Graphics Forum, p. 12, 2021.
    """
    def __init__(self, coords_channel=3,data_channel=1,features=256,layers=5,w0=30,res=False,output_act=False,**kwargs):
        super().__init__()
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(coords_channel,features),nn.ReLU(inplace=True)))
        for i in range(layers-2):
            self.net.append(nn.Sequential(nn.Linear(features,features),nn.ReLU(inplace=True)))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel),nn.ReLU(inplace=True)))
        else:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.Sequential(*self.net)
        # self.net.apply(sine_init)
        # self.net[0].apply(first_layer_sine_init)
    def forward(self,coords):
        output = self.net(coords)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,res,**kwargs):
        if res:
            param_count=coords_channel*features+features+2*(layers-2)*(features**2+features)+features*data_channel+data_channel
        else:
            param_count=coords_channel*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,res,**kwargs):
        if res:
            a = (layers-2)*2
            b = coords_channel+1+2*layers-4+data_channel
            c = -param_count+data_channel
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        else:
            a = layers-2
            b = coords_channel+1+layers-2+data_channel
            c = -param_count+data_channel
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features
class SIREN_SIGMOID(nn.Module):
    """
    V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, 
    “Implicit Neural Representations with Periodic Activation Functions,” 
    arXiv:2006.09661 [cs, eess], Jun. 2020, Accessed: May 04, 2021. [Online].
    Available: http://arxiv.org/abs/2006.09661

    Y. Lu, K. Jiang, J. A. Levine, and M. Berger,
    “Compressive Neural Representations,” 
    Computer Graphics Forum, p. 12, 2021.
    """
    def __init__(self, coords_channel=3,data_channel=1,features=256,layers=5,w0=30,res=False,output_act=False,**kwargs):
        super().__init__()
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(coords_channel,features),nn.Sigmoid()))
        for i in range(layers-2):
            self.net.append(nn.Sequential(nn.Linear(features,features),nn.Sigmoid()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel),nn.Sigmoid()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.Sequential(*self.net)
        # self.net.apply(sine_init)
        # self.net[0].apply(first_layer_sine_init)
    def forward(self,coords):
        output = self.net(coords)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,res,**kwargs):
        if res:
            param_count=coords_channel*features+features+2*(layers-2)*(features**2+features)+features*data_channel+data_channel
        else:
            param_count=coords_channel*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,res,**kwargs):
        if res:
            a = (layers-2)*2
            b = coords_channel+1+2*layers-4+data_channel
            c = -param_count+data_channel
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        else:
            a = layers-2
            b = coords_channel+1+layers-2+data_channel
            c = -param_count+data_channel
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features
###########################
class MFNBase(nn.Module):
    def __init__(
        self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

        return

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out
class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))
class MFNFourier(MFNBase):
    """
    R. Fathony, D. Willmott, A. K. Sahu, and J. Z. Kolter,
     “MULTIPLICATIVE FILTER NETWORKS,” p. 11, 2021.
    """
    def __init__(
        self,
        coords_channel=3,
        features=256,
        data_channel=1,
        layers=5,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
        **kwargs
    ):
        super().__init__(
            features, data_channel, layers-2, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                FourierLayer(coords_channel, features, input_scale / np.sqrt(layers-1))
                for _ in range(layers-1)
            ]
        )
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,**kwargs):
        param_count=(layers-2)*(features**2+features)+features*data_channel+data_channel+(layers-1)*(coords_channel*features+features)
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,**kwargs):
        a = layers-2
        b = layers-2+data_channel+(layers-1)*(1+1*coords_channel)
        c = -param_count+data_channel
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features
class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])
class MFNGabor(MFNBase):
    """
    R. Fathony, D. Willmott, A. K. Sahu, and J. Z. Kolter,
     “MULTIPLICATIVE FILTER NETWORKS,” p. 11, 2021.
    """
    def __init__(
        self,
        coords_channel=3,
        features=256,
        data_channel=1,
        layers=5,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        bias=True,
        output_act=False,
        **kwargs
    ):
        super().__init__(
            features, data_channel, layers-2, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    coords_channel,
                    features,
                    input_scale / np.sqrt(layers-1),
                    alpha / (layers-1),
                    beta,
                )
                for _ in range(layers-1)
            ]
        )
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,**kwargs):
        param_count=(layers-2)*(features**2+features)+features*data_channel+data_channel+(layers-1)*(2*coords_channel*features+2*features)
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,**kwargs):
        a = layers-2
        b = layers-2+data_channel+(layers-1)*(2+2*coords_channel)
        c = -param_count+data_channel
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features
ALLPHI={'SIREN':SIREN,'FFN':FFN,'NeRF':NeRF,'MFNFourier':MFNFourier,'MFNGabor':MFNGabor,'SIRENFT':SIRENFT,'SIREN_Pyramid':SIREN_Pyramid,'SIRENPS':SIRENPS,'SIREN_RELU':SIREN_RELU,'SIREN_SIGMOID':SIREN_SIGMOID,'SIRENPos':SIRENPos}
ALL_CALC_PHI_FEATURES = {'SIREN':SIREN.calc_features,'FFN':FFN.calc_features,'NeRF':NeRF.calc_features,'MFNFourier':MFNFourier.calc_features,'MFNGabor':MFNGabor.calc_features,'SIRENFT':SIRENFT.calc_features,'SIREN_Pyramid':SIREN_Pyramid.calc_features,'SIRENPS':SIRENPS.calc_features,'SIREN_RELU':SIREN_RELU.calc_features,'SIREN_SIGMOID':SIREN_SIGMOID.calc_features,'SIRENPos':SIRENPos.calc_features}
ALL_CALC_PHI_PARAM_COUNT = {'SIREN':SIREN.calc_param_count,'FFN':FFN.calc_param_count,'NeRF':NeRF.calc_param_count,'MFNFourier':MFNFourier.calc_param_count,'MFNGabor':MFNGabor.calc_param_count,'SIRENFT':SIRENFT.calc_param_count,'SIREN_Pyramid':SIREN_Pyramid.calc_param_count,'SIRENPS':SIRENPS.calc_param_count,'SIREN_RELU':SIREN_RELU.calc_param_count,'SIREN_SIGMOID':SIREN_SIGMOID.calc_param_count,'SIRENPos':SIRENPos.calc_param_count}
ALL_CHECK_PARAM_COUNT = {'SIRENFT':SIRENFT.check_param_count,'SIREN_Pyramid':SIREN_Pyramid.check_param_count,'SIRENPS':SIRENPS.check_param_count}

def init_phi(kwargs) -> Union[SIREN,FFN,NeRF,MFNFourier,MFNGabor]:
    kwargs = copy.deepcopy(kwargs)
    return ALLPHI[kwargs.pop('name')](**kwargs)
class Projector(nn.Module):
    def __init__(self,layers,in_channel,hidden_channel,out_channel,kernel_size,act,init,bias,output_act) -> None:
        super().__init__()
        self.init = init
        self.bias = bias
        self.net = []
        assert layers > 0
        act_fun_dict = {'relu':nn.ReLU(),'leakyrelu':nn.LeakyReLU(),'none':nn.Identity()}
        act_fun = act_fun_dict[act]
        if output_act:
            output_act_fun = act_fun
        else:
            output_act_fun = nn.Identity()
        if layers == 1:
            self.net.append(nn.Sequential(
                act_fun,
                nn.Conv3d(in_channel,out_channel,kernel_size,padding='same',bias=bias),
                output_act_fun
                ))
        else:
            self.net.append(nn.Sequential(
                act_fun,
                nn.Conv3d(in_channel,hidden_channel,kernel_size,padding='same',bias=bias)
                ))
            for _ in range(layers-2):
                self.net.append(nn.Sequential(
                    act_fun,
                    nn.Conv3d(hidden_channel,hidden_channel,kernel_size,padding='same',bias=bias)
                ))
            self.net.append(nn.Sequential(
                act_fun,
                nn.Conv3d(hidden_channel,out_channel,kernel_size,padding='same',bias=bias),
                output_act_fun
                ))
        self.net = nn.Sequential(*self.net)
        self.net.apply(self.weight_init)
    def weight_init(self,layer):
        if isinstance(layer,nn.Conv3d):
            if self.init == 'ones':
                torch.nn.init.ones_(layer.weight.data)
            elif self.init == 'dirac':
                torch.nn.init.dirac_(layer.weight.data)
            elif self.init == 'none':
                pass
            else:
                raise NotImplementedError
    def forward(self,x,reshape=True):
        if reshape:
            x = rearrange(x,'n d h w c -> n c d h w')
        for layer in self.net:
            x = layer(x)
        if reshape:
            x = rearrange(x,'n c d h w -> n d h w c')
        return x
def calc_phi_hyperparam(param_count:float,name:str,layers:int,coords_channel:int=3,data_channel:int=1,res:bool=False,frequencies:int=10,skip:bool=True,embsize:int=256,**kwargs):
    if name == 'SIREN':
        #####
        # w=inc*f+f+(l-2)(f**2+f)+f*outc+outc
        # (l-2)f**2+(inc+1+l-2+outc)f-w+outc=0
        #####
        if res:
            a = (layers-2)*2
            b = coords_channel+1+2*layers-4+data_channel
            c = -param_count+data_channel
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        else:
            a = layers-2
            b = coords_channel+1+layers-2+data_channel
            c = -param_count+data_channel
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
    elif name == 'NeRF':
        #####
        # if not skip
        # w=d*f+f+(l-2)(f**2+f)+f*outc+outc 
        # (l-2)f**2+(d+1+l-2+outc)f-w+outc=0
        # if skip
        # w=d*f+f+(l-2)(f**2+f)+d*f+f*outc+outc 
        # (l-2)f**2+(2*d+1+l-2+outc)f-w+outc=0
        #####
        d =  coords_channel + 2 * coords_channel * frequencies
        a = layers-2
        if skip:
            b = 2*d+1+layers-2+data_channel
        else:
            b = d+1+layers-2+data_channel
        c = -param_count+data_channel
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
    elif name == 'FFN':
        #####
        # if not skip
        # w=inc*d+d*f+f+(l-2)(f**2+f)+f*outc+outc 
        # (l-2)f**2+(d+1+l-2+outc)f-w+outc+inc*d=0
        # if skip
        # w=inc*d+d*f+f+(l-2)(f**2+f)+d*f+f*outc+outc 
        # (l-2)f**2+(2*d+1+l-2+outc)f-w+outc+inc*d=0
        #####
        d =  2 * embsize
        a = layers-2
        if skip:
            b = 2*d+1+ layers-2+data_channel
        else:
            b = d+1+ layers-2+data_channel
        c = -param_count+data_channel+d*coords_channel
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
    elif name == 'MFNFourier':
        #####
        # w=(l-2)(f**2+f)+f*outc+outc+(l-1)(inc*f+f+inc*f+f)
        # (l-2)f**2+(l-2+outc+(l-1)(inc*2+2))f-w+outc=0
        #####
        a = layers-2
        b = layers-2+data_channel+(layers-1)*(2+2*coords_channel)
        c = -param_count+data_channel
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
    elif name == 'MFNGabor':
        #####
        # w=(l-2)(f**2+f)+f*outc+outc+(l-1)(inc*f+f)
        # (l-2)f**2+(l-2+outc+(l-1)(inc+1))f-w+outc=0
        #####
        a = layers-2
        b = layers-2+data_channel+(layers-1)*(1+1*coords_channel)
        c = -param_count+data_channel
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
    else:
        raise NotImplementedError
    return features
###########################
class Modulator(nn.Module):
    """
    I. Mehta, M. Gharbi, C. Barnes, E. Shechtman, R. Ramamoorthi, and M. Chandraker, 
    “Modulated Periodic Activations for Generalizable Local Functional Representations,” 
    p. 10.
    """
    def __init__(self,y_channel, dim_hidden, layers):
        super().__init__()
        self.net = nn.ModuleList([])
        for ind in range(layers):
            is_first = ind == 0
            dim = y_channel if is_first else (dim_hidden + y_channel)
            self.net.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))
    def forward(self, y):
        mod = y
        mods = []
        for layer in self.net:
            mod = layer(mod)
            mods.append(mod)
            mod = torch.cat((mod, y),dim=-1) 
        return mods
class CropModulator(nn.Module):
    """
    I. Mehta, M. Gharbi, C. Barnes, E. Shechtman, R. Ramamoorthi, and M. Chandraker, 
    “Modulated Periodic Activations for Generalizable Local Functional Representations,” 
    p. 10.
    """
    def __init__(self,y_channel, dim_hidden, layers):
        super().__init__()
        self.net = nn.ModuleList([])
        for ind in range(layers):
            is_first = ind == 0
            dim = y_channel if is_first else (dim_hidden + y_channel)
            self.net.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))
    def forward(self, y):
        n,c,pc_d,pc_h,pc_w = y.shape
        y = rearrange(y,'n c pc_d pc_h pc_w -> (n pc_d pc_h pc_w) c')
        mod = y
        mods = []
        for layer in self.net:
            mod = layer(mod)
            mods.append(rearrange(mod,'(n pc_d pc_h pc_w) c -> n c pc_d pc_h pc_w',n=n,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w))
            mod = torch.cat((mod, y),dim=-1) 
        return mods
ALLGMOD={'Modulator':Modulator,'CropModulator':CropModulator}
def init_gmod(**kwargs) -> Union[CropModulator,Modulator]:
    return ALLGMOD[kwargs.pop('name')](**kwargs)

###########################################
class Conv3dStridedownPooling(nn.Module):
    def __init__(self,data_channel,y_channel,stridedown_layers):
        super().__init__()
        self.stridedown = []
        adaptivepoolshape = (4,10,10)
        dim_hidden = y_channel//(adaptivepoolshape[0]*adaptivepoolshape[1]*adaptivepoolshape[2])
        for ind in range(stridedown_layers):
            if ind == 0:
               self.stridedown.append(nn.Sequential(
                    nn.Conv3d(data_channel,dim_hidden//2**(stridedown_layers-ind-1),(5,5,5),(2,2,2),(2,2,2)),
                    nn.ReLU()
                ))
            else:
                self.stridedown.append(nn.Sequential(
                    nn.Conv3d(dim_hidden//2**(stridedown_layers-ind),dim_hidden//2**(stridedown_layers-ind-1),(5,5,5),(2,2,2),(2,2,2)),
                    nn.ReLU()
                ))
        self.stridedown = nn.Sequential(*self.stridedown)
        self.avgpool = nn.AdaptiveAvgPool3d(adaptivepoolshape)
        self.maxpool = nn.AdaptiveMaxPool3d(adaptivepoolshape)
        self.tail = nn.Sequential(
            nn.Conv3d(dim_hidden*2,dim_hidden,1,1,0),
            Rearrange('n c d h w -> n (c d h w)')
        )
    def forward(self, data):
        outputs = self.stridedown(data)
        outputs_avg = self.avgpool(outputs)
        outputs_max = self.maxpool(outputs)
        y = self.tail(torch.cat([outputs_avg,outputs_max],dim=1))
        return y
class CropConv3dStridedown(nn.Module):
    """
     J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:"Variational Image Compression with a Scale Hyperprior"
    Int. Conf. on Learning Representations (ICLR), 2018 https://arxiv.org/abs/1802.01436 
    """
    def __init__(self,ps_d,ps_h,ps_w,data_channel,y_channel):
        super().__init__()
        ps_min = min(ps_d,ps_h,ps_w)
        Downsample_layer = int(math.log2(ps_min))
        self.net = []
        for ind in range(Downsample_layer):
            if ind == 0 :
                self.net.append(nn.Sequential(
                    nn.Conv3d(data_channel,y_channel,(3,3,3),(2,2,2),(1,1,1)),
                    nn.ReLU()# GDN(y_channel) #FIXME GD t 
                ))
            elif ind == Downsample_layer-1:
                self.net.append(nn.Sequential(
                    nn.Conv3d(y_channel,y_channel,(3,3,3),(2,2,2),(1,1,1)),
                    nn.AdaptiveMaxPool3d((1,1,1)),
                    nn.Conv3d(y_channel,y_channel,(1,1,1),(1,1,1),(0,0,0)),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Conv3d(y_channel,y_channel,(3,3,3),(2,2,2),(1,1,1)),
                    nn.ReLU()# GDN(y_channel)
                ))
        self.net = nn.Sequential(*self.net)
    def forward(self, data,bs:int=0):
        """
        Args:
            bs (int, optional) gp  fo y 
        """
        n,pc_d,pc_h,pc_w,_,_,_,_ = data.shape
        data = rearrange(data,'n pc_d pc_h pc_w c ps_d ps_h ps_w -> (n pc_d pc_h pc_w) c ps_d ps_h ps_w')
        if bs:
            y = []
            for idx in range(math.ceil(n*pc_d*pc_h*pc_w/bs)):
                y.append(self.net(data[idx*bs:idx*bs+bs,...]))
            y = torch.cat(y)
        else:
            y = self.net(data)
        y = rearrange(y,'(n pc_d pc_h pc_w) c i1 i2 i3 -> n (c i1 i2 i3) pc_d pc_h pc_w',pc_d=pc_d, pc_h=pc_h, pc_w=pc_w)
        return y
ALLHY={'Conv3dStridedownPooling':Conv3dStridedownPooling,'CropConv3dStridedown':CropConv3dStridedown}
def init_hy(**kwargs) -> Union[Conv3dStridedownPooling,CropConv3dStridedown]:
    return ALLHY[kwargs.pop('name')](**kwargs)  
##########################################
class UnivariateNonParametricEntropyModel(EntropyBottleneck):
    def __init__(self, channels: int, *args: Any, tail_mass: float = 1e-9, init_scale: float=10, filters: Tuple[int, ...] = (3, 3, 3, 3), optimizer_name_quantiles:str,lr_quantiles:float,**kwargs: Any):
        """
        Args:
            optimizer_name_quantiles (str): used for updating quantiles
            lr_quantiles (float): used for updating quantiles
        """
        super().__init__(channels, *args, tail_mass=tail_mass, init_scale=init_scale, filters=filters, **kwargs)
        self.optimizer_quantiles = configure_optimizer([{'params':self.quantiles}],optimizer_name_quantiles,lr_quantiles)
    def update_quantiles_once(self,):
        loss = self.loss()
        gradient_descent(loss,[self.optimizer_quantiles])
        return loss.item()
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
class GaussianConditionalEntropyModel(GaussianConditional):
    def __init__(self,scale_bound: float, tail_mass: float):
        super().__init__(scale_table=None,scale_bound=scale_bound, tail_mass=tail_mass)
        self.update_scale_table(get_scale_table())
ALLEMY={'UnivariateNonParametricEntropyModel':UnivariateNonParametricEntropyModel}
def init_emy(**kwargs) :
    return ALLEMY[kwargs.pop('name')](**kwargs) 
ALLEMZ={'UnivariateNonParametricEntropyModel':UnivariateNonParametricEntropyModel}
def init_emz(**kwargs) :
    return ALLEMZ[kwargs.pop('name')](**kwargs) 
ALLEMYZ={'GaussianConditionalEntropyModel':GaussianConditionalEntropyModel}
def init_emyz(**kwargs):
    return ALLEMYZ[kwargs.pop('name')](**kwargs) 
###########################################
class PlainConv3dChannelShrink(nn.Module):
    """
     Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).
    """
    def __init__(self,y_channel,z_channel,layers):
        super().__init__()
        self.net = []
        for ind in range(layers):
            if ind == 0 :
                self.net.append(nn.Sequential(
                    nn.Conv3d(y_channel,z_channel,(3,3,3),(1,1,1),(1,1,1)),
                    nn.LeakyReLU(inplace=True)
                ))
            elif ind == layers-1:
                self.net.append(nn.Sequential(
                    nn.Conv3d(z_channel,z_channel,(3,3,3),(1,1,1),(1,1,1)),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Conv3d(z_channel,z_channel,(3,3,3),(1,1,1),(1,1,1)),
                    nn.LeakyReLU(inplace=True)
                ))
        self.net = nn.Sequential(*self.net)
    def forward(self,y:torch.Tensor):
        z = self.net(y)
        return z
ALLHZ={'PlainConv3dChannelShrink':PlainConv3dChannelShrink}
def init_hz(**kwargs):
    return ALLHZ[kwargs.pop('name')](**kwargs)  
###########################################
class PlainConv3dMeanScale(nn.Module):
    """
     Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).
    """
    def __init__(self,y_channel,z_channel,layers):
        super().__init__()
        self.net = []
        for ind in range(layers):
            if ind == 0 :
                self.net.append(nn.Sequential(
                    nn.Conv3d(z_channel,y_channel,(3,3,3),(1,1,1),(1,1,1)),
                    nn.LeakyReLU(inplace=True)
                ))
            elif ind == layers-1:
                self.net.append(nn.Sequential(
                    nn.Conv3d(y_channel,y_channel*2,(3,3,3),(1,1,1),(1,1,1)),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Conv3d(y_channel,y_channel,(3,3,3),(1,1,1),(1,1,1)),
                    nn.LeakyReLU(inplace=True)
                ))
        self.net = nn.Sequential(*self.net)
    def forward(self,z:torch.Tensor):
        y_gaussian_params = self.net(z)
        scales_hat, means_hat = y_gaussian_params.chunk(2, 1)
        return scales_hat, means_hat
ALLGY={'PlainConv3dMeanScale':PlainConv3dMeanScale}
def init_gy(**kwargs):
    return ALLGY[kwargs.pop('name')](**kwargs)  

###########################################
if __name__=='__main__':
    # quick test your model
    nerfnet=NeRF(skip=2)
    data = torch.rand((1,1000,3))
    output = nerfnet(data)
    print(output.shape)

    ffn=FFN(skip=-1)
    data = torch.rand((1,1000,3))
    output = ffn(data)
    print(output.shape)

    siren=SIREN(res=False)
    data = torch.rand((1,1000,3))
    output = siren(data)
    print(output.shape)

    siren_res=SIREN(res=True)
    data = torch.rand((1,1000,3))
    output = siren_res(data)
    print(output.shape)

    gabornet = MFNGabor(3,256,1)
    data = torch.rand((1,1000,3))
    output = gabornet(data)
    print(output.shape)

    fouriernet = MFNFourier(3,256,1)
    data = torch.rand((1,1000,3))
    output = fouriernet(data)
    print(output.shape)