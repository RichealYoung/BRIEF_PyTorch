from functools import reduce
import sys
from typing import List, Tuple, Dict, Union
from einops import rearrange,repeat
from torch.types import Number
from utils.Typing import CompressFrameworkOpt,ModuleOpt
from utils.Networks import init_phi,init_gmod,init_hy,init_emy,init_gy,init_emz,init_emyz,init_hz
from utils.io import *
from utils.misc import *
from utils.dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from utils.Logger import MyLogger
from tqdm import tqdm
from compressai.entropy_models import EntropyBottleneck,GaussianConditional
from skimage.metrics import structural_similarity
class _BaseCompressFramerwork:
    def __init__(self,opt: CompressFrameworkOpt) -> None:
        super().__init__()
        self.opt = opt
        self.module:Dict[str,Union[nn.Module,EntropyBottleneck,GaussianConditional]]={}
    def init_module(self,):
        """Initialize every Module in self.module.
        """
        raise NotImplementedError
    def load_module(self,module_path:str,serializing_method:str='torchsave'):#module_state_dict:Dict[str,Dict[str,torch.Tensor]]
        """Load every Module in self.module from the trained module.
        """
        if serializing_method in ['torchsave','torchsave_wo_new_zipfile']:
            module_state_dict = torch.load(module_path,map_location=torch.device('cpu'))
        elif serializing_method in ['torchsave_7z','torchsave_wo_new_zipfile_7z']:
            module_state_dict = read_7z(module_path)
            module_state_dict = torch.load(next(iter(module_state_dict.values())),map_location=torch.device('cpu'))  
        elif serializing_method == 'numpysave':
            module_state_dict = {k:self.module[k].cpu().state_dict() for k in self.module.keys()}
            for n in module_state_dict:
                for k in module_state_dict[n]:
                    data = np.load(opj(module_path,n,k,'0.npy'))
                    data = torch.from_numpy(data)
                    module_state_dict[n][k] = data
        elif serializing_method == 'numpysave_7z':
            module_extracted_dir = opj(opd(module_path),'temp_extracted')
            extract_7z(module_path,module_extracted_dir)
            module_state_dict = {k:self.module[k].cpu().state_dict() for k in self.module.keys()}
            for n in module_state_dict:
                for k in module_state_dict[n]:
                    data = np.load(opj(module_extracted_dir,'0',n,k,'0.npy'))
                    data = torch.from_numpy(data)
                    module_state_dict[n][k] = data
            os.system('rm -rf -r {}'.format(module_extracted_dir))
        for k in module_state_dict.keys():
            self.module[k].load_state_dict(module_state_dict[k])
    def save_module(self,save_path:str,serializing_method:str='torchsave'):
        """Save the trained every Module in self.module into save_path e.g. xxx.pt.
        """
        module_state_dict = {k:self.module[k].cpu().state_dict() for k in self.module.keys()}
        if serializing_method == 'torchsave':
            torch.save(module_state_dict,save_path)
            module_size = os.path.getsize(save_path)
        elif serializing_method == 'torchsave_7z':
            torch.save(module_state_dict,save_path)
            write_7z([save_path],save_path+'_7z',[opb(save_path)])
            os.remove(save_path)
            os.rename(save_path+'_7z',save_path)
            module_size = os.path.getsize(save_path)
        elif serializing_method == 'torchsave_wo_new_zipfile':
            torch.save(module_state_dict,save_path,_use_new_zipfile_serialization=False)
            module_size = os.path.getsize(save_path)
        elif serializing_method == 'torchsave_wo_new_zipfile_7z':
            torch.save(module_state_dict,save_path,_use_new_zipfile_serialization=False)
            write_7z([save_path],save_path+'_7z',[opb(save_path)])
            os.remove(save_path)
            os.rename(save_path+'_7z',save_path)
            module_size = os.path.getsize(save_path)        
        elif serializing_method == 'numpysave':
            os.makedirs(save_path)
            for n in module_state_dict:
                for k in module_state_dict[n]:
                    os.makedirs(opj(save_path,n,k))
                    np.save(opj(save_path,n,k,'data.npy'),module_state_dict[n][k].numpy())
            module_size = get_folder_size(save_path) 
        elif serializing_method == 'numpysave_7z':
            os.makedirs(save_path)
            for n in module_state_dict:
                for k in module_state_dict[n]:
                    os.makedirs(opj(save_path,n,k))
                    np.save(opj(save_path,n,k,'0.npy'),module_state_dict[n][k].numpy())
            write_7z([save_path],save_path+'_7z','0')
            os.system('rm -rf -r {}'.format(save_path))
            os.rename(save_path+'_7z',save_path)
            module_size = os.path.getsize(save_path)  
        return module_size
    def move_module_to(self,device:str):
        """Move every Module in self.module to the given device.
        """
        for k in self.module.keys():
            self.module[k] = self.module[k].to(device)
    def set_module_eval(self,):
        """Set every Module in self.module in evaluation mode.
        """
        for k in self.module.keys():
            self.module[k].eval()
    def set_module_train(self,):
        """Set every Module in self.module in training mode.
        """
        for k in self.module.keys():
            self.module[k].train()
    def set_module_no_grad(self,):
        """Let autograd not record operations on parameters in every Module in self.module
        """
        for k in self.module.keys():
            self.module[k].requires_grad_(False)
    def set_module_grad(self,):
        """Let autograd record operations on parameters in every Module in self.module
        """
        for k in self.module.keys():
            self.module[k].requires_grad_(True)
    def module_parameters(self,) -> List[nn.Parameter]:
        """Return parameters in every Module in self.module.
        """
        parameters = []
        for module in self.module.values():
            parameters.extend(list(module.parameters()))
        return parameters
    def sample_nf(self,coords:torch.Tensor) -> torch.Tensor:
        """Given sampled coords, calculate the corresponding sampled data from the Neural Fileds.
        """
        raise NotImplementedError
    def train(self,save_dir:str):
        """Train every Module in self.module.
        Args:
            save_dir (str): save the trained self.module in this dir. save the files while evaluating performance in this dir.
        """
        raise NotImplementedError
    def compress(self,data_path_list:List[str],save_path:str,) -> torch.Tensor:
        """Compress a batch of data read from the given data_path_list, then save the compressed data into save_path.
        """
        raise NotImplementedError
    def decompress(self,compressed_data_path:str,compressed_data:Dict[str,Union[torch.Tensor,str,dict]],save_path_list:List[str],) -> torch.Tensor:
        """Decompress a batch of data from the given compressed_data or compressed_data_path, then save the decompressed datas into save_path_list.
        """
        raise NotImplementedError
class _BaseNeuralFiledsLocalRep(_BaseCompressFramerwork):
    def __init__(self, opt: CompressFrameworkOpt,) -> None:
        super().__init__(opt)
        self.y_channel = self.opt.Module.gmod.y_channel
        self.data_channel = self.opt.Module.phi.data_channel
    def sample_nf(self, coords: torch.Tensor,mods:List[torch.Tensor]) -> torch.Tensor:
        data_hat = self.module['phi'].forward_syn(coords,mods)
        return data_hat
    def loss_Distortion_func(self, coords: torch.Tensor,y:torch.Tensor,data_gt: torch.Tensor) -> torch.Tensor:
        mods = self.module['gmod'](y)
        data_hat = self.module['phi'].forward_syn(coords,mods)
        loss_Distortion = F.mse_loss(data_hat,data_gt)
        return loss_Distortion
    #@profile_me
    def train(self,save_dir:str,Log:MyLogger):
        os.makedirs(opj(save_dir,'trained_module'),exist_ok=True)
        # device
        device = 'cuda' if self.opt.Train.gpu else 'cpu'
        # module
        self.move_module_to(device)
        # dataset
        train_data_path_list = gen_pathlist_fromimgdir(self.opt.Train.train_data_dir)
        val_data_path_list_list = gen_data_path_list_list(self.opt.Train.val_data_dir,self.opt.Train.val_data_quanity)
        dataset = CropDataset(self.opt.Train.batch_size,self.opt.Train.sample_size,self.opt.Normalize,self.opt.Train.transform,self.opt.Module.crop,True,True,data_path_list=train_data_path_list)
        # optimizer_module
        optimizer_module = configure_optimizer(self.module_parameters(),self.opt.Train.optimizer_name_module,self.opt.Train.lr_module)
        # gradient descent
        max_steps = self.opt.Train.max_steps
        log_every_n_step = self.opt.Train.log_every_n_step
        val_every_n_step = self.opt.Train.val_every_n_step 
        val_every_n_epoch = self.opt.Train.val_every_n_epoch
        def start():
            pbar = tqdm(total=max_steps,desc='Training',file=sys.stdout)
            steps = 0
            for epoch in range(int(1e8)):
                for (cropped_data,cropped_sampler,pc_shape,sideinfos) in dataset:
                    pbar.set_description_str("Training Epoch={}({} steps per epoch = {} dataset_iters* {} cropped_sampler_iters)".format(
                        epoch,dataset.__len__()*cropped_sampler.__len__(),dataset.__len__(),cropped_sampler.__len__()))
                    cropped_data = cropped_data.to(device)
                    # loss_batchdata_value = []
                    for (sampled_coords,sampled_data) in cropped_sampler:
                        sampled_coords = sampled_coords.to(device)
                        sampled_data = sampled_data.to(device)
                        loss_sampledata_value = self.training_step(cropped_data,pc_shape,sideinfos,sampled_coords,sampled_data,optimizer_module,device)
                        # loss_batchdata_value.append(loss_sampledata_value)
                        pbar.update(1)
                        steps += 1
                        if steps % log_every_n_step == 0:
                            Log.log_metrics({'remaining steps/train':max_steps - steps},steps)
                            Log.log_metrics({'loss/train':loss_sampledata_value},steps)
                            pbar.set_postfix_str('loss/train={:.6f}'.format(loss_sampledata_value))
                        if steps % val_every_n_step == 0:
                            # evaluate performance
                            self.save_module(opj(save_dir,'trained_module','epoch_{}_step_{}.pt'.format(epoch,steps)))
                            eval_save_dir = opj(save_dir,'eval_results','epoch_{}_step_{}'.format(epoch,steps))
                            performance = eval_performance(val_data_path_list_list,self,eval_save_dir,True)
                            Log.log_metrics({'performance/mse':performance['mse'].mean(),'performance/psnr':performance['psnr'].mean(),
                        'performance/compress_ratio':performance['compress_ratio'].mean(),'epoch':epoch},steps)
                        if steps == max_steps:
                            self.save_module(opj(save_dir,'trained_module','epoch_{}_step_{}.pt'.format(epoch,steps)))
                            eval_save_dir = opj(save_dir,'eval_results','epoch_{}_step_{}'.format(epoch,steps))
                            performance = eval_performance(val_data_path_list_list,self,eval_save_dir,True)
                            Log.log_metrics({'performance/mse':performance['mse'].mean(),'performance/psnr':performance['psnr'].mean(),
                                'performance/compress_ratio':performance['compress_ratio'].mean(),'epoch':epoch},steps)
                            return            
                if (epoch+1)%val_every_n_epoch==0:
                    # evaluate performance
                    self.save_module(opj(save_dir,'trained_module','epoch_{}_step_{}.pt'.format(epoch,steps)))
                    eval_save_dir = opj(save_dir,'eval_results','epoch_{}_step_{}'.format(epoch,steps))
                    performance = eval_performance(val_data_path_list_list,self,eval_save_dir,True)
                    Log.log_metrics({'performance/mse':performance['mse'].mean(),'performance/psnr':performance['psnr'].mean(),
                        'performance/compress_ratio':performance['compress_ratio'].mean(),'epoch':epoch},steps)
                    
        start()
        self.move_module_to('cpu')
        torch.cuda.empty_cache()
    def training_step(self,cropped_data:torch.Tensor,pc_shape:tuple,sideinfos:dict,sampled_coords:torch.Tensor,sampled_data:torch.Tensor,optimizer_module:torch.optim.Optimizer,device:str) -> Number:
        """Main step of training data
        """
        raise NotImplementedError
    def compress(self, data_path_list: List[str], save_path: str=None):
        # device
        device = 'cuda' if self.opt.Compress.gpu else 'cpu'
        # module
        self.move_module_to(device)
        self.set_module_eval()
        self.set_module_no_grad()
        # dataset
        dataset = CropDataset(len(data_path_list),self.opt.Compress.sample_size,self.opt.Normalize,None,self.opt.Module.crop,False,True,data_path_list=data_path_list)
        cropped_data,cropped_sampler,pc_shape,sideinfos = iter(dataset).__next__()
        cropped_data = cropped_data.to(device)
        compressed_data = self.compressing_data(cropped_data,cropped_sampler,pc_shape,sideinfos,device,save_path)
        self.set_module_train()
        self.set_module_grad()
        torch.cuda.empty_cache()
        return compressed_data
    def compressing_data(self,cropped_data:torch.Tensor,cropped_sampler:CroppedSampler,pc_shape:tuple,sideinfos:dict,device:str,save_path:str) -> Dict[str,Union[torch.Tensor,dict,str]]:
        """Main step of compressing data
        """
        raise NotImplementedError
    def decompress(self, compressed_data_path: str=None, compressed_data:Dict[str,Union[torch.Tensor,Dict]]=None,save_path_list: List[str]=None):
        # device
        device = 'cuda' if self.opt.Decompress.gpu else 'cpu'
        # module
        self.move_module_to(device)
        # decompress
        y,sideinfos = self.decompressing_data(compressed_data_path,compressed_data,device)
        data_shape = sideinfos['data_shape']
        # calc mods Avoiding repetitive computation 
        mods = self.module['gmod'](y)
        # sample from nf
        data = reconstruct_cropped(data_shape,self.opt.Decompress.sample_size,mods,self.sample_nf,**self.opt.Module.crop,device=device)
        data = invnormalize_data(data,sideinfos,**self.opt.Normalize)
        if save_path_list is not None:
            save_data_batch(data,save_path_list)
        return data
    def decompressing_data(self,compressed_data_path:str,compressed_data:Dict[str,Union[torch.Tensor,Dict]],device:str) -> Tuple[torch.Tensor,dict]:
        """Main step of decompressing data
        """
        raise NotImplementedError
class _NFLR(_BaseNeuralFiledsLocalRep):
    def init_compressing_var_optimizer(self,cropped_data:torch.Tensor,sideinfos:dict,pc_shape:tuple,device:str):
        raise NotImplementedError
    def compressing_data(self,cropped_data:torch.Tensor,cropped_sampler:CroppedSampler,pc_shape:tuple,sideinfos:dict,device:str,save_path:str) -> Dict[str,Union[torch.Tensor,dict,str]]:
        y,optimizer_y = self.init_compressing_var_optimizer(cropped_data,sideinfos,pc_shape,device)
        pbar = tqdm(total=self.opt.Compress.max_steps,desc='Compressing',leave=False,file=sys.stdout)
        steps = 0
        for i in range(int(1e8)):
            for (sampled_coords,sampled_data) in cropped_sampler:
                sampled_coords = sampled_coords.to(device)
                sampled_data = sampled_data.to(device)
                loss = self.loss_Distortion_func(sampled_coords,y,sampled_data)
                gradient_descent(loss,[optimizer_y])
                pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
                pbar.update(1)
                steps += 1
                if steps == self.opt.Compress.max_steps:
                    compressed_data = {'sideinfos':sideinfos,'y':y.data.cpu()}
                    if save_path is not None:
                        torch.save(compressed_data,save_path)
                    return compressed_data
    def decompressing_data(self,compressed_data_path:str,compressed_data:Dict[str,Union[torch.Tensor,Dict]],device:str) -> Tuple[torch.Tensor,dict]:
        if compressed_data_path is not None:
            compressed_data = torch.load(compressed_data_path)
        sideinfos = compressed_data['sideinfos']
        y = compressed_data['y'].to(device)
        return y,sideinfos

class NFLR_AutoDecoder(_NFLR):
    def __init__(self, opt: CompressFrameworkOpt) -> None:
        super().__init__(opt)
        self.init_module()
    def init_module(self):
        self.module['phi'] = init_phi(**self.opt.Module.phi)
        self.module['gmod']  = init_gmod(dim_hidden=self.opt.Module.phi.features,layers=self.opt.Module.phi.layers-1,**self.opt.Module.gmod)
    def training_step(self,cropped_data:torch.Tensor,pc_shape:tuple,sideinfos:dict,sampled_coords:torch.Tensor,sampled_data:torch.Tensor,optimizer_module:torch.optim.Optimizer,device:str) -> Number:
        y = init_y(sideinfos['data_shape'][0],self.y_channel,pc_shape,device=device)
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Train.optimizer_name_y,self.opt.Train.lr_y)
        for i in range(self.opt.Train.argmin_steps):
            loss = self.loss_Distortion_func(sampled_coords,y,sampled_data)
            gradient_descent(loss,[optimizer_y])
        loss = self.loss_Distortion_func(sampled_coords,y,sampled_data)
        gradient_descent(loss,[optimizer_module])
        return loss.detach()
    def init_compressing_var_optimizer(self,cropped_data:torch.Tensor,sideinfos:dict,pc_shape:tuple,device:str) -> Tuple[torch.Tensor,torch.optim.Optimizer]:
        y = init_y(sideinfos['data_shape'][0],self.y_channel,pc_shape,device=device)
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Compress.optimizer_name_y,self.opt.Compress.lr_y)
        return y,optimizer_y

class NFLR_AutoEncoder(_NFLR):
    def __init__(self, opt: CompressFrameworkOpt) -> None:
        super().__init__(opt)
        self.init_module()
    def init_module(self):
        self.module['phi'] = init_phi(**self.opt.Module.phi)
        self.module['gmod']  = init_gmod(dim_hidden=self.opt.Module.phi.features,layers=self.opt.Module.phi.layers-1,**self.opt.Module.gmod)
        self.module['hy'] = init_hy(ps_d=self.opt.Module.crop.ps_d,ps_h=self.opt.Module.crop.ps_h,ps_w=self.opt.Module.crop.ps_w,
            y_channel=self.y_channel,data_channel=self.data_channel,**self.opt.Module.hy)
    def training_step(self,cropped_data:torch.Tensor,pc_shape:tuple,sideinfos:dict,sampled_coords:torch.Tensor,sampled_data:torch.Tensor,optimizer_module:torch.optim.Optimizer,device:str) -> Number:
        y = self.module['hy'](cropped_data)
        loss = self.loss_Distortion_func(sampled_coords,y,sampled_data)
        gradient_descent(loss,[optimizer_module])
        return loss.detach()
    def init_compressing_var_optimizer(self,cropped_data:torch.Tensor,sideinfos:dict,pc_shape:tuple,device:str) -> Tuple[torch.Tensor,torch.optim.Optimizer]:
        y = self.module['hy'](cropped_data,self.opt.Compress.hy_bs)
        y.requires_grad=True
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Compress.optimizer_name_y,self.opt.Compress.lr_y)
        return y,optimizer_y
class _NFLR_Coding(_NFLR):
    def loss_RateDistortion_UN(self,coords:torch.Tensor,data_gt:torch.Tensor,y:torch.Tensor,Lambda:float) -> torch.Tensor:
        # relax y by adding noise & calc likelihoods
        y_noisy,y_likelihoods = self.module['emy'](y,training=True) # set training=True can add uniform noise and calculate the noisy_var's likelihood
        # calc loss
        loss = loss_bpp_func(y_likelihoods)+Lambda*self.loss_Distortion_func(coords,y_noisy,data_gt)
        return loss
    def loss_RateDistortion_SGA(self,coords:torch.Tensor,data_gt:torch.Tensor,y:torch.Tensor,Lambda:float,tau:float) -> torch.Tensor:
        epsilon = 1e-5
        # relax y by SGA
        y_floor = torch.floor(y)
        y_ceil = torch.ceil(y)
        y_bds = torch.stack([y_floor, y_ceil], axis=-1)
        ry_logits = torch.stack([-torch.atanh(torch.clip(y - y_floor, -1 + epsilon, 1 - epsilon)) / tau,-torch.atanh(torch.clip(y_ceil - y, -1 + epsilon, 1 - epsilon)) / tau],
                        axis=-1)  # last dim are logits for DOWN or UP; clip to prevent NaN as temperature -> 0
        ry_dist = RelaxedOneHotCategorical(tau, logits=ry_logits)  # technically we can use a different temperature here
        ry_sample = ry_dist.sample()
        y_tilde = torch.sum(y_bds * ry_sample, axis=-1)  # inner product in last dim
        # calc likelihoods
        if len(y_tilde.shape) == 4:
            n,c,h,w = y_tilde.shape
            y_likelihoods = self.module['emy']._likelihood(rearrange(y_tilde,'n c h w -> c () (n h w)'))
            y_likelihoods = rearrange(y_likelihoods,'c i (n h w) -> n (c i) h w',n=n,h=h,w=w)
        elif len(y_tilde.shape) == 5:
            n,c,d,h,w = y_tilde.shape
            y_likelihoods = self.module['emy']._likelihood(rearrange(y_tilde,'n c d h w -> c () (n d h w)'))
            y_likelihoods = rearrange(y_likelihoods,'c i (n d h w) -> n (c i) d h w',n=n,d=d,h=h,w=w)
        # calc loss
        loss = loss_bpp_func(y_likelihoods)+Lambda*self.loss_Distortion_func(coords,y_tilde,data_gt)
        return loss
    def compressing_data(self,cropped_data:torch.Tensor,cropped_sampler:CroppedSampler,pc_shape:tuple,sideinfos:dict,device:str,save_path:str) -> Dict[str,Union[torch.Tensor,dict,str]]:
        y,optimizer_y = self.init_compressing_var_optimizer(cropped_data,sideinfos,pc_shape,device)
        pbar = tqdm(total=self.opt.Compress.max_steps,desc='Compressing',leave=False,file=sys.stdout)
        steps = 0
        for i in range(int(1e8)):
            for (sampled_coords,sampled_data) in cropped_sampler:
                tau = annealed_temperature(i, r=self.opt.Compress.annealing_rate, ub=self.opt.Compress.init_temperature, scheme=self.opt.Compress.annealing_scheme, t0=self.opt.Compress.t0)
                sampled_coords = sampled_coords.to(device)
                sampled_data = sampled_data.to(device)
                loss = self.loss_RateDistortion_SGA(sampled_coords,sampled_data,y,self.opt.Train.Lambda,tau)
                gradient_descent(loss,[optimizer_y])
                pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
                pbar.update(1)
                steps += 1
                if steps == self.opt.Compress.max_steps:
                    self.module['emy'].update()
                    y_strings = self.module['emy'].compress(y)
                    sideinfos['y_shape'] = list(y.shape[2:])
                    if save_path is not None:
                        write_binary_yaml_zip({'y_strings':y_strings},sideinfos,save_path)
                    return {'y_strings':y_strings,'sideinfos':sideinfos}
    def decompressing_data(self,compressed_data_path:str,compressed_data:Dict[str,Union[torch.Tensor,Dict]],device:str) -> Tuple[torch.Tensor,dict]:
        if compressed_data_path is not None:
            binary_list_dict,sideinfos_dict = read_binary_yaml_zip(['y_strings'],compressed_data_path)
            compressed_data = {**binary_list_dict,**{'sideinfos':sideinfos_dict}}
        sideinfos = compressed_data['sideinfos']
        y_strings = compressed_data['y_strings']
        self.module['emy'].update()
        y = self.module['emy'].decompress(y_strings, sideinfos['y_shape'])
        y = y.to(device)
        return y,sideinfos

class NFLR_Coding_AutoDecoder(_NFLR_Coding):
    def __init__(self, opt: CompressFrameworkOpt) -> None:
        super().__init__(opt)
        self.init_module()
    def init_module(self):
        self.module['phi'] = init_phi(**self.opt.Module.phi)
        self.module['gmod']  = init_gmod(dim_hidden=self.opt.Module.phi.features,layers=self.opt.Module.phi.layers-1,**self.opt.Module.gmod)
        self.module['emy'] = init_emy(channels = self.y_channel,**self.opt.Module.emy)
    def training_step(self,cropped_data:torch.Tensor,pc_shape:tuple,sideinfos:dict,sampled_coords:torch.Tensor,sampled_data:torch.Tensor,optimizer_module:torch.optim.Optimizer,device:str) -> Number:
        y = init_y(sideinfos['data_shape'][0],self.y_channel,pc_shape,device=device)
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Train.optimizer_name_y,self.opt.Train.lr_y)
        for i in range(self.opt.Train.argmin_steps):
            loss = self.loss_RateDistortion_UN(sampled_coords,sampled_data,y,self.opt.Train.Lambda)
            gradient_descent(loss,[optimizer_y])
        loss = self.loss_RateDistortion_UN(sampled_coords,sampled_data,y,self.opt.Train.Lambda)
        gradient_descent(loss,[optimizer_module])
        self.module['emy'].update_quantiles_once()
        return loss.detach()
    def init_compressing_var_optimizer(self,cropped_data:torch.Tensor,sideinfos:dict,pc_shape:tuple,device:str) -> Tuple[torch.Tensor,torch.optim.Optimizer]:
        y = init_y(sideinfos['data_shape'][0],self.y_channel,pc_shape,device=device)
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Compress.optimizer_name_y,self.opt.Compress.lr_y)
        return y,optimizer_y

class NFLR_Coding_AutoEncoder(_NFLR_Coding):
    def __init__(self, opt: CompressFrameworkOpt) -> None:
        super().__init__(opt)
        self.init_module()
    def init_module(self):
        self.module['phi'] = init_phi(**self.opt.Module.phi)
        self.module['gmod']  = init_gmod(dim_hidden=self.opt.Module.phi.features,layers=self.opt.Module.phi.layers-1,**self.opt.Module.gmod)
        self.module['emy'] = init_emy(channels = self.y_channel,**self.opt.Module.emy)
        self.module['hy'] = init_hy(ps_d=self.opt.Module.crop.ps_d,ps_h=self.opt.Module.crop.ps_h,ps_w=self.opt.Module.crop.ps_w,
            y_channel=self.y_channel,data_channel=self.data_channel,**self.opt.Module.hy)
    def training_step(self,cropped_data:torch.Tensor,pc_shape:tuple,sideinfos:dict,sampled_coords:torch.Tensor,sampled_data:torch.Tensor,optimizer_module:torch.optim.Optimizer,device:str) -> Number:
        y = self.module['hy'](cropped_data)
        loss = self.loss_RateDistortion_UN(sampled_coords,sampled_data,y,self.opt.Train.Lambda)
        gradient_descent(loss,[optimizer_module])
        self.module['emy'].update_quantiles_once()
        return loss.detach()
    def init_compressing_var_optimizer(self,cropped_data:torch.Tensor,sideinfos:dict,pc_shape:tuple,device:str) -> Tuple[torch.Tensor,torch.optim.Optimizer]:
        y = self.module['hy'](cropped_data,self.opt.Compress.hy_bs)
        y.requires_grad=True
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Compress.optimizer_name_y,self.opt.Compress.lr_y)
        return y,optimizer_y

class _NFLR_Coding_Hyper(_NFLR_Coding):
    def loss_RateDistortion_hyper_UN(self,coords:torch.Tensor,data_gt:torch.Tensor,y:torch.Tensor,z:torch.Tensor,Lambda:float) -> torch.Tensor:
        # relax z by adding noise & calc likelihoods
        z_noisy,z_likelihoods = self.module['emz'](z,training=True) # set training=True can add uniform noise and calculate the noisy_var's likelihood
        # estimate y_gaussian_params
        scales_hat, means_hat = self.module['gy'](z_noisy)
        scales_hat = torch.exp(scales_hat) # make positive
        # relax y by adding noise & calc likelihoods
        y_noisy, y_likelihoods = self.module['emyz'](y, scales_hat, means=means_hat,training=True) # add uniform noise and calculate the noisy_var's likelihood
        # calc loss
        loss = loss_bpp_func(y_likelihoods)+loss_bpp_func(z_likelihoods)+Lambda*self.loss_Distortion_func(coords,y_noisy,data_gt)
        return loss
    def loss_RateDistortion_hyper_SGA(self,coords:torch.Tensor,data_gt:torch.Tensor,y:torch.Tensor,z:torch.Tensor,Lambda:float,tau:float) -> torch.Tensor:
        epsilon = 1e-5
        # relax z by SGA
        z_floor = torch.floor(z)
        z_ceil = torch.ceil(z)
        z_bds = torch.stack([z_floor, z_ceil], axis=-1)
        rz_logits = torch.stack([-torch.atanh(torch.clip(z - z_floor, -1 + epsilon, 1 - epsilon)) / tau,-torch.atanh(torch.clip(z_ceil - z, -1 + epsilon, 1 - epsilon)) / tau],
                        axis=-1)  # last dim are logits for DOWN or UP; clip to prevent NaN as temperature -> 0
        rz_dist = RelaxedOneHotCategorical(tau, logits=rz_logits)  # technically we can use a different temperature here
        rz_sample = rz_dist.sample()
        z_tilde = torch.sum(z_bds * rz_sample, axis=-1)  # inner product in last dim
        # calc likelihoods
        if len(z_tilde.shape) == 4:
            n,c,h,w = z_tilde.shape
            z_likelihoods = self.module['emz']._likelihood(rearrange(z_tilde,'n c h w -> c () (n h w)'))
            z_likelihoods = rearrange(z_likelihoods,'c i (n h w) -> n (c i) h w',n=n,h=h,w=w)
        elif len(z_tilde.shape) == 5:
            n,c,d,h,w = z_tilde.shape
            z_likelihoods = self.module['emz']._likelihood(rearrange(z_tilde,'n c d h w -> c () (n d h w)'))
            z_likelihoods = rearrange(z_likelihoods,'c i (n d h w) -> n (c i) d h w',n=n,d=d,h=h,w=w)
        # estimate y_gaussian_params
        scales_hat, means_hat = self.module['gy'](z_tilde)
        scales_hat = torch.exp(scales_hat) # make positive
        # relax y by SGA
        y_floor = torch.floor(y)
        y_ceil = torch.ceil(y)
        y_bds = torch.stack([y_floor, y_ceil], axis=-1)
        ry_logits = torch.stack([-torch.atanh(torch.clip(y - y_floor, -1 + epsilon, 1 - epsilon)) / tau,-torch.atanh(torch.clip(y_ceil - y, -1 + epsilon, 1 - epsilon)) / tau],
                        axis=-1)  # last dim are logits for DOWN or UP; clip to prevent NaN as temperature -> 0
        ry_dist = RelaxedOneHotCategorical(tau, logits=ry_logits)  # technically we can use a different temperature here
        ry_sample = ry_dist.sample()
        y_tilde = torch.sum(y_bds * ry_sample, axis=-1)  # inner product in last dim
        # calc likelihoods
        y_likelihoods = self.module['emyz']._likelihood(y_tilde,scales_hat, means_hat)
        # calc loss
        loss = loss_bpp_func(y_likelihoods)+loss_bpp_func(z_likelihoods)+Lambda*self.loss_Distortion_func(coords,y_tilde,data_gt)
        return loss
    def compressing_data(self,cropped_data:torch.Tensor,cropped_sampler:CroppedSampler,pc_shape:tuple,sideinfos:dict,device:str,save_path:str) -> Dict[str,Union[torch.Tensor,dict,str]]:
        y,z,optimizer_y,optimizer_z = self.init_compressing_var_optimizer(cropped_data,sideinfos,pc_shape,device)
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Compress.optimizer_name_y,self.opt.Compress.lr_y)
        optimizer_z = configure_optimizer([{'params':z}],self.opt.Compress.optimizer_name_z,self.opt.Compress.lr_z)
        pbar = tqdm(total=self.opt.Compress.max_steps,desc='Compressing',leave=False,file=sys.stdout)
        steps = 0
        for i in range(int(1e8)):
            for (sampled_coords,sampled_data) in cropped_sampler:
                tau = annealed_temperature(i, r=self.opt.Compress.annealing_rate, ub=self.opt.Compress.init_temperature, scheme=self.opt.Compress.annealing_scheme, t0=self.opt.Compress.t0)
                sampled_coords = sampled_coords.to(device)
                sampled_data = sampled_data.to(device)
                loss = self.loss_RateDistortion_hyper_SGA(sampled_coords,sampled_data,y,z,self.opt.Train.Lambda,tau)
                gradient_descent(loss,[optimizer_y,optimizer_z])
                pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
                pbar.update(1)
                steps += 1
                if steps == self.opt.Compress.max_steps:
                    self.module['emz'].update()
                    z_strings = self.module['emz'].compress(z)
                    z_rec = self.module['emz'].decompress(z_strings, z.size()[2:])
                    scales, means = self.module['gy'](z_rec)
                    scales = torch.exp(scales) # make positive
                    indexes = self.module['emyz'].build_indexes(scales)
                    y_strings = self.module['emyz'].compress(y,indexes,means=means)
                    sideinfos['z_shape'] = list(z.shape[2:])
                    if save_path is not None:
                        write_binary_yaml_zip({'y_strings':y_strings,'z_strings':z_strings},sideinfos,save_path)
                    return {'y_strings':y_strings,'z_strings':z_strings,'sideinfos':sideinfos}
    def decompressing_data(self,compressed_data_path:str,compressed_data:Dict[str,Union[torch.Tensor,Dict]],device:str) -> Tuple[torch.Tensor,dict]:
        if compressed_data_path is not None:
            binary_list_dict,sideinfos_dict = read_binary_yaml_zip(['y_strings','z_strings'],compressed_data_path)
            compressed_data = {**binary_list_dict,**{'sideinfos':sideinfos_dict}}
        sideinfos = compressed_data['sideinfos']
        y_strings = compressed_data['y_strings']
        z_strings = compressed_data['z_strings']
        self.module['emz'].update()
        z = self.module['emz'].decompress(z_strings, sideinfos['z_shape'])
        scales, means = self.module['gy'](z)
        scales = torch.exp(scales) # make positive
        indexes = self.module['emyz'].build_indexes(scales)
        y = self.module['emyz'].decompress(y_strings,indexes,means=means)
        y = y.to(device)
        return y,sideinfos
class NFLR_Coding_Hyper_AutoDecoder(_NFLR_Coding_Hyper):
    def __init__(self, opt: CompressFrameworkOpt) -> None:
        super().__init__(opt)
        self.init_module()
    def init_module(self):
        self.z_channel = self.opt.Module.gy.z_channel
        self.module['phi'] = init_phi(**self.opt.Module.phi)
        self.module['gmod']  = init_gmod(dim_hidden=self.opt.Module.phi.features,layers=self.opt.Module.phi.layers-1,**self.opt.Module.gmod)
        self.module['gy'] = init_gy(y_channel = self.y_channel,**self.opt.Module.gy)
        self.module['emyz'] = init_emyz(**self.opt.Module.emyz)
        self.module['emz'] = init_emz(channels = self.z_channel,**self.opt.Module.emz)
    def training_step(self,cropped_data:torch.Tensor,pc_shape:tuple,sideinfos:dict,sampled_coords:torch.Tensor,sampled_data:torch.Tensor,optimizer_module:torch.optim.Optimizer,device:str) -> Number:
        y = init_y(sideinfos['data_shape'][0],self.y_channel,pc_shape,device=device)
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Train.optimizer_name_y,self.opt.Train.lr_y)
        z = init_z(sideinfos['data_shape'][0],self.z_channel,pc_shape,device=device)
        optimizer_z = configure_optimizer([{'params':z}],self.opt.Train.optimizer_name_z,self.opt.Train.lr_z)
        for i in range(self.opt.Train.argmin_steps):
            loss = self.loss_RateDistortion_hyper_UN(sampled_coords,sampled_data,y,z,self.opt.Train.Lambda)
            gradient_descent(loss,[optimizer_y,optimizer_z])
        loss = self.loss_RateDistortion_hyper_UN(sampled_coords,sampled_data,y,z,self.opt.Train.Lambda)
        gradient_descent(loss,[optimizer_module])
        return loss.detach()
    def init_compressing_var_optimizer(self,cropped_data:torch.Tensor,sideinfos:dict,pc_shape:tuple,device:str) -> Tuple[torch.Tensor,torch.Tensor,torch.optim.Optimizer,torch.optim.Optimizer]:
        y = init_y(sideinfos['data_shape'][0],self.y_channel,pc_shape,device=device)
        z = init_z(sideinfos['data_shape'][0],self.z_channel,pc_shape,device=device)
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Compress.optimizer_name_y,self.opt.Compress.lr_y)
        optimizer_z = configure_optimizer([{'params':z}],self.opt.Compress.optimizer_name_z,self.opt.Compress.lr_z)
        return y,z,optimizer_y,optimizer_z
class NFLR_Coding_Hyper_AutoEncoder(_NFLR_Coding_Hyper):
    def __init__(self, opt: CompressFrameworkOpt) -> None:
        super().__init__(opt)
        self.init_module()
    def init_module(self):
        self.z_channel = self.opt.Module.gy.z_channel
        self.module['phi'] = init_phi(**self.opt.Module.phi)
        self.module['gmod']  = init_gmod(dim_hidden=self.opt.Module.phi.features,layers=self.opt.Module.phi.layers-1,**self.opt.Module.gmod)
        self.module['gy'] = init_gy(y_channel = self.y_channel,**self.opt.Module.gy)
        self.module['emyz'] = init_emyz(**self.opt.Module.emyz)
        self.module['emz'] = init_emz(channels = self.z_channel,**self.opt.Module.emz)
        self.module['hy'] = init_hy(ps_d=self.opt.Module.crop.ps_d,ps_h=self.opt.Module.crop.ps_h,ps_w=self.opt.Module.crop.ps_w,
            y_channel=self.y_channel,data_channel=self.data_channel,**self.opt.Module.hy)
        self.module['hz'] = init_hz(y_channel = self.y_channel,z_channel = self.z_channel,layers = self.opt.Module.gy.layers,**self.opt.Module.hz)
    def training_step(self,cropped_data:torch.Tensor,pc_shape:tuple,sideinfos:dict,sampled_coords:torch.Tensor,sampled_data:torch.Tensor,optimizer_module:torch.optim.Optimizer,device:str) -> Number:
        y = self.module['hy'](cropped_data)
        z = self.module['hz'](y)
        loss = self.loss_RateDistortion_hyper_UN(sampled_coords,sampled_data,y,z,self.opt.Train.Lambda)
        gradient_descent(loss,[optimizer_module])
        return loss.detach()
    def init_compressing_var_optimizer(self,cropped_data:torch.Tensor,sideinfos:dict,pc_shape:tuple,device:str) -> Tuple[torch.Tensor,torch.Tensor,torch.optim.Optimizer,torch.optim.Optimizer]:
        y = self.module['hy'](cropped_data,self.opt.Compress.hy_bs)
        z = self.module['hz'](y)
        y.requires_grad=True
        z.requires_grad=True
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Compress.optimizer_name_y,self.opt.Compress.lr_y)
        optimizer_z = configure_optimizer([{'params':z}],self.opt.Compress.optimizer_name_z,self.opt.Compress.lr_z)
        return y,z,optimizer_y,optimizer_z


ALLCF={'NFLR_AutoDecoder':NFLR_AutoDecoder,'NFLR_AutoEncoder':NFLR_AutoEncoder,'NFLR_Coding_AutoDecoder':NFLR_Coding_AutoDecoder,'NFLR_Coding_AutoEncoder':NFLR_Coding_AutoEncoder,
        'NFLR_Coding_Hyper_AutoDecoder':NFLR_Coding_Hyper_AutoDecoder,'NFLR_Coding_Hyper_AutoEncoder':NFLR_Coding_Hyper_AutoEncoder}
def init_compressframework(opt:CompressFrameworkOpt) -> Union[NFLR_AutoDecoder,NFLR_AutoEncoder,NFLR_Coding_AutoDecoder,NFLR_Coding_AutoEncoder,NFLR_Coding_Hyper_AutoDecoder,NFLR_Coding_Hyper_AutoEncoder]:
    return ALLCF[opt.Name](opt)

def eval_performance(data_path_list_list:List[List[str]],compressframework:_BaseCompressFramerwork,save_dir:str,keep_data:bool=True,max=None) -> pd.DataFrame:
    """Evaluate the given compressframework's performace.
    Args:
        data_path_list_list (List[List[str]]): the data path list to evaluate performance
        save_dir (str): the dir to save compressed data,decompressed data,performance.
        keep_data (bool, optional): whether to keep the compressed data and decompressed data. Defaults to True.
    """
    os.makedirs(save_dir,exist_ok=True)
    metrics = ['data_path','mse','psnr','ssim','compressed_data_Mbytes','orig_data_Mbytes','compress_ratio',]
    performance = pd.DataFrame(np.zeros((len(data_path_list_list),len(metrics))),columns=metrics)
    for idx,data_path_list in enumerate(tqdm(data_path_list_list,'Evaluating performance',leave=False)):
        compressed_data_save_path = opj(save_dir,opb(ops(data_path_list[0])[0])+'_compressed')
        decompressed_data_save_path_list = [opj(save_dir,opb(ops(data_path)[0])+'_decompressed'+ops(data_path)[-1]) for data_path in data_path_list]
        compressed_data = compressframework.compress(data_path_list=data_path_list,save_path=compressed_data_save_path)
        decompressed_data = compressframework.decompress(compressed_data=compressed_data,save_path_list=decompressed_data_save_path_list)
        orig_data = read_data_batch(data_path_list)
        dtype = orig_data.dtype.name
        if max is None:
            if dtype == 'uint8':
                max = 255
            elif dtype == 'uint12':
                max = 4098
            elif dtype == 'uint16':
                max = 65535
            else:
                raise NotImplementedError
        mse = np.mean(np.power(orig_data/max-decompressed_data/max,2))
        psnr = -10*np.log10(mse)
        ssim_list = []
        if len(orig_data.shape) == 5:
            for i in range(orig_data.shape[0]):
                orig_data_ = rearrange(orig_data[i],'c d h w -> d h w c')
                decompressed_data_ = rearrange(decompressed_data[i],'c d h w -> d h w c')
                ssim_list.append(structural_similarity(orig_data_,decompressed_data_,data_range=max,multichannel=True)) 
        else:
            raise NotImplementedError
        ssim = sum(ssim_list)/len(ssim_list)
        orig_data_Mbytes = reduce(lambda x,y:x*y,[os.path.getsize(data_path)/2**20 for data_path in data_path_list])
        compressed_data_Mbytes = os.path.getsize(compressed_data_save_path)/2**20
        compress_ratio = orig_data_Mbytes/compressed_data_Mbytes
        performance.iloc[idx]=[str(data_path_list),mse,psnr,ssim,compressed_data_Mbytes,orig_data_Mbytes,compress_ratio]
        if not keep_data:
            os.remove(compressed_data_save_path)
            for decompressed_data_save_path in decompressed_data_save_path_list:
                os.remove(decompressed_data_save_path)
    performance.to_csv(opj(save_dir,'performance.csv'))
    return performance
