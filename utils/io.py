import math
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import shutil
import time
from typing import Callable, List, Tuple,Dict
import torch
import torch.optim
import numpy as np
from utils.tool import read_img, save_img
import py7zr
from py7zr import FILTER_BZIP2,FILTER_LZMA,FILTER_ZSTD
import yaml
import zipfile
SEVENZFILTER={'default':None,'bzip2':[{'id': FILTER_BZIP2}],'lzma':[{'id': FILTER_LZMA}],'zstd':[{'id': FILTER_ZSTD, 'level': 3}]}
def gen_pathlist_fromimgdir(imgdir)->list:
    imgnamelist=os.listdir(imgdir)
    imgpathlist=[opj(imgdir,imgname) for imgname in imgnamelist]
    imgpathlist.sort()
    return imgpathlist
def gen_pathlist_fromimgdirdir(imgdirdir)->list:
    imgpathlist=[]
    imgdirnamelist=os.listdir(imgdirdir)
    imgdirlist=[opj(imgdirdir,imgdirname) for imgdirname in imgdirnamelist]
    imgdirlist.sort()
    for imgdir in imgdirlist:
        imgnamelist=os.listdir(imgdir)
        imgpathlist_=[opj(imgdir,imgname) for imgname in imgnamelist]
        imgpathlist_.sort()
        imgpathlist.extend(imgpathlist_)
    return imgpathlist
def gen_data_path_list_list(data_dir:str,batch_size:int,drop_last:bool=False) -> List[List[str]]:
    data_path_list_list = []
    data_path_list = gen_pathlist_fromimgdir(data_dir)
    batch_idx = -1
    for batch_idx in range(math.ceil(len(data_path_list)/batch_size)-1):
        data_path_list_list.append([data_path_list[idx] for idx in range(batch_idx*batch_size,batch_idx*batch_size+batch_size)])
    if (batch_idx+2)*batch_size > len(data_path_list):
        if drop_last:
            pass
        else:
            data_path_list_list.append([data_path_list[idx] for idx in range((batch_idx+1)*batch_size,len(data_path_list))])
    else:
        data_path_list_list.append([data_path_list[idx] for idx in range((batch_idx+1)*batch_size,len(data_path_list))])
    return data_path_list_list



def read_data_batch(data_path_list: List[str]) -> np.ndarray:
    data_batch = []
    for data_path in data_path_list:
        data = read_img(data_path)
        data = data[None,None,...]
        data_batch.append(data)
    data_batch = np.concatenate(data_batch,axis=0)
    return data_batch
def save_data_batch(data_batch: np.ndarray,save_path_list: List[str]):
    for idx,save_path in enumerate(save_path_list):
        data = data_batch[idx,0,...]
        save_img(save_path,data)

def normalize_data(data:np.ndarray,name: str,min=None,max=None) -> Tuple[torch.Tensor,dict]:
    #TODO
    if 'minmaxany' in name:
        scale_min,scale_max=name.split('_')[1:]
        scale_min,scale_max = float(scale_min),float(scale_max)
        dtype = data.dtype.name
        data = data.astype(np.float32)
        if min is None:
            min = float(data.min())
        if max is None:
            max = float(data.max())
        data = (data-min)/(max-min)
        data *= (scale_max-scale_min)
        data += scale_min
        data = torch.tensor(data,dtype=torch.float)
        return data,{'dtype':dtype,'min':min,'max':max,'normalized_min':data.min().item(),'normalized_max':data.max().item()}
    elif name == 'minmax01_0mean':
        dtype = data.dtype.name
        data = data.astype(np.float32)
        min = float(data.min())
        max = float(data.max())
        data = (data-min)/(max-min)
        mean = data.mean()
        data = data-mean
        data = torch.tensor(data,dtype=torch.float)
        return data,{'dtype':dtype,'min':min,'max':max,'mean':mean,'normalized_min':-mean,'normalized_max':1-mean}
    elif name == 'minmax01_0mean1std':
        dtype = data.dtype.name
        data = data.astype(np.float32)
        min = float(data.min())
        max = float(data.max())
        data = (data-min)/(max-min)
        mean = data.mean()
        std = data.std()
        data = (data-mean)/std
        data = torch.tensor(data,dtype=torch.float)
        return data,{'dtype':dtype,'min':min,'max':max,'mean':mean,'std':std,'normalized_min':(-mean)/std,'normalized_max':(1-mean)/std}
    elif name == 'none':
        dtype = data.dtype.name
        data = data.astype(np.float32)
        min = float(data.min())
        max = float(data.max())
        data = torch.tensor(data,dtype=torch.float)
        return data,{'dtype':dtype,'min':min,'max':max,'normalized_min':min,'normalized_max':max}
    else:
        raise NotImplementedError
def invnormalize_data(data:torch.Tensor,sideinfos:dict,name: str) -> np.ndarray:
    dtype = sideinfos['dtype']
    if dtype == 'uint8':
        max = 255
        dtype = np.uint8
    elif dtype == 'uint12':
        max = 4098
        dtype = np.uint12
    elif dtype == 'uint16':
        max = 65535
        dtype = np.uint16
    elif dtype == 'float32':
        max = 1e8
        dtype = np.float32
    elif dtype == 'float64':
        max = 1e8
        dtype = np.float64
    elif dtype == 'int16':
        dtype = np.int16     
    else:
        raise NotImplementedError
    if name == 'zeromean_depth':
        data = torch.clip((data + 0.5)*max,0,max)
        data = np.array(data,dtype=dtype)
        return data
    elif 'minmaxany' in name:
        scale_min,scale_max=name.split('_')[1:]
        scale_min,scale_max = float(scale_min),float(scale_max)
        dtype = sideinfos['dtype']
        min = sideinfos['min']
        max = sideinfos['max']
        data -= scale_min
        data /= (scale_max - scale_min)
        data = torch.clip(data,0,1)
        data = data*(max-min)+min
        data = np.array(data,dtype=dtype)
        return data
    elif name == 'minmax01':
        dtype = sideinfos['dtype']
        min = sideinfos['min']
        max = sideinfos['max']
        data = torch.clip(data,0,1)
        data = data*(max-min)+min
        data = np.array(data,dtype=dtype)
        return data
    elif name == 'minmaxn11':
        dtype = sideinfos['dtype']
        min = sideinfos['min']
        max = sideinfos['max']
        data = torch.clip(data,-1,1)
        data = data/2+0.5
        data = data*(max-min)+min
        data = np.array(data,dtype=dtype)
        return data
    elif name == 'minmax01_0mean':
        dtype = sideinfos['dtype']
        min = sideinfos['min']
        max = sideinfos['max']
        mean = sideinfos['mean']
        data = data+mean
        data = torch.clip(data,0,1)
        data = data*(max-min)+min
        data = np.array(data,dtype=dtype)
        return data
    elif name == 'minmax01_0mean_scale10':
        dtype = sideinfos['dtype']
        min = sideinfos['min']
        max = sideinfos['max']
        mean = sideinfos['mean']
        data = data/10+mean
        data = torch.clip(data,0,1)
        data = data*(max-min)+min
        data = np.array(data,dtype=dtype)
        return data
    elif name == 'minmax01_0mean_scale100':
        dtype = sideinfos['dtype']
        min = sideinfos['min']
        max = sideinfos['max']
        mean = sideinfos['mean']
        data = data/100+mean
        data = torch.clip(data,0,1)
        data = data*(max-min)+min
        data = np.array(data,dtype=dtype)
        return data
    elif name == 'minmax01_0mean1std':
        dtype = sideinfos['dtype']
        min = sideinfos['min']
        max = sideinfos['max']
        mean = sideinfos['mean']
        std = sideinfos['std']
        data = data*std+mean
        data = torch.clip(data,0,1)
        data = data*(max-min)+min
        data = np.array(data,dtype=dtype)
        return data
    elif name == 'none':
        dtype = sideinfos['dtype']
        min = sideinfos['min']
        max = sideinfos['max']
        data = torch.clip(data,min,max)
        data = np.array(data,dtype=dtype)
        return data
    else:
        raise NotImplementedError

def get_folder_size(folder_path:str):
    total_size = 0
    if os.path.isdir(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    else:
        total_size = os.path.getsize(folder_path)
    return total_size
def write_yaml(dict,save_path):
    with open(save_path, 'w') as file:
        yaml.dump(dict, file)
def read_yaml(file_path=None,file=None):
    if file is not None:
        dict = yaml.load(file, Loader=yaml.FullLoader)
    else:
        with open(file_path, 'r') as file:
            dict = yaml.load(file, Loader=yaml.FullLoader)
    return dict
def write_binary(binary,save_path):
    with open(save_path, 'wb') as file:
        file.write(binary)
def read_binary(file_path):
    with open(file_path,'rb') as file:
        binary = file.read()
    return binary
def write_zip(file_path_list,save_path,arcname_list=None):
    with zipfile.ZipFile(save_path, 'w') as zip_file:
        for file_path,arcname in zip(file_path_list,arcname_list):
            zip_file.write(file_path,arcname)
def read_zip(file_path):
    zip_data = {}
    with zipfile.ZipFile(file_path,) as zip_file:
        for name in zip_file.NameToInfo.keys():
            with zip_file.open(name) as file:
                zip_data[os.path.basename(name)] = file.read()
    return zip_data
def write_7z(file_path_list,save_path,arcname_list=None,method='default'):
    if arcname_list is None:
        arcname_list = [None for _ in file_path_list]
    filters = SEVENZFILTER[method]
    with py7zr.SevenZipFile(save_path, 'w',filters=filters) as archive:
        archive.writeall(file_path_list[0],arcname_list[0])
    if len(file_path_list)>1:
        with py7zr.SevenZipFile(save_path, 'a',filters=filters) as archive:
            for file_path,arcname in zip(file_path_list[1:],arcname_list[1:]):
                archive.writeall(file_path,arcname)
def extract_7z(file_path:str,extract_to_dir:str):
    with py7zr.SevenZipFile(file_path, 'r') as zip:
        zip.extractall(extract_to_dir)
def read_7z(file_path:str):
    sevenzip_data = {}
    with py7zr.SevenZipFile(file_path, 'r') as zip:
        sevenzip_data = zip.readall()
    # sevenzip_data = {opb(k):sevenzip_data[k] for k in sevenzip_data}
    return sevenzip_data
def get_folder_size(folder_path:str):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size
def write_binary_yaml_zip(binary_list_dict:Dict[str,list],sideinfos_dict:dict,save_path):
    timestamp = time.strftime("_%Y%m%d%H%M%S")
    temp_dir = 'temp'+timestamp
    temp_path_list = []
    os.makedirs(temp_dir,exist_ok=False)
    for key in binary_list_dict.keys():
        for batch_idx,binary in enumerate(binary_list_dict[key]):
            temp_binary_path = os.path.join(temp_dir,key+'_{}'.format(batch_idx))
            temp_path_list.append(temp_binary_path)
            write_binary(binary_list_dict[key][batch_idx],temp_binary_path)
    temp_yaml_path = os.path.join(temp_dir,'sideinfos.yaml')
    temp_path_list.append(temp_yaml_path)
    write_yaml(sideinfos_dict,temp_yaml_path)
    write_zip(temp_path_list,save_path)
    shutil.rmtree(temp_dir)
def read_binary_yaml_zip(binary_name_list:List[str],file_path):
    zip_data = read_zip(file_path)
    sideinfos_dict_file = zip_data['sideinfos.yaml']
    sideinfos_dict = read_yaml(file=sideinfos_dict_file)
    binary_list_dict = {}
    for binary_name in binary_name_list:
        binary_list = []
        for batch_idx in range(int(1e8)):
            binary_path_basename = binary_name+'_{}'.format(batch_idx)
            if binary_path_basename in zip_data.keys():
                binary_list.append(zip_data[binary_path_basename])
            else:
                break
        binary_list_dict[binary_name] = binary_list
    return binary_list_dict,sideinfos_dict