import numpy as np
import os
import struct
import torch
import shutil
import copy

def load_model(model,model_path,device:str='cpu'):
    if hasattr(model, "net"):
        files = os.listdir(model_path)
        for file in files:
            file_path = os.path.join(model_path,file)
            with open(file_path, 'rb') as data_file:
                if 'weight' in file:
                    _,l,shape0,shape1 = file.split('-')
                    l,shape0,shape1 = int(l),int(shape0),int(shape1)
                    weight = np.array(struct.unpack('f'*shape0*shape1, data_file.read())).astype(np.float32).reshape(shape0,shape1)
                    weight = torch.tensor(weight).to(device)
                    with torch.no_grad():
                        model.net[l][0].weight.data = weight
                elif 'bias' in file:
                    _,l,length = file.split('-')
                    l,length = int(l),int(length)
                    bias = np.array(struct.unpack('f'*length, data_file.read())).astype(np.float32)
                    bias = torch.tensor(bias).to(device)
                    with torch.no_grad():
                        model.net[l][0].bias.data = bias
    else:
        model.load_state_dict(torch.load(model_path))
    return model

def save_model(model,save_path,devive:str='cpu'):
    if hasattr(model, "net"):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        for l in range(len(model.net)):
            weight = model.net[l][0].weight.data.to('cpu')
            bias = model.net[l][0].bias.data.to('cpu')
            # weight = copy.deepcopy(weight).to('cpu')
            # bias = copy.deepcopy(bias).to('cpu')
            weight_save_path = os.path.join(save_path,'weight-{}-{}-{}'.format(l,weight.shape[0],weight.shape[1]))
            weight = np.array(weight).reshape(-1)
            with open(weight_save_path, 'wb') as data_file:
                data_file.write(struct.pack('f'*len(weight), *weight))
            bias_save_path = os.path.join(save_path,'bias-{}-{}'.format(l,len(bias)))
            with open(bias_save_path, 'wb') as data_file:
                data_file.write(struct.pack('f'*len(bias), *bias))
            weight = model.net[l][0].weight.data.to(devive)
            bias = model.net[l][0].bias.data.to(devive)
    else:
        model = torch.save(model.state_dict(), save_path)

def CopyDir(old_dir,new_dir):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    files = os.listdir(old_dir)
    for file in files:
        old_path = os.path.join(old_dir,file)
        new_path = os.path.join(new_dir,file)
        shutil.copy(old_path, new_path)
        
    