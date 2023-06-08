import gurobipy as gp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import tifffile
import cv2
from math import log, sqrt, ceil
import os
import numpy as np
import copy
from utils.tool import get_dimension, read_img, save_img
import math

def cal_feature(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft(np.fft.fft(gray,axis=0),axis=1)
    elif len(image.shape) == 4:
        f = np.fft.fft(np.fft.fft(np.fft.fft(image,axis=0),axis=1),axis=2)
    f = np.abs(f)
    feature = int(f.max())/int(f.sum())
    return feature
class Patch2d():
    def __init__(self, optim_model, parent, level, orderx, ordery) -> None:
        self.level = level
        self.orderx = orderx
        self.ordery = ordery
        self.parent = parent
        self.children = []
        self.optim_model = optim_model
        self.active = self.optim_model.addVar(vtype=gp.GRB.BINARY,name=f"{self.level}-{self.orderx}-{self.ordery}")
        self.prune = False
    def get_children(self):
        for i in range(2):
            for j in range(2):
                orderx = 2*self.orderx + j
                ordery = 2*self.ordery + i
                child = Patch2d(self.optim_model,parent=self,level=self.level+1,orderx=orderx,ordery=ordery)
                self.children.append(child)
        return self.children
    def init_data(self,data,h,w):
        self.h = h//(2**self.level)
        self.w = w//(2**self.level)
        self.y = self.h*self.ordery
        self.x = self.w*self.orderx
        self.data = copy.deepcopy(data[self.y:self.y+self.h,self.x:self.x+self.w])
        return self.data
    def get_feature(self, Type):
        self.feature = cal_feature(self.data)
        return self.feature
    def active(self):
        self.prune = False
        self.active = self.optim_model.addVar(vtype=gp.GRB.BINARY,name=f"{self.level}-{self.orderx}-{self.ordery}-{self.orderz}")
    def deactive(self):
        self.prune = True
        self.optim_model.remove(self.active)

class QuadTree():
    def __init__(self, data, max_level, var_thr, e_thr):
        # data
        self.data = data
        self.h = data.shape[0] 
        self.w = data.shape[1]
        self.max_level = max_level
        assert len(self.data.shape) == 2,"data must be 2d!"
        assert self.h%(2**max_level) == 0 and self.w%(2**max_level) == 0,"image size error!"
        # optimizer
        self.optim_model = gp.Model()
        # quadtree and addvar
        self.tree = Patch2d(self.optim_model,parent=None,level=0, orderx=0, ordery=0)
        self.patch_list = []    
        self.patch_dict = {}   
        self.init_tree(self.tree,0)
        self.tree2list(self.tree)
        self.tree2dict(self.tree)
        self.init_data()
        self.prune(var_thr,e_thr)
        self.get_feature()
        self.optim_model.update()
    def init_tree(self,parent,level):
        # self.patch_list.append(parent)
        if level < self.max_level:
            children = parent.get_children() 
            for child in children:
                self.init_tree(child,level+1)
    def tree2list(self,patch):  # dfs
        self.patch_list.append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2list(child)
    def tree2dict(self,patch):  # dfs
        if not (str(patch.level) in self.patch_dict):
            self.patch_dict[str(patch.level)] = [patch]
        else:
            self.patch_dict[str(patch.level)].append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2dict(child)
    def init_data(self):
        for patch in self.patch_list:
            patch.init_data(self.data,self.h,self.w)
    def get_depth(self):
        patch = self.tree
        while len(patch.children) != 0:
            patch = patch.children[0]
        return patch.level
    def get_feature(self):
        for patch in self.patch_list:
            if patch.prune == False: 
                patch.get_feature(self.Type)
    def draw(self,data:np.array=None):
        if data.any() == None:
            data = copy.deepcopy(self.data)
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    x,y,w,h = patch.x,patch.y,patch.w,patch.h
                    cv2.rectangle(data, (x, y), (x+w, y+h), (0, 0, 255), 2)
        return data
    def get_descendants(self,patch):    # Get the descendant node of this node (excluding itself)
        descendants = []
        children = patch.children
        descendants += children
        if len(children) == 0:
            return []
        for child in children:
            descendants += self.get_descendants(child)
        return descendants
    def get_genealogy(self,patch):      # Get all parent nodes of this node (including itself)
        genealogy = [patch]
        while (patch.parent != None):
            genealogy.append(patch.parent)
            patch = patch.parent
        return genealogy
    def solve_optim(self, Nb, min_level):
        self.Nb = Nb
        Obj = []
        Constr = []
        for patch in self.patch_list:
            if patch.prune == False:
                Obj.append(patch.feature*patch.active/(4**patch.level))
                Constr.append(patch.active)
                # 4.the active chunk's level should larger than the min_level
                if patch.level < min_level:
                    self.optim_model.addConstr(patch.active == 0)
        self.optim_model.setObjective(gp.quicksum(Obj), gp.GRB.MAXIMIZE)
        # Add constraints
        # 1.the total numbers of the active chunks should not be larger than the set value
        self.optim_model.addConstr(gp.quicksum(Constr) <= self.Nb)
        # 2.only one menber can be active in the same genealogy
        depth = self.get_depth()
        deepest_layer = self.patch_dict[str(depth)]
        for patch in deepest_layer:
            genealogy = self.get_genealogy(patch)
            actives = []
            for patch in genealogy:
                if patch.prune == False: 
                    actives.append(patch.active)
            # 3.if one member is pruned, the numbers of the other active members in the same genealogy should lease than one
            if len(actives) < len(genealogy) and len(actives) >= 2:
                self.optim_model.addConstr(gp.quicksum(actives) <= 1)
                # print(len(genealogy)-len(actives))
            elif len(actives) == len(genealogy):
                self.optim_model.addConstr(gp.quicksum(actives) == 1)
        # Solve it!
        self.optim_model.optimize()
        print(f"Optimal objective value: {self.optim_model.objVal}")
    def get_active(self):
        self.active_patch_list = []
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    self.active_patch_list.append(patch)
        return self.active_patch_list
    def prune(self,var_thr:float=0,e_thr:float=0):
        count = 0
        for patch in self.patch_list:
            if ((patch.data-patch.data.mean())**2).mean() <= var_thr and abs(patch.data.mean())<=e_thr:
                patch.deactive()
                count += 1
                descendants = self.get_descendants(patch)
                for descendant in descendants:
                    descendant.deactive()
                    count += 1
        print(f'prune numbers:{count}')
    def draw_tree(self):
        actives = {}
        for patch in self.patch_list:
            if not (str(patch.level) in actives):
                actives[str(patch.level)] = [patch.active.x]
            else:
                actives[str(patch.level)].append(patch.active.x)
        for key in actives.keys():
            print(actives[key])
class Patch3d():
    def __init__(self, optim_model, parent, level, orderx, ordery, orderz) -> None:
        self.level = level
        self.orderx = orderx
        self.ordery = ordery
        self.orderz = orderz
        self.parent = parent
        self.children = []
        self.optim_model = optim_model
        self.active()
    def get_children(self):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    orderx = 2*self.orderx + k
                    ordery = 2*self.ordery + j
                    orderz = 2*self.orderz + i
                    child = Patch3d(self.optim_model,parent=self,level=self.level+1,orderx=orderx,ordery=ordery,orderz=orderz)
                    self.children.append(child)
        return self.children
    def init_data(self,data,d,h,w):
        self.d = d//(2**self.level)
        self.h = h//(2**self.level)
        self.w = w//(2**self.level)
        self.z = self.d*self.orderz
        self.y = self.h*self.ordery
        self.x = self.w*self.orderx
        self.data = copy.deepcopy(data[self.z:self.z+self.d,self.y:self.y+self.h,self.x:self.x+self.w])
        return self.data
    def get_feature(self):
        self.feature = cal_feature(self.data)
        return self.feature
    def active(self):
        self.prune = False
        self.active = self.optim_model.addVar(vtype=gp.GRB.BINARY,name=f"{self.level}-{self.orderx}-{self.ordery}-{self.orderz}")
    def deactive(self):
        self.prune = True
        self.optim_model.remove(self.active)

class OctTree():
    def __init__(self, data, max_level, var_thr, e_thr):
        # data (d,h,w)
        self.data = data
        self.d = data.shape[0]  
        self.h = data.shape[1]
        self.w = data.shape[2]
        self.max_level = max_level
        assert len(self.data.shape) == 3 or (len(self.data.shape) == 4 and self.data.shape[-1]==1),"data must be 3d!"
        assert self.d%(2**max_level) ==0 and self.h%(2**max_level) == 0 and self.w%(2**max_level) == 0,"image size error!"
        # optimizer
        self.optim_model = gp.Model()
        # octtree and addvar
        self.tree = Patch3d(self.optim_model,parent=None,level=0, orderx=0, ordery=0, orderz=0)
        self.patch_list = []    
        self.patch_dict = {}   
        self.init_tree(self.tree,0)
        self.tree2list(self.tree)
        self.tree2dict(self.tree)
        self.init_data()
        self.prune(var_thr,e_thr)
        self.get_feature()
        self.optim_model.update()
    def init_tree(self,parent,level):
        if level < self.max_level:
            children = parent.get_children() 
            for child in children:
                self.init_tree(child,level+1)
    def tree2list(self,patch):
        self.patch_list.append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2list(child)
    def tree2dict(self,patch):
        if not (str(patch.level) in self.patch_dict):
            self.patch_dict[str(patch.level)] = [patch]
        else:
            self.patch_dict[str(patch.level)].append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2dict(child)
    def init_data(self):
        for patch in self.patch_list:
            patch.init_data(self.data,self.d,self.h,self.w)
    def get_depth(self):
        patch = self.tree
        while len(patch.children) != 0:
            patch = patch.children[0]
        return patch.level
    def get_feature(self):
        for patch in self.patch_list:
            if patch.prune == False: 
                patch.get_feature()
    def get_descendants(self,patch):
        descendants = []
        children = patch.children
        descendants += children
        if len(children) == 0:
            return []
        for child in children:
            descendants += self.get_descendants(child)
        return descendants
    def get_genealogy(self,patch):
        genealogy = [patch]
        while (patch.parent != None):
            genealogy.append(patch.parent)
            patch = patch.parent
        return genealogy
    def solve_optim(self, Nb, min_level):
        self.Nb = Nb
        Obj = []
        Constr = []
        for patch in self.patch_list:
            if patch.prune == False:
                Obj.append(patch.feature*patch.active/(8**patch.level))
                Constr.append(patch.active)
                # 4.the active chunk's level should larger than the min_level
                if patch.level < min_level:
                    self.optim_model.addConstr(patch.active == 0)
        self.optim_model.setObjective(gp.quicksum(Obj), gp.GRB.MAXIMIZE)
        # Add constraints
        # 1.the total numbers of the active chunks should not be larger than the set value
        self.optim_model.addConstr(gp.quicksum(Constr) <= self.Nb)
        # 2.only one member can be active in the same genealogy
        depth = self.get_depth()
        deepest_layer = self.patch_dict[str(depth)]
        for patch in deepest_layer:
            genealogy = self.get_genealogy(patch)
            actives = []
            for patch in genealogy:
                if patch.prune == False: 
                    actives.append(patch.active)
            # 3.if one member is pruned, the numbers of the other active members in the same genealogy should lease than one
            if len(actives) < len(genealogy) and len(actives) >= 2:
                self.optim_model.addConstr(gp.quicksum(actives) <= 1)
                # print(len(genealogy)-len(actives))
            elif len(actives) == len(genealogy):
                self.optim_model.addConstr(gp.quicksum(actives) == 1)
        # Solve it!
        self.optim_model.optimize()
        # print(f"Optimal objective value: {self.optim_model.objVal}")
    def prune(self,var_thr:float=0,e_thr:float=0):
        count = 0
        for patch in self.patch_list:
            if ((patch.data-patch.data.mean())**2).mean() <= var_thr and abs(patch.data.mean())<=e_thr:
                # print(patch.variance)
                patch.deactive()
                count += 1
                descendants = self.get_descendants(patch)
                for descendant in descendants:
                    descendant.deactive()
                    count += 1
        print(f'prune numbers:{count}')
    def get_active(self):
        self.active_patch_list = []
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    self.active_patch_list.append(patch)
        return self.active_patch_list
    def draw(self,data:np.array=None):
        if data.any() == None:
            data = copy.deepcopy(self.data)
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    x,y,z,w,h,d = patch.x,patch.y,patch.z,patch.w,patch.h,patch.d
                    data[z,y:y+h,x:x+w] = 2000
                    data[z+d-1,y:y+h,x:x+w] = 2000
                    data[z:z+d,y,x:x+w] = 2000
                    data[z:z+d,y+h-1,x:x+w] = 2000
                    data[z:z+d,y:y+h,x] = 2000
                    data[z:z+d,y:y+h,x+w-1] = 2000
        return data
    def draw_tree(self):
        actives = {}
        for patch in self.patch_list:
            if not (str(patch.level) in actives):
                actives[str(patch.level)] = [int(not patch.prune)]
            else:
                actives[str(patch.level)].append(int(not patch.prune))
        for key in actives.keys():
            print(actives[key])

# gpu_limit: 80*80*80*2 = 1024000
def adaptive_cal_tree(img_path,param_size,var_thr:float=-1,e_thr:float=-1,gpu_limit:int=1024000,maxl:int=-1,minl:int=-1,Nb:int=-1):
    dimension = get_dimension(img_path)
    img = read_img(img_path)
    data = copy.deepcopy(img)
    if len(data.shape) == 4:
        if data.shape[-1] == 3:
            gray = np.zeros(data.shape[:-1]).astype(data.dtype.name)
            for i in range(data.shape[0]):
                gray[i] = cv2.cvtColor(data[i],cv2.COLOR_RGB2GRAY)
            data = gray
        if Nb == -1:
            Nb = int(param_size/(4*1361))
            if Nb <= 0:
                Nb = 1
        # minl as uniform as possible
        minl = math.floor(log(Nb,2**dimension))
        maxl = minl + 2
        # Create an octree
        tree = OctTree(data,maxl,var_thr,e_thr)
    elif len(data.shape) == 3:
        if len(data.shape) == 3:
            data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
        # The number of blocks NB 2 * 10 + 10 + 3 * (10 * 10 + 10) + 10 * 1 + 1 = 371 is calculated by taking the ear 
        # parameters of siren network with f = 10, l = 5 and float32 as the average value of block parameters
        if Nb == -1:
            Nb = int(param_size/(4*1361))
            if Nb <= 0:
                Nb = 1
        minl = math.floor(log(Nb,2**dimension))
        maxl = minl + 2
        # Create a quadtree
        tree = QuadTree(data,maxl,var_thr,e_thr)
    tree.solve_optim(Nb,minl)
    save_data = copy.deepcopy(img)
    save_data = tree.draw(save_data)
    info = 'maxl:{},minl:{},var_thr:{},e_thr:{},Nb:{}'.format(maxl,minl,var_thr,e_thr,Nb)
    print(info)
    print('number of blocks:{}'.format(len(tree.get_active())))
    return tree, save_data, dimension

def cal_factor(n):
    fac = [1]
    for i in range(2, n):
        if n % i == 0:
            fac.append(i)
    return fac

def cal_divide_num(d,h,w,Nb,param_size):
    fac_d = cal_factor(d)
    fac_h = cal_factor(h)
    fac_w = cal_factor(w)
    num_max = 0
    if Nb <= 0:
        Nb = int(param_size/(4*1361))
        if Nb <= 0:
            Nb = 1
    for nd in fac_d:
        for nh in fac_h:
            for nw in fac_w:
                num = nd*nh*nw
                if num <= Nb:
                    if num > num_max:
                        num_max = num
                        number = np.array([nd,nh,nw])
                        # var_min = ((number - number.mean())**2).mean()
                        size = np.array([d/nd,h/nh,w/nw])
                        var_min = ((size - size.mean())**2).mean()
                    elif num == num_max:
                        number_tem = np.array([nd,nh,nw])
                        # var_tem = ((number_tem - number_tem.mean())**2).mean()
                        size_tem = np.array([d/nd,h/nh,w/nw])
                        var_tem = ((size_tem - size_tem.mean())**2).mean()
                        if var_tem < var_min:
                            number = number_tem
                            var_min = var_tem
    return number