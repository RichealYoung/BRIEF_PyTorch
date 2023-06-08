from utils.misc import omegaconf2dotlist
import numpy as np
import pandas as pd
from utils.Typing import MultiTaskOpt, SingleTaskOpt
from itertools import product
from utils.TasksManager import Queue, Task
import sys
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from typing import List
from omegaconf import OmegaConf
import omegaconf.dictconfig
import omegaconf.listconfig
import time
import logging
import pynvml
import argparse
import shutil
import copy
logging.basicConfig(level=logging.DEBUG)
pynvml.nvmlInit()
timestamp = time.strftime("_%Y_%m%d_%H%M%S")

def dict2dotlist_list(optdict):
    if 'PRODUCT' in optdict.keys():
        dotlist_list = PRODUCT(optdict['PRODUCT'])
    elif 'CONCAT' in optdict.keys():
        dotlist_list = CONCAT(optdict['CONCAT'])
    else:
        dotlist = []
        for k,v in optdict.items():
            dotlist.append('{}={}'.format(k,v))
        dotlist_list = [dotlist]
    return dotlist_list

def PRODUCT(optlist):
    dotlist_list_list = []
    for opt in optlist:
        dotlist_list_list.append(dict2dotlist_list(opt))
    dottuple_list = list(product(*dotlist_list_list))
    dotlist_list = []
    for dottuple in dottuple_list:
        dotlist = []
        for dl in dottuple:
            dotlist.extend(dl)
        dotlist_list.append(dotlist)
    return dotlist_list

def CONCAT(optlist):
    dotlist_list = []
    for opt in optlist:
        dotlist_list.extend(dict2dotlist_list(opt))
    return dotlist_list

def refine_dynamic_for_task_table(dynamic:List[str]) -> dict:
    dynamic_refined = {}
    for dl in dynamic:
        dynamic_refined[dl[:dl.find('=')]] = dl[dl.find('=')+1:]
    return dynamic_refined
def gen_task_list(yaml_path:str,main_script_path:str):
    task_list = []
    opt :MultiTaskOpt = OmegaConf.load(yaml_path)
    temp_dir = opj(opd(yaml_path),'temp_opt_'+opt.Static.Log.project_name)
    os.makedirs(temp_dir,exist_ok=True)
    static = omegaconf2dotlist(opt.Static)
    dynamic_list = CONCAT(opt.Dynamic)
    dotlist_list = [static+dynamic for dynamic in dynamic_list]
    # generate task table for post contrast experiment
    columns = ['name','yaml_path']
    # task
    for task_idx, dotlist in enumerate(dotlist_list):
        task_opt :SingleTaskOpt = OmegaConf.from_dotlist(dotlist)
        source = task_opt.pop('Source')
        task_name = 'exp_{:03}'.format(task_idx)
        # write .yaml file
        task_opt_yaml_path = opj(temp_dir,task_name+'.yaml')
        OmegaConf.save(task_opt,task_opt_yaml_path)
        # instance Task
        command = "python {} -p {}".format(main_script_path,task_opt_yaml_path)
        task_list.append(Task(command,task_name,source.gpucost,source.cpucost))
    return task_list,temp_dir
def main():
    task_list,temp_dir = gen_task_list(yaml_path,singletask_script_path)
    try:
        queue = Queue(task_list,gpu_list)
        queue.init_sharecost_dict()
        queue.start(time_interval=time_interval,max_task=max_task,debug=debug,log=log,autogpu=autogpu)
        shutil.rmtree(temp_dir)
    except:
        shutil.rmtree(temp_dir)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Batch Compress')
    parser.add_argument('-stp',type=str,default='main.py',help='the singletask script path')
    parser.add_argument('-p',type=str,default='opt/MultiTask/default.yaml',help='yaml file path')
    parser.add_argument('-g', help='availabel gpu list',default='0,1',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-t',type=float,default=10,help='the time interval between each task')
    parser.add_argument('-m',type=int,default=30,help='the max nums of task in running')
    parser.add_argument('-debug',action='store_true')
    parser.add_argument('-log',action='store_true')
    parser.add_argument('-onebyone',action='store_true')
    args = parser.parse_args()
    singletask_script_path = args.stp
    yaml_path = args.p
    gpu_list = args.g
    time_interval = args.t
    max_task = args.m
    debug = args.debug
    log = args.log
    onebyone = args.onebyone
    if onebyone:
        max_task = 1
        autogpu = False
    else:
        autogpu = True
    # while True:
    #     key = input('Make sure yaml_path={} gpu_list={}. Press Y/y to continue, N/n to quite.\n'.format(yaml_path,gpu_list))
    #     if key in ['Y','y']:
    #         break
    #     elif key in ['N','n']:
    #         quit()
    main()
