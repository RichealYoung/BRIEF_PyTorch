import sys
from utils.misc import omegaconf2dict, omegaconf2list
from omegaconf import OmegaConf
import os
from os.path import join as opj
from os.path import dirname as opd
from typing import Dict, Union
import time
from torch.utils.tensorboard import SummaryWriter
timestamp = time.strftime("_%Y_%m%d_%H%M%S")
class MyLogger():
    def __init__(self, project_name:str, stdlog:bool=True, tensorboard:bool=True, outputs_dir:str='outputs', time:bool=False):
        self.project_dir = opj(outputs_dir, project_name)
        self.stdlog = stdlog
        self.tensorboard = tensorboard
        if time:
            self.project_dir += timestamp
        temp_name = self.project_dir
        for i in range(10):
            if not os.path.exists(temp_name):
                break
            temp_name = self.project_dir + '-' + str(i)
        self.project_dir = temp_name
        self.logdir = self.project_dir
        self.logger_dict = {}
        if tensorboard:
            self.tensorboard_init()
        else:
            os.makedirs(self.project_dir, exist_ok=True)
        if stdlog:
            self.stdlog_init()
        self.dir_init()

    def stdlog_init(self):
        stderr_handler=open(opj(self.logdir,'stderr.log'), 'w')
        sys.stderr=stderr_handler
        
    def tensorboard_init(self,):
        self.tblogger = SummaryWriter(self.logdir, flush_secs=30)
        self.logger_dict['tblogger']=self.tblogger
    
    def dir_init(self,):
        self.script_dir = opj(self.project_dir, 'script')
        # self.model_dir = opj(self.project_dir, 'model')
        # self.predict_dir = opj(self.project_dir, 'predict')
        # self.info_dir = opj(self.project_dir, 'info')
        os.mkdir(self.script_dir)
        # os.mkdir(self.model_dir)
        # os.mkdir(self.predict_dir)
        # os.mkdir(self.info_dir)

    def log_metrics(self, metrics_dict: Dict[str, float], iters):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'csvlogger':
                self.logger_dict[logger_name].log_metrics(metrics_dict, iters)
                self.logger_dict[logger_name].save()
            elif logger_name == 'clearml_logger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].report_scalar(k, k, metrics_dict[k], iters)
            elif logger_name == 'tblogger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].add_scalar(k, metrics_dict[k], iters)

    def close(self):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                self.logger_dict[logger_name].close()