from dataclasses import dataclass
from typing import Any, List
#################
@dataclass
class divideopt:
    divide_type: str
    zslice: str
    param_alloc: str
    param_size_thres: float
    exception: dict
    module_7z: bool
@dataclass
class sampleropt:
    name: str
    cube_count: int
    cube_len: List[int]
    sample_size: int
    gpu_force: bool
@dataclass
class preprocessopt:
    denoise: Any
    clip: List[int]
class paramopt:
    filesize_ratio: int
    given_size: float
@dataclass
class lossopt:
    name: str
    beta: float
    weight: List[str]
    weight_thres: float
#############
@dataclass
class CompressOpt:
    divide: divideopt
    half: bool
    module_serializing_method: str
    sampler: sampleropt
    coords_mode: str
    preprocess: preprocessopt
    param: paramopt
    loss: lossopt
    gpu: bool
    max_steps: int
    checkpoints: str
    loss_log_freq: int
    lr_phi: float
    optimizer_name_phi: str
    lr_scheduler_phi: Any
    decompress: bool
@dataclass
class DecompressOpt:
    gpu: bool
    sample_size: int
    postprocess: preprocessopt
    keep_decompressed: bool
    mip: bool
    ssim: bool
    aoi_thres_list: List[int]
    mse_mape_thres_list: List[int]
    psnr_type_list: List[dict]
@dataclass
class CropOpt:
    ps_d: int # 2**k
    ps_h: int # 2**k
    ps_w: int # 2**k
    ol_d: int
    ol_h: int
    ol_w: int
@dataclass
class ModuleOpt:
    phi: Any
    projector: Any
    gmod: Any
    gf: Any
    hy: Any
    emy: Any
    gy: Any
    emz: Any
    emyz: Any
    hz: Any
    crop: CropOpt = CropOpt

@dataclass
class NormalizeOpt:
    name: str
@dataclass
class Datasetopt:
    data_path: str
@dataclass
class TransformOpt:
    Crop3d: Any
    RandomCrop3d: Any
    Resize3d: Any
    RandomResize3d: Any
    FlipRoat3d: Any
@dataclass
class TrainOpt:
    train_data_dir: str
    val_data_dir: str
    sample_size: int # sample #sample_size per patch
    batch_size: int
    max_steps: int
    gpu: bool
    log_every_n_step: int
    val_every_n_step: int
    val_every_n_epoch: int
    val_data_quanity: int # compress a batch of #val_data_quanity data when eval performance
    optimizer_name_module: str
    lr_module: float
    argmin_steps: int # to solve argmin ,gd #steps_y in every step
    optimizer_name_y: str
    lr_y: float
    optimizer_name_z: str
    lr_z: float
    Lambda: float # R+Lambda*D
    transform: TransformOpt = TransformOpt
@dataclass
class CompressFrameworkOpt:
    Name: str # NFLR_AutoDecoder,NFLR_AutoEncoder,NFLR_Coding_AutoDecoder,NFLR_Coding_AutoEncoder,NFLR_Coding_Hyper_AutoDecoder,NFLR_Coding_Hyper_AutoEncoder
    Compress: CompressOpt = CompressOpt
    Decompress: DecompressOpt = DecompressOpt
    Module: ModuleOpt = ModuleOpt
    Normalize: NormalizeOpt = NormalizeOpt

@dataclass
class LogOpt:
    logdir:str
    project_name: str
    task_name: str
    stdlog: bool
    tensorboard: bool
@dataclass
class ReproducOpt:    
    seed: int
    benchmark: bool
    deterministic: bool
@dataclass
class SingleTaskOpt:
    Reproduc: ReproducOpt = ReproducOpt
    CompressFramework: CompressFrameworkOpt = CompressFrameworkOpt
    Log: LogOpt = LogOpt
    Dataset: Datasetopt = Datasetopt
@dataclass
class MultiTaskOpt:
    Dynamic: Any
    Static: SingleTaskOpt = SingleTaskOpt