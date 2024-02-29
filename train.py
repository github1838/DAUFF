import os
from core.cfgs import cfg
cfg.CUDA_VISIBLE_DEVICES = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.CUDA_VISIBLE_DEVICES)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_IFNAME'] = 'enp2s0'
import time
# time.sleep(10)

import random
import numpy as np
import torch
from core.cfgs import  parse_args_extend
# SEED_VALUE = 2023
# seed = 2023
# if SEED_VALUE >= 0:
#     # os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
#     # random.seed(SEED_VALUE)
#     # np.random.seed(SEED_VALUE)
#     # torch.manual_seed(SEED_VALUE)
#     # torch.cuda.manual_seed(SEED_VALUE)      
#     # torch.cuda.manual_seed_all(SEED_VALUE)
#     # torch.backends.cudnn.deterministic = True

#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # torch.use_deterministic_algorithms(True, warn_only=True)

#     torch.backends.cudnn.deterministic = True

#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.benchmark = False

import socket
from core.train_options import TrainOptions
from utils.train_utils import prepare_env

from core.trainer import Trainer
import torch.distributed as dist

from core.cfgs import global_logger as logger
import warnings
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告


# CUDA_VISIBLE_DEVICES=0,3 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --distributed --multiprocessing_distributed

def get_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def main(gpu, ngpus_per_node, options):
    parse_args_extend(options)

    options.batch_size = cfg.TRAIN.BATCH_SIZE
    options.workers = cfg.TRAIN.NUM_WORKERS
    options.gpu = gpu
    options.ngpus_per_node = ngpus_per_node
    cfg.GPU = options.gpu
    # global_logger.info('this is train')
    freeport = "49155"
    
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='localhost', master_port=freeport)

    if options.distributed:
        options.dist_url = dist_init_method # "env://" # 
        # print("gpu" + str(gpu) + " start init_process_group ")
        # print("gpu " + str(gpu) + " backend " + options.dist_backend)
        # print("gpu " + str(gpu) + " init_method " + options.dist_url)
        # print("gpu " + str(gpu) + " world_size " + str(options.world_size))
        # print("gpu " + str(gpu) + " rank " + str(options.local_rank))
        
        dist.init_process_group(backend=options.dist_backend, init_method=options.dist_url,
                                world_size=options.world_size, rank=options.local_rank)
        # print("gpu" + str(gpu) + "end init_process_group")
        

    if options.multiprocessing_distributed:
        print("options.multiprocessing_distributed")
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        options.rank, world_size = dist.get_rank(), dist.get_world_size()
        assert options.rank == options.local_rank
        assert world_size == options.world_size

    trainer = Trainer(options)
    # latest_checkpoint = trainer.fit()
    trainer.fit()
    # return latest_checkpoint


if __name__ == '__main__':

    options = TrainOptions().parse_args()
    # 是否继续,生成日志,载入配置
    parse_args_extend(options)
    # 配置日志目录路径
    if options.local_rank == 0:
        prepare_env(options)
    else:
        options.checkpoint_dir = '/home/wz/work_dir/PyMAFtest/core'

    # cfg.CUDA_VISIBLE_DEVICES = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[options.local_rank])
    if cfg.SEED_VALUE >= 0:
        logger.info(f'Seed value for the experiment {cfg.SEED_VALUE}')

    ngpus_per_node = torch.cuda.device_count()
    path = '/home/wz/work_dir/PyMAFtest/logs/pymaf_res50/pymaf_res50_'
    pathmix = '/home/wz/work_dir/PyMAFtest/logs/pymaf_res50_mix/pymaf_res50_'
    # options.multiprocessing_distributed = True

    # options.pretrained_checkpoint = pathmix+"mix_as_lp3_mlp256-128-64-5_Jun01-22-00-26-Tra/checkpoints/model_epoch_20.pt"
    
    options.distributed = (ngpus_per_node > 1) or options.multiprocessing_distributed
    if options.multiprocessing_distributed:
        options.world_size = ngpus_per_node * options.world_size
        main(options.local_rank, ngpus_per_node, options)
    else:
        # Simply call main_worker function
        # main_worker(args.gpu, ngpus_per_node, args)
        main(None, ngpus_per_node, options) # gpu, ngpus_per_node, options
