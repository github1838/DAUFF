"""
# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/datasets/mixed_dataset.py
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset
from core.cfgs import global_logger as logger
from core import path_config
from .coco_occlusion import load_pascal_occluders

class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, options, **kwargs):
        # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco-full', 'mpi-inf-3dhp']
        # # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        # self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco-full': 4, 'mpi-inf-3dhp': 5}

        # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco-full', 'mpi-inf-3dhp', '3dpw']
        # self.num_datasets = len(self.dataset_list)
        # self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco-full': 4, 'mpi-inf-3dhp': 5, '3dpw': 6}

        self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco-full', 'mpi-inf-3dhp']
        self.num_datasets = len(self.dataset_list)
        self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco-full': 4, 'mpi-inf-3dhp': 5}

        occluders = None
        if options.use_synthetic_occlusion:
            logger.info('Loading synthetic occluders for dataset.')
            occluders = load_pascal_occluders(pascal_voc_root_path=path_config.PASCAL_ROOT)
            logger.info(f'Found {len(occluders)} suitable '
                        f'objects from {options.occ_aug_dataset} dataset')
                
        self.datasets = [BaseDataset(options, ds, occluders=occluders, **kwargs) for ds in self.dataset_list]
        self.dataset_length = {self.dataset_list[idx]: len(ds) for idx, ds in enumerate(self.datasets)}
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        # length_itw = sum([len(ds) for ds in self.datasets[1:4]])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        # self.partition = [
        #                     .3,
        #                     .6*len(self.datasets[1])/length_itw,
        #                     .6*len(self.datasets[2])/length_itw,
        #                     .6*len(self.datasets[3])/length_itw,
        #                     .6*len(self.datasets[4])/length_itw,
        #                     0.1]
        self.partition = [
                            .5,
                            .3 * len(self.datasets[1]) / length_itw,
                            .3 * len(self.datasets[2]) / length_itw, # 0.075
                            .3 * len(self.datasets[3]) / length_itw,
                            .3 * len(self.datasets[4]) / length_itw,
                            0.2]
        # self.partition = [
        #                     0.5, 
        #                     0.21 * len(self.datasets[1]) / length_itw,
        #                     0.21 * len(self.datasets[2]) / length_itw, # 0.07
        #                     0.21 * len(self.datasets[3]) / length_itw,
        #                     0.09, 
        #                     0.2]
        # self.partition = [
        #                     0.3, 
        #                     0.18 * len(self.datasets[1]) / length_itw,
        #                     0.18 * len(self.datasets[2]) / length_itw, # 0.06
        #                     0.18 * len(self.datasets[3]) / length_itw,
        #                     0.12,
        #                     0.2,
        #                     0.2]
        
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(self.num_datasets):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
    
    def __name__(self):
        return self.dataset_list
