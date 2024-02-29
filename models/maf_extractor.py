# This script is borrowed and extended from https://github.com/shunsukesaito/PIFu/blob/master/lib/model/SurfaceClassifier.py

from packaging import version
import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from core.cfgs import cfg
from utils.geometry import projection

from core.cfgs import global_logger as logger

class MAF_Extractor(nn.Module):
    ''' Mesh-aligned Feature Extrator

    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    '''

    def __init__(self, filter_channels=None, filter_channels2=None, device=torch.device('cuda')):
        super().__init__()

        self.device = device
        self.filters = []
        self.filters2 = []
        # filter_channels2 = [256, 128, 64, 5]
        self.num_views = 1
        self.last_op = nn.ReLU(True) 

        # filter_channels: [256, 128, 64, 5]
        if filter_channels is not None:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))
                self.add_module("conv%d" % l, self.filters[l])
        
        # filter_channels: [257, 128, 64, 5]
        for l in range(0, len(filter_channels2) - 1):
            if 0 != l:
                self.filters2.append(
                    nn.Conv1d(
                        filter_channels2[l] + filter_channels2[0],
                        filter_channels2[l + 1],
                        1))
            else:
                self.filters2.append(nn.Conv1d(
                    filter_channels2[l],
                    filter_channels2[l + 1],
                    1))

            self.add_module("convq%d" % l, self.filters2[l])
        ######
        self.im_feat = None
        self.cam = None
        self.img_focal =None

        # downsample SMPL mesh and assign part labels
        # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
        smpl_mesh_graph = np.load('data/mesh_downsampling.npz', allow_pickle=True, encoding='latin1')

        A = smpl_mesh_graph['A']
        U = smpl_mesh_graph['U']
        D = smpl_mesh_graph['D'] # shape: (2,)

        # downsampling
        ptD = []
        for i in range(len(D)):
            d = scipy.sparse.coo_matrix(D[i])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))
        
        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890]
        # ptD[1].to_dense() - Size: [431. 1723]
        Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense()) # 6890 -> 431 [431, 6890] 
        self.register_buffer('Dmap', Dmap)

        self.points_mapping = np.load('my_indices.npz', allow_pickle=True)

    def reduce_dim(self, feature, att=None):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p, N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        if att is None or att==1:
            filters = self.filters2
        else:
            filters = self.filters
        for i, f in enumerate(filters):
            if att is None or att==1:
                y = self._modules['convq' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            else:
                # y = conv(y)
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        y = self.last_op(y)
        # [B, C_s, N] -> [B, C_p, N]  -> [B, C_p x N] 
        # y = y.view(y.shape[0], -1)
        return y

    def sampling(self, points, im_feat=None, z_feat=None, reduce_dim=True):
        '''
        Given 2D points, sample the point-wise features for each point, 
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps 
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        if im_feat is None:
            im_feat = self.im_feat

        batch_size = im_feat.shape[0]

        if version.parse(torch.__version__) >= version.parse('1.3.0'):
            # Default grid_sample behavior has changed to align_corners=False since 1.3.0.
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=False)[..., 0]
        else:
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2))[..., 0] 
        # [bs, C_s, H_s, W_s]  + [bs, N, 2]  = bs * C_s * N  point_feat

        if reduce_dim:
            mesh_align_feat = self.reduce_dim(point_feat, att=1)
            return mesh_align_feat
        else:
            return point_feat

    def forward(self, p, s_feat=None, cam=None, reduce_dim=True, img_focal=None, **kwargs):
        ''' Returns mesh-aligned features for the 3D mesh points.

        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            s_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        '''
        if cam is None:
            cam = self.cam
        if img_focal is None:
            img_focal = self.img_focal
        p_proj_2d = projection(p, cam, img_focal, retain_z=False)
        mesh_align_feat = self.sampling(p_proj_2d, s_feat, reduce_dim=reduce_dim)
        return mesh_align_feat
