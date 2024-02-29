import torch
import torch.nn as nn
import numpy as np
import collections
import math
import time
import spconv.pytorch as spconv
from .pose_resnet import get_resnet_encoder
from .hr_module import get_hrnet_encoder
from core.cfgs import cfg
from utils.geometry import rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis
from .maf_extractor import MAF_Extractor
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .hmr import ResNet_Backbone
from .iuv_predictor import IUV_predict_layer
from core import path_config, constants
from .xx import SEAttention
from .qq import Unpool, FCRNDecoder, DEPTH_predict_layer, reduce_dim_layer, MultiLinear
from .attention import get_att_block
from utils.renderer import IUV_Renderer
from utils.myutils import bilinear_sampler
from utils.iuvmap import iuv_img2map, depth_img2map

from core.cfgs import global_logger as logger

BN_MOMENTUM = 0.1

class Regressor(nn.Module):
    def __init__(self, feat_dim, smpl_mean_params):
        super().__init__()

        npose = 24 * 6
        ncam = 3
        nshape = 10
        # 1
        self.fc1 = nn.Linear(feat_dim + npose + nshape + ncam, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose) # B,144
        self.decshape = nn.Linear(1024, nshape) # B,10
        self.deccam = nn.Linear(1024, ncam) # B,3
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=32,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    # 7
    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None, focal_length=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            # 2
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam, focal_length)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis
        
        opt_output = self.smpl(
            betas=pred_shape,
            body_pose=pose[:,3:],
            global_orient=pose[:,:3],
        )
        opt_vertices = opt_output.vertices

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            'opt_vertices' : opt_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,          # 测试
            'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output

    def forward_init(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None, focal_length=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose.contiguous()).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam, focal_length)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints, 
            'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output

def get_attention_modules(module_keys, img_feature_dim_list, hidden_feat_dim, n_total_points, n_iter, num_attention_heads=1):

    align_attention = nn.ModuleDict()
    for k in module_keys:
        align_attention[k] = nn.ModuleList()
        for i in range(n_iter):
            align_attention[k].append(get_att_block(img_feature_dim=img_feature_dim_list[k][i], 
                                                    hidden_feat_dim=hidden_feat_dim, 
                                                    n_points=n_total_points[k],
                                                    num_attention_heads=num_attention_heads))

    return align_attention

def get_fusion_modules(module_keys, ma_feat_dim, grid_feature_dim, n_iter, out_feat_len, alfeature_dim=None):
    if alfeature_dim is not None:
        feat_fusion = nn.ModuleDict()
        for k in module_keys:
            feat_fusion[k] = nn.ModuleList()
            for i in range(n_iter):
                feat_fusion[k].append(nn.Linear(grid_feature_dim + ma_feat_dim[k] + alfeature_dim, out_feat_len[k]))

    else:
        feat_fusion = nn.ModuleDict()
        for k in module_keys:
            feat_fusion[k] = nn.ModuleList()
            for i in range(n_iter):
                feat_fusion[k].append(nn.Linear(grid_feature_dim + ma_feat_dim[k], out_feat_len[k]))

    return feat_fusion

class PyMAF(nn.Module):
    """ PyMAF based Deep Regressor for Human Mesh Recovery
    PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, pretrained=True):
        super().__init__()
        self.global_mode = not cfg.MODEL.PyMAF.MAF_ON # 是否使用avgpool无deconv,即不变，默认False
        # self.feature_extractor = get_resnet_encoder(cfg, global_mode=self.global_mode)
        # body_sfeat_dim = list(cfg.POSE_RES_MODEL.EXTRA.NUM_DECONV_FILTERS)
        
        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            self.feature_extractor = get_resnet_encoder(cfg, global_mode=self.global_mode)
            body_sfeat_dim = list(cfg.POSE_RES_MODEL.EXTRA.NUM_DECONV_FILTERS) # 268 268 268 
        elif cfg.MODEL.PyMAF.BACKBONE == 'hr48':
            self.feature_extractor = get_hrnet_encoder(cfg, global_mode=self.global_mode)
            body_sfeat_dim = list(cfg.HR_MODEL.EXTRA.STAGE4.NUM_CHANNELS)
            body_sfeat_dim.reverse()
            body_sfeat_dim = body_sfeat_dim[1:] # 192 96 48
            body_sfeat_dim[-1] = cfg.FINAL_CHANNEL
        
        # deconv layers
        self.inplanes = self.feature_extractor.inplanes
        if cfg.MODEL.PyMAF.MARKER:
            self.ssm = np.load(path_config.SMPL_MARKER)
            self.numpoints = len(self.ssm)

        # if cfg.RE_DEEP:
        #     body_sfeat_dim[0] = 67*4 - 1
        
        if cfg.MODEL.PyMAF.MASK:
            self.indices_part = torch.from_numpy(np.load('indices_part.npz', allow_pickle=True)['value'])

        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            self.deconv_layers = self._make_deconv_layer( # 三层反卷积3，256*3，4*3
                cfg.RES_MODEL.NUM_DECONV_LAYERS,
                cfg.RES_MODEL.NUM_DECONV_FILTERS,
                cfg.RES_MODEL.NUM_DECONV_KERNELS,
            )

            self.deconv_depth_layers = self._make_FCRNdeconv_layer()
        # deconv layers
        self.inplanes = self.feature_extractor.inplanes

       

        # self.relu = nn.ReLU(inplace=True)
        # self.c = nn.Embedding(431, 8)
        # self.voxel_size_num = [0.05, 0.05, 0.05]
        # self.voxel_size = torch.tensor(self.voxel_size_num).to(torch.device('cuda', cfg.GPU))

        grid_size = 21
        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)])
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * cfg.MODEL.PyMAF.MLP_DIM[-1]
        
        self.fuse_grid_align = cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT or cfg.MODEL.PyMAF.GRID_ALIGN.USE_FC
        assert not (cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT and cfg.MODEL.PyMAF.GRID_ALIGN.USE_FC)
        bhf_att_feat_dim = {}
        self.att_starts = 10
        if self.fuse_grid_align:
            self.att_starts = cfg.MODEL.PyMAF.GRID_ALIGN.ATT_STARTS # 
            n_iter_att = cfg.MODEL.PyMAF.N_ITER - self.att_starts
            att_feat_dim_idx = - cfg.MODEL.PyMAF.GRID_ALIGN.ATT_FEAT_IDX # -2
            num_att_heads = cfg.MODEL.PyMAF.GRID_ALIGN.ATT_HEAD
            hidden_feat_dim = cfg.MODEL.PyMAF.MLP_DIM[att_feat_dim_idx]
        
        self.maf_extractor = nn.ModuleList() # 三层提取特征器
        if cfg.MODEL.PyMAF.MARKER:
            self.flow_layer = nn.ModuleList() # 二层
        # self.sparse_convnet = nn.ModuleList() # 二层
        # for _ in range(cfg.MODEL.PyMAF.N_ITER):
        #     self.maf_extractor.append(MAF_Extractor())

        filter_channels_default = cfg.MODEL.PyMAF.MLP_DIM 
        sfeat_dim = [body_sfeat_dim[-1], body_sfeat_dim[-1], body_sfeat_dim[-1]]
        for i in range(cfg.MODEL.PyMAF.N_ITER):
            for f_i, f_dim in enumerate(filter_channels_default):
                if sfeat_dim[i] > f_dim:
                    filter_start = f_i
                    break
            filter_channels = [sfeat_dim[i]] + filter_channels_default[filter_start:]

            if cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT:
                self.maf_extractor.append(MAF_Extractor(filter_channels=filter_channels_default[att_feat_dim_idx:],
                                                        filter_channels2=filter_channels))
                # self.sparse_convnet.append(SparseConvNet())
                if cfg.MODEL.PyMAF.MARKER:
                    self.flow_layer.append(MultiLinear(self.numpoints, 4*(2*3+1)**2, 7))
            
        self.maf_extractor0 = MAF_Extractor(filter_channels2=filter_channels)    
            
        
        self.smpl_ds_len = self.maf_extractor[-1].Dmap.shape[0]
        ma_feat_len = self.maf_extractor[-1].Dmap.shape[0] * cfg.MODEL.PyMAF.MLP_DIM[-1]  

        bhf_names = []
        bhf_names.append('body')      
        # spatial alignment attention
        if cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT:
            hfimg_feat_dim_list = {}
            n_total_points = {}
            if 'body' in bhf_names:
                # hfimg_feat_dim_list['body'] = body_sfeat_dim[-n_iter_att:]
                hfimg_feat_dim_list['body'] = [body_sfeat_dim[-1], body_sfeat_dim[-1], body_sfeat_dim[-1]]
                n_total_points['body'] = grid_size**2 + self.smpl_ds_len

            self.align_attention = get_attention_modules(bhf_names, hfimg_feat_dim_list, 
                                                         hidden_feat_dim, 
                                                         n_total_points=n_total_points, n_iter=n_iter_att, 
                                                         num_attention_heads=num_att_heads)
        
        if cfg.MODEL.PyMAF.MARKER and self.fuse_grid_align:
            self.att_feat_reduce = get_fusion_modules(bhf_names, ma_feat_dim={'body': ma_feat_len}, 
                                                      grid_feature_dim=grid_feat_len, n_iter=n_iter_att, 
                                                      out_feat_len={'body': 2048}, alfeature_dim=self.numpoints*7)
        elif self.fuse_grid_align:
            self.att_feat_reduce = get_fusion_modules(bhf_names, ma_feat_dim={'body': ma_feat_len}, 
                                                      grid_feature_dim=grid_feat_len, n_iter=n_iter_att, 
                                                      out_feat_len={'body': 2048})
        self.regressor = nn.ModuleList()
        for i in range(cfg.MODEL.PyMAF.N_ITER):
            ref_infeat_dim = 2048

            # if i == 0:
            #     ref_infeat_dim = grid_feat_len
            #     # ref_infeat_dim = 256*3*3
            # else:
            #     ref_infeat_dim = ma_feat_len
            #     # ref_infeat_dim = 256*3*3
            self.regressor.append(Regressor(feat_dim=ref_infeat_dim, smpl_mean_params=smpl_mean_params))

        self.regressor0 = Regressor(feat_dim=grid_feat_len, smpl_mean_params=smpl_mean_params)
        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            dp_feat_dim = 256
            dep_deat_dim = 128
        elif cfg.MODEL.PyMAF.BACKBONE == 'hr48':
            dp_feat_dim = body_sfeat_dim[-1]
            dep_deat_dim = body_sfeat_dim[-1]
        
        self.with_uv = cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0
        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)
            self.depth_head = DEPTH_predict_layer(feat_dim=dep_deat_dim)
            if cfg.MODEL.PyMAF.BACKBONE == 'res50':
                self.reduce_head = reduce_dim_layer()

            self.iuv_maker = IUV_Renderer(output_size=56,device=torch.device('cuda', cfg.GPU))

        if cfg.RE_DEEP:
            self.deptran_layers = self._make_deptran_layer()
            self.depfusion0 = nn.Sequential(
                nn.Conv2d(body_sfeat_dim[-1]+64, body_sfeat_dim[-1], kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(body_sfeat_dim[-1]),
                nn.ReLU(inplace=True))
            self.depfusion1 = nn.Sequential(
                nn.Conv2d(body_sfeat_dim[-1]+64, body_sfeat_dim[-1], kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(body_sfeat_dim[-1]),
                nn.ReLU(inplace=True))
            self.depfusion2 = nn.Sequential(
                nn.Conv2d(body_sfeat_dim[-1]+64, body_sfeat_dim[-1], kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(body_sfeat_dim[-1]),
                nn.ReLU(inplace=True))
        
    def _make_deptran_layer(self):# 3，256*3，4*3
        layers = []
        for i in range(3):
            layers.append(nn.Conv2d(
                in_channels=90,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1))
            layers.append(nn.BatchNorm2d(64, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):# 3，256*3，4*3
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        
        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i) # kernel 4 padding 1 output_padding 0

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    
    def _make_FCRNdeconv_layer(self):
        layers = []
        num_channels = 2048
        num_layers = 3

        layers.append(nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False))
        layers.append(nn.BatchNorm2d(num_channels//2))

        in_channels= num_channels//2
        for i in range(num_layers):
            upconv = nn.Sequential(collections.OrderedDict([
            ('unpool',    Unpool(in_channels)),
            ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=3,stride=1,padding=1,bias=False)),
            ('batchnorm', nn.BatchNorm2d(in_channels//2)),
            ('relu',      nn.ReLU()),
            ]))
            in_channels = in_channels//2
            layers.append(upconv)

        return nn.Sequential(*layers)
    
    # 4
    def forward(self, x, img_focal, J_regressor=None):

        batch_size = x.shape[0]
        # batch_indice = torch.arange(batch_size).to(torch.device('cuda', cfg.GPU)) 
        # batch_indices = batch_indice.unsqueeze(1).repeat(1, 431).unsqueeze(-1).reshape(-1,1)

        # spatial features and global features
        s_feat, g_feat = self.feature_extractor(x) # s_feat是从PoseResNet的输出而得到
        s_feat_i = s_feat
        s_feat_f = s_feat

        # assert cfg.MODEL.PyMAF.N_ITER >= 0 and cfg.MODEL.PyMAF.N_ITER <= 3
        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]


            depth_deconv_blocks = [self.deconv_depth_layers[0:2], 
                                self.deconv_depth_layers[2:3], 
                                self.deconv_depth_layers[3:4],
                                self.deconv_depth_layers[4:5]]
            if cfg.DEEP:
                s_feat_f = depth_deconv_blocks[0](s_feat_f) 
        
        if cfg.RE_DEEP:
            deptran_blocks = [self.deptran_layers[0:3],
                            self.deptran_layers[3:6],
                            self.deptran_layers[6:9]]
    
        out_list = {}

        # initial parameters
        # TODO: remove the initial mesh generation during forward to reduce runtime
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        smpl_output = self.regressor[0].forward_init(g_feat, J_regressor=J_regressor, focal_length=img_focal)

        out_list['smpl_out'] = [smpl_output]
        out_list['dp_out'] = []
        out_list['depth_out'] = []
        out_list['dp_re_out'] = []
        out_list['depth_re_out'] = []

        # for visulization
        vis_feat_list = []

        if cfg.MODEL.PyMAF.USE_MAX_MAP:
            if cfg.MODEL.PyMAF.BACKBONE == 'res50':
                for i in range(3):
                    s_feat_i = deconv_blocks[i](s_feat_i) 
                    # vis_feat_list.append(s_feat_i.detach())
                    if cfg.DEEP:
                    # 深度
                        s_feat_f = depth_deconv_blocks[i + 1](s_feat_f).to(torch.device('cuda', cfg.GPU))
                        
                        # self.maf_extractor[rf_i].im_feat = s_feat_i
                        if i == 2:
                            s_feat_e = torch.cat((s_feat_i,s_feat_f),dim=1)

                            # 残差？
                            im_feat0 = self.reduce_head(s_feat_e, i)
                    else:
                        im_feat0 = s_feat_i
            if cfg.MODEL.PyMAF.BACKBONE == 'hr48':
                im_feat0 = s_feat
        
        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            iuv_out_dict = self.dp_head(s_feat_i)
            out_list['dp_out'].append(iuv_out_dict)
            if cfg.DEEP:
                # 深度图估计
                depth_out_dict = self.depth_head(s_feat_f)
                out_list['depth_out'].append(depth_out_dict)

        # 初始化
        im_feat = im_feat0
        pred_cam = smpl_output['pred_cam']
        pred_shape = smpl_output['pred_shape']
        pred_pose = smpl_output['pred_pose']

        pred_cam = pred_cam.detach()
        pred_shape = pred_shape.detach()
        pred_pose = pred_pose.detach()

        # if not cfg.MODEL.PyMAF.USE_MAX_MAP:
        #     s_feat_i = deconv_blocks[0](s_feat_i) 
        #     vis_feat_list.append(s_feat_i.detach())

        #     # 深度
        #     s_feat_f = depth_deconv_blocks[1](s_feat_f).to(torch.device('cuda', cfg.GPU))
        #     s_feat_e = torch.cat((s_feat_i,s_feat_f),dim=1)
            
        #     im_feat = self.reduce_head(s_feat_e, 0)
        
        pred_smpl_verts = smpl_output['verts'].detach() # bs * 6890 *3 

        self.maf_extractor0.im_feat = im_feat
        self.maf_extractor0.cam = pred_cam
        self.maf_extractor0.img_focal = img_focal


        # 网格采样 ref_feature 0降维
        sample_points = torch.transpose(self.points_grid.expand(batch_size, -1, -1), 1, 2) # bs * 441 * 2
        ref_feature = self.maf_extractor0.sampling(sample_points, reduce_dim=True) # bs * 2205
        ref_feature = ref_feature.view(batch_size, -1)
        smpl_output = self.regressor0(ref_feature, pred_pose, pred_shape, pred_cam, n_iter=2, J_regressor=J_regressor, focal_length=img_focal)
        out_list['smpl_out'].append(smpl_output)

        pred_cam = smpl_output['pred_cam']
        if cfg.RE_DEEP:
            render_verts = smpl_output['opt_vertices'].detach() # bs * 6890 *3 
            render_cam = torch.stack([pred_cam[:, 1],
                            pred_cam[:, 2],
                            2 * 5000. / (224. * pred_cam[:, 0] + 1e-9)], dim=-1)

            iuv_image, depthimg = self.iuv_maker.verts2iuvimg(render_verts, cam=render_cam)  # [B, 3, 56, 56]
            iuv_output = iuv_img2map(iuv_image)
            # out_list['dp_re_out'].append(iuv_output)
            # depth_output = depth_img2map(depthimg)
            # out_list['depth_re_out'].append(depth_output)

        # parameter predictions
        for rf_i in range(cfg.MODEL.PyMAF.N_ITER):
            im_feat = im_feat0
            pred_cam = smpl_output['pred_cam']
            pred_shape = smpl_output['pred_shape']
            pred_pose = smpl_output['pred_pose']

            pred_cam = pred_cam.detach()
            pred_shape = pred_shape.detach()
            pred_pose = pred_pose.detach()

            # bs * 256 * 14 * 14
            # bs * 256 * 28 * 28
            # bs * 256 * 56 * 56
            # simple baseline
            if not cfg.MODEL.PyMAF.USE_MAX_MAP:
                s_feat_i = deconv_blocks[rf_i](s_feat_i) 
                # vis_feat_list.append(s_feat_i.detach())

                # 深度
                s_feat_f = depth_deconv_blocks[rf_i + 1](s_feat_f).to(torch.device('cuda', cfg.GPU))
                # self.se = SEAttention(channel=256,reduction=s_feat_r.shape[2]).to(torch.device('cuda'))
                # s_feat_r = se(s_feat_r)
                
                # self.maf_extractor[rf_i].im_feat = s_feat_i
                s_feat_e = torch.cat((s_feat_i,s_feat_f),dim=1)
                
                im_feat = self.reduce_head(s_feat_e, rf_i)
            
            pred_smpl_verts = smpl_output['verts'].detach() # bs * 6890 *3 

            if cfg.RE_DEEP:
                iuv_map = torch.cat(iuv_output, dim=1)

                iuv_map = deptran_blocks[rf_i](iuv_map)

                im_feat = torch.cat((im_feat, iuv_map), dim=1)
                if rf_i == 0:
                    im_feat = self.depfusion0(im_feat)
                if rf_i == 1:
                    im_feat = self.depfusion1(im_feat)
                if rf_i == 2:
                    im_feat = self.depfusion2(im_feat)
            
            vis_feat_list.append(im_feat.detach())
            self.maf_extractor[rf_i].im_feat = im_feat
            self.maf_extractor[rf_i].cam = pred_cam
            self.maf_extractor[rf_i].img_focal = img_focal

            reduce_dim = not self.fuse_grid_align

            # 顶点采样 ref_feature 1、2不降维
            # pred_smpl_verts_ds = torch.matmul(self.maf_extractor[rf_i].Dmap.unsqueeze(0), pred_smpl_verts) # [B, 431, 3]
            indices = torch.nonzero(self.maf_extractor[rf_i].Dmap == 1).squeeze()[:,1]
            pred_smpl_verts_ds = pred_smpl_verts[:, indices, :]
            ref_feature = self.maf_extractor[rf_i](pred_smpl_verts_ds, reduce_dim=reduce_dim) # [B, n_feat, 431] # bs * 2155
            # 随机对部位mask
            if cfg.MODEL.PyMAF.MASK and J_regressor is None:
                mask = torch.ones(batch_size, ref_feature.shape[2]).to(torch.device('cuda', cfg.GPU))
                p = cfg.MODEL.PyMAF.MASKED_PROB
                zero_idxs = []
                for bs in range(batch_size):
                    zero_idxs.append([int(i) for i in torch.nonzero(torch.rand(24) < p)])
                    for value in zero_idxs[-1]:
                        indices = torch.nonzero(self.indices_part.to(torch.device('cuda', cfg.GPU)) == value)
                        if indices.numel() > 0:
                            mask[bs][indices.squeeze()] = 0
                mask = mask.unsqueeze(1).expand_as(ref_feature)
                ref_feature = ref_feature * mask
            # 随机对点mask
            # p = 0.2
            # # 生成一个与输入Tensor相同形状的随机mask
            # mask = np.random.choice([0, 1], size=ref_feature.shape[2], p=[p, 1-p])
            # mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(torch.device('cuda', cfg.GPU))
            # mask = mask.expand_as(ref_feature)
            # ref_feature = ref_feature * mask
            if cfg.MODEL.PyMAF.MARKER:
                pred_smpl_verts_ds_marker = pred_smpl_verts[:, self.ssm]
                pred_smpl_verts_ds_marker_2d = projection(pred_smpl_verts_ds_marker, pred_cam, img_focal)
                pred_smpl_verts_ds_marker_2d = pred_smpl_verts_ds_marker_2d * 28.    # [-1,1] -> [-28,28]   # [B, N, 2]
                pred_smpl_verts_ds_marker_2d = pred_smpl_verts_ds_marker_2d.detach()  
                marker_feature = self.lookup(im_feat, pred_smpl_verts_ds_marker_2d) 
                marker_feature = self.flow_layer[rf_i - 1](marker_feature)

            # 坐标融合    
            # 图像特征 [B*N, num_features] 
            # point_feats = self.maf_extractor[rf_i](pred_smpl_verts_ds, reduce_dim=reduce_dim).transpose(1, 2) # [B, N, C]
            # emb = self.c(torch.arange(0, 431).to(torch.device('cuda', cfg.GPU)))
            # enbedding = emb.unsqueeze(0).repeat(batch_size,1,1)
            # feats = torch.cat((enbedding, point_feats), dim=-1).reshape(-1,256 + 8)

            # 坐标转换体素坐标
            # coord = pred_smpl_verts_ds.reshape(-1,3)
            # min_xyz = torch.min(coord, 0)[0]
            # max_xyz = torch.max(coord, 0)[0]
            # coord = (coord - min_xyz) // self.voxel_size
                    
            # cat创建批次索引
            # indices = torch.cat((batch_indices, coord), dim=-1).to(torch.int32)  

            # 体素特征空间大小 20, 36, 16
            # out_sh = np.ceil((max_xyz.cpu().numpy() - min_xyz.cpu().numpy()) / self.voxel_size_num).astype(np.int32)
            # x = 4
            # out_sh = (out_sh | (x - 1)) + 1

            # 创建 SparseConvTensor
            # xyzc = spconv.SparseConvTensor(feats, indices, out_sh, batch_size)

            # xyzc_features = self.sparse_convnet[rf_i-1](xyzc)
            # vol_feat = xyzc_features.mean(dim=(2, 3, 4))


            # 分割采样
            # part_feature_list = []
            # indices = torch.nonzero(self.maf_extractor[rf_i].Dmap == 1).squeeze()[:,1]
            # pred_smpl_verts_ds2 = pred_smpl_verts[:, indices, :]
            # for i in range(24):
            #     part_i = pred_smpl_verts_ds2[:, self.maf_extractor[rf_i].points_mapping["point"][i], :]
            #     part_feature_i = self.maf_extractor[rf_i](part_i, reduce_dim=reduce_dim)
            #     part_feature_list.append(part_feature_i)         
            # part_feature = torch.cat(part_feature_list, dim=2)
            # # print(np.array_equal(np.array(pred_smpl_verts_ds.cpu()), np.array(pred_smpl_verts_ds2.cpu())))
            # print(np.array_equal(np.array(ref_feature), np.array(part_feature)))
            
            # 分割采样后运动链组合
            # ref_feature_list = []
            # for i in range(24):
            #     ref_feature_i_list = []
            #     for j in constants.POSE_MAPPING_INDEX[i]:
            #         ref_feature_i_list.append(part_feature_list[j-1])
            #     ref_feature_i = torch.cat(ref_feature_i_list, dim=2)
            #     ref_feature_list.append(ref_feature_i.view(batch_size, -1))

            # 注意力和两种特征级联
            grid_feature = self.maf_extractor[rf_i].sampling(sample_points, reduce_dim=reduce_dim) # bs * 2205
            grid_ref_feat = torch.cat([grid_feature, ref_feature], dim=-1)

            grid_ref_feat = grid_ref_feat.permute(0, 2, 1)

            if cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT:
                # 拼接网格和点采样特征，进入SAA ([32, 872, 256]) -> ([32, 872, 64]) ->([32, 872, 64])
                att_ref_feat = self.align_attention['body'][rf_i-self.att_starts](grid_ref_feat)[0]
            # ([32, 872, 64]) -> ([32, 872, 5]) -. ([32, 5, 872])
            att_ref_feat = self.maf_extractor[rf_i].reduce_dim(att_ref_feat.permute(0, 2, 1), att=2)
            att_ref_feat = att_ref_feat.view(batch_size, -1)
            
            if cfg.MODEL.PyMAF.MARKER:
                marker_feature = marker_feature.view(batch_size, -1)
                att_ref_feat = torch.cat((att_ref_feat, marker_feature), dim=-1)
                ref_feature = self.att_feat_reduce['body'][rf_i-self.att_starts](att_ref_feat)
            else:
                # 线性层融降维
                # att_ref_feat = torch.cat((att_ref_feat, vol_feat), dim=-1)
                ref_feature = self.att_feat_reduce['body'][rf_i-self.att_starts](att_ref_feat)

            smpl_output = self.regressor[rf_i](ref_feature, pred_pose, pred_shape, pred_cam, n_iter=2, J_regressor=J_regressor, focal_length=img_focal)
            out_list['smpl_out'].append(smpl_output)
            pred_cam = smpl_output['pred_cam']
            pred_cam = pred_cam.detach()
            if cfg.RE_DEEP and rf_i < 2:
                render_verts = smpl_output['opt_vertices'].detach() # bs * 6890 *3 
                render_cam = torch.stack([pred_cam[:, 1],
                                pred_cam[:, 2],
                                2 * 5000. / (224. * pred_cam[:, 0] + 1e-9)], dim=-1)

                iuv_image, depthimg = self.iuv_maker.verts2iuvimg(render_verts, cam=render_cam)  # [B, 3, 56, 56]
                iuv_output = iuv_img2map(iuv_image)
                # out_list['dp_re_out'].append(iuv_output)
                # depth_output = depth_img2map(depthimg)
                # out_list['depth_re_out'].append(depth_output)

        return out_list, vis_feat_list
    
    def lookup(self, corr, coords):
        r = 3
        h, w = corr.shape[-2:]
        device = corr.device

        h, w = corr.shape[-2:]   # (batch, j*c, 56, 56)
        bn, j = coords.shape[:2] # (batch, 67, 2)

        dx = torch.linspace(-r, r, 2*r+1, device=device)
        dy = torch.linspace(-r, r, 2*r+1, device=device)

        # lookup window
        centroid = coords.reshape(bn*j, 1, 1, 2)   # (bn*j, 1, 1, 2)

        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        delta = delta.view(1, 2*r+1, 2*r+1, 2)
        window = centroid + delta                  # (bn*j, 2r+1, 2r+1, 2)

        # feature map
        corr = corr                                # (bn, j*c, 56, 56)
        fmap = corr.view(bn*j, -1, h, w)           # (bn*j, c, 56, 56)
        feature = bilinear_sampler(fmap, window)   # (bn*j, c, 2r+1, 2r+1)
        feature = feature.view(bn, j, -1)          # (bn, j, c*(2r+1)*(2r+1))

        return feature

def pymaf_net(smpl_mean_params, pretrained=True):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyMAF(smpl_mean_params, pretrained)
    # print(model)
    return model

