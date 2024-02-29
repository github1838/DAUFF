import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# import cv2
import torch
import torchvision
import argparse
# import scipy.io
import numpy as np
# import torchgeometry as tgm
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import BaseDataset
from models import hmr, SMPL, pymaf_net
from core import constants, path_config
from core.cfgs import parse_args
from core.cfgs import cfg
cfg.CUDA_VISIBLE_DEVICES = 1
# from utils.imutils import uncrop
from utils.uv_vis import vis_smpl_iuv
from utils.renderer import  IUV_Renderer
# from utils.pose_utils import reconstruction_error
# from utils.part_utils import PartRenderer # used by lsp

from core.cfgs import global_logger as logger
import cv2
import scipy
import random
from utils.renderer import PyRenderer

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='/home/wz/work_dir/PyMAFtest/logs/pymaf_hr48_mix/pymaf_hr48_mix_as_lp3_mlp256-128-64-5_Oct19-20-48-53-Urr/checkpoints/model_best_72.541_43.128_83.229.pt', help='Path to network checkpoint')
parser.add_argument('--dataset', choices=['h36m-p1', 'h36m-p2', 'h36m-p2-mosh', 'lsp', '3dpw', 'mpi-inf-3dhp', '3doh50k'],
                                            default='3dpw', help='Choose evaluation dataset')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--regressor', type=str, choices=['hmr', 'pymaf_net'], default='pymaf_net', help='Name of the SMPL regressor.')
parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml', help='config file path for PyMAF.')
parser.add_argument('--misc', default=None, type=str, nargs="*", help='other parameters')

parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--vis_demo', default=True, action='store_true', help='result visualization')
parser.add_argument('--ratio', default=1, type=int, help='image size ration for visualization')


def run_evaluation(model, dataset):
    """Run evaluation on the datasets and metrics we report in the paper. """
    
    shuffle = args.shuffle
    log_freq = args.log_freq
    batch_size = args.batch_size
    dataset_name = args.dataset
    num_workers = args.num_workers
    vis_imname = 'downtown_rampAndStairs_00/image_00819.jpg'

    device = torch.device('cuda') if torch.cuda.is_available() \
                                else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(path_config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(path_config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(path_config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    # renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(path_config.JOINT_REGRESSOR_H36M)).float()

    # Disable shuffling if you want to save the results
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Pose metrics
    pve = np.zeros(len(dataset))

    iuv_maker = IUV_Renderer(output_size=cfg.MODEL.PyMAF.DP_HEATMAP_SIZE, device=torch.device('cuda', cfg.GPU))

    eval_pose = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == 'h36m-p2-mosh' \
       or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp' or dataset_name == '3doh50k':
        eval_pose = True

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # if step < 500:
        #     continue
        imgnames = [i_n.split('/')[-2] + '/' + i_n.split('/')[-1] for i_n in batch['imgname']]
        name_hit = False
        for i_n in imgnames:
            if vis_imname in i_n:
                name_hit = True
                print('vis: ' + i_n)
        if not name_hit:
            continue
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_smpl_out = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_vertices_nt = gt_smpl_out.vertices
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = images.shape[0]
        img_focal = batch["img_focal"].to(device).float()
        valid_fit = batch['has_smpl'].to(torch.bool)

        with torch.no_grad():
            if args.regressor == 'hmr':
                pred_rotmat, pred_betas, pred_camera = model(images)
                # torch.Size([32, 24, 3, 3]) torch.Size([32, 10]) torch.Size([32, 3])
            elif args.regressor == 'pymaf_net':
                preds_dict, vis_feat_list = model(images, img_focal)
                # print(model)
                new_model = torchvision.models._utils.IntermediateLayerGetter(model, {'feature_extractor': 'feat1'})
                out = new_model(images)

                # print((k, v.shape) for k, v in out.items())

                tensor_ls=[(k,v) for  k,v in out.items()]

                #这里选取layer2的输出画特征图
                v=tensor_ls[0][1][0]
                v=v.data.squeeze(0)  # torch.Size([32, 96, 56, 56])


                pred_rotmat = preds_dict['smpl_out'][-1]['rotmat'].contiguous().view(-1, 24, 3, 3)
                pred_betas = preds_dict['smpl_out'][-1]['theta'][:, 3:13].contiguous()
                pred_camera = preds_dict['smpl_out'][-1]['theta'][:, :3].contiguous()

            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices


        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name or '3doh50k' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            if '3dpw' in dataset_name:
                per_vertex_error = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            else:
                per_vertex_error = torch.sqrt(((pred_vertices - gt_vertices_nt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error

        if args.vis_demo:
            imgnames = [i_n.split('/')[-1] for i_n in batch['imgname']]

            if args.regressor == 'hmr':
                iuv_pred = None
            else:
                iuv_pred = preds_dict['dp_out'][-1]
            if cfg.DEEP:
                depth_pred = preds_dict['depth_out'][-1]

            iuv_image_gt = torch.zeros((batch_size, 3, cfg.MODEL.PyMAF.DP_HEATMAP_SIZE, cfg.MODEL.PyMAF.DP_HEATMAP_SIZE)).to(device)
            depth_gt = torch.zeros((batch_size, 1, 56, 56)).to(device)   
    
            if '3dpw' in dataset_name:
                iuv_image_gt[valid_fit], temp = iuv_maker.verts2iuvimg(gt_vertices[valid_fit], pred_camera[valid_fit])  # [B, 3, 56, 56]
            else:
                iuv_image_gt[valid_fit], temp = iuv_maker.verts2iuvimg(pred_output.vertices[valid_fit], pred_camera[valid_fit])  # [B, 3, 56, 56]
            
            depth_gt[valid_fit] = temp
            

            images_vis = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images_vis = images_vis + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
            vis_smpl_iuv(images_vis.cpu().numpy(), pred_camera.cpu().numpy(), pred_output.vertices,
                        smpl_neutral.faces, iuv_pred, 100 * per_vertex_error, imgnames,
                        os.path.join('./visimg', dataset_name, args.checkpoint.split('/')[-3]), args, 
                        depth_pred, iuv_image_gt, depth_gt, v)
        if len(vis_imname) > 0:
            exit()


if __name__ == '__main__':
    args = parser.parse_args()
    parse_args(args)

    if args.regressor == 'pymaf_net':
        model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=False)
    elif args.regressor == 'hmr':
        model = hmr(path_config.SMPL_MEAN_PARAMS)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        strict = args.regressor != 'hmr'
        model.load_state_dict(checkpoint['model'], strict=strict)

    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(args, args.dataset, is_train=False)

    # Run evaluation
    run_evaluation(model, dataset)
