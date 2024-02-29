import os
import torch
import numpy as np
import torch.nn.functional as F
from skimage.transform import resize
# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import cv2
import scipy
import random


# from .renderer import OpenDRenderer
from .renderer import PyRenderer


def iuv_map2img(U_uv, V_uv, Index_UV, AnnIndex=None, uv_rois=None, ind_mapping=None):
    device_id = U_uv.get_device()
    batch_size = U_uv.size(0)
    K = U_uv.size(1)
    heatmap_size = U_uv.size(2)

    Index_UV_max = torch.argmax(Index_UV, dim=1)
    if AnnIndex is None:
        Index_UV_max = Index_UV_max.to(torch.int64)
    else:
        AnnIndex_max = torch.argmax(AnnIndex, dim=1)
        Index_UV_max = Index_UV_max * (AnnIndex_max > 0).to(torch.int64)

    outputs = []

    for batch_id in range(batch_size):
        output = torch.zeros([3, U_uv.size(2), U_uv.size(3)], dtype=torch.float32).cuda(device_id)
        output[0] = Index_UV_max[batch_id].to(torch.float32)
        if ind_mapping is None:
            output[0] /= float(K - 1)
        else:
            for ind in range(len(ind_mapping)):
                output[0][output[0] == ind] = ind_mapping[ind] * (1. / 24.)

        for part_id in range(1, K):
            CurrentU = U_uv[batch_id, part_id]
            CurrentV = V_uv[batch_id, part_id]
            output[1, Index_UV_max[batch_id] == part_id] = CurrentU[Index_UV_max[batch_id] == part_id]
            output[2, Index_UV_max[batch_id] == part_id] = CurrentV[Index_UV_max[batch_id] == part_id]

        if uv_rois is None:
            outputs.append(output.unsqueeze(0))
        else:
            roi_fg = uv_rois[batch_id][1:]
            w = roi_fg[2] - roi_fg[0]
            h = roi_fg[3] - roi_fg[1]

            aspect_ratio = float(w) / h

            if aspect_ratio < 1:
                new_size = [heatmap_size, max(int(heatmap_size * aspect_ratio), 1)]
                output = F.interpolate(output.unsqueeze(0), size=new_size, mode='nearest')
                paddingleft = int(0.5 * (heatmap_size - new_size[1]))
                output = F.pad(output, pad=(paddingleft, heatmap_size - new_size[1] - paddingleft, 0, 0))
            else:
                new_size = [max(int(heatmap_size / aspect_ratio), 1), heatmap_size]
                output = F.interpolate(output.unsqueeze(0), size=new_size, mode='nearest')
                paddingtop = int(0.5 * (heatmap_size - new_size[0]))
                output = F.pad(output, pad=(0, 0, paddingtop, heatmap_size - new_size[0] - paddingtop))

            outputs.append(output)

    return torch.cat(outputs, dim=0)


def vis_smpl_iuv(image, cam_pred, vert_pred_tensor, face, pred_uv, vert_errors_batch, image_name, save_path, opt, 
                 pred_dep, iuv_image_gt_batch, depth_gt_batch, vis_feat_list):
    device = torch.device('cuda') if torch.cuda.is_available() \
                                else torch.device('cpu')
    
    vert_pred = vert_pred_tensor.cpu().numpy()
    # save_path = os.path.join('./notebooks/output/demo_results-wild', ids[f_id][0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # dr_render = OpenDRenderer(ratio=opt.ratio)
    dr_render = PyRenderer()

    focal_length = 5000.
    orig_size = 224.

    depth_image_gt_render = depth_gt_batch.clamp(min = -0.0025).expand(-1,3,-1,-1)
    depth_image_gt_render = (depth_image_gt_render + torch.ones_like(depth_image_gt_render) * 0.0025) * 400 / 401

    for draw_i in range(len(cam_pred)):
        err_val = '{:06d}_'.format(int(10 * vert_errors_batch[draw_i]))
        draw_name = err_val + image_name[draw_i]
        K = np.array([[focal_length, 0., orig_size / 2.],
                      [0., focal_length, orig_size / 2.],
                      [0., 0., 1.]])

        render_imgs = [] 
        img_vis = np.transpose(image[draw_i], (1, 2, 0)) * 255
        img_vis = img_vis[:,:,[2,1,0]]
        img_vis = img_vis.astype(np.uint8)

        # render_imgs.append(img_vis)

        img_smpl = dr_render(
                vert_pred[draw_i],
                face,
                img=img_vis,
                cam=cam_pred[draw_i],
        )
        # render_imgs.append(img_smpl)

        # if pred_uv is not None:
        #     iuv_pred = [pred_uv['predict_u'][draw_i:draw_i+1], pred_uv['predict_v'][draw_i:draw_i+1], \
        #                                 pred_uv['predict_uv_index'][draw_i:draw_i+1], pred_uv['predict_ann_index'][draw_i:draw_i+1]]
        #     iuv_img = iuv_map2img(*iuv_pred)[0].detach().cpu().numpy()
        #     iuv_img = np.transpose(iuv_img, (1, 2, 0)) * 255
        #     iuv_img = iuv_img[:,:,[2,1,0]]
        #     iuv_image_resized = resize(iuv_img, (img_vis.shape[0], img_vis.shape[1]),
        #                                                     preserve_range=True, anti_aliasing=True)

        #     render_imgs.append(iuv_image_resized)

        # if pred_dep is not None:
        #     depth = pred_dep['predict_depth']# b,1,h,w
        #     depth_bc = pred_dep['predict_bc']# b,1,h,w
        #     depth_bc = torch.round(torch.sigmoid(depth_bc))

        #     depth_image_render = torch.zeros((depth.shape[0], 3, 56, 56)).to(device)
        #     # depth_bc_render = torch.zeros((depth.shape[0], 3, 56, 56)).to(self.device)
        #     depth_temp = torch.zeros((depth.shape[0], 56, 56)).to(device)
            
        #     depth_batch = depth[draw_i].clamp(min = -0.0025)
        #     depth_batch = (depth_batch + torch.ones_like(depth_batch) * 0.0025) * 400 / 401 #  + (self.depth_bins - 1) / 2 # -10 0-40

        #     # 可视化只取二分类预测中前景部分
        #     depth_temp[draw_i, depth_bc[draw_i] != 0] = depth_batch[depth_bc[draw_i] != 0]

        #     depth_image_render = depth_temp.unsqueeze(1).expand(-1,3,-1,-1)
        #     depth_image_pred = depth_image_render[draw_i].detach().cpu().numpy()
        #     depth_image_pred = np.transpose(depth_image_pred, (1, 2, 0)) * 255
        #     depth_image_pred_resized = resize(depth_image_pred, (img_vis.shape[0], img_vis.shape[1]),
        #                                                 preserve_range=True, anti_aliasing=True)
        #     depth_image_pred_resized = depth_image_pred_resized[:,:,[2,1,0]]
        #     render_imgs.append(depth_image_pred_resized.astype(np.uint8))


        smpl_mesh_graph = np.load('data/mesh_downsampling.npz', allow_pickle=True, encoding='latin1')
        D = smpl_mesh_graph['D'] # shape: (2,)
        # downsampling
        ptD = []
        for i in range(len(D)):
            d = scipy.sparse.coo_matrix(D[i])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))
        Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense()) # 6890 -> 431 [431, 6890] 
        verts_ds = torch.matmul(Dmap.unsqueeze(0).to(device), vert_pred_tensor)
        verts_ds_pred = verts_ds.cpu().numpy()

        img_points = dr_render(
                verts_ds_pred[draw_i],
                # img=img_vis,
                cam=cam_pred[draw_i],
                renderpoints=True,
        )
        kernel = np.ones((5,5), np.uint8)  # 这里的(5,5)表示腐蚀核的大小，你可以根据需要调整

        # 进行腐蚀操作
        # img_points = cv2.erode(img_points, kernel, iterations=1)
        render_imgs.append(img_points)


        # iuv_image_gt = iuv_image_gt_batch[draw_i].detach().cpu().numpy()
        # iuv_image_gt = np.transpose(iuv_image_gt, (1, 2, 0)) * 255
        # iuv_image_gt = iuv_image_gt[:,:,[2,1,0]]
        # iuv_image_gt_resized = resize(iuv_image_gt, (img_vis.shape[0], img_vis.shape[1]),
        #                                         preserve_range=True, anti_aliasing=True)
        # render_imgs.append(iuv_image_gt_resized.astype(np.uint8))


        # depth_image_gt = depth_image_gt_render[draw_i].detach().cpu().numpy()
        # depth_image_gt = np.transpose(depth_image_gt, (1, 2, 0)) * 255
        # depth_image_gt = depth_image_gt[:,:,[2,1,0]]
        # depth_image_gt_resized = resize(depth_image_gt, (img_vis.shape[0], img_vis.shape[1]),
        #                                         preserve_range=True, anti_aliasing=True)
        # render_imgs.append(depth_image_gt_resized.astype(np.uint8))
        
        # # feat = vis_feat_list[draw_i].squeeze(0)  # Size([96, 56, 56])
        # # feat = torch.sum(feat, dim=0).detach().cpu().numpy()
        # # feat = torch.mean(feat, dim=0).detach().cpu().numpy()
        # render_vis=[]
        # a = [9,18,25,44,48,58,66,70,83,95]
        # a_label= [-1, 1, -1, 1, -1, 1, 1, -1, 1, 1]
        # for i in range(len(a)):
        #     channel = a[i]
        #     feat = vis_feat_list[draw_i].squeeze(0)[channel,:,:].squeeze(0).detach().cpu().numpy()

        #     # Size([56, 56])

        #     # mean = np.mean(feat)
        #     # std = np.std(feat)
        #     # feat = (feat - mean)   / std
        #     # feat = 255 * (feat + 1)  /  2

        #     max = np.max(feat)
        #     min = np.min(feat)
        #     feat = (feat - min) / (max -min)
        #     feat = 255 * feat 
        #     feat[feat < 0] = 0
        #     feat[feat > 255] = 255
        #     if a_label[i] == -1:
        #         feat = 255 - feat

        #     # feat = cv2.applyColorMap(feat.astype(np.uint8), cv2.COLORMAP_JET)
            
        #     # feat = resize(feat, (img_vis.shape[0], img_vis.shape[1]),
        #     #                                         preserve_range=True, anti_aliasing=True)
            
        #     render_vis.append(feat.astype(np.uint8))
        
        # vis = np.sum(render_vis,axis=0) / len(a)

        # # mean = np.mean(vis)
        # # std = np.std(vis)
        # # vis = (vis - mean)   / std
        # # vis = 255 * (feat + 1)  /  2


        # max = np.max(vis)
        # min = np.min(vis)
        # vis = (vis - min) / (max -min)
        # vis = 255 * vis

        # vis[vis < 0] = 0
        # vis[vis > 255] = 255
        # # vis = 255 - vis

        # vis = vis.astype(np.uint8)

        # vis = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        # # vis = resize(vis, (img_vis.shape[0], img_vis.shape[1]), preserve_range=True, anti_aliasing=False)
        # vis = cv2.resize(vis, (img_vis.shape[0], img_vis.shape[1]), interpolation=cv2.INTER_NEAREST)
        

        # render_imgs.append(vis)

        # hist = cv2.calcHist([vis], [0], )

        # matplotlib.image.imsave(os.path.join(save_path, draw_name[:-4] + '.png'), feat.astype(np.uint8))



        img = np.concatenate(render_imgs, axis=1)
        cv2.imwrite(os.path.join(save_path, draw_name[:-4] + '.png'),img)

        
        # img_orig, img_resized, img_smpl, render_smpl_rgba = dr_render(
        #     verts=vert_pred[draw_i],
        #     faces=face,
        #     mesh_filename=draw_name[:-4],
        #     img=image[draw_i],
        #     cam=cam_pred[draw_i],
        #     rgba=True
        # )

        # ones_img = np.ones(img_smpl.shape[:2]) * 255
        # ones_img = ones_img[:, :, None]
        # img_smpl_rgba = np.concatenate((img_smpl * 255, ones_img), axis=2)
        # img_resized_rgba = np.concatenate((img_resized * 255, ones_img), axis=2)

        # render_img = np.concatenate((img_resized_rgba, img_smpl_rgba, render_smpl_rgba * 255), axis=1)
        # render_img[render_img < 0] = 0
        # render_img[render_img > 255] = 255
        # matplotlib.image.imsave(os.path.join(save_path, draw_name[:-4] + '.png'), render_img.astype(np.uint8))

        # if pred_uv is not None:
        #     # estimated global IUV
        #     global_iuv = iuv_img[draw_i].cpu().numpy()
        #     global_iuv = np.transpose(global_iuv, (1, 2, 0))
        #     global_iuv = resize(global_iuv, img_resized.shape[:2])
        #     global_iuv[global_iuv > 1] = 1
        #     global_iuv[global_iuv < 0] = 0
        #     matplotlib.image.imsave(os.path.join(save_path, 'pred_uv_' + draw_name[:-4] + '.png'), global_iuv)