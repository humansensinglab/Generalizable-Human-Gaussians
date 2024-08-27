import torch
from torch import nn
from core.extractor import UnetExtractor
from lib.ghg.gs_parm_network import GSRegressor
from lib.ghg.deep_fill_v2 import GatedGenerator
from lib.loss import sequence_loss
from lib.utils import repeat_interleave
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
import cv2
import pdb

def dilate(mask, kernel_size=5):
    device = mask.device
    shape = mask.shape
    mask = mask.detach().cpu().numpy() * 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))) / 255
    mask = torch.tensor(mask, device=device).view(shape)
    return mask

class GaussianRegressor(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.train_iters = self.cfg.raft.train_iters
        self.val_iters = self.cfg.raft.val_iters

        # self.img_encoder encodes the RGB aggregated on each scaffold
        # in_channel = 3 (RGB) x 5 (scaffolds = 1 base SMPL surface + 4 outer surfaces)
        self.img_encoder = UnetExtractor(in_channel=3*5,
                                         encoder_dim=self.cfg.raft.encoder_dims)

        self.gs_parm_regressor = GSRegressor(self.cfg, rgb_dim=3, depth_dim=1)

        self.generator = GatedGenerator()

    def forward(self, data, is_train=True):

        bs = data['pos'].shape[0]

        # extract UV map foreground region
        valid = (data['pos'] != 0.0)
        valid = torch.logical_or(torch.logical_or(valid[:, 0:1, :, :], valid[:, 1:2, :, :]), valid[:, 2:3, :, :])
        valid = valid.permute(0, 2, 3, 1)

        # 0th level scaffold (base SMPL surface)
        aggregated_image_2d, inpaint_mask_2d = self.aggregate_rgb_image_visibility(
            data,
            data['pos'],
            data['input_view']['visibility'],
            valid,
            bs, shell_idx=0)

        # 1st level scaffold
        out_shell_rgb_1, out_shell_mask_1 = self.aggregate_rgb_image_visibility(
            data, data['outer_pos_1'], data['input_view']['outer_visibility_1'],
            valid, bs, shell_idx=1)
        valid_out_shell_1 = torch.logical_and(valid, torch.logical_not(out_shell_mask_1.permute(0, 2, 3, 1)))
        # 2nd level scaffold
        out_shell_rgb_2, out_shell_mask_2 = self.aggregate_rgb_image_visibility(
            data, data['outer_pos_2'], data['input_view']['outer_visibility_2'],
            valid, bs, shell_idx=2)
        valid_out_shell_2 = torch.logical_and(valid, torch.logical_not(out_shell_mask_2.permute(0, 2, 3, 1)))
        # 3rd level scaffold
        out_shell_rgb_3, out_shell_mask_3 = self.aggregate_rgb_image_visibility(
            data, data['outer_pos_3'], data['input_view']['outer_visibility_3'],
            valid, bs, shell_idx=3)
        valid_out_shell_3 = torch.logical_and(valid, torch.logical_not(out_shell_mask_3.permute(0, 2, 3, 1)))
        # 4th level scaffold
        out_shell_rgb_4, out_shell_mask_4 = self.aggregate_rgb_image_visibility(
            data, data['outer_pos_4'], data['input_view']['outer_visibility_4'],
            valid, bs, shell_idx=4)
        valid_out_shell_4 = torch.logical_and(valid, torch.logical_not(out_shell_mask_4.permute(0, 2, 3, 1)))


        data['out_shell_1'] = {}
        data['out_shell_2'] = {}
        data['out_shell_3'] = {}
        data['out_shell_4'] = {}

        data['out_shell_1']['rgb_maps'] = out_shell_rgb_1
        data['out_shell_2']['rgb_maps'] = out_shell_rgb_2
        data['out_shell_3']['rgb_maps'] = out_shell_rgb_3
        data['out_shell_4']['rgb_maps'] = out_shell_rgb_4


        data['out_shell_1']['pts_valid'] = valid_out_shell_1.view(bs, -1)
        data['out_shell_2']['pts_valid'] = valid_out_shell_2.view(bs, -1)
        data['out_shell_3']['pts_valid'] = valid_out_shell_3.view(bs, -1)
        data['out_shell_4']['pts_valid'] = valid_out_shell_4.view(bs, -1)

        data['in_shell'] = {}


        # inpainting
        mask = inpaint_mask_2d
        fg_mask = valid.permute(0,3,1,2)*1.0
        inpaint_input = fg_mask*aggregated_image_2d + (1-fg_mask)*(-1)
        _, inpainted_image = self.generator(inpaint_input, mask)
        combined_image = aggregated_image_2d * (1-mask) + inpainted_image * mask
        data['in_shell']['rgb_maps'] = combined_image

        # appearance cue (RGB maps aggregated on each scaffold)
        appearance_map = torch.concatenate(
            [aggregated_image_2d, out_shell_rgb_1, out_shell_rgb_2,
             out_shell_rgb_3,out_shell_rgb_4], 1)


        with autocast(enabled=self.cfg.raft.mixed_precision):

            appearance_feat = self.img_encoder(appearance_map)

            data = self.decode_gsparms(appearance_map, appearance_feat, data, bs, valid)

            data['in_shell']['inpaint_input'] = aggregated_image_2d
            data['in_shell']['inpaint_mask'] = inpaint_mask_2d

            return data

    def decode_gsparms(self, appearance_map, appearance_feat, data, bs, valid):


        data['in_shell']['xyz'] = data['pos'].permute(0, 2, 3, 1).view(bs, -1,3)
        data['out_shell_1']['xyz'] = data['outer_pos_1'].permute(0,2,3,1).view(bs,-1,3)
        data['out_shell_2']['xyz'] = data['outer_pos_2'].permute(0,2,3,1).view(bs,-1,3)
        data['out_shell_3']['xyz'] = data['outer_pos_3'].permute(0,2,3,1).view(bs,-1,3)
        data['out_shell_4']['xyz'] = data['outer_pos_4'].permute(0,2,3,1).view(bs,-1,3)


        data['in_shell']['pts_valid'] = valid.view(bs, -1) # [1, 1048576]

        # compute offset maps (offset between neighboring scaffolds)
        pos_offset_0 = data['outer_pos_1'] - data['pos']
        pos_offset_1 = data['outer_pos_2'] - data['outer_pos_1']
        pos_offset_2 = data['outer_pos_3'] - data['outer_pos_2']
        pos_offset_3 = data['outer_pos_4'] - data['outer_pos_3']

        geometry_map = torch.concatenate([pos_offset_0,pos_offset_1,pos_offset_2,pos_offset_3],1)

        rot_maps, scale_maps, opacity_maps = self.gs_parm_regressor(
            appearance_map,
            geometry_map,
            appearance_feat)
        opacity_maps = opacity_maps.type(torch.float32)

        for shell_idx, shell_name in enumerate(['in_shell','out_shell_1','out_shell_2','out_shell_3','out_shell_4']):

            data[shell_name]['rot_maps'] = rot_maps[:,shell_idx*4:shell_idx*4+4,:,:]
            data[shell_name]['scale_maps'] = scale_maps[:,shell_idx*3:shell_idx*3+3, :, :]
            data[shell_name]['opacity_maps'] = opacity_maps[:,shell_idx*1:shell_idx*1+1, :, :]

        return data

    def aggregate_rgb_image_visibility(self, data, pos, visibility_masks, valid, bs, shell_idx):

        visibility_masks = torch.clamp(visibility_masks,0.0,1.0)

        # transform valid 3D points into 2D
        input_imgs = data['input_view']['img']
        input_masks = data['input_view']['mask']
        input_imgs = torch.concat([input_imgs, input_masks[:, :, 0:1, :, :]], dim=2)

        uvmap_shape = data['pos'].shape[-2:]
        rgb_scale = torch.tensor(
            [input_imgs.shape[-1], input_imgs.shape[-2]]).cuda()
        rgb_scale = rgb_scale / (rgb_scale - 1) * 2.0

        valid = valid.view(bs, -1)
        xyz = pos.permute(0, 2, 3, 1)
        xyz = xyz.view(bs, -1, 3)

        input_R = data['input_view']['extr'][..., :3]
        input_T = data['input_view']['extr'][..., 3:]
        input_K = data['input_view']['intr']


        aggregated_image_2d_list = []
        inpaint_mask_2d_list = []

        for i in range(bs):

            # # inputs denotes the number of input views
            input_R_i = input_R[i].reshape(-1, 3, 3) # [# inputs, 3, 3]
            input_T_i = input_T[i].reshape(-1, 3, 1) # [# inputs, 3, 1]
            input_K_i = input_K[i].reshape(-1, 3, 3) # # [# inputs, 3, 3]

            xyz_i = xyz[i]
            valid_i = valid[i]
            xyz_i = xyz_i[valid_i][None]

            pts_3d = repeat_interleave(xyz_i, input_R_i.shape[0])  # [# inputs, n_valid, 3]
            pts_3d_rot = torch.matmul(input_R_i[:, None], pts_3d.unsqueeze(-1))[..., 0]
            pts_cam = pts_3d_rot + input_T_i[:, None, :3, 0]
            pts_2d = torch.matmul(input_K_i[:, None], pts_cam.unsqueeze(-1))[..., 0]
            uv = pts_2d[:, :, :2] / pts_2d[:, :, 2:]  # [# inputs, n_valid, 3]
            input_imgs_i = input_imgs[i]


            image_shape = input_imgs.shape[-2:]
            rgb_value = self.sample_from_feature_map(input_imgs_i,rgb_scale, image_shape,uv)
            input_imgs_i = input_imgs_i[:, :3, :, :]
            visibility_i = visibility_masks[i]  # [# inputs, 1024, 1024, 1]

            # select a view that has the highest visibility
            max_visibility_mask, indices = torch.max(visibility_i,dim=0,keepdim=True)
            tmp_viz = torch.zeros_like(visibility_i) + torch.range(0,visibility_i.shape[0]-1).view(-1,1,1,1).cuda()
            tmp_viz_mask = (tmp_viz==indices)
            visibility_i = tmp_viz_mask*visibility_i


            visibility_i = visibility_i.view(input_R_i.shape[0], -1, 1)  # [# inputs, 1024*1024, 1]
            visibility_i = visibility_i[:, valid_i]  # [# inputs, n_valid, 1]

            # foreground mask in the UV space
            fg_mask = rgb_value[..., 3:4, :]
            fg_mask = fg_mask.permute(0, 2, 1)  # [# inputs, n_valid, 1]
            visibility_i = visibility_i * fg_mask

            rgb_value = rgb_value[..., :3, :]
            rgb_value = rgb_value.permute(0, 2, 1)

            aggregated_img = torch.zeros_like(data['pos'][0]).permute(1, 2, 0)

            visibility_sum_i = visibility_i.sum(axis=0)
            inpaint_mask = (visibility_sum_i == 0).type(torch.float32)

            mask = torch.zeros_like(aggregated_img)[..., 0]
            mask = mask.view(-1, 1)
            mask[valid_i] = inpaint_mask

            inpaint_mask_2d = mask.reshape(image_shape[0], image_shape[1], 1)
            if shell_idx == 0:
                inpaint_mask_2d = dilate(inpaint_mask_2d, kernel_size=5)
            inpaint_mask_2d_list.append(inpaint_mask_2d.permute(2, 0, 1))

            weighted_rgb = (visibility_i * rgb_value).sum(axis=0)
            weighted_rgb = weighted_rgb / (visibility_sum_i + 1e-6)

            aggregated_img = aggregated_img.view(-1, 3)
            aggregated_img[valid_i] = weighted_rgb
            aggregated_img_2d = aggregated_img.reshape(1024, 1024, 3)
            aggregated_image_2d_list.append(aggregated_img_2d.permute(2, 0, 1))

        aggregated_image_2d_tensor = torch.stack(aggregated_image_2d_list)
        inpaint_mask_2d_tensor = torch.stack(inpaint_mask_2d_list)

        return aggregated_image_2d_tensor, inpaint_mask_2d_tensor


    def sample_from_feature_map(self, feat_map, feat_scale, image_shape, uv):

        h = image_shape[0]
        w = image_shape[1]
        image_shape = torch.tensor((w, h)).cuda()
        scale = feat_scale / image_shape

        uv = uv * scale - 1.0
        uv = uv.unsqueeze(2)

        samples = F.grid_sample(
            feat_map,
            uv,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )

        return samples[:, :, :, 0]

    def projection_3D_to_2D(self, pts_3d, R, T, K):

        input_R = R.reshape(-1, 3, 3)
        input_T = T.reshape(-1, 3, 1)
        input_K = K.reshape(-1, 3, 3)

        pts_3d = pts_3d[None]
        pts_3d_rot = torch.matmul(input_R[:, None], pts_3d.unsqueeze(-1))[..., 0]

        pts_cam = pts_3d_rot + input_T[:, None, :3, 0]

        pts_2d = torch.matmul(input_K[:, None], pts_cam.unsqueeze(-1))[..., 0]
        pts_2d = pts_2d[:, :, :3] / pts_2d[:, :, 2:]

        return pts_2d