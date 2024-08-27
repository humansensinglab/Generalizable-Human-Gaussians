import torch
from gaussian_renderer import render
import pdb

def pts2render(data, bg_color, phase='train'):
    bs = data['pos'].shape[0]

    render_novel_list = []
    render_depth_list = []
    render_alpha_list = []

    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []

        for shell in ['in_shell', 'out_shell_1', 'out_shell_2', 'out_shell_3', 'out_shell_4']:


            valid_i = data[shell]['pts_valid'][i, :]
            xyz_i = data[shell]['xyz'][i, :, :]
            rgb_i = data[shell]['rgb_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)


            rot_i = data[shell]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = data[shell]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1,3)

            if (shell == 'in_shell') and (phase == 'test'):
                scale_i = torch.clamp_min(scale_i,0.0013)

            opacity_i = data[shell]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))

            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))


        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)

        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        multi_view_renders = []
        multi_depth_renders = []
        multi_alpha_renders = []

        if phase == 'test':
            novel_view_name_list = ['novel_view']
        elif phase == 'train':
            novel_view_name_list = ['novel_view_0', 'novel_view_1', 'novel_view_2']

        for novel_view in novel_view_name_list:

            render_novel_view_i, render_novel_depth_i, render_novel_alpha_i \
                = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i,
                         opacity_i, bg_color=bg_color,
                         novel_view_name=novel_view)

            multi_view_renders.append(render_novel_view_i)
            multi_depth_renders.append(render_novel_depth_i)
            multi_alpha_renders.append(render_novel_alpha_i)

        # for multi-view supervision
        multi_view_renders_tensor = torch.stack(multi_view_renders, 0)
        multi_depth_renders_tensor = torch.stack(multi_depth_renders, 0)
        multi_alpha_renders_tensor = torch.stack(multi_alpha_renders, 0)

        render_novel_list.append(multi_view_renders_tensor.unsqueeze(0))
        render_depth_list.append(multi_depth_renders_tensor.unsqueeze(0))
        render_alpha_list.append(multi_alpha_renders_tensor.unsqueeze(0))


    if phase == 'test':

        predictions = torch.concat(render_novel_list, dim=0)
        data['novel_view']['img_pred'] = predictions[:, 0]

        depth_predictions = torch.concat(render_depth_list, dim=0)
        data['novel_view']['depth_pred'] = depth_predictions[:, 0]

        alpha_predictions = torch.concat(render_alpha_list, dim=0)
        data['novel_view']['alpha_pred'] = alpha_predictions[:, 0]

    elif phase == 'train':

        predictions = torch.concat(render_novel_list,dim=0)
        data['novel_view_0']['img_pred'] = predictions[:, 0]
        data['novel_view_1']['img_pred'] = predictions[:, 1]
        data['novel_view_2']['img_pred'] = predictions[:, 2]

        depth_predictions = torch.concat(render_depth_list, dim=0)
        data['novel_view_0']['depth_pred'] = depth_predictions[:, 0]
        data['novel_view_1']['depth_pred'] = depth_predictions[:, 1]
        data['novel_view_2']['depth_pred'] = depth_predictions[:, 2]

        alpha_predictions = torch.concat(render_alpha_list, dim=0)
        data['novel_view_0']['alpha_pred'] = alpha_predictions[:, 0]
        data['novel_view_1']['alpha_pred'] = alpha_predictions[:, 1]
        data['novel_view_2']['alpha_pred'] = alpha_predictions[:, 2]

    return data
