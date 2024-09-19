import os
import smplx
import trimesh
import pickle
import torch
import numpy as np
from tqdm import tqdm

def generate_smplx(split_path, smplx_model_path, transform_root, param_root, output_data_root):

    model_init_params = dict(
        gender='male',
        model_type='smplx',
        model_path=smplx_model_path,
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_transl=False,
        num_pca_comps=12)

    smpl_model = smplx.create(**model_init_params)



    human_list = []
    with open(split_path, 'r') as f:
        for line in f:
            human_name = line.strip()
            human_list.append(human_name)
    human_list.sort()


    n_humans = len(human_list)

    for human_idx, human in tqdm(enumerate(human_list), total=n_humans):

        print(human)

        # read the original THuman smplx parameters
        param_fp = os.path.join(param_root,human,'smplx_param.pkl')

        param = np.load(param_fp, allow_pickle=True)
        for key in param.keys():
            param[key] = torch.as_tensor(param[key]).to(torch.float32)

        model_forward_params = dict(betas=param['betas'],
                                    global_orient=param['global_orient'],
                                    body_pose=param['body_pose'],
                                    left_hand_pose=param['left_hand_pose'],
                                    right_hand_pose=param['right_hand_pose'],
                                    jaw_pose=param['jaw_pose'],
                                    leye_pose=param['leye_pose'],
                                    reye_pose=param['reye_pose'],
                                    expression=param['expression'],
                                    return_verts=True)

        smpl_out = smpl_model(**model_forward_params)

        smpl_verts = (
            (smpl_out.vertices[0] * param['scale'] + param['translation'])).detach()

        transform_fp = os.path.join(transform_root,human,"{}_transform.npy".format(human))
        transform = np.load(transform_fp,allow_pickle=True).item()

        vy_max = transform['vy_max']
        vy_min = transform['vy_min']
        human_height = transform['human_height']
        offset = transform['offset']
        smpl_verts[:, :3] = smpl_verts[:, :3] / (vy_max - vy_min) * human_height
        smpl_verts[:, 1] -= offset

        move_delta_axis_0 = transform['move_delta_axis_0']
        move_delta_axis_2 = transform['move_delta_axis_2']
        smpl_verts[:, 0] += move_delta_axis_0
        smpl_verts[:, 2] += move_delta_axis_2

        regenerated_smpl_mesh = trimesh.Trimesh(smpl_verts,
                                    smpl_model.faces,
                                    process=False,
                                    maintain_order=True)

        regenerated_mesh_fname = os.path.join(output_data_root,'{}.obj'.format(human))
        regenerated_smpl_mesh.export(regenerated_mesh_fname)

        print('human {}/{} processed!'.format(str(human_idx),n_humans))



if __name__ == '__main__':

    for phase in ['train', 'val']:

        split_path = 'datasets/THuman/split_{}.txt'.format(phase)
        # 'model_path' should be the root directory of SMPL-X directory.
        # e.g., model_path - smplx - SMPLX_NEUTRAL.pkl
        smplx_model_path = 'datasets/THuman/models'

        transform_root = 'datasets/THuman/THuman2.0_Release'

        # please set this to the root directory of the original THuman 2.0 smplx parameters
        param_root = 'datasets/THuman/THuman2.0_smplx'

        output_data_root = 'datasets/THuman/{}/smplx_obj'.format(phase)
        if not os.path.exists(output_data_root):
            os.makedirs(output_data_root)

        generate_smplx(split_path, smplx_model_path, transform_root, param_root, output_data_root)