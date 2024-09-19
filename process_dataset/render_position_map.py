'''
MIT License

Copyright (c) 2024 Youngjoong Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import sys
import math
import trimesh
import os

def find_3d_vertices_for_uv(faces):
    uv_to_vertices = {}

    vertices = []
    uvs = []

    for face in faces:
        for idx in range(3):

            vertex_index, uv_index = face[idx]

            if uv_index in uv_to_vertices:
                uv_to_vertices[uv_index].add(vertex_index)
            else:
                uv_to_vertices[uv_index] = set()
                uv_to_vertices[uv_index].add(vertex_index)

    return uv_to_vertices

def load_obj(file_path):

    vertices = []
    faces = []
    uvs = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == 'v':

                x, y, z = map(float, tokens[1:4])
                vertices.append((x, y, z))

            elif tokens[0] == 'vt':

                u, v = map(float, tokens[1:3])
                uvs.append((u, v))

            elif tokens[0] == 'f':

                face = []
                for token in tokens[1:]:
                    vertex_info = token.split('/')
                    vertex_index = int(vertex_info[0]) - 1
                    uv_index = int(vertex_info[1]) - 1 if len(
                        vertex_info) > 1 else None
                    face.append((vertex_index, uv_index))
                faces.append(face)

    return vertices, faces, uvs


num_angles = 16
phase = 'train'  # 'val' #'train'
data_root = 'datasets/THuman/{}'.format(phase)
calib_dir = os.path.join(data_root, 'parm')
depth_dir = os.path.join(data_root, 'depth')
human_list = set()

for dir_name in os.listdir(os.path.join(data_root, 'img')):
    subject_name = dir_name.split('_')[0]
    human_list.add(subject_name)
human_list = list(human_list)
human_list.sort()

resolution = 1024

# image plane shape
image_height = 1024
image_width = 1024

glctx = dr.RasterizeCudaContext()

smplx_fp = "datasets/THuman/smplx_uv.obj"
vertices_tpose, faces, uvs = load_obj(smplx_fp)

vertices_tpose = np.array(vertices_tpose)
faces = np.array(faces)
uvs = np.array(uvs)
uv_pts_mapping = find_3d_vertices_for_uv(faces)

n_uvs = uvs.shape[0]

pos = uvs
pos = 2 * pos - 1
final_pos = np.stack(
    [pos[..., 0], pos[..., 1], np.zeros_like(pos[..., 0]),
     np.ones_like(pos[..., 0])], axis=-1)
final_pos = final_pos.reshape((1, -1, 4))

pos_uv = torch.from_numpy(final_pos).to(dtype=torch.float32, device='cuda')
tri_uv = torch.from_numpy(faces[...,1]).to(dtype=torch.int32, device='cuda')
rast_uv_space, _ = dr.rasterize(glctx, pos_uv, tri_uv, resolution=[resolution, resolution])

face_id_raw = rast_uv_space[..., 3:]
face_id = face_id_raw[0]

for scale_idx in range(5):

    scale = scale_idx * 0.01

    if scale_idx == 0:
        position_dir = os.path.join(data_root,'position_map_uv_space')

    elif scale_idx > 0:
        position_dir = os.path.join(data_root,'position_map_uv_space_outer_shell_{}'.format(scale_idx))

    if not os.path.exists(position_dir):
        os.makedirs(position_dir)

    for human_idx in range(len(human_list)):

        human = human_list[human_idx]

        obj_file_path = os.path.join(data_root, 'smplx_obj','{}.obj'.format(human))

        vertices, _, _ = load_obj(obj_file_path)
        vertices = np.array(vertices)
        vertices = torch.from_numpy(vertices).to(dtype=torch.float32, device='cuda')

        mesh = trimesh.load(obj_file_path, process=False)
        normals = np.array(mesh.vertex_normals)
        normals = torch.from_numpy(normals).to(dtype=torch.float32, device='cuda')

        vertices = vertices + scale * normals

        attr = []
        for uv_idx in range(len(uvs)):
            for vertex_idx in uv_pts_mapping[uv_idx]:
                attr.append(vertices[vertex_idx])

        attr = torch.stack(attr, dim=0)
        attr = attr[None]


        out2, _ = dr.interpolate(attr, rast_uv_space, tri_uv)

        bg_indices = torch.nonzero(face_id <= 0)
        out2[:, bg_indices[:, 0], bg_indices[:, 1], 0] = 0
        out2[:, bg_indices[:, 0], bg_indices[:, 1], 1] = 0
        out2[:, bg_indices[:, 0], bg_indices[:, 1], 2] = 0
        out3 = out2.cpu().numpy()[0, ::-1, :, :]


        position_map_name = os.path.join(position_dir,'{}_{}.npy'.format(human,resolution))
        out3 = out3.astype(np.float32)
        np.save(position_map_name, out3)