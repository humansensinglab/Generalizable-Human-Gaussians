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

def compute_fov_from_focal_length(focal_length, image_size):

    return 2 * math.atan(image_size / (2 * focal_length))

def compute_near_far(vertices, extr, device='cuda'):

    extr_4x4 = torch.cat([extr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)], dim=0).to(device)

    correction_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    extr_opengl = torch.matmul(correction_matrix, extr_4x4)

    ones = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)
    vertices_h = torch.cat([vertices, ones], dim=1).to(device)

    vertices_cam = torch.matmul(vertices_h, extr_opengl.T)

    z_vals = vertices_cam[:, 2]

    near = torch.max(z_vals)
    far = torch.min(z_vals)

    return near, far

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

def perspective_projection_opencv_to_opengl(fx, fy, cx, cy, near, far, width, height, device='cuda'):

    fovy_rad = compute_fov_from_focal_length(fy, height)
    fovx_rad = compute_fov_from_focal_length(fx, width)

    near = -near
    far = -far

    cot_half_fov = 1 / math.tan(fovy_rad / 2)

    aspect = width / height

    P_clip = torch.tensor([
        [1 / (aspect * math.tan(fovy_rad / 2)), 0, 0, 0],
        [0, 1 / math.tan(fovy_rad / 2), 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=torch.float32, device=device)

    c_x = cx
    c_y_flipped = height - cy

    p_x = c_x - width / 2
    p_y = c_y_flipped - height / 2

    # Handle scaling, if necessary
    scale = 0.001
    scale *= 2.0

    translate = torch.tensor([
        [1, 0, 0, -scale * p_x],
        [0, 1, 0, scale * p_y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    return P_clip, translate

def world_to_clip_space(vertices, extr, P_clip, translate, device='cuda'):

    extr_4x4 = torch.cat([extr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)], dim=0)

    correction_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    extr_corrected = torch.matmul(correction_matrix, extr_4x4)

    ones = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)
    vertices_h = torch.cat([vertices, ones], dim=1)

    vertices_cam = torch.matmul(vertices_h, extr_corrected.T)

    mtx = torch.matmul(P_clip, translate)

    vertices_clip = torch.matmul(vertices_cam, mtx.T)

    vertices_clip = vertices_clip / vertices_clip[:, 3:4]

    return vertices_clip[:, :4]


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

for scale_idx in range(5):

    scale = scale_idx * 0.01

    if scale_idx == 0:
        visibility_dir = os.path.join(data_root,'visibility_map_uv_space')

    elif scale_idx > 0:
        visibility_dir = os.path.join(data_root,'visibility_map_uv_space_outer_shell_{}'.format(scale_idx))

    if not os.path.exists(visibility_dir):
        os.makedirs(visibility_dir)

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

        for angle_idx in range(num_angles):

            angle = str(angle_idx).zfill(3)

            for subangle_idx in range(1):

                subangle = str(subangle_idx)

                intr_name = os.path.join(calib_dir, '{}_{}'.format(human, angle),
                                         '{}_intrinsic.npy'.format(subangle))
                extr_name = os.path.join(calib_dir, '{}_{}'.format(human, angle),
                                         '{}_extrinsic.npy'.format(subangle))

                intr, extr = np.load(intr_name), np.load(extr_name)
                intr = torch.from_numpy(intr).to(dtype=torch.float32, device='cuda')
                extr = torch.from_numpy(extr).to(dtype=torch.float32, device='cuda')

                f_x = intr[0][0]
                f_y = intr[1][1]
                c_x = intr[0][2]
                c_y = intr[1][2]

                near, far = compute_near_far(vertices, extr)

                P_clip, translate = perspective_projection_opencv_to_opengl(f_x, f_y,
                                                                            c_x, c_y,
                                                                            near,
                                                                            far,
                                                                            image_width,
                                                                            image_height)

                pos_clip = world_to_clip_space(vertices, extr, P_clip, translate)
                pos_clip = pos_clip.unsqueeze(0) # [1, 10475, 4]


                tri = torch.from_numpy(faces[..., 0]).to(dtype=torch.int32, device='cuda')
                rast, _ = dr.rasterize(glctx, pos_clip.contiguous(), tri,
                                       resolution=[image_width, image_height])

                face_id_raw = rast[..., 3:]
                face_id = face_id_raw[0]

                visible_faces = face_id.unique().type(torch.long) - 1
                visible_faces.sort()
                visible_faces = visible_faces[1:]
                packed_faces = torch.from_numpy(faces[..., 0]).type(torch.long).cuda()

                vertex_visibility_map = torch.zeros(vertices.shape[0]).cuda()

                visible_verts_idx = packed_faces[visible_faces]
                unique_visible_verts_idx = torch.unique(visible_verts_idx)

                vertex_visibility_map[unique_visible_verts_idx] = 1.0

                attr = []
                for uv_idx in range(len(uvs)):
                    for vertex_idx in uv_pts_mapping[uv_idx]:
                        attr.append(vertex_visibility_map[vertex_idx])

                attr = torch.tensor(attr,dtype=torch.float32, device='cuda')

                attr = attr[None][..., None]

                rast_visibility, _ = dr.interpolate(attr, rast_uv_space, tri_uv)

                visibility_np = rast_visibility.cpu().numpy()[0, ::-1,:]

                visibility_np = visibility_np.astype(np.float32)
                visibility_np = np.clip(visibility_np, 0,1)
                visibility_fp = os.path.join(visibility_dir,'{}_{}.npy'.format(human, angle))

                np.save(visibility_fp,visibility_np)

                print('Processed human {} angle {} - {} ! {}/{} '.format(human,
                                                                         str(angle),
                                                                         str(subangle),
                                                                         str(human_idx),
                                                                         str(len(
                                                                             human_list))))