import taichi_three as t3
import numpy as np
from taichi_three.transform import *
from pathlib import Path
from tqdm import tqdm
import os
import cv2
import pickle
import pdb

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def save(pid, data_id, vid, save_path, extr, intr, depth, img, mask,
         img_hr=None):
    img_save_path = os.path.join(save_path, 'img', data_id + '_' + '%03d' % pid)
    depth_save_path = os.path.join(save_path, 'depth',
                                   data_id + '_' + '%03d' % pid)
    mask_save_path = os.path.join(save_path, 'mask',
                                  data_id + '_' + '%03d' % pid)
    parm_save_path = os.path.join(save_path, 'parm',
                                  data_id + '_' + '%03d' % pid)
    Path(img_save_path).mkdir(exist_ok=True, parents=True)
    Path(parm_save_path).mkdir(exist_ok=True, parents=True)
    Path(mask_save_path).mkdir(exist_ok=True, parents=True)
    Path(depth_save_path).mkdir(exist_ok=True, parents=True)

    depth = depth * 2.0 ** 15
    cv2.imwrite(os.path.join(depth_save_path, '{}.png'.format(vid)),
                depth.astype(np.uint16))
    img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
    mask = (np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(img_save_path, '{}.jpg'.format(vid)), img)
    if img_hr is not None:
        img_hr = (np.clip(img_hr, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :,
                 ::-1]
        cv2.imwrite(os.path.join(img_save_path, '{}_hr.jpg'.format(vid)),
                    img_hr)
    cv2.imwrite(os.path.join(mask_save_path, '{}.png'.format(vid)), mask)
    np.save(os.path.join(parm_save_path, '{}_intrinsic.npy'.format(vid)), intr)
    np.save(os.path.join(parm_save_path, '{}_extrinsic.npy'.format(vid)), extr)


class StaticRenderer:
    def __init__(self):
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        self.scene = t3.Scene()
        self.N = 10

    def change_all(self):
        save_obj = []
        save_tex = []
        for model in self.scene.models:
            save_obj.append(model.init_obj)
            save_tex.append(model.init_tex)
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        print('init')
        self.scene = t3.Scene()
        for i in range(len(save_obj)):
            model = t3.StaticModel(self.N, obj=save_obj[i], tex=save_tex[i])
            self.scene.add_model(model)

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj['vi'].shape[0], self.N)
        self.N = max(obj['f'].shape[0], self.N)
        if not (obj['vt'] is None):
            self.N = max(obj['vt'].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()
            self.camera_light()

    def add_model(self, obj, tex=None):
        self.check_update(obj)
        model = t3.StaticModel(self.N, obj=obj, tex=tex)
        self.scene.add_model(model)

    def modify_model(self, index, obj, tex=None):
        self.check_update(obj)
        self.scene.models[index].init_obj = obj
        self.scene.models[index].init_tex = tex
        self.scene.models[index]._init()

    def camera_light(self):
        camera = t3.Camera(res=(1024, 1024))
        self.scene.add_camera(camera)

        camera_hr = t3.Camera(res=(2048, 2048))
        self.scene.add_camera(camera_hr)

        light_dir = np.array([0, 0, 1])
        light_list = []
        for l in range(6):
            rotate = np.matmul(
                rotationX(math.radians(np.random.uniform(-30, 30))),
                rotationY(math.radians(360 // 6 * l)))
            dir = [*np.matmul(rotate, light_dir)]
            light = t3.Light(dir, color=[1.0, 1.0, 1.0])
            light_list.append(light)
        lights = t3.Lights(light_list)
        self.scene.add_lights(lights)


def render_data(renderer, smplx_path, data_path, phase, data_id, save_path, cam_nums, res,
                dis=1.0, is_thuman=False):


    # data_path 'datasets/THuman/THuman2.0_Release/'
    # smplx_path 'datasets/THuman/THuman2.0_smplx/'

    obj_path = os.path.join(data_path, data_id, '%s.obj' % data_id)
    smpl_obj_path = os.path.join(smplx_path, data_id, 'mesh_smplx.obj')
    texture_path = data_path
    img_path = os.path.join(texture_path, data_id, 'material0.jpeg')
    texture = cv2.imread(img_path)[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]
    obj = t3.readobj(obj_path, scale=1)
    smpl_obj = t3.readobj(smpl_obj_path, scale=1)
    original_smpl_obj = t3.readobj(smpl_obj_path, scale=1)

    # to save the transform
    transform_dict = {}

    # height normalization

    ### 1. original scan
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    height_delta = np.random.uniform(-0.05, 0.05, 1)
    print(height_delta)
    human_height = 1.80 + height_delta
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    offset = np.min(obj['vi'][:, 1])
    obj['vi'][:, 1] -= offset

    transform_dict['vy_max'] = vy_max
    transform_dict['vy_min'] = vy_min
    transform_dict['height_delta'] = height_delta
    transform_dict['human_height'] = human_height
    transform_dict['offset'] = offset

    ### 2. SMPL-X obj

    smpl_obj['vi'][:, :3] = smpl_obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    smpl_obj['vi'][:, 1] -= offset

    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    # randomly move the scan

    ### 1. original scan

    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01

    move_delta_axis_0 = np.random.uniform(-move_range, move_range, 1)
    move_delta_axis_2 = np.random.uniform(-move_range, move_range, 1)
    print(move_delta_axis_0)
    obj['vi'][:, 0] += move_delta_axis_0
    obj['vi'][:, 2] += move_delta_axis_2
    output_obj_path = os.path.join(data_path, data_id,
                                   '%s_modified.obj' % data_id)
    t3.save_modified_obj(obj_path, list(obj['vi']), output_obj_path)
    transform_dict['delta_x'] = delta_x
    transform_dict['delta_z'] = delta_z
    transform_dict['move_range'] = move_range
    transform_dict['move_delta_axis_0'] = move_delta_axis_0
    transform_dict['move_delta_axis_2'] = move_delta_axis_2

    ### 2. SMPL obj

    smpl_obj['vi'][:, 0] += move_delta_axis_0
    smpl_obj['vi'][:, 2] += move_delta_axis_2

    # save the modified scans
    output_transform_path = output_obj_path = os.path.join(data_path, data_id, '%s_transform.npy' % data_id)
    output_smpl_obj_path = os.path.join(data_path, data_id, '%s_smplx_modified.obj'% data_id)
    output_original_smpl_obj_path = os.path.join(data_path, data_id,'%s_smplx.obj' % data_id)



    t3.save_modified_obj(smpl_obj_path, list(smpl_obj['vi']), output_smpl_obj_path)
    t3.save_modified_obj(smpl_obj_path, list(original_smpl_obj['vi']), output_original_smpl_obj_path)
    np.save(output_transform_path, transform_dict)

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)


    degree_interval = 360 / cam_nums
    angle_list1 = list(range(360 - int(degree_interval // 2), 360))
    angle_list2 = list(range(0, 0 + int(degree_interval // 2)))
    angle_list = angle_list1 + angle_list2
    angle_base = np.random.choice(angle_list, 1)[0]

    if is_thuman:
        # thuman needs a normalization of orientation
        smpl_path = os.path.join(smplx_path, data_id,
                                 'smplx_param.pkl')
        with open(smpl_path, 'rb') as f:
            smpl_para = pickle.load(f)

        y_orient = smpl_para['global_orient'][0][1]
        angle_base += (y_orient * 180.0 / np.pi)

    for pid in range(cam_nums):
        angle = angle_base + pid * degree_interval

        def render(dis, angle, look_at_center, p, renderer, render_2k=False):
            ori_vec = np.array([0, 0, dis])
            rotate = np.matmul(rotationY(math.radians(angle)),
                               rotationX(math.radians(p)))
            fwd = np.matmul(rotate, ori_vec)
            cam_pos = look_at_center + fwd

            x_min = 0
            y_min = -25
            cx = res[0] * 0.5
            cy = res[1] * 0.5
            fx = res[0] * 0.8
            fy = res[1] * 0.8
            _cx = cx - x_min
            _cy = cy - y_min
            renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
            renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
            renderer.scene.cameras[0]._init()

            if render_2k: # False
                fx = res[0] * 0.8 * 2
                fy = res[1] * 0.8 * 2
                _cx = (res[0] * 0.5 - x_min) * 2
                _cy = (res[1] * 0.5 - y_min) * 2
                renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
                renderer.scene.cameras[1].set(pos=cam_pos,
                                              target=look_at_center)
                renderer.scene.cameras[1]._init()

                renderer.scene.render()
                camera = renderer.scene.cameras[0]
                camera_hr = renderer.scene.cameras[1]
                extrinsic = camera.export_extrinsic()
                intrinsic = camera.export_intrinsic()
                depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
                img = camera.img.to_numpy().swapaxes(0, 1)
                img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
                mask = camera.mask.to_numpy().swapaxes(0, 1)
                return extrinsic, intrinsic, depth_map, img, mask, img_hr

            renderer.scene.render()
            camera = renderer.scene.cameras[0]
            extrinsic = camera.export_extrinsic()
            intrinsic = camera.export_intrinsic()
            depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
            img = camera.img.to_numpy().swapaxes(0, 1)
            mask = camera.mask.to_numpy().swapaxes(0, 1)
            return extrinsic, intrinsic, depth_map, img, mask

        extr, intr, depth, img, mask = render(dis, angle, look_at_center,
                                              base_cam_pitch, renderer)
        save(pid, data_id, 0, save_path, extr, intr, depth, img, mask)
        extr, intr, depth, img, mask = render(dis,
                                              (angle + degree_interval) % 360,
                                              look_at_center, base_cam_pitch,
                                              renderer)
        save(pid, data_id, 1, save_path, extr, intr, depth, img, mask)

        # three novel viewpoints between source views
        angle1 = (angle + (np.random.uniform() * degree_interval / 2)) % 360
        angle2 = (angle + degree_interval / 2) % 360
        angle3 = (angle + degree_interval - (
                    np.random.uniform() * degree_interval / 2)) % 360

        extr, intr, depth, img, mask, img_hr = render(dis, angle1,
                                                      look_at_center,
                                                      base_cam_pitch, renderer,
                                                      render_2k=True)
        save(pid, data_id, 2, save_path, extr, intr, depth, img, mask, img_hr)
        extr, intr, depth, img, mask, img_hr = render(dis, angle2,
                                                      look_at_center,
                                                      base_cam_pitch, renderer,
                                                      render_2k=True)
        save(pid, data_id, 3, save_path, extr, intr, depth, img, mask, img_hr)
        extr, intr, depth, img, mask, img_hr = render(dis, angle3,
                                                      look_at_center,
                                                      base_cam_pitch, renderer,
                                                      render_2k=True)
        save(pid, data_id, 4, save_path, extr, intr, depth, img, mask, img_hr)


if __name__ == '__main__':

    np.random.seed(42)

    cam_nums = 16
    scene_radius = 2.0
    res = (1024, 1024)
    smplx_root = 'datasets/THuman/THuman2.0_smplx/'
    thuman_root = 'datasets/THuman/THuman2.0_Release/'
    source_root = 'datasets/THuman'
    save_root = 'datasets/THuman/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    renderer = StaticRenderer()

    for phase in ['train', 'val']:

        split_file = os.path.join(source_root,'split_{}.txt'.format(phase))
        thuman_list = []

        with open(split_file, 'r') as f:
            for line in f:
                human_name = line.strip()
                thuman_list.append(human_name)
        thuman_list.sort()

        save_path = os.path.join(save_root, phase)

        for data_id in tqdm(thuman_list):

            render_data(renderer, smplx_root, thuman_root, phase, data_id, save_path,
                        cam_nums, res, dis=scene_radius, is_thuman=True)
