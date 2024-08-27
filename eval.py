from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

from lib.ghg.human_loader import HumanDataset
from lib.ghg.network_eval import GaussianRegressor
from config.default_config import HumanConfig as config
from lib.ghg.utils import get_eval_calib
from lib.ghg.GaussianRender import pts2render

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class HumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file

        self.bs = self.cfg.batch_size
        self.generator_dict = None

        self.dataset_name = 'THuman2.0'

        self.model = GaussianRegressor(self.cfg, with_gs_render=True)
        self.dataset = HumanDataset(self.cfg.dataset, phase=phase)

        # debug dataloader here
        #self.dataset.get_item_debug(0)

        self.model.cuda()
        if self.cfg.restore_ckpt and self.cfg.generator_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt, self.cfg.generator_ckpt)
        self.model.eval()

    def infer_static(self, view_select, novel_view_nums, bg_color):

        total_samples = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))

        subangle_map={0:0, 1:2, 2:3, 3:4}


        for idx in tqdm(range(total_samples)):

            # for quantitative evaluation
            item = self.dataset.get_test_item(idx, source_id=view_select)
            data = self.fetch_data(item)

            with torch.no_grad():

                data = self.model(data, is_train=False)

                for i in range(novel_view_nums):
                    subangle = subangle_map[i]
                    ratio_tmp = (i+0.5)*(1/novel_view_nums)

                    data_i = get_eval_calib(data, self.cfg.dataset, data['name'],subangle)

                    if bg_color == 'black':
                        data_i = pts2render(data_i, bg_color=[0,0,0],phase='test')
                    elif bg_color == 'white':
                        data_i = pts2render(data_i,bg_color=[1,1,1],phase='test')

                    render_novel = self.tensor2np(data_i['novel_view']['img_pred'])
                    cv2.imwrite(self.cfg.test_out_path + '/%s_novel%s.jpg' % (data_i['name'], str(i).zfill(2)), render_novel)


    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):

        for key in data.keys():
            if key in ['pos','outer_pos','outer_pos_1','outer_pos_2','outer_pos_3','outer_pos_4']: # position map
                data[key] = data[key].cuda().unsqueeze(0)
            elif key in ['input_view']:
                for sub_key in data[key].keys():
                    data[key][sub_key] = data[key][sub_key].cuda().unsqueeze(0)

        return data

    def load_ckpt(self, regressor_path, inpaintor_path):

        assert os.path.exists(regressor_path)
        logging.info(f"Loading checkpoint from {regressor_path} ...")
        ckpt = torch.load(regressor_path, map_location='cuda')

        missing_keys, unexpected_keys = self.model.load_state_dict(ckpt['network'], strict=False)
        generator_state_dict = torch.load(inpaintor_path,map_location='cuda')
        generator_prefix = 'generator.'
        generator_specific_dict = {generator_prefix + k: v for k, v in
                                   generator_state_dict.items()}
        missing_keys, unexpected_keys = self.model.load_state_dict(generator_specific_dict, strict=False)

        print("Weights loaded!")





if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--regressor_path', type=str, required=True) # Gaussian Regressor
    parser.add_argument('--inpaintor_path', type=str, required=True) # Inpaint Net

    parser.add_argument('--novel_view_nums', type=int, default=4)
    parser.add_argument('--bg_color', type=str, default='black')

    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'config.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = arg.regressor_path
    cfg.generator_ckpt = arg.inpaintor_path


    exp_name = 'GHG'

    cfg.test_out_path = os.path.join('./outputs/eval',exp_name)


    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = HumanRender(cfg, phase='test')

    render.infer_static(view_select=[0, 1], novel_view_nums=arg.novel_view_nums, bg_color=arg.bg_color)
