from __future__ import print_function, division

import logging
import torch.nn as nn

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from lib.ghg.human_loader import HumanDataset
from lib.ghg.network_train_nightly_ver import GaussianRegressor
from config.default_config import HumanConfig as config
from lib.train_recorder import Logger, file_backup
from lib.ghg.GaussianRender import pts2render
from lib.loss import l1_loss, ssim, psnr

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import imageio


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = GaussianRegressor(self.cfg, with_gs_render=True)
        self.train_set = HumanDataset(self.cfg.dataset, phase='train')


        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size*2, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = HumanDataset(self.cfg.dataset, phase='val')

        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=True,
                                     num_workers=4, pin_memory=True)

        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)

        self.generator_dict = None


        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)


        self.model.train()

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)

        self.total_steps = 0

        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)
        self.foreground_loss = nn.BCELoss()

    def train(self):
        for _ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            data = self.model(data, is_train=True)

            #  Gaussian Render
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            # Multi-view Supervision
            loss = 0.0
            for novel_view in ['novel_view_0','novel_view_1','novel_view_2']:
                render_novel = data[novel_view]['img_pred']
                gt_novel = data[novel_view]['img'].cuda()

                render_fg = data[novel_view]['alpha_pred']
                gt_fg = data[novel_view]['mask'].cuda()
                Ll1 = l1_loss(render_novel, gt_novel)
                Lssim = 1.0 - ssim(render_novel, gt_novel)

                # foreground loss
                Lfg = self.foreground_loss(render_fg,gt_fg)

                loss += 0.8 * Ll1 + 0.2 * Lssim + 0.02*Lfg

            loss = loss/3.0

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)

            if self.total_steps and self.total_steps % 10000 == 0:
                self.save_ckpt(save_path=Path('%s/%s_%s.pth' % (
                cfg.record.ckpt_path, cfg.name, str(self.total_steps))),
                               show_log=False)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()

            self.total_steps += 1

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):

        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        psnr_list = []
        fg_list = []

        for eval_idx in range(5):

            data = self.fetch_data(phase='val')

            with torch.no_grad():

                data = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                psnr_value = 0.0
                fg_value = 0.0
                for novel_view in ['novel_view_0', 'novel_view_1', 'novel_view_2']:
                    render_novel = data[novel_view]['img_pred']
                    gt_novel = data[novel_view]['img'].cuda()


                    render_fg = data[novel_view]['alpha_pred']
                    gt_fg = data[novel_view]['mask'].cuda()

                    tmp_psnr = psnr(render_novel, gt_novel).mean().double()
                    psnr_value += tmp_psnr

                    # foreground loss
                    tmp_fg = self.foreground_loss(render_fg, gt_fg)
                    fg_value += tmp_fg

                psnr_value = psnr_value / 3.0
                fg_value = fg_value / 3.0

                psnr_list.append(psnr_value.item())
                fg_list.append(fg_value.item())

                if eval_idx == 0:
                    nv_idx = 0
                    for novel_view in ['novel_view_0', 'novel_view_1', 'novel_view_2']:

                        subject_name = data['name'][0]
                        gt = data[novel_view]['img']
                        gt = gt[0].detach().permute(1, 2, 0).cpu().numpy()
                        gt = 255 * gt
                        gt = gt.astype(np.uint8)
                        gt_name = '%s/iter_%s_gt_view_%s_%s.jpg' % (cfg.record.show_path, str(self.total_steps).zfill(7),str(nv_idx),subject_name)
                        imageio.imsave(gt_name, gt)

                        pred = data[novel_view]['img_pred']
                        pred = pred[0].detach().permute(1, 2,0).cpu().numpy()
                        pred = 255 * pred
                        pred = pred.astype(np.uint8)
                        pred_name = '%s/iter_%s_pred_view_%s_%s.jpg' % (cfg.record.show_path, str(self.total_steps).zfill(7),str(nv_idx),subject_name)
                        imageio.imsave(pred_name, pred)


                        gt_mask = data[novel_view]['mask']
                        gt_mask = gt_mask[0].detach().permute(1, 2, 0).cpu().numpy()
                        gt_mask = 255 * gt_mask
                        gt_mask = gt_mask.astype(np.uint8)
                        gt_mask_name = '%s/iter_%s_gt_mask_view_%s_%s.jpg' % (
                        cfg.record.show_path, str(self.total_steps).zfill(7),str(nv_idx),subject_name)
                        imageio.imsave(gt_mask_name, gt_mask[...,0])

                        pred_mask = data[novel_view]['alpha_pred']
                        pred_mask = pred_mask[0].detach().permute(1, 2, 0).cpu().numpy()
                        pred_mask = 255 * pred_mask
                        pred_mask = pred_mask.astype(np.uint8)
                        pred_mask_name = '%s/iter_%s_pred_mask_view_%s_%s.jpg' % (
                        cfg.record.show_path, str(self.total_steps).zfill(7),
                        str(nv_idx),subject_name)
                        imageio.imsave(pred_mask_name, pred_mask[...,0])

                        nv_idx = nv_idx + 1


                    input_views = data['input_view']['img'][0].detach().permute(0,2,3,1).cpu().numpy()
                    input_views = 0.5*(input_views + 1)
                    input_views = 255*input_views
                    input_views = input_views.astype(np.uint8)
                    for input_idx in range(input_views.shape[0]):
                        input_name = '%s/iter_%s_input_%d_%s.jpg' % (cfg.record.show_path, str(self.total_steps).zfill(7),input_idx,subject_name)
                        imageio.imsave(input_name, input_views[input_idx])

                    for out_shell_name in ['in_shell','out_shell_1','out_shell_2','out_shell_3','out_shell_4']:
                        out_shell_uvmap = data[out_shell_name]['rgb_maps'][0].detach().permute(1, 2, 0).cpu().numpy()
                        out_shell_uvmap = 0.5 * (out_shell_uvmap + 1)
                        out_shell_uvmap = 255 * out_shell_uvmap
                        out_shell_uvmap = out_shell_uvmap.astype(np.uint8)
                        out_shell_uvmap_name = '%s/iter_%s_%s_uvmap_%s.jpg' % (
                        cfg.record.show_path, str(self.total_steps).zfill(7),out_shell_name,subject_name)
                        imageio.imsave(out_shell_uvmap_name, out_shell_uvmap)

                    inpaint_input = data['in_shell']['inpaint_input'][0].detach().permute(1, 2, 0).cpu().numpy()
                    inpaint_input = 0.5 * (inpaint_input + 1)
                    inpaint_input = 255 * inpaint_input
                    inpaint_input = inpaint_input.astype(np.uint8)
                    inpaint_input_name = '%s/iter_%s_inpaint_input_%s.jpg' % (cfg.record.show_path, str(self.total_steps).zfill(7),subject_name)
                    imageio.imsave(inpaint_input_name, inpaint_input)

                    inpaint_mask = data['in_shell']['inpaint_mask'][0].detach().permute(1, 2, 0).cpu().numpy()
                    inpaint_mask = 255 * inpaint_mask
                    inpaint_mask = inpaint_mask.astype(np.uint8)
                    inpaint_mask_name = '%s/iter_%s_inpaint_mask_%s.jpg' % (
                    cfg.record.show_path, str(self.total_steps).zfill(7),subject_name)
                    imageio.imsave(inpaint_mask_name, inpaint_mask[...,0])


        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        val_fg = np.round(np.mean(np.array(fg_list)), 4)

        logging.info(f"Validation Metrics ({self.total_steps}): psnr {val_psnr}, fg {val_fg}")

        self.logger.write_dict( {'val_psnr': val_psnr, 'val_fg': val_fg},write_step=self.total_steps)
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for key in data.keys():
            if key in ['pos','outer_pos','outer_pos_1','outer_pos_2','outer_pos_3','outer_pos_4']:
                data[key] = data[key].cuda()
            elif key in ['input_view','novel_view_0','novel_view_1','novel_view_2']:
                for sub_key in data[key].keys():
                    data[key][sub_key] = data[key][sub_key].cuda()

        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")


    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/config.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
