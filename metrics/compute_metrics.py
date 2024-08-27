###########################################
# imports
###########################################

import sys, glob
import imageio.v2 as imageio
import skimage.metrics
import numpy as np
import torch
import cv2
import os
from lpips import LPIPS
import shutil
import pdb


# USAGE: python metrics/compute_metrics.py
###########################################

tmp_ours = './outputs/metrics/eval_img_dilate_11_inpaint_3_black_input_2048/pred'
tmp_gt = './outputs/metrics/eval_img_dilate_11_inpaint_3_black_input_2048/gt'
tmperr = './outputs/metrics/eval_img_dilate_11_inpaint_3_black_input_2048/error'

def mae(imageA, imageB):
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    return err


###########################################

def mse(imageA, imageB):
    errImage = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2, 2)
    errImage = np.sqrt(errImage)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    return err, errImage


###########################################

def func(g_path, t_path, mask):


    H = 1000
    W = 500

    psnrs, ssims, mses, maes = [], [], [], []


    humans = set()
    for file_name in os.listdir(g_path):
        file_name_split = file_name[:-4].split('_')
        human_name = file_name_split[0]
        humans.add(human_name)

    humans = list(humans)
    humans.sort()


    for human_idx in range(len(humans)):

        human = humans[human_idx]

        for angle in [3, 8, 13]:

            sample_name = '{}_{}_novel00'.format(human, str(angle).zfill(3))
            print(sample_name)

            ours_path = os.path.join(g_path, '{}_{}_novel00.jpg'.format(human,str(angle).zfill(3)))
            g = imageio.imread(ours_path).astype('float32') / 255.

            gt_mask = os.path.join(mask, '{}_{}/0.png'.format(human,str(angle).zfill(3)))
            gt_mask = imageio.imread(gt_mask).astype('float32') / 255.

            gt_path = os.path.join(t_path, '{}_{}/0.jpg'.format(human,str(angle).zfill(3)))

            t = imageio.imread(gt_path).astype('float32') / 255.


            if (g.shape[0] != 1024) or (g.shape[1] != 1024):
                g = cv2.resize(g, (t.shape[1], t.shape[0]))

            h0, w0 = h, w = g.shape[0], g.shape[1]

            # ---------------

            ii, jj = np.where(
                ~(t == 0).all(-1))  # all background pixel coordinates

            try:
                # bounds for V direction
                hmin, hmax = np.min(ii), np.max(ii)
                uu = (H - (hmax + 1 - hmin)) // 2
                vv = H - (hmax - hmin) - uu
                if hmin - uu < 0:
                    hmin, hmax = 0, H
                elif hmax + vv > h:
                    hmin, hmax = h - H, h
                else:
                    hmin, hmax = hmin - uu, hmax + vv

                # bounds for U direction
                wmin, wmax = np.min(jj), np.max(jj)
                uu = (W - (wmax + 1 - wmin)) // 2
                vv = W - (wmax - wmin) - uu
                if wmin - uu < 0:
                    wmin, wmax = 0, W
                elif wmax + vv > w:
                    wmin, wmax = w - W, w
                else:
                    wmin, wmax = wmin - uu, wmax + vv

            except ValueError:
                print(f"target is empty")
                continue

            # crop images
            g = g[hmin: hmax, wmin: wmax]
            t = t[hmin: hmax, wmin: wmax]


            h, w = g.shape[0], g.shape[1]

            assert (h == H) and (w == W), f"error {hmin} {hmax} {wmin} {wmax} {h0} {w0} {uu} {vv}"

            mseValue, errImg = mse(g, t)

            errImg = (errImg * 255.0).astype(np.uint8)
            errImg = cv2.applyColorMap(errImg, cv2.COLORMAP_JET)

            subject_angle_name = '{}_{}_novel00.png'.format(human,
                                                            str(angle).zfill(3))
            cv2.imwrite(os.path.join(tmperr, subject_angle_name), errImg)


            mseValue_ours_gt, errImg_ours_gt = mse(g, t)
            maeValue = mae(g, t)
            psnr = 10 * np.log10((1 ** 2) / mseValue_ours_gt)

            imageio.imsave("{}/{}_source.png".format(tmp_ours, sample_name),
                           (g * 255).astype('uint8'))  # ours

            imageio.imsave("{}/{}_target.png".format(tmp_gt, sample_name),
                           (t * 255).astype('uint8'))  # gt


            psnrs += [psnr]
            ssims += [skimage.metrics.structural_similarity(g, t, channel_axis=2,data_range=1)]
            # maes += [maeValue]
            mses += [mseValue]



    return np.asarray(psnrs), np.asarray(ssims), np.asarray(mses), np.asarray(maes)


###########################################


def evaluateErr(ours, target, mask):

    psnrs, ssims, mses, maes = func(g_path=ours, t_path=target, mask=mask)

    ###########################################
    # PSNR & SSIM
    psnr = psnrs.mean()
    print(f"PSNR mean {psnr}", flush=True)
    ssim = ssims.mean()
    print(f"SSIM mean {ssim}", flush=True)

    ###########################################
    # LPIPS

    lpips = LPIPS(net='alex', version='0.1')
    if torch.cuda.is_available():
        lpips = lpips.cuda()

    g_files = sorted(glob.glob(tmp_ours + '/*_source.png'))
    t_files = sorted(glob.glob(tmp_gt + '/*_target.png'))

    lpipses = []
    for i in range(len(g_files)):

        g = imageio.imread(g_files[i]).astype('float32') / 255.
        t = imageio.imread(t_files[i]).astype('float32') / 255.
        g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        if torch.cuda.is_available():
            g = g.cuda()
            t = t.cuda()
        lpipses += [lpips(g, t).item()]
    lpips = np.mean(lpipses)
    print(f"LPIPS Alex Mean {lpips}", flush=True)


    ###########

    lpips = LPIPS(net='vgg', version='0.1')
    if torch.cuda.is_available():
        lpips = lpips.cuda()

    g_files = sorted(glob.glob(tmp_ours + '/*_source.png'))
    t_files = sorted(glob.glob(tmp_gt + '/*_target.png'))

    lpipses = []
    for i in range(len(g_files)):
        g = imageio.imread(g_files[i]).astype('float32') / 255.
        t = imageio.imread(t_files[i]).astype('float32') / 255.
        g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        if torch.cuda.is_available():
            g = g.cuda()
            t = t.cuda()
        lpipses += [lpips(g, t).item()]
    lpips = np.mean(lpipses)
    print(f"LPIPS VGG mean {lpips}", flush=True)


    ###########################################
    # FID

    os.system('python -m pytorch_fid --device cuda {} {}'.format(tmp_ours, tmp_gt))


######################################################################################
# parameters
######################################################################################


# path to GT rgb
target = './datasets/THuman/val/img/'
# path to GT mask
mask = './datasets/THuman/val/mask/'
# path to prediction
ours = './outputs/eval/GHG/'


######################################################################################


print('###############################################', flush=True)

if not os.path.exists(tmperr):
    os.makedirs(tmperr, exist_ok=True)

if not os.path.exists(tmp_ours):
    os.makedirs(tmp_ours, exist_ok=True)

if not os.path.exists(tmp_gt):
    os.makedirs(tmp_gt, exist_ok=True)

evaluateErr(ours, target, mask)

print(ours)
######################################################################################
