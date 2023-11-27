import numpy as np
import torch
import torch.nn.functional as F
from itertools import chain
from torchvision.utils import save_image
from skimage.io import imread
from torch.utils import data
import time
from utils import load_yml2args, text_color
from torch import autograd, optim
from model import *
from model.datasets import *
torch.manual_seed(4321)
np.random.seed(4321)

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

COARSE_SIZE = (256, 448)
txt_color = text_color()

import argparse

def init_loggger(cfgs):
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
    logger.add(cfgs.Syn_cfgs.log_path+"log.txt", colorize=False, format=log_format)

def compute_loss(frame_syn, frame_gt, epsilon):
    frame_gt_flattened = frame_gt.flatten()
    frame_syn_flattened = frame_syn.flatten()
    charbonnierLoss = 0.
    censusLoss = 0.

    charbonnierLoss = torch.mean(torch.sqrt(torch.pow(frame_gt_flattened-frame_syn_flattened,2) + epsilon*epsilon))

    return charbonnierLoss + censusLoss
def train(cfgs, dataloaders, BiFormer, Upsampler, SynNet, syn_optim, syn_scheduler):
    flow_bw_dict, flow_fw_dict = {}, {}
    for param in SynNet.parameters():
        param.requires_grad = True

    # pre-compute flow and pre-store
    for data in tqdm(dataloaders['train'], desc="Precompute flow"):
        # img1: ndarray:{h,w,c}
        # img2: ndarray:{h,w,c}
        f1 = np.asarray(data['frame1'])[:, :, 0:3]
        f3 = np.asarray(data['frame3'])[:, :, 0:3]
        f2 = np.asarray(data['frame2'])
        idx = data['index']

        frame1 = torch.tensor(f1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        frame3 = torch.tensor(f3).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        frame2 = torch.tensor(f2).unsqueeze(2).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # frame1 = torch.from_numpy(f1[:, :, 0:3]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        # frame3 = torch.from_numpy(f3[:, :, 0:3]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        #
        # frame2 = torch.from_numpy(f2[:, :, 0:3]).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        frame1 = frame1.cuda()
        frame3 = frame3.cuda()
        # frame2 = frame2.cuda()

        _, _, H_ori, W_ori = frame1.shape
        assert (H_ori >= 256 * 8) and (W_ori >= 448 * 8), 'Only 4K resolution available'
        img1_prev = F.interpolate(frame1, COARSE_SIZE, mode='bilinear')
        img3_prev = F.interpolate(frame3, COARSE_SIZE, mode='bilinear')

        flow_fw = BiFormer(img1_prev, img3_prev)

        for iter in reversed(range(1, 3)):
            H_ = H_ori // (2 ** iter)
            W_ = W_ori // (2 ** iter)
            img1_prev = F.interpolate(frame1, (H_, W_), mode='bilinear')
            img3_prev = F.interpolate(frame3, (H_, W_), mode='bilinear')

            _, _, H_c, W_c = flow_fw.shape
            flow_fw = F.interpolate(flow_fw, (H_, W_), mode='bilinear')
            flow_fw[:, 0, :, :] *= W_ / float(W_c)
            flow_fw[:, 1, :, :] *= H_ / float(H_c)

            flow_fw = Upsampler(img1_prev, img3_prev, flow_fw)

        _, _, H_c, W_c = flow_fw.shape
        flow_fw = F.interpolate(flow_fw, (H_ori, W_ori), mode='bilinear')
        flow_fw[:, 0, :, :] *= W_ori / float(W_c)
        flow_fw[:, 1, :, :] *= H_ori / float(H_c)

        # Based on linear motion assumption
        flow_bw = flow_fw * (-1)

        flow_bw_dict[idx] = flow_bw.to('cpu')
        flow_fw_dict[idx] = flow_fw.to('cpu')

    # training start
    for epoch in tqdm(range(cfgs.Syn_cfgs.total_epoch), desc="Optimizing synNet"):
        total_loss = 0
        valid_count = 0
        for index, data in enumerate(dataloaders['train']):
            SynNet.zero_grad()
            # img1: ndarray:{h,w,c}
            # img2: ndarray:{h,w,c}
            f1 = np.asarray(data['frame1'])[:, :, 0:3]
            f3 = np.asarray(data['frame3'])[:, :, 0:3]
            f2 = np.asarray(data['frame2'])
            idx = data['index']

            frame1 = torch.tensor(f1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            frame3 = torch.tensor(f3).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            frame2 = torch.tensor(f2).unsqueeze(2).permute(2, 0, 1).float().unsqueeze(0) / 255.0



            # frame1 = torch.from_numpy(f1[:, :, 0:3]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            # frame3 = torch.from_numpy(f3[:, :, 0:3]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            #
            # frame2 = torch.from_numpy(f2[:, :, 0:3]).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            frame1 = frame1.cuda()
            frame3 = frame3.cuda()
            frame2 = frame2.cuda()

            # TODO: flow_bw, flow_fw
            flow_bw, flow_fw = flow_bw_dict[idx].cuda(), flow_fw_dict[idx].cuda()

            output = SynNet(frame1, frame3, flow_bw, flow_fw)
            ma = torch.max(output.clone())
            mi = torch.min(output.clone())
            output_mask = (output.clone() - mi) / (ma - mi)
            save_image(output_mask, args.output + 'epoch_' +str(epoch)+'_index_'+ str(index) + '.png', value_range=(0, 1))
            loss = compute_loss(output, frame2, 1e-6)
            loss.requires_grad_(True)
            total_loss += loss.item()
            loss.backward(retain_graph=True)
            valid_count += 1
            if torch.isnan(loss):
                valid_count -= 1
                continue
            syn_optim.step()

        cur_avg_loss = total_loss / valid_count
        syn_scheduler.step()
        logger.info(f"Epoch {epoch} - avg loss: {cur_avg_loss}")
        # save_image(output.clone(), args.output+'_'+str(epoch), value_range=(0, 1))

        if epoch % cfgs.Syn_cfgs.model_save_fre == 0:
            synNet_name = "synNet_" + str(epoch) +\
                            "_charbonnierLoss_" + str(cur_avg_loss) + \
                          "_time_" + str(np.round(np.mean(np.array(time.time())), 6)) + ".pth"

            torch.save(SynNet.state_dict(), cfgs.Syn_cfgs.model_save_pth + synNet_name)

        ## This is not the correct PSNR computation due to datatype ##
        if args.target is not None:
            from math import log10

            target = torch.from_numpy(imread(args.target)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            target = target.cuda()

            output = output.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
            output = output.float() / 255.0

            mse = F.mse_loss(output, target)
            psnr = 10 * log10(1 / mse.item())
            print(f'PSNR: {psnr:.06f}dB')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--target', type=str, required=False)
    parser.add_argument('--configs', type=str, default='configs/BiFormer_paper.yaml')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/BiFormer_weights.pth')
    args = parser.parse_args()

    cfgs = load_yml2args(args.configs)

    init_loggger(cfgs)

    BiFormer = BiFormer(**cfgs.transformer_cfgs)
    Upsampler = Upsampler(**cfgs.Upsampler_cfgs)
    SynNet = SynNet()

    print("load model BiFormer and UpSampler")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    BiFormer.load_state_dict(checkpoint['BiFormer_state_dict'], strict=True)
    Upsampler.load_state_dict(checkpoint['Upsampler_state_dict'], strict=True)
    # SynNet.load_state_dict(checkpoint['SynNet_state_dict'], strict=True)

    BiFormer = BiFormer.cuda()
    Upsampler = Upsampler.cuda()
    SynNet = SynNet.cuda()

    for param in chain(BiFormer.parameters(), Upsampler.parameters()):
        param.requires_grad = False
    # for param in SynNet.parameters():
    #     param.requires_grad = True
    # SynNet.parameters()已经初始化过了，这是我们需要优化的参数
    SynNet_param = [_ for _ in SynNet.parameters()]
    syn_optim = optim.Adam(
        SynNet_param,
        lr=cfgs.Syn_cfgs.lr,
        betas=(0.9, 0.99), eps=1e-15
    )
    syn_scheduler = optim.lr_scheduler.StepLR(
        syn_optim,
        step_size=cfgs.Syn_cfgs.scheduler_step_size,
        gamma=cfgs.Syn_cfgs.scheduler_gamma
    )
    dataset = MultiResolutionDataset(cfgs.Syn_cfgs.scene_pth, device='cuda:0')
    init_train_dataloader = dataset.dataloader
    dataloaders = {'train': init_train_dataloader}
    train(cfgs, dataloaders, BiFormer, Upsampler, SynNet, syn_optim, syn_scheduler)



