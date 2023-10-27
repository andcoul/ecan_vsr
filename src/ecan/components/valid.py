from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from ecan.components.model import Net as RCAN
from evaluation import psnr2, ssim
import numpy as np
import argparse
import os
from ecan.utils import common
from ecan.constants import *
from ecan import logger

parser = argparse.ArgumentParser(description="PyTorch Ecan")
parser.add_argument("--dataset_dir", default='../data',
                    type=str, help="test_dataset dir")
parser.add_argument("--test_results_dir",
                    default='../results/Exp19_nFr7_batch64_k20_l5Res12_m10_ecan_lab_dpw3_no_att', type=str, help="test_results dir")
parser.add_argument(
    "--model", default='../logs/ecan/Exp19_nFr7_batch64_k20_l5Res12_m10_ecan_lab_dpw3_no_att/ecan.pth.tar', type=str,
    help="checkpoint")
parser.add_argument("--gpu", type=int, default=0, help="Test gpu")
parser.add_argument("--datasets", type=str,
                    default=['vid4', 'SPMC-11'], help="Test batch size")

global opt, model
opt = parser.parse_args()
hyp = common.read_yaml(HYP_FILE_PATH)
torch.cuda.set_device(opt.gpu)


def demo_test(net, test_loader, scale_factor, dataset_name, video_name):
    PSNR_list = []
    SSIM_list = []
    with torch.no_grad():
        for idx_iter, (LR, HR, SR_bicubic) in enumerate(test_loader):
            LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
            SR = net(LR)
            SR = torch.clamp(SR, 0, 1)

            PSNR_list.append(psnr2(SR, HR[:, :, 3, :, :]))
            SSIM_list.append(ssim(SR, HR[:, :, 3, :, :]))

            if not os.path.exists(opt.test_results_dir + '/' + dataset + '/' + video_name):
                os.makedirs(opt.test_results_dir + '/' +
                            dataset + '/' + video_name)

            SR_bicubic[:, 0, :, :] = SR[:, 0, :, :].cpu()
            SR_rgb = (ycbcr2rgb(SR_bicubic[0, :, :, :].permute(2, 1, 0))).permute(
                2, 1, 0)
            SR_rgb = torch.clamp(SR_rgb, 0, 1)
            SR_img = transforms.ToPILImage()(SR_rgb)
            SR_img.save(
                opt.test_results_dir + '/' + dataset_name + '/' + video_name + '/sr_rgb_' + str(idx_iter + 1).rjust(2,
                                                                                                                    '0') + '.png')

        PSNR_mean = float(torch.cat(PSNR_list, 0)[2:-2].data.cpu().mean())
        SSIM_mean = float(torch.cat(SSIM_list, 0)[2:-2].data.cpu().mean())
        logger.info(video_name + ' psnr: ' + str(PSNR_mean) +
              ' ssim: ' + str(SSIM_mean))
        return PSNR_mean, SSIM_mean


def main(dataset_name):
    net = RCAN(opt.scale_factor).cuda()
    model = torch.load(MODEL_FILE_PATH)
    net.load_state_dict(model['state_dict'])

    PSNR_dataset = []
    SSIM_dataset = []

    if dataset_name == 'vid4' or dataset_name == 'SPMC-11':
        video_list = os.listdir(opt.test_dataset_dir + '/' + dataset_name)
        for i in range(0, len(video_list)):
            video_name = video_list[i]
            test_set = TestSetLoader(opt.test_dataset_dir + '/' + dataset_name + '/' + video_name,
                                     scale_factor=opt.scale_factor)
            test_loader = DataLoader(
                dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
            psnr, ssim = demo_test(
                net, test_loader, opt.scale_factor, dataset_name, video_name)
            PSNR_dataset.append(psnr)
            SSIM_dataset.append(ssim)
        logger.info(dataset_name + ' psnr: ' + str(float(np.array(PSNR_dataset).mean())) + '  ssim: ' + str(
            float(np.array(SSIM_dataset).mean())))

if __name__ == '__main__':
    for i in range(len(opt.datasets)):
        dataset = opt.datasets[i]
        if not os.path.exists(opt.test_results_dir + '/' + dataset):
            os.makedirs(opt.test_results_dir + '/' + dataset)
        import time

        start = time.time()
        main(dataset)
        end = time.time()
        print(end - start)
