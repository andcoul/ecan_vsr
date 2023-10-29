import os
import argparse
import time
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from ecan.components.data_preparation import DataPreparation
from ecan.entity.config_entity import TrainingConfig
from ecan.utils.evaluation import psnr
from ecan.utils.loss import *
from ecan.components.base_model import Net as ECAN
from ecan.config.configuration import ConfigurationManager

# Training parameters
# torch.cuda.set_device(0)
torch.manual_seed(1)


class DataTraining:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self, train_loader):
        # if not torch.cuda.is_available():
        #     raise Exception('No Gpu found, please run with gpu')
        # else:
        #     use_gpu = torch.cuda.is_available()
        # if use_gpu:
        #     cudnn.benchmark = True
        #     torch.cuda.manual_seed_all(1)

        config = ConfigurationManager

        net = ECAN(self.config.params_scale_factor)

        # if torch.cuda.device_count() > 1:
        #     net = nn.DataParallel(net)

        epoch_state = 0
        loss_list = []
        psnr_list = []
        # ssim_list = []
        loss_epoch = []
        psnr_epoch = []
        # ssim_epoch = []

        # Load a cuda pre-trained model and
        # Specify that the model should be loaded on the 'cpu' device, instead of the 'cuda' device.
        ckpt = torch.load(self.config.updated_base_model_path, map_location=torch.device('cpu'))

        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        loss_list = ckpt['loss']
        psnr_list = ckpt['test_psnr']

        optimizer = torch.optim.Adam(
            net.parameters(), lr=4e-4, betas=(0.9, 0.999))
        # criterion_CL = torch.nn.MSELoss().cuda()
        criterion_CL = CharbonnierLoss().cuda()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=1e-7)

        for idx_epoch in range(epoch_state, self.config.params_epochs):
            for idx_iter, (LR, HR) in enumerate(train_loader):
                LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
                # Input shape for the network: [BxCxNxHxW] [B: mini-batch] x [C: channels] x [N depth] x [H height] x [W: width]
                VSR = net(LR)

                loss = criterion_CL(VSR, HR[:, :, 3, :, :])
                loss_epoch.append(loss.detach().cpu())
                psnr_epoch.append(psnr(VSR, HR[:, :, 3, :, :]))
                # ssim_epoch.append(ssim(VSR, HR[:, :, 3, :, :]))
                t0 = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t1 = time.time()
                print("===> Epoch[{}/{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(idx_epoch + 1,
                                                                                            self.config.params_epochs,
                                                                                            idx_iter +
                                                                                            1, len(
                        train_loader),
                                                                                            loss.item(), (t1 - t0)))
            scheduler.step()
            if idx_epoch % 1 == 0:
                loss_list.append(float(np.array(loss_epoch).mean()))
                psnr_list.append(float(np.array(psnr_epoch).mean()))
                # ssim_list.append(float(np.array(ssim_epoch).mean()))
                print(time.ctime()[4:-5] + ' Epoch---%d, loss_epoch---%f, PSNR---%f' % (
                    idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))
                print('MAX TEST PSNR:::::: +======> ', max(psnr_list))
                model = {
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': loss_list,
                    'test_psnr': psnr_list,
                    # 'test_ssim': ssim_list,
                }

                psnr_epoch_mean = float(np.array(psnr_epoch).mean())
                if psnr_epoch_mean == max(psnr_list):
                    self.save_checkpoint(model, save_path=self.config.trained_model_path,
                                         filename='base_model.pth.tar')

                loss_epoch = []
                psnr_epoch = []

                # valid(net, model, scale_factor, idx_epoch)

    def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(state, os.path.join(save_path, filename))

        return None

    def main(self):
        video_list = os.listdir(str(self.config.training_data))
        for i in range(0, len(video_list)):
            video_name = video_list[i]
            self.config.training_data = str(self.config.training_data) + "/" + video_name
            train_set = DataPreparation(config=self.config)
            train_loader = DataLoader(
                dataset=train_set, num_workers=16, batch_size=self.config.params_batch_size, pin_memory=True,
                shuffle=False)
            self.train(train_loader)

    if __name__ == '__main__':
        main()
