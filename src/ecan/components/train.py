import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import *
from evaluation import psnr
from loss import *
from src.ecan.model.model import Net as ECAN

# Training parameters
parser = argparse.ArgumentParser(description='ECAN Super-Resolution Settings')
parser.add_argument("--save", default='../logs/Settings/nFr7_batch64_K20L5M10_ecan_lab_cnn3_no_att', type=str,
                    help="Save path")
parser.add_argument("--resume", default="", type=str,
                    help="Resume path (default: none)")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--train_dataset_dir",
                    default='../data/vimeo', type=str, help="train_dataset")
parser.add_argument("--inType", type=str, default='y',
                    help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=64,
                    help="Training batch size")
parser.add_argument("--nFrames", type=int, default=7,
                    help="Number of input frame to train for")
parser.add_argument("--nEpochs", type=int, default=200,
                    help="Number of epochs to train for")
parser.add_argument("--gpu", default=0, type=int, help="gpu ids (default: 0)")
parser.add_argument('--seed', type=int, default=1,
                    help='random seed to use. Default=0')
parser.add_argument("--lr", type=float, default=4e-4,
                    help="Learning Rate. Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--t_max', type=int, default=200,
                    help='Maximum number of iterations in CosineAnnealingLR, default=50')
parser.add_argument("--step", type=int, default=6,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=16,
                    help="Number of threads for data loader to use, Default: 1")

opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
VALID_PSNR_LIST = []
VALID_SSIM_LIST = []


def train(train_loader, scale_factor, epoch_num):
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)

    net = ECAN(scale_factor).cuda()

    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)

    epoch_state = 0
    loss_list = []
    psnr_list = []
    # ssim_list = []
    loss_epoch = []
    psnr_epoch = []
    # ssim_epoch = []
    global VALID_PSNR_LIST

    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        loss_list = ckpt['loss']
        psnr_list = ckpt['test_psnr']
        VALID_PSNR_LIST = ckpt['valid_psnr']

    optimizer = torch.optim.Adam(
        net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    # criterion_CL = torch.nn.MSELoss().cuda()
    criterion_CL = CharbonnierLoss().cuda()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.t_max, eta_min=1e-7)

    for idx_epoch in range(epoch_state, epoch_num):
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
            print("===> Epoch[{}/{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(idx_epoch + 1, epoch_num,
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
                save_checkpoint(model, save_path=opt.save,
                                filename='ecan.pth.tar')

            loss_epoch = []
            psnr_epoch = []
            ssim_epoch = []

            valid(net, model, scale_factor, idx_epoch)

    print(f"=> BEST TEST PSNR :::::: {max(psnr_list)}")
    print(f"=> BEST TRAINING LOSS :::::: {min(loss_list)}")
    print(f"=> BEST VALID PSNR :::::: {max(VALID_PSNR_LIST)}")


def valid(net, model, scale_factor, idx_epoch):
    global VALID_PSNR_LIST
    global VALID_SSIM_LIST
    valid_set = ValidSetLoader(
        opt.train_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    valid_loader = DataLoader(
        dataset=valid_set, num_workers=opt.threads, batch_size=8, shuffle=True)
    psnr_list = []
    ssim_list = []
    for idx_iter, (LR, HR) in enumerate(valid_loader):
        LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
        SR = net(LR)
        psnr_list.append(psnr(SR.detach(), HR[:, :, 3, :, :].detach()))
        # ssim_list.append(ssim(SR.detach(), HR[:, :, 3, :, :].detach()))
    print('valid PSNR---%f' % (float(np.array(psnr_list).mean())))

    VALID_PSNR_LIST.append(float(np.array(psnr_list).mean()))
    # VALID_SSIM_LIST.append(float(np.array(ssim_list).mean()))
    model['valid_psnr'] = VALID_PSNR_LIST
    print('MAX VALID PSNR:::::: +======> ', max(VALID_PSNR_LIST))

    save_checkpoint(model, save_path=opt.save,
                    filename='model' + str(scale_factor) + '_epoch' + str(idx_epoch + 1) + '.pth.tar')


def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path, filename))

    return None


def main():
    train_set = TrainSetLoader(
        opt.train_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    train_loader = DataLoader(
        dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True, shuffle=True)
    train(train_loader, opt.scale_factor, opt.nEpochs)


if __name__ == '__main__':
    main()
