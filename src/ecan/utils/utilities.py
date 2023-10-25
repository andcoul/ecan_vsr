import cv2
import math
import os
import pickle
import subprocess
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from math import log10

matplotlib.use('Agg')


######################################################################################
#################################  Model utility  ####################################
######################################################################################

def save_model(model, optimizer, opts):
    # save opts
    opts_filename = os.path.join(opts.model_dir, "opts.pth")
    print("Save %s" % opts_filename)
    with open(opts_filename, 'wb') as f:
        pickle.dump(opts, f)
    # serialize model and optimizer to dict
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    model_filename = os.path.join(
        opts.model_dir, "model_epoch_%d.pth" % model.epoch)
    print("Save %s" % model_filename)
    torch.save(state_dict, model_filename)


def load_model(model, optimizer, opts, epoch):
    # load model
    model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" % epoch)
    print("Load %s" % model_filename)
    state_dict = torch.load(model_filename)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    # move optimizer state to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    model.epoch = epoch  # reset model epoch
    return model, optimizer


class SubsetSequentialSampler(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def create_data_loader(data_set, opts, mode):
    # generate random index
    if mode == 'train':
        total_samples = opts.train_epoch_size * opts.batch_size
    else:
        total_samples = opts.valid_epoch_size * opts.batch_size
    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))
    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]
    # generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    print('data_set', len(data_set), 'num_epochs', num_epochs)
    data_loader = DataLoader(dataset=data_set, num_workers=8, batch_size=opts.batch_size,
                             pin_memory=True)  # opts.threads:8
    return data_loader


def learning_rate_decay(opts, epoch):
    # 1 ~ offset              : lr_init
    # offset ~ offset + step       : lr_init * drop^1
    # offset + step ~ offset + step * 2   : lr_init * drop^2
    # ...
    if opts.lr_drop == 0:  # constant learning rate
        decay = 0
    else:
        assert (opts.lr_step > 0)
        decay = math.floor(float(epoch) / opts.lr_step)
        decay = max(decay, 0)  # decay = 1 for the first lr_offset iterations
    lr = opts.lr_init * math.pow(opts.lr_drop, decay)
    lr = max(lr, opts.lr_init * opts.lr_min)
    return lr


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])
    return N


######################################################################################
#################################  Loss utility  ####################################
######################################################################################

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                'GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


######################################################################################
#################################  Evaluation utility  ###############################
######################################################################################

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def calculate_psnr(prediction, target):
    # prediction and target have range [0, 255]
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def psnr(img1, img2):
    img1 = (img1 * 255.0).int()
    img2 = (img2 * 255.0).int()
    img1 = img1.float() / 255.0
    img2 = img2.float() / 255.0
    img1_ = img1[:, :, 8: -8, 8: -8]
    img2_ = img2[:, :, 8: -8, 8: -8]
    mse = torch.sum((img1_ - img2_) ** 2) / img1_.numel()
    psnr = 10 * log10(1 / mse)
    return psnr

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=False):
    _, channel, h, w = img1.size()

    img1_ = img1[:, :, 8: -8, 8: -8]
    img2_ = img2[:, :, 8: -8, 8: -8]
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1_.get_device())
    window = window.type_as(img1_)
    return _ssim(img1_, img2_, window, window_size, channel, size_average)


######################################################################################
#################################  Image utility  ####################################
######################################################################################

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def img2tensor(img):
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))
    return img_t


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def rotate_image(img, degree, interp=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, degree, 1.)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    img_out = cv2.warpAffine(
        img, rotation_mat, (bound_w, bound_h), flags=interp + cv2.WARP_FILL_OUTLIERS)
    return img_out


def numpy_to_PIL(img_np):
    # input image is numpy array in [0, 1]
    # convert to PIL image in [0, 255]
    img_PIL = np.uint8(img_np * 255)
    img_PIL = Image.fromarray(img_PIL)
    return img_PIL


def PIL_to_numpy(img_PIL):
    img_np = np.asarray(img_PIL)
    img_np = np.float32(img_np) / 255.0
    return img_np


def read_img(filename, grayscale=0):
    # read image and convert to RGB in [0, 1]
    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Image %s does not exist" % filename)
        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)
        if img is None:
            raise Exception("Image %s does not exist" % filename)
        img = img[:, :, ::-1]  # BGR to RGB
    img = np.float32(img) / 255.0
    return img


######################################################################################
#################################  Other utility  ####################################
######################################################################################

def save_vector_to_txt(matrix, filename):
    with open(filename, 'w') as f:
        print("Save %s" % filename)
        for i in range(matrix.size):
            line = "%f" % matrix[i]
            f.write("%s\n" % line)


def run_cmd(cmd):
    print(cmd)
    subprocess.call(cmd, shell=True)


def make_video(input_dir, img_fmt, video_filename, fps=24):
    cmd = "ffmpeg -y -loglevel error -framerate %s -i %s/%s -vcodec libx264 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" %s" \
          % (fps, input_dir, img_fmt, video_filename)
    run_cmd(cmd)
