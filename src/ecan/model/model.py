import torch
import torch.nn as nn
from networks import *
import torch.nn.functional as F
import functools
import time


class Net(nn.Module):
    def __init__(self, scale_factor, base_filter=1, K=20, L=5, M=10, nin=64, nf=64, kernel_size=3):
        super(Net, self).__init__()
        self.kernel_size = kernel_size
        self.upscale_factor = scale_factor
        # ------------------------------ Features Extraction ------------------------------- #
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=base_filter, out_channels=nf,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.residual3dpw = self.make_layer(
            functools.partial(Residual3d_dpw, nf), K)
        self.TA = nn.Conv2d(7 * nin, nf, kernel_size=1,
                            stride=1, padding=0, bias=True)
        # ------------------------------- SR Reconstruction --------------------------------- #
        # Dense Cross-Attention Module
        # self.RCA = self.make_layer(functools.partial(RCA, nf), L)
        self.DAB = self.make_layer(functools.partial(RestGroup, nf, 12), L)
        self.residual2d = self.make_layer(
            functools.partial(Residual2d, nf), M)
        # -------------------------------------- SR ----------------------------------------- #
        self.upscale = nn.Sequential(
            nn.Conv2d(nf, nf * scale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(nf, base_filter, 3, 1, 1, bias=False),
            nn.Conv2d(base_filter, base_filter, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        # [BxCxNxHxW] [B: mini-batch] x [C: channels] x [N depth] x [H height] x [W: width]
        b, c, n, h, w = x.size()
        residual = F.interpolate(x[:, :, n // 2, :, :], scale_factor=self.upscale_factor, mode='bilinear',
                                 align_corners=False)
        # ------------------------------ Features Extraction ------------------------------- #
        out = self.conv1(x)
        out = self.residual3dpw(out)
        # ------------------------------ Efficient Temporal Alignment ------------------------------- #
        ta = self.TA(out.permute(0, 2, 1, 3, 4).contiguous().view(b, -1, h, w))
        out = self.DAB(ta)
        # ------------------------------- SR Reconstruction --------------------------------- #
        out = self.residual2d(out)
        # out = self.residual2d(out + ta)
        out = self.upscale(out)
        return torch.add(out, residual)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = Net(4).cuda()
    from thop import profile
    from thop import clever_format
    # from torchsummary import summary

    input = torch.randn(1, 1, 7, 32, 32).cuda()
    flops, params = profile(net, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('   Params(M) ::: ', params)
    print('   FLOPs(T) ::: ', flops)

    # total_params = sum(p.numel() for p in net.parameters())
    # print(f"[INFO]: {total_params:,} total parameters.")
    # total_trainable_params = sum(
    #     p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"[INFO]: {total_trainable_params:,} trainable parameters.")
    # summary(net, (1, 7, 32, 32))
    import torch

    print(torch.__version__)
