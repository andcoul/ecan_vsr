import math
import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, activation='lrelu',
                 bn=True, bias=False, convolution="3d"):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.convolution = convolution
        if self.convolution == '3d':
            self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, bias=bias)
            self.bn = nn.BatchNorm3d(out_channels) if bn else None
        if self.convolution == '2d':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels) if bn else None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.act(x)
        return x


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(
            input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseBlock, self).__init__()
        self.cn0 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             activation=None)
        self.convAct = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 convolution='3d', activation=None)
        self.conv = ConvBlock(in_channels=out_channels, out_channels=1,
                              kernel_size=kernel_size, convolution='3d')
        self.temp = None

    def forward(self, x):
        x1 = self.convAct(x)
        self.temp = torch.add(x, x1)
        x2 = self.convAct(self.temp)
        self.temp = torch.add(self.temp, x2)
        x3 = self.convAct(self.temp)
        self.temp = torch.add(self.temp, x3)
        x4 = self.convAct(self.temp)
        self.temp = torch.add(self.temp, x4)
        out = self.conv(self.temp)
        return out + x


class DenseLayer(nn.Module):
    def __init__(self, nin, nout):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class DAB(nn.Module):
    def __init__(self, nin, growth_rate, nl):
        super(DAB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(nin + growth_rate * i, growth_rate) for i in range(nl)])

        # attention module
        self.CA = CrossAttention()

        # local feature fusion
        self.lff = nn.Conv2d(nin + growth_rate * nl,
                             growth_rate, kernel_size=1)

    def forward(self, x):
        out = self.layers(x)
        out = self.CA(out)
        return x + self.lff(out)  # local residual learning


class RestLayer(nn.Module):
    def __init__(self, nf):
        super(RestLayer, self).__init__()
        self.RCA = RCA(nf)

    def forward(self, x):
        return self.RCA(x)


class RestGroup(nn.Module):
    def __init__(self, nf, nl):
        super(RestGroup, self).__init__()
        self.layers = nn.Sequential(
            *[RestLayer(nf) for i in range(nl)]
        )

    def forward(self, x):
        return x + self.layers(x)  # local residual learning


class RestLayerX(nn.Module):
    def __init__(self, nf):
        super(RestLayerX, self).__init__()
        self.DAB = DAB(nf, nf, 4)

    def forward(self, x):
        return self.DAB(x)


class RestGroupX(nn.Module):
    def __init__(self, nf, nl):
        super(RestGroupX, self).__init__()
        self.layers = nn.Sequential(
            *[RestLayerX(nf) for i in range(nl)]
        )

    def forward(self, x):
        return x + self.layers(x)  # local residual learning


class RCA(nn.Module):
    def __init__(self, nf):
        super(RCA, self).__init__()
        self.cnn0 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.cnn1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.CA = CrossAttention()

    def forward(self, x):
        # return self.CA(self.cnn1(self.relu(self.cnn0(x)))) + x
        return self.cnn1(self.relu(self.cnn0(x))) + x


class ResBlock_d3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_d3d, self).__init__()
        self.dcn0 = DeformConvPack_d(
            nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(
            nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x


class Residual2d(nn.Module):
    def __init__(self, nf):
        super(Residual2d, self).__init__()
        self.con0 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.con1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.con1(self.relu(self.con0(x))) + x


class Residual3d_dpw(nn.Module):
    def __init__(self, nf):
        super(Residual3d_dpw, self).__init__()
        # self.dpw0 = depthwise_separable_conv(nf, nf, kernel_size=3, padding=1)
        # self.dpw1 = depthwise_separable_conv(nf, nf, kernel_size=3, padding=1)
        self.con0 = nn.Conv3d(nf, nf, kernel_size=3, padding=1)
        self.con1 = nn.Conv3d(nf, nf, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.con1(self.relu(self.con0(x))) + x


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv3d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class depthwise_separable_conv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_conv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class AttentionGate(nn.Module):
    """
        ---------++++++++--------
         Attention branch(gates)
          --------++++++++--------
    """

    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        # self.dilatedconv = ConvBlock(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=2, dilation=2,
        #                              activation='sigmoid')
        self.conv = ConvBlock(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, convolution='2d',
                              activation='sigmoid')
        # self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        return x * x_out


class CrossAttention(nn.Module):
    """
        ---------++++++++--------
         Cross-domain Attention
          --------++++++++--------
    """

    def __init__(self, no_spatial=False):
        super(CrossAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class ResidualAttentionBlock(nn.Module):
    """
        ---------++++++++--------
         Residual Cross-domain Attention Block
          --------++++++++--------
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               activation=None)
        self.CA = CrossAttention()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        at = self.CA(x2)
        return at + x


class ResidualAttentionGroup(nn.Module):
    """
        ---------++++++++--------
         Residual Cross-domain Attention Group
          --------++++++++--------
    """

    def __init__(self, in_channels, out_channels, kernel_size, M):
        super(ResidualAttentionGroup, self).__init__()
        self.M = M
        self.RCAB = ResidualAttentionBlock(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv = ConvBlock(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        res = x
        for i in range(self.M - 1):
            x = self.RCAB(x)
        out = self.conv(x) + res
        return out


class ResidualAttentionModule(nn.Module):
    """
        ---------++++++++--------
         Residual Cross-domain Attention Module
          --------++++++++--------
    """

    def __init__(self, in_channels, out_channels, kernel_size, L, M):
        super(ResidualAttentionModule, self).__init__()
        self.L = L
        self.M = M
        self.RCAG = ResidualAttentionGroup(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           M=M)
        self.conv = ConvBlock(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        res = x
        for i in range(self.L - 1):
            x = self.RCAG(x)
        out = self.conv(x) + res
        return out


class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=1, stride=1, padding=0, bias=True,
                 activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor ** 2, kernel_size, stride, padding,
                                    bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class UpsampleBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, bias=True, upsample='deconv', activation='relu',
                 norm='batch'):
        super(UpsampleBlock, self).__init__()
        scale_factor = scale_factor
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out


def pixel_unshuffle(input, upscale_factor):
    batch_size, output, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class Upsampling(nn.Module):
    def __init__(self, scale_factor):
        super(Upsampling, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.scale_factor)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1,
                                     1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(2))
            if bn:
                modules.append(torch.nn.BatchNorm2d(n_feat))
            # modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)

        self.activation = act
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out
