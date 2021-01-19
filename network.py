import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

from torch.nn import functional as F
from torch.nn import Parameter
# from utils import *

import pdb

## for contextual attention
def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    batch_size, channel, height, width = images.size()
    images = same_padding(images, ksizes, strides, rates)
    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)

    # (Pdb) images.size()
    # torch.Size([1, 192, 128, 170])
    # (Pdb) patches.size()
    # torch.Size([1, 3072, 5440])

    patches = patches.view(batch_size, channel, ksizes[0], ksizes[1], -1)
    patches = patches.permute(0, 4, 1, 2, 3)    # shape: [B, L, C, k, k]
    # torch.Size([1, 5440, 192, 4, 4])

    return patches  # [B, L, C, k, k], L is the total number of such blocks

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4

    # pdb.set_trace()
    # ksizes = [4, 4]
    # strides = [2, 2]
    # rates = [1, 1]
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = padding_rows // 2
    padding_left = padding_cols // 2
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left

    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def reduce_mean(x, axis=None, keepdim=True):
    # axis = [1, 2, 3]
    # keepdim = True
    # torch.Size([5440, 1, 3, 3])
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    # pdb.set_trace()
    # torch.Size([5440, 1, 1, 1])

    return x

def reduce_sum(x, axis=None, keepdim=True):
    # pdb.set_trace()
    # axis = [1, 2, 3]
    # keepdim = True
    # (Pdb) x.size()
    # torch.Size([5440, 192, 3, 3])

    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)

    # x.size()
    # torch.Size([5440, 1, 1, 1])

    return x

#-----------------------------------------------
#                Normal ConvBlock
#-----------------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, activation = 'elu', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme

        self.pad = nn.ZeroPad2d(padding)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
        pdb.set_trace()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.activation:
            x = self.activation(x)
        return x

#-----------------------------------------------
#                Gated ConvBlock
#-----------------------------------------------
class GatedConv2d(nn.Module):
    # latent_channels*8, latent_channels*4, 3, 1, 1, activation = 'elu'

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, activation = 'elu', sn = False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        self.pad = nn.ZeroPad2d(padding)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        # (Pdb) x.size()
        # torch.Size([1, 4, 516, 684])
        conv = self.conv2d(x)
        if self.activation:
            conv = self.activation(conv)

        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)

        x = conv * gated_mask
        # (Pdb) x.size()
        # torch.Size([1, 48, 512, 680])

        # (Pdb) conv.size()
        # torch.Size([1, 48, 512, 680])
        # (Pdb) mask.size()
        # torch.Size([1, 48, 512, 680])

        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, activation = 'lrelu', sn = True, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, activation, sn)

    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor)
        x = self.gated_conv2d(x)
        return x

#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
        # !!!!!!!!!!!!!!!!!!
        # name === 'weight'
        # power_iterations === 1
        # !!!!!!!!!!!!!!!!!!

        # self = SpectralNorm(
        #   (module): Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1))
        # )
        # module = Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1))

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        # Performs a matrix-vector product of the matrix input and the vector vec.
        # (Pdb) pp u.data.size(), v.data.size(), w.data.size()
        # (torch.Size([96]), torch.Size([1728]), torch.Size([96, 192, 3, 3]))

        # xxxx8888
        for _ in range(self.power_iterations):
            # v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            # u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
            v.data = l2normalize(torch.mm(torch.t(w.view(height,-1).data), u.data.view(-1, 1)).squeeze(1))
            u.data = l2normalize(torch.mm(w.view(height,-1).data, v.data.view(-1, 1)).squeeze(1))

        # xxxx8888
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.data * (w.view(height, -1).data).mm(v.data.view(-1, 1)).squeeze(1)
        sigma = sigma.sum()

        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        # (Pdb) pp type(w)
        # <class 'torch.nn.parameter.Parameter'>
        # (Pdb) pp w.size()
        # torch.Size([96, 192, 3, 3])

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        # (Pdb) pp height, width
        # (96, 1728 = 192x3x3)

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        # (Pdb) pp u.data.size(), v.data.size(), w.data.size()
        # (torch.Size([96]), torch.Size([1728]), torch.Size([96, 192, 3, 3]))

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=True, use_cuda=True):
        # ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda

        # ksize = 3
        # stride = 1
        # rate = 2
        # fuse_k = 3
        # softmax_scale = 10
        # fuse = True
        # use_cuda = True

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
            Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w

        # extract patches from background with stride and rate
        # raw_w is extracted for reconstruction, b -- backgroud

        # ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True
        raw_b_patches = extract_image_patches(b, ksizes=[2 * self.rate, 2 * self.rate],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        raw_b_groups = torch.split(raw_b_patches, 1, dim=0)
        # (Pdb) len(raw_b_groups) -- 1
        # (Pdb) raw_b_groups[0].size()
        # torch.Size([1, 5440, 192, 4, 4])

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        # self.rate == 2
        f = F.interpolate(f, scale_factor=1./self.rate)
        b = F.interpolate(b, scale_factor=1./self.rate)
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # pdb.set_trace()
        # (Pdb) len(f_groups), f_groups[0].size()
        # (1, torch.Size([1, 192, 64, 85]))

        b_patches = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        b_groups = torch.split(b_patches, 1, dim=0)

        # process mask
        mask = F.interpolate(mask, scale_factor=1./self.rate)
        m_patches = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        m = m_patches[0]    # m shape: [B, C, k, k]

        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(self.fuse_k).view(1, 1, self.fuse_k, self.fuse_k)  # 1*1*k*k
        escape_NaN = torch.FloatTensor([1e-4])

        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()
            escape_NaN = escape_NaN.cuda()

        for xi, bi, raw_bi in zip(f_groups, b_groups, raw_b_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            bi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_bi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare

            # xi.size(), bi.size(), raw_bi.size()
            # (torch.Size([1, 192, 64, 85]), torch.Size([1, 5440, 192, 3, 3]), torch.Size([1, 5440, 192, 4, 4]))
            bi = bi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(bi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = bi / max_wi
            # (Pdb) wi_normed.size()
            # torch.Size([5440, 192, 3, 3])
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [self.fuse_k, self.fuse_k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [self.fuse_k, self.fuse_k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_bi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        return y


def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)

#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(nn.Module):
    def __init__(self):
        super(GatedGenerator, self).__init__()

        in_channels = 4
        out_channels = 3
        latent_channels = 48
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, activation = 'elu'),
            GatedConv2d(latent_channels, latent_channels * 2, 3, 2, 1, activation = 'elu'),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 2, 1, activation = 'elu'),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, activation = 'elu'),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, activation = 'elu'),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels, latent_channels//2, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels//2, out_channels, 3, 1, 1, activation = 'none'),
            nn.Tanh()
      )
        
        self.refine_conv = nn.Sequential(
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, activation = 'elu'),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1, activation = 'elu'),
            GatedConv2d(latent_channels, latent_channels*2, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels*2, latent_channels*2, 3, 2, 1, activation = 'elu'),
            GatedConv2d(latent_channels*2, latent_channels*4, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8, activation = 'elu'),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16, activation = 'elu')
        )
        self.refine_atten_1 = nn.Sequential(
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, activation = 'elu'),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1, activation = 'elu'),
            GatedConv2d(latent_channels, latent_channels*2, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels*2, latent_channels*4, 3, 2, 1, activation = 'elu'),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, activation = 'relu')
        )
        self.refine_atten_2 = nn.Sequential(
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, activation = 'elu')
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(latent_channels*8, latent_channels*4, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, activation = 'elu'),
            TransposeGatedConv2d(latent_channels * 4, latent_channels*2, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels*2, latent_channels*2, 3, 1, 1, activation = 'elu'),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels, latent_channels//2, 3, 1, 1, activation = 'elu'),
            GatedConv2d(latent_channels//2, out_channels, 3, 1, 1, activation = 'none'),
            nn.Tanh()
        )
        self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True)
        # pdb.set_trace()
        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse
        # (Pdb) img.size(), mask.size()
        # (torch.Size([1, 3, 512, 680]), torch.Size([1, 1, 512, 680]))
        # (Pdb) mask.mean()
        # tensor(0.0975, device='cuda:0')

        H, W = img.shape[2], img.shape[3]

        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), dim=1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in)                           # out: [B, 3, H, W]
        first_out = nn.functional.interpolate(first_out, (H, W))

        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)

        # (Pdb) second_in.size()
        # torch.Size([1, 4, 512, 680])
        refine_conv = self.refine_conv(second_in)
        refine_atten = self.refine_atten_1(second_in)
        # (Pdb) refine_conv.size()
        # torch.Size([1, 192, 128, 170])
        # (Pdb) refine_atten.size()
        # torch.Size([1, 192, 128, 170])
        mask_s = nn.functional.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)

        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (H, W))

        # (Pdb) first_out.size(), second_out.size()
        # (torch.Size([1, 3, 512, 680]), torch.Size([1, 3, 512, 680]))
        return first_out, second_out

#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()

        in_channels = 4 
        latent_channels = 48

        # Down sampling
        self.block1 = Conv2dLayer(in_channels, latent_channels, 7, 1, 3, activation = 'elu', sn = True)
        self.block2 = Conv2dLayer(latent_channels, latent_channels * 2, 4, 2, 1, activation = 'elu', sn = True)
        self.block3 = Conv2dLayer(latent_channels * 2, latent_channels * 4, 4, 2, 1, activation = 'elu', sn = True)
        self.block4 = Conv2dLayer(latent_channels * 4, latent_channels * 4, 4, 2, 1, activation = 'elu', sn = True)
        self.block5 = Conv2dLayer(latent_channels * 4, latent_channels * 4, 4, 2, 1, activation = 'elu', sn = True)
        self.block6 = Conv2dLayer(latent_channels * 4, 1, 4, 2, 1, activation = 'none', sn = True)
        
    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x

def export_onnx_model():
    """Export onnx model."""

    import onnx
    from onnx import optimizer

    # 1. Create and load model.
    model = GatedGenerator()
    model_name = 'pretrained_model/deepfillv2_WGAN_G_epoch40_batchsize4.pth'
    model.load_state_dict(torch.load(model_name))
    model.eval()
    model = model.cuda()

    onnx_file = "results/model.onnx"

    # 2. Model export
    print("Export model ...")
    image_input = torch.randn(1, 3, 512, 512).cuda()
    mask_input = torch.randn(1, 1, 512, 512).cuda()

    input_names = ["input", "mask"]
    output_names = ["output"]
    # variable lenght axes
    dynamic_axes = {'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},
                    'mask': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},
                    'output': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}

    torch.onnx.export(model, (image_input, mask_input), onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True,
                      export_params=True,
                      dynamic_axes=dynamic_axes)

    pdb.set_trace()

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer",
              "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('models/image_color.onnx')"


if __name__ == "__main__":
    export_onnx_model()