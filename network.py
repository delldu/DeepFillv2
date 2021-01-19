import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

from network_module import *

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
        # pdb.set_trace()
        # (Pdb) img.size(), mask.size()
        # (torch.Size([1, 3, 512, 680]), torch.Size([1, 1, 512, 680]))
        # (Pdb) mask.mean()
        # tensor(0.0975, device='cuda:0')

        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), dim=1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in)                           # out: [B, 3, H, W]
        # (Pdb) img.shape
        # torch.Size([1, 3, 512, 680])
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))

        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)

        refine_conv = self.refine_conv(second_in)
        refine_atten = self.refine_atten_1(second_in)
        # (Pdb) second_in.size()
        # torch.Size([1, 4, 512, 680])
        # (Pdb) refine_conv.size()
        # torch.Size([1, 192, 128, 170])
        # (Pdb) refine_atten.size()
        # torch.Size([1, 192, 128, 170])
        mask_s = nn.functional.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        # mask_s.size()
        # torch.Size([1, 1, 128, 170])
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)

        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))

        # pdb.set_trace()
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
