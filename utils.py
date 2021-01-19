import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv

import network
import pdb

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GatedGenerator()
    print('Generator is created!')
    network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator()
    print('Discriminator is created!')
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_perceptualnet():
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    print('Perceptual network is created!')
    return perceptualnet
    
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_names(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.jpg'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

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
    patches = patches.permute(0, 4, 1, 2, 3)    # raw_shape: [B, L, C, k, k]
    # torch.Size([1, 5440, 192, 4, 4])

    return patches  # [N, C*k*k, L], L is the total number of such blocks

# xxxx5555
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

# xxxx5555
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
