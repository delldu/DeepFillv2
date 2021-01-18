import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import network
import test_dataset
import utils

import pdb

def export_onnx_model(model):
    """Export onnx model."""

    import onnx
    from onnx import optimizer

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


def WGAN_tester(opt):
    
    # Save the model if pre_train == True
    def load_model_generator(net, epoch, opt):
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, 4)
        model_name = os.path.join('pretrained_model', model_name)
        pretrained_dict = torch.load(model_name)
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

    # Build networks
    generator = utils.create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    load_model_generator(generator, opt.epoch, opt)
    print('-------------------------Pretrained Model Loaded-------------------------')

    # To device
    generator = generator.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = test_dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # export_onnx_model(generator)

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    for batch_idx, (img, mask) in enumerate(dataloader):
        img = img.cuda()
        mask = mask.cuda()

        # Generator output
        with torch.no_grad():
            first_out, second_out = generator(img, mask)

        # forward propagation
        first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
        second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

        # masked_img = img * (1 - mask) + mask
        # mask = torch.cat((mask, mask, mask), 1)

        img_list = [first_out_wholeimg, second_out_wholeimg]
        name_list = ['first_out', 'second_out']
        utils.save_sample_png(sample_folder = opt.results_path, sample_name = '%d' % (batch_idx + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        print('----------------------batch_idx%d' % (batch_idx + 1) + ' has been finished----------------------')
