import time
import argparse
import random
from imageio import imwrite
import torch
import numpy as np

from NN.pretrained.vgg import Vgg16Pretrained
import NN.misc as misc
from NN.misc import load_path_for_pytorch
from NN.stylize import produce_stylization

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

misc.USE_GPU = torch.cuda.is_available()
cnn = misc.to_device(Vgg16Pretrained())
phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

sz = 1024 if misc.USE_GPU else 768
iter = 150 if misc.USE_GPU else 30
content_weight = 0


async def loadNN(images_ids):
    content_path = 'photos/' + images_ids[1] + '.jpg'
    style_path = 'photos/' + images_ids[0] + '.jpg'
    output_path = 'photos/' + images_ids[0] + images_ids[1] + '.jpg'
    
    content_im_orig = misc.to_device(load_path_for_pytorch(content_path, target_size=sz)).unsqueeze(0)
    style_im_orig = misc.to_device(load_path_for_pytorch(style_path, target_size=sz)).unsqueeze(0)
    
    if misc.USE_GPU: torch.cuda.synchronize()
    
    return content_im_orig, style_im_orig, output_path

def stylization(content_im_orig, style_im_orig, output_path):
    startTime = time.time()
    output = produce_stylization(content_im_orig, style_im_orig, phi,
                            iter,
                            2e-3,
                            content_weight,
                            3,
                            False,
                            True,
                            False)
    endTime = time.time()
    new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)
    save_im = (new_im_out * 255).astype(np.uint8)
    imwrite(output_path, save_im)	
    if misc.USE_GPU: torch.cuda.empty_cache()
    return endTime - startTime
    
