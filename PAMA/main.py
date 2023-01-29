import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import save_image
from PIL import Image, ImageFile
from PAMA.net import Net
from PAMA.utils import DEVICE, train_transform, test_transform, FlatFolderDataset, InfiniteSamplerWrapper, plot_grad_flow, adjust_learning_rate
Image.MAX_IMAGE_PIXELS = None  
ImageFile.LOAD_TRUNCATED_IMAGES = True

from argparse import Namespace


def train(args):
    logging.basicConfig(filename='training.log',
                    format='%(asctime)s %(levelname)s: %(message)s', 
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

    mes = "current pid: " + str(os.getpid())
    print(mes)
    logging.info(mes)
    model = Net(args)
    model.train()
    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(DEVICE)

    tf = train_transform()
    content_dataset = FlatFolderDataset(args.content_folder, tf)
    style_dataset = FlatFolderDataset(args.style_folder, tf)
    content_iter = iter(data.DataLoader(
                        content_dataset, batch_size=args.batch_size,
                        sampler=InfiniteSamplerWrapper(content_dataset),
                        num_workers=args.num_workers))
    style_iter = iter(data.DataLoader(
                      style_dataset, batch_size=args.batch_size,
                      sampler=InfiniteSamplerWrapper(style_dataset),
                      num_workers=args.num_workers))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for img_index in range(args.iterations):
        print("iteration :", img_index+1)
        optimizer.zero_grad()
        Ic = next(content_iter).to(DEVICE)
        Is = next(style_iter).to(DEVICE)
        
        loss = model(Ic, Is)
        print(loss)
        loss.sum().backward()
        
        #plot_grad_flow(GMMN.named_parameters())
        optimizer.step()

        if (img_index+1)%args.log_interval == 0:
            print("saving...")
            mes = "iteration: " + str(img_index+1) + " loss: "  + str(loss.sum().item())
            logging.info(mes)
            model.module.save_ckpts()
            adjust_learning_rate(optimizer, img_index, args)


def eval(args):
    mes = "current pid: " + str(os.getpid())
    print(mes)
    logging.info(mes)
    print(args, type(args))
    model = Net(args)
    model.eval()
    model = model.to(DEVICE)
    
    tf = test_transform()
    if args.run_folder == True:
        content_dir = args.content 
        style_dir = args.style
        for content in os.listdir(content_dir):
            for style in os.listdir(style_dir):
                name_c = content_dir + content
                name_s = style_dir + style
                Ic = tf(Image.open(name_c)).to(DEVICE)
                Is = tf(Image.open(name_s)).to(DEVICE)
                Ic = Ic.unsqueeze(dim=0)
                Is = Is.unsqueeze(dim=0)
                with torch.no_grad():
                    Ics = model(Ic, Is)

                name_cs = "ics/" + os.path.splitext(content)[0]+"--"+style 
                save_image(Ics[0], name_cs)
    else:
        Ic = tf(Image.open(args.content)).to(DEVICE)
        Is = tf(Image.open(args.style)).to(DEVICE)

        Ic = Ic.unsqueeze(dim=0)
        Is = Is.unsqueeze(dim=0)
        
        with torch.no_grad():
            Ics = model(Ic, Is)

        name_cs = args.style[:-4]+args.content[7:]
        print(name_cs)
        save_image(Ics[0], name_cs)
        print('DONE')
        
def mainPAMA(content, style):
    args = Namespace(content=content, pretrained=True, requires_grad=True, run_folder=False, style=style, subcommand='eval', training=False)
    eval(args)
    
if __name__ == "__main__":
    mainPAMA()
