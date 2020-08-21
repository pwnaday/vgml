""" ML Imports """
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os

""" Vector graphics imports """
import gizeh as gz
from math import pi

""" Config """
dataroot = os.getcwd() + "/data"
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 29
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta = 0.5
npgu = 1
""""""

def make_yinyang(size=200, r=80, filename="yin_yang.png"):
    surface = gz.Surface(size, size, bg_color=(0, .3 , .6))
    yin_yang = gz.Group([
	 gz.arc(r, pi/2, 3*pi/2, fill = (1,1,1)), 
	 gz.arc(r, -pi/2, pi/2, fill = (0,0,0)), 
	 gz.arc(r/2, -pi/2, pi/2, fill = (1,1,1), xy = [0,-r/2]), 
	 gz.arc(r/2, pi/2, 3*pi/2, fill = (0,0,0), xy = [0, r/2]),  
	 gz.circle(r/8, xy = [0,  +r/2], fill = (1,1,1)), 
	 gz.circle(r/8, xy = [0,  -r/2], fill = (0,0,0)) ]) 
    yin_yang.translate([size/2,size/2]).draw(surface)
    surface.write_to_png(filename)
    return 0

def main() :
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([\
					transforms.Resize(image_size),\
					transforms.CenterCrop(image_size),\
					transforms.ToTensor(),\
					transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))\
			    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("training imgs")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
main()
