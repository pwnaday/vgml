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

