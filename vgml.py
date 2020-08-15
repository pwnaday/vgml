""" ML Imports """
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
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

