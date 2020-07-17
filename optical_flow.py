from PIL import Image
from scipy import signal
import numpy as np
from utils import *


# get list of the image file paths
img_paths = get_image_paths()

# open first image (handle)
curr_frame = Image.open(img_paths[0])

# open second image (handle)
next_frame = Image.open(img_paths[1])


'''
	let the filter in x-direction be Gx = 0.25*[[-1,1],[-1,1]]
	let the filter in y-direction be Gy = 0.25*[[-1,-1],[1,1]]
	let the filter in xy-direction be Gt = 0.25*[[1,1],[1, 1]]
	**1/4 = 0.25** for a 2x2 filter
	'''


def compute_optical_flow(img_1, img_2):
    """ returns computed optical flow between two given frames I_{t-1} and I_{t}"""
    pass

def get_gradients(img_1, img_2):
    """ returns gradients of image I(x,y,t) along x, y, t as tuple """
    pass 

def compute_horizontal_gradient(img_1, img_2):
    """ returns gradient of the image (I_x) along the horizontal direction """
    pass

def compute_vertical_gradient(img_1, img_2):
    """ returns gradient of the image (I_y) along the vertical direction """
    pass

def compute_temporal_gradient(img_1, img_2):
    """ returns temporal gradient between two frames """
    pass


