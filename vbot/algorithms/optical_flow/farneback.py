import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# add vca\ to sys.path
vca_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if vca_path not in sys.path:
    sys.path.append(vca_path)

from utils import *



def compute_optical_flow_farneback(img_1, 
                                   img_2, 
                                   farneback_params):
    """ returns computed optical flow using farneback algorithm """
    # use opencv library function to compute flow 
    flow = cv.calcOpticalFlowFarneback(img_1, img_2, None, **farneback_params)
    u = flow[..., 0]
    v = flow[..., 1]
    return u,v


if __name__ == "__main__":
    # create dummy test images 
    height = 20
    width = 20
    data_path = generate_synth_data(img_size=(height, width), 
                                    path=os.path.join(vca_path,'../datasets'), 
                                    num_images=4, 
                                    folder_name='synth_data')

    # set some path paramaters as dictionaries, holding image path and types
    synth_path_params = {'path':data_path, 'image_type':'jpg'}
    dimetrodon_path_params = {'path':os.path.join(vca_path,'../datasets/Dimetrodon'), 'image_type':'png'}
    rubber_path_params = {'path':os.path.join(vca_path,'../datasets/RubberWhale'), 'image_type':'png'}
    car_path_params = {'path':'C:\MY DATA\Code Valley\MATLAB\determining-optical-flow-master\horn-schunck', 'image_type':'png'}

    # keep a dictionary of path parameters
    path_params = {'synth':synth_path_params, 
                   'dimetrodon':dimetrodon_path_params, 
                   'rubber':rubber_path_params, 
                   'car':car_path_params}
    
    # list out the image path
    path_params_key = 'car'
    img_paths = get_image_paths(**path_params[path_params_key])

    # read and preprocess
    img_1 = preprocess_image(cv.imread(img_paths[0]))
    img_2 = preprocess_image(cv.imread(img_paths[1]))

    # set farneback parameters
    PYR_SCALING_RATIO = 0.5         
    PYR_NUM_LEVELS = 3
    WINDOW_SIZE = 15
    NUM_ITER = 3
    POLY_NEIGHBORHOOD = 5
    POLY_SIGMA = 1.1
    OPERATION_FLAGS = 0

    # create dictionary of farneback params
    farneback_params = dict( pyr_scale=PYR_SCALING_RATIO,
                             levels=PYR_NUM_LEVELS,
                             winsize=WINDOW_SIZE,
                             iterations=NUM_ITER,
                             poly_n=POLY_NEIGHBORHOOD,
                             poly_sigma=POLY_SIGMA,
                             flags=OPERATION_FLAGS )

    # get optical flow 
    u, v = compute_optical_flow_farneback(img_1, img_2, farneback_params)
    
    # draw color encoded optical flow
    img_OF_color = get_OF_color_encoded(u, v)
    cv.imwrite(os.path.join(vca_path, f'algorithms/optical_flow/results/optical_flow_farn_{path_params_key}.jpg'), img_OF_color)

    # display optical flow 
    cv.imshow('Optical Flow color encoded', img_OF_color)
    cv.waitKey(0)
    cv.destroyAllWindows()
        