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


""" 
calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow
.   @brief Computes a dense optical flow using the Gunnar Farneback's algorithm.
.
.   @param prev first 8-bit single-channel input image.
.   @param next second input image of the same size and the same type as prev.
.   @param flow computed flow image that has the same size as prev and type CV_32FC2.
.   @param pyr_scale parameter, specifying the image scale (\<1) to build pyramids for each image;
.   pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous
.   one.
.   @param levels number of pyramid layers including the initial image; levels=1 means that no extra
.   layers are created and only the original images are used.
.   @param winsize averaging window size; larger values increase the algorithm robustness to image
.   noise and give more chances for fast motion detection, but yield more blurred motion field.
.   @param iterations number of iterations the algorithm does at each pyramid level.
.   @param poly_n size of the pixel neighborhood used to find polynomial expansion in each pixel;
.   larger values mean that the image will be approximated with smoother surfaces, yielding more
.   robust algorithm and more blurred motion field, typically poly_n =5 or 7.
.   @param poly_sigma standard deviation of the Gaussian that is used to smooth derivatives used as a
.   basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a
.   good value would be poly_sigma=1.5.
.   @param flags operation flags that can be a combination of the following:
.    -   **OPTFLOW_USE_INITIAL_FLOW** uses the input flow as an initial flow approximation.
.    -   **OPTFLOW_FARNEBACK_GAUSSIAN** uses the Gaussian \f$\texttt{winsize}\times\texttt{winsize}\f$
.        filter instead of a box filter of the same size for optical flow estimation; usually, this
.        option gives z more accurate flow than with a box filter, at the cost of lower speed;
.        normally, winsize for a Gaussian window should be set to a larger value to achieve the same
.        level of robustness.
"""



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
    img_paths = get_image_paths(**path_params['dimetrodon'])

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
    cv.imwrite(os.path.join(vca_path,'out_imgs/optical_flow_farneback.jpg'), img_OF_color)

    # display optical flow 
    cv.imshow('Optical Flow color encoded', img_OF_color)
    cv.waitKey(0)
    cv.destroyAllWindows()
        