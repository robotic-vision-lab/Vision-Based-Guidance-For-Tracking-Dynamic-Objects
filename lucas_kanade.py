import cv2 as cv
import numpy as np

from img_utils import *
from window_utils import *
from optical_flow_utils import *
from synth_data import generate_synth_data

def compute_optical_flow_LK():
    pass


if __name__ == "__main__":
    """ test Lucas Kanade implementation """

    # create dummy test images 
    height = 20
    width = 20
    data_path = generate_synth_data( img_size=(height, width), 
                                     path='./', 
                                     num_images=4, 
                                     folder_name='synth_data' )

    # gather the path params needed in a dictionary
    synth_path_params = {'path':data_path, 'image_type':'jpg'}
    dimetrodon_path_params = {'path':'./datasets/Dimetrodon', 'image_type':'png'}
    rubber_path_params = {'path':'./datasets/RubberWhale', 'image_type':'png'}
    car_path_params = {'path':'C:\MY DATA\Code Valley\MATLAB\determining-optical-flow-master\horn-schunck', 'image_type':'png'}
    venus_path_params = {'path':'./datasets/Venus', 'image_type':'png'}

    path_params = { 'synth':synth_path_params, 
                    'dimetrodon':dimetrodon_path_params, 
                    'rubber':rubber_path_params, 
                    'car':car_path_params,
                    'venus':venus_path_params }
    
    # list out the image path
    img_paths = get_image_paths(**path_params['car'])

    # read and preprocess
    img_1 = preprocess_image(cv.imread(img_paths[0]))
    img_2 = preprocess_image(cv.imread(img_paths[1]))

    # get images 




