import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from img_utils import *
from optical_flow_utils import *
from window_utils import *
from synth_data import generate_synth_data



def compute_optical_flow_HS(img_1, 
                            img_2, 
                            alpha=0.5,
                            num_iter=128):
    """ returns computed optical flow using Horn-Schunk algorithm """
    # compute gradients along x,y,t
    I_x, I_y, I_t = compute_gradients(img_1, img_2)

    # pre-compute smoothness error term
    e_smooth = alpha**2 + I_x**2 + I_y**2               # note operations are element-wise

    # initialize velocity field components u and v with zeros  
    u = np.zeros_like(img_1)
    v = np.zeros_like(img_1)
    avg_ker = get_average_kernel()

    # perform iterative solution
    for i in range(num_iter):
        # compute average of previous velocity estimates
        u_avg = cv.filter2D(u, -1, avg_ker)
        v_avg = cv.filter2D(v, -1, avg_ker)

        # compute the brightness error term
        e_bright = I_x * u_avg + I_y * v_avg + I_t      # note operations are element-wise

        # compute new velocities from estimated derivatives
        e_quotient = e_bright / e_smooth
        u = u_avg - I_x * e_quotient
        v = v_avg - I_y * e_quotient

    return u,v


if __name__ == "__main__":
    # create dummy test images 
    height = 20
    width = 20

    data_path = generate_synth_data( img_size=(height,width), 
                                     path='./', 
                                     num_images=4, 
                                     folder_name='synth_data' )


    # set some path paramaters as dictionaries, holding image path and types
    synth_path_params = {'path':data_path, 'image_type':'jpg'}
    dimetrodon_path_params = {'path':'./datasets/Dimetrodon', 'image_type':'png'}
    rubber_path_params = {'path':'./datasets/RubberWhale', 'image_type':'png'}
    car_path_params = {'path':'C:\MY DATA\Code Valley\MATLAB\determining-optical-flow-master\horn-schunck', 'image_type':'png'}

    # keep a dictionary of path parameters
    path_params = {'synth':synth_path_params, 
                   'dimetrodon':dimetrodon_path_params, 
                   'rubber':rubber_path_params, 
                   'car':car_path_params}
    
    # list out the image path
    img_paths = get_image_paths(**path_params['car'])

    # read and preprocess
    img_1 = preprocess_image(cv.imread(img_paths[0]))
    img_2 = preprocess_image(cv.imread(img_paths[1]))

    # initialize parameters alpha and number of iterations (higher alpha enforces smoother flow field)
    alpha = 0.5
    num_iter = 128

    # get optical flow 
    u, v = compute_optical_flow_HS(img_1, img_2, alpha, num_iter)
    
    # draw color encoded optical flow
    img_OF_color = get_OF_color_encoded(u, v)
    cv.imwrite('optical_flow_horn_schunk.jpg', img_OF_color)

    # display optical flow 
    cv.imshow('Optical Flow color encoded', img_OF_color)
    cv.waitKey(0)
    cv.destroyAllWindows()
        