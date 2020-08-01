import cv2 as cv
import numpy as np
from img_utils import *
from window_utils import *
from math import sin, cos
from synth_data import generate_synth_data
import matplotlib.pyplot as plt

def compute_gradients(img_1, img_2):
    """ returns gradients of image I(x,y,t) along x, y, t as tuple """
    I_x = compute_horizontal_gradient(img_1, img_2)
    I_y = compute_vertical_gradient(img_1, img_2)
    I_t = compute_temporal_gradient(img_1, img_2)

    return (I_x, I_y, I_t)

def compute_horizontal_gradient(img_1, img_2):
    """ returns gradient of the image (I_x) along the horizontal direction """
    kernel_x = 0.25 * np.array([[-1, 1], [-1, 1]]).astype('float32')

    d_dx_img_1 = cv.filter2D(img_1.astype('float32'), -1, kernel_x)
    d_dx_img_2 = cv.filter2D(img_2.astype('float32'), -1, kernel_x)
    d_dx_img = d_dx_img_1 + d_dx_img_2

    return d_dx_img/2

def compute_vertical_gradient(img_1, img_2):
    """ returns gradient of the image (I_y) along the vertical direction """
    kernel_y = 0.25 * np.array([[-1, -1], [1, 1]]).astype('float32')

    d_dy_img_1 = cv.filter2D(img_1.astype('float32'), -1, kernel_y)
    d_dy_img_2 = cv.filter2D(img_2.astype('float32'), -1, kernel_y)
    d_dy_img = d_dy_img_1 + d_dy_img_2

    return d_dy_img/2

def compute_temporal_gradient(img_1, img_2):
    """ returns temporal gradient between two frames """
    kernel_t = 0.25 * np.array([[1, 1], [1, 1]]).astype('float32')

    d_dt_img_1 = cv.filter2D(img_1.astype('float32'), -1, kernel_t)
    d_dt_img_2 = cv.filter2D(img_2.astype('float32'), -1, kernel_t)
    d_dt_img = d_dt_img_2 - d_dt_img_1

    return d_dt_img/2

def preprocess_image(img, blur=True):
    """ returns grayscaled, gaussian blurred and normalised image """
    img = convert_to_grayscale(img)
    if blur:
        img = cv.GaussianBlur(img, (3,3), 0)
    img = normalize_to_unit_range(img)

    return img

def get_color_scheme(img_size=(512, 512)):
    """ returns the color scheme used for optical flow representation """
    # size validation
    if not len(img_size)==2:
        print(f'img_size needs two integers.')
        return 
    
    # initialise u, v
    u = np.zeros(img_size, dtype='float32')
    v = np.zeros(img_size, dtype='float32')

    # build u, v
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            u[i][j] = np.sign(j - img_size[1]//2) * (j - img_size[1]//2)**2
            v[i][j] = np.sign(i - img_size[0]//2) * (i - img_size[0]//2)**2

    # return the color code
    return get_OF_color_encoded(u,v)

def get_OF_color_encoded(u, v):
    """ Takes in u and v components of optical flow and draws color encoded image. 
        First an HSV image is built using polar coordinates of the flow.
        
        note:
        OpenCV uses HSV ranges between (0-180, 0-255, 0-255).

        Then, its converted to RGB and displayed.
    """
    # create tensor with all zeros of frame shape
    # note: flow field u must be a 2D array
    hsv = np.zeros((*get_image_shape(u), 3), dtype=np.uint8)

    # get polar representation of the optical flow components
    magnitude, angle = cv.cartToPolar(u, v)

    # set saturation to maximum
    hsv[..., 1] = 255

    # set image hue and value
    hsv[..., 0] = angle * 90 / np.pi    # angle is 0.0-2pi, we need 0-180
    hsv[..., 2] = normalize_to_255_range(magnitude)
    
    # convert HSV to RGB (BGR in opencv)
    color_img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return color_img

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
    data_path = generate_synth_data(img_size=(height,width), 
                                    path='./', 
                                    num_images=4, 
                                    folder_name='synth_data')


    synth_path_params = {'path':data_path, 'image_type':'jpg'}
    dimetrodon_path_params = {'path':'./datasets/Dimetrodon', 'image_type':'png'}
    rubber_path_params = {'path':'./datasets/RubberWhale', 'image_type':'png'}
    car_path_params = {'path':'C:\MY DATA\Code Valley\MATLAB\determining-optical-flow-master\horn-schunck', 'image_type':'png'}

    path_params = {'synth':synth_path_params, 
                   'dimetrodon':dimetrodon_path_params, 
                   'rubber':rubber_path_params, 
                   'car':car_path_params}
    
    # list out the image path
    img_paths = get_image_paths(**path_params['synth'])

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
        