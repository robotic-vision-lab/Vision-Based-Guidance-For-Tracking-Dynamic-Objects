from PIL import Image
from scipy import signal
import numpy as np
from img_utils import *
import cv2 as cv


# get list of the image file paths
img_paths = get_image_paths()

# read first image
curr_frame = cv.imread(img_paths[0])

# read second image
next_frame = cv.imread(img_paths[1])


def compute_optical_flow_HS(img_1, img_2, alpha=0.5, num_iter=128):
    """ returns computed optical flow between two given frames I_{t-1} and I_{t}"""
    I_x, I_y, I_t = compute_gradients(img_1, img_2)

    img_shape = get_image_shape(img_1)

    # initialize velocity field
    u = np.zeros(img_shape)
    v = np.zeros(img_shape)

    # pre-compute the smoothness error term
    e_smooth = alpha**2 + I_x**2 + I_y**2               # note operations are element-wise
    laplacian_kernel = np.array([[1/12, 1/6, 1/12], [1/6, -1, 1/6], [1/12, 1/6, 1/12]])

    for _ in range(num_iter):
        # compute average of previous velocity estimates
        u_avg = cv.filter2D(u, -1, laplacian_kernel)
        v_avg = cv.filter2D(v, -1, laplacian_kernel)

        # compute the brightness error term
        e_bright = I_x * u_avg + I_y * v_avg + I_t      # note operations are element-wise

        # compute new velocities from estimated derivatives
        e_quotient = e_bright / e_smooth
        u = u_avg - I_x * e_quotient
        v = v_avg - I_y * e_quotient

    return u,v


def display_optical_flow(u, v):
    """ display the optical flow plot in color encoding """
    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros(curr_frame.shape, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow("colored flow", bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
def compute_gradients(img_1, img_2):
    """ returns gradients of image I(x,y,t) along x, y, t as tuple """
    I_x = compute_horizontal_gradient(img_1, img_2)
    I_y = compute_vertical_gradient(img_1, img_2)
    I_t = compute_temporal_gradient(img_1, img_2)

    return (I_x, I_y, I_t)

def compute_horizontal_gradient(img_1, img_2):
    """ returns gradient of the image (I_x) along the horizontal direction """
    kernel_x = 0.25 * np.array([[-1, 1], [-1, 1]])

    d_dx_img_1 = cv.filter2D(img_1, -1, kernel_x)
    d_dx_img_2 = cv.filter2D(img_2, -1, kernel_x)
    d_dx_img = d_dx_img_1 + d_dx_img_2

    return d_dx_img

def compute_vertical_gradient(img_1, img_2):
    """ returns gradient of the image (I_y) along the vertical direction """
    kernel_y = 0.25 * np.array([[-1, -1], [1, 1]])

    d_dy_img_1 = cv.filter2D(img_1, -1, kernel_y)
    d_dy_img_2 = cv.filter2D(img_2, -1, kernel_y)
    d_dy_img = d_dy_img_1 + d_dy_img_2

    return d_dy_img

def compute_temporal_gradient(img_1, img_2):
    """ returns temporal gradient between two frames """
    kernel_t = 0.25 * np.array([[1, 1], [1, 1]])

    d_dt_img_1 = cv.filter2D(img_1, -1, kernel_t)
    d_dt_img_2 = cv.filter2D(img_2, -1, -kernel_t)
    d_dt_img = d_dt_img_1 + d_dt_img_2

    return d_dt_img

if __name__ == "__main__":
    img_1 = convert_to_grayscale(curr_frame)
    img_2 = convert_to_grayscale(next_frame)
    u, v = compute_optical_flow_HS(img_1, img_2)
    print(u)
    print(v)
    cv.imshow('u', u)
    cv.imshow('v', v)

    cv.waitKey(0)
    cv.destroyAllWindows()
    # display_optical_flow(u, v)
