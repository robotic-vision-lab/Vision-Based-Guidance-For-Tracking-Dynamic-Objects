import numpy as np
import cv2 as cv

if __name__ == "__main__":
    from img_utils import *
else:
    from .img_utils import *

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


def compute_gradients(img_1, img_2):
    """ returns gradients of image I(x,y,t) along x, y, t as tuple """
    I_x = compute_horizontal_gradient(img_1, img_2)
    I_y = compute_vertical_gradient(img_1, img_2)
    I_t = compute_temporal_gradient(img_1, img_2)

    return (I_x, I_y, I_t)


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

    # set value to maximum
    hsv[..., 2] = 255

    # set image hue and saturation
    hsv[..., 0] = angle * 90 / np.pi    # angle is 0.0-2pi, we need 0-180
    hsv[..., 1] = normalize_to_255_range(magnitude)
    
    # convert HSV to RGB (BGR in opencv)
    color_img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return color_img


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
            u[i][j] = (j - img_size[1]//2)**1  #* np.sign(j - img_size[1]//2)**2 
            v[i][j] = (i - img_size[0]//2)**1  #* np.sign(i - img_size[0]//2)**2 

    # return the color code
    return get_OF_color_encoded(u,v)


def draw_sparse_optical_flow_arrows(img, prev_points, cur_points, thickness=1, arrow_scale=3.0, color=(0, 255, 255)):
    """takes in image alongwith previous and current points (tracked features).
        returns the image with arrows drawn on it
        For instance, prev_point was (1, 2), cur_point is (2, 2)
        returned image should have an arrow representing flow
        (cur_point - prev_point) i.e., (1, 0) drawn at (2, 2). 

    Args:
        img (np.ndarray): img over which to draw the arrows
        prev_points (np.ndarray): vector of 2D points (float32)
        cur_points (np.ndarray): output vector of 2D points (float32)
    """
    for i, (cur_point, prev_point) in enumerate(zip(cur_points, prev_points)):
        c1, c2 = cur_point.ravel()
        p1, p2 = prev_point.ravel()
        from_point = (int(c1), int(c2))
        to_point = (int(arrow_scale*(c1-p1) + c1), int(arrow_scale*(c2-p2) + c2))
        img = cv.arrowedLine(img=img, 
                             pt1=from_point, 
                             pt2=to_point, 
                             color=color, 
                             thickness=thickness, 
                             line_type=None, 
                             shift=None, 
                             tipLength=0.2)
    return img

    



if __name__ == "__main__":
    import os

    # make the color_scheme_path i.e., uri for color scheme image 
    out_imgs_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../out_imgs'))
    color_scheme_path = os.path.join(out_imgs_path, 'color_scheme.jpg')

    # get the color_scheme
    color_scheme = get_color_scheme()

    # write the scheme image, show it 
    cv.imwrite(color_scheme_path, color_scheme)
    cv.imshow('scheme', color_scheme)

    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
