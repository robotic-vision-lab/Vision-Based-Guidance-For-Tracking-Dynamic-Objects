"""Optical Flow utils

This script consists of utility functions for optical flow tasks like gradient computation, drawing optical flow as a color coded image or as a vector at keypoints.
Additionally, it consists of helper functions to fetch optical flow color scheme used.

This file will be imported as module to access the utility functions to aid optical flow based tasks. It contains the following functions:
    * compute_horizontal_gradient - returns horizontal gradients
    * compute_vertical_gradient - returns vertical gradients
    * compute_temporal_gradient - returns temporal gradients
    * compute_gradients - returns tuple of spatio-temporal gradients (I_x, I_y, I_t)
    * get_OF_color_encoded - returns dense optical flow field in color encoded representation
    * get_color_scheme - returns color scheme used to color encode dense optical flow field
    * draw_sparse_optical_flow_arrows - returns image with arrows representing flow at keypoints
    * draw_tracks - returns image with circles drawn at keypoints added with track lines over a mask
"""

import numpy as np
import cv2 as cv

if __name__ == "__main__":
    from img_utils import *
else:
    from .img_utils import *

def compute_horizontal_gradient(img_1, img_2):
    """Returns gradient of the image (I_x) along the horizontal direction

    Args:
        img_1 (np.ndarray): previous image
        img_2 (np.ndarray): next image

    Returns:
        np.ndarray: 2D grid of horizontal gradient values
    """
    kernel_x = 0.25 * np.array([[-1, 1], [-1, 1]]).astype('float32')

    d_dx_img_1 = cv.filter2D(img_1.astype('float32'), -1, kernel_x)
    d_dx_img_2 = cv.filter2D(img_2.astype('float32'), -1, kernel_x)
    d_dx_img = d_dx_img_1 + d_dx_img_2

    return d_dx_img/2


def compute_vertical_gradient(img_1, img_2):
    """Returns gradient of the image (I_y) along the vertical direction

    Args:
        img_1 (np.ndarry): previous image
        img_2 (np.ndarray): next image

    Returns:
        np.ndarray: 2D grid of vertical gradient values
    """
    kernel_y = 0.25 * np.array([[-1, -1], [1, 1]]).astype('float32')

    d_dy_img_1 = cv.filter2D(img_1.astype('float32'), -1, kernel_y)
    d_dy_img_2 = cv.filter2D(img_2.astype('float32'), -1, kernel_y)
    d_dy_img = d_dy_img_1 + d_dy_img_2

    return d_dy_img/2


def compute_temporal_gradient(img_1, img_2):
    """Returns temporal gradient between two frames

    Args:
        img_1 (np.ndarray): previous image
        img_2 (np.ndarray): next image

    Returns:
        np.ndarray: 2D grid of temporal gradient values
    """
    kernel_t = 0.25 * np.array([[1, 1], [1, 1]]).astype('float32')

    d_dt_img_1 = cv.filter2D(img_1.astype('float32'), -1, kernel_t)
    d_dt_img_2 = cv.filter2D(img_2.astype('float32'), -1, kernel_t)
    d_dt_img = d_dt_img_2 - d_dt_img_1

    return d_dt_img/2


def compute_gradients(img_1, img_2):
    """Returns gradients of image I(x,y,t) along x, y, t as tuple

    Args:
        img_1 (np.ndarray): previous image
        img_2 (np.ndarray): next image

    Returns:
        tuple: tuple consisting of horizontal (I_x), vertical (I_y) and temporal (I_t) gradients between given images
    """
    I_x = compute_horizontal_gradient(img_1, img_2)
    I_y = compute_vertical_gradient(img_1, img_2)
    I_t = compute_temporal_gradient(img_1, img_2)

    return (I_x, I_y, I_t)


def get_OF_color_encoded(u, v):
    """Takes in u and v components of optical flow and draws color encoded image.
        First an HSV image is built using polar coordinates of the flow.

        note:
        OpenCV uses HSV ranges between (0-180, 0-255, 0-255).

        Then, its converted to RGB and displayed.

    Args:
        u (np.ndarray): optical flow field x component
        v (np.ndarray): optical flow field y component

    Returns:
        np.ndarray: image color encoded by optical flow
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
    """Returns the color scheme used for optical flow representation

    Args:
        img_size (tuple, optional): image size. Defaults to (512, 512).

    Returns:
        np.ndarray: image with color scheme used to encode optical flow
    """
    # size validation
    if not len(img_size) == 2:
        print(f'img_size needs two integers.')
        return

    # initialise u, v
    u = np.zeros(img_size, dtype='float32')
    v = np.zeros(img_size, dtype='float32')

    # build u, v
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            u[i][j] = (j - img_size[1]//2)**1
            v[i][j] = (i - img_size[0]//2)**1

    # return the color code
    return get_OF_color_encoded(u, v)


def draw_sparse_optical_flow_arrows(img,
                                    prev_points,
                                    cur_points,
                                    thickness=1,
                                    arrow_scale=3.0,
                                    color=(0, 255, 255)):
    """Takes in image alongwith previous and current points (tracked features).
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
        c_1, c_2 = cur_point.ravel()
        p_1, p_2 = prev_point.ravel()
        from_point = (int(c_1), int(c_2))
        to_point = (int(arrow_scale*(c_1-p_1) + c_1), int(arrow_scale*(c_2-p_2) + c_2))
        img = cv.arrowedLine(img=img,
                             pt1=from_point,
                             pt2=to_point,
                             color=color,
                             thickness=thickness,
                             line_type=None,
                             shift=None,
                             tipLength=0.2)
    return img


def draw_tracks(img, old_pts,
                new_pts,
                colors=None,
                mask=None,
                track_thickness=1,
                radius=3,
                circle_thickness=-1):
    """Draw tracks on a mask, circles on the image, adds them and returns it.
        Given new and old points,
        draws a track between each pair of a new point and old point.
        Also, draws a filled circle at each new point.

    Args:
        img (np.ndarray): Image over which tracks are to be drawn
        new_pts (np.ndarray): Vector coordinates of new points
        old_pts (np.ndarray): Vector coordinates of old points
        colors (np.ndarray): Corresponding colors for each pair of new and old points
        mask (np.ndarray): Mask on which lines are to be drawn or already drawn previously
        track_thickness (int): Thickness of track lines
        radius (int): Radius of filled circle to be drawn on img
        circle_thickness (int): Thickness of circle. -1 indicates fill. Defaults to -1.

    """
    # if image is grayscale it need to be converted to BGR
    img = convert_grayscale_to_BGR(img)

    # create a mask if not provided
    if mask is None:
        mask = np.zeros_like(img)

    # add tracks on mask, circles on img
    if new_pts is not None:
        for i, new in enumerate(new_pts):
            a, b = new.astype(np.int).ravel()
            if len(colors) == 1:
                color = colors[0]
            else:
                color = (102, 255, 102) if colors is None else colors[i].tolist()
            img = cv.circle(img, (a, b), radius, color, circle_thickness)

    if new_pts is not None and old_pts is not None:
        for i, (new, old) in enumerate(zip(new_pts, old_pts)):
            a, b = new.astype(np.int).ravel()
            c, d = old.astype(np.int).ravel()
            if len(colors) == 1:
                color = colors[0]
            else:
                color = (102, 255, 102) if colors is None else colors[i].tolist()
            mask = cv.line(mask, (a, b), (c, d), color, track_thickness)

    # add mask to img
    img[mask > 0] = 0
    img = cv.add(img, mask)

    return img, mask


if __name__ == "__main__":
    import os

    # prep the color_scheme_path i.e., uri for color scheme image
    _OUT_IMGS_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../out_imgs'))
    COLOR_SCHEME_PATH = os.path.join(_OUT_IMGS_PATH, 'color_scheme.jpg')

    # get the color_scheme
    color_scheme = get_color_scheme()

    # write the scheme image, show it
    cv.imwrite(COLOR_SCHEME_PATH, color_scheme)
    cv.imshow('scheme', color_scheme)
    cv.waitKey(0)

    cv.destroyAllWindows()
