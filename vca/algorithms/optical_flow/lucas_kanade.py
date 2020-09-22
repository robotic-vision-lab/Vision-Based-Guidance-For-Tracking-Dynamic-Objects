import os
import sys
import cv2 as cv
import numpy as np


# add vca\ to sys.path
vca_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if vca_path not in sys.path:
    sys.path.append(vca_path)

from utils import *


def compute_optical_flow_LK(img_1,
                            img_2,
                            pts_1=None,
                            lk_params=None,
                            feature_params=None):
    """Implements Lucas-Kanade. Takes in images I_1 and I_2

    Args:
        img_1 (np.ndarray): Previous frame image 
        img_2 (np.ndarray): Current frame image
        lk_params (dict): Parameters to be fed to LK optical flow function. Defaults to None.
        pts_1 (np.ndarray): Good feature points in previous image. Defaults to None.
        feature_params (dict): Parameters to be fed to the goodFeaturesToTrack function. Defaults to None.
    """
    # if previous feature points for previous frame are not given, compute them!
    if pts_1 is None:
        # if feature_params are not given the initialise a default one
        if feature_params is None:
            feature_params = dict( maxCorners = 100,
                                   qualityLevel = 0.3,
                                   minDistance = 7,
                                   blockSize = 7 )
        
        # compute good feature points
        pts_1 = cv.goodFeaturesToTrack(img_1, mask=None, **feature_params)
    
    # compute LK optical flow
    if lk_params is None:
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 3,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03) )

    pts_2, st, err = cv.calcOpticalFlowPyrLK( prevImg=img_1, 
                                              nextImg=img_2, 
                                              prevPts=pts_1, 
                                              nextPts=None,
                                              **lk_params )

    return pts_1, pts_2, st, err


if __name__ == "__main__":
    """ test Lucas Kanade implementation """

    # create dummy test images 
    height = 20
    width = 20
    data_path = generate_synth_data( img_size=(height, width), 
                                     path=os.path.join(vca_path,'../datasets'), 
                                     num_images=4, 
                                     folder_name='synth_data' )

    # gather the path params needed in a dictionary
    synth_path_params = {'path':data_path, 'image_type':'jpg'}
    dimetrodon_path_params = {'path':os.path.join(vca_path, '../datasets/Dimetrodon'), 'image_type':'png'}
    rubber_path_params = {'path':os.path.join(vca_path,'../datasets/RubberWhale'), 'image_type':'png'}
    car_path_params = {'path':'C:\MY DATA\Code Valley\MATLAB\determining-optical-flow-master\horn-schunck', 'image_type':'png'}
    venus_path_params = {'path':os.path.join(vca_path, '../datasets/Venus'), 'image_type':'png'}

    path_params = { 'synth':synth_path_params, 
                    'dimetrodon':dimetrodon_path_params, 
                    'rubber':rubber_path_params, 
                    'car':car_path_params,
                    'venus':venus_path_params }
    
    # list out the image path
    img_paths = get_image_paths(**path_params['car'])

    # read and preprocess
    img_1 = normalize_to_255_range(preprocess_image(cv.imread(img_paths[0])))
    img_2 = normalize_to_255_range(preprocess_image(cv.imread(img_paths[1])))

    # set params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # set parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 3,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03) )

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    img_gray_1 = convert_to_grayscale(img_1)
    img_gray_2 = convert_to_grayscale(img_2)

    # Create a mask image for drawing purposes later
    mask = np.zeros_like(cv.cvtColor(img_1, cv.COLOR_GRAY2BGR))

    
    # calculate optical flow
    pts_1 = cv.goodFeaturesToTrack(img_gray_1, mask=None, **feature_params)
    _, pts_2, st, err = compute_optical_flow_LK( img_1=img_gray_1, 
                                                 img_2=img_gray_2, 
                                                 pts_1=p1, 
                                                 lk_params=lk_params )

    # select good points
    good_1 = pts_1[st==1]
    good_2 = pts_2[st==1]

    # draw tracks
    for i, (new, old) in enumerate(zip(good_2, good_1)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b), (c,d), color[i].tolist(), 2) 
        frame = cv.circle(cv.cvtColor(img_2, cv.COLOR_GRAY2BGR), (a,b), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)   # lines and circles drawn on img_2
    cv.imshow('before', img)
    # draw arrows
    img = draw_optical_flow_arrows(img, good_1, good_2)
    
    cv.imshow('frame', img)
    k = cv.waitKey(0)

