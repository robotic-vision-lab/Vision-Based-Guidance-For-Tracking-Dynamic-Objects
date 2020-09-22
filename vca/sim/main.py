import os
import sys
import cv2 as cv
import numpy as np
import shutil
import pygame 

from pygame.locals import *
from settings import *
from optical_flow_config import (FARNEBACK_PARAMS,
                                 FARN_TEMP_FOLDER,
                                 FEATURE_PARAMS, 
                                 LK_PARAMS,
                                 LK_TEMP_FOLDER)

# add vca\ to sys.path
vca_path = os.path.abspath(os.path.join('..'))
if vca_path not in sys.path:
    sys.path.append(vca_path)

from game import Game
from game_utils import _prep_temp_folder
from utils.vid_utils import create_video_from_images
from utils.optical_flow_utils import (get_OF_color_encoded, 
                                      draw_sparse_optical_flow_arrows,
                                      draw_tracks)
from utils.img_utils import convert_to_grayscale
from algorithms.optical_flow import (compute_optical_flow_farneback, 
                                     compute_optical_flow_HS, 
                                     compute_optical_flow_LK)


def run_simulation():
    """Runs the game simulation. 
        Lets us record the screen frames into the temp folder set in settings.
    """
    # instantiate game 
    car_sim_game = Game()

    # start new game
    car_sim_game.start_new()

    # run
    car_sim_game.run()


def make_video(video_name, folder_path):
    """Looks for frames in given folder,
    writes them into a video file, with the given name. 
    Also removes the folder after creating the video.
    """
    if os.path.isdir(folder_path):
        create_video_from_images(folder_path, 'jpg', video_name, FPS)

        # delete folder
        shutil.rmtree(folder_path)


def run_farneback(video_name):
    """uses farneback to compute optical flow of video.
    Saves results in temp folder

    Args:
        video_name (str): name of video file. eg: 'vid_out_car.avi'
    """
    # prep temp folder
    _prep_temp_folder(FARN_TEMP_FOLDER)

    # create video capture and capture first frame
    vid_cap = cv.VideoCapture(video_name)
    ret, frame_1 = vid_cap.read()
    cur = convert_to_grayscale(frame_1)

    # capture frames from video, compute OF, save flow images
    _frame_num = 0
    while True:
        ret, frame_2 = vid_cap.read()
        if not ret:
            break
        nxt = convert_to_grayscale(frame_2)

        # compute optical flow between current and next frame
        u, v = compute_optical_flow_farneback(cur, nxt, FARNEBACK_PARAMS)

        # form the color encoded flow image
        img_flow_color = get_OF_color_encoded(u, v)

        # save image
        _frame_num += 1
        img_name = f'frame_{str(_frame_num).zfill(4)}.jpg'
        img_path = os.path.join(FARN_TEMP_FOLDER, img_name)
        cv.imwrite(img_path, img_flow_color)

        cur = nxt # .copy() ?

    vid_cap.release()


def run_lk(video_name):
    """uses lucas kanade to compute optical flow from video file,
    Also tracks good features, save results in temp folder

    Args:
        video_name (str): name of input video file. eg: 'vid_out_car.avi'
    """
    # prep temp folder
    _prep_temp_folder(LK_TEMP_FOLDER)

    # create some random colors one for each corner
    colors = np.random.randint(0, 255, (FEATURE_PARAMS['maxCorners'], 3))

    # create video capture; capture first frame
    vid_cap = cv.VideoCapture(video_name)
    ret, frame_1 = vid_cap.read()
    cur_frame = convert_to_grayscale(frame_1)
    cur_points = cv.goodFeaturesToTrack(cur_frame, mask=None, **FEATURE_PARAMS)

    # create a mask for drawing tracks
    mask = np.zeros_like(frame_1)

    # capture frames from video, compute OF, save flow images
    _frame_num = 0
    while True:
        ret, frame_2 = vid_cap.read()
        if not ret:
            break
        nxt_frame = convert_to_grayscale(frame_2)


        # compute optical flow between current and next frame
        cur_points, nxt_points, stdev, err = compute_optical_flow_LK(cur_frame, nxt_frame, cur_points, LK_PARAMS)

        # select good points, with standard deviation 1. use numpy index trick
        good_cur = cur_points[stdev==1]
        good_nxt = nxt_points[stdev==1]

        # create img with added tracks for all point pairs on next frame
        img, mask = draw_tracks(frame_2, good_cur, good_nxt, colors, mask, track_thickness=2)

        # add optical flow arrows 
        img = draw_sparse_optical_flow_arrows(img, good_cur, good_nxt, thickness=2, arrow_scale=10.0, color=RED_CV)

        # save image
        _frame_num += 1
        img_name = f'frame_{str(_frame_num).zfill(4)}.jpg'
        img_path = os.path.join(LK_TEMP_FOLDER, img_name)
        cv.imwrite(img_path, img)

        # ready for next iteration
        cur_frame = nxt_frame.copy()
        cur_points = good_nxt.reshape(-1, 1, 2) # -1 indicates to infer that dim size

        # every n seconds (n*FPS frames), get good points
        num_seconds = 1
        if _frame_num % (num_seconds*FPS) == 0:
            cur_points = cv.goodFeaturesToTrack(cur_frame, mask=None, **FEATURE_PARAMS)
            # for every point in good point if its not there in cur points, add , update color too
            

    vid_cap.release()  


if __name__ == "__main__":
    # note :
    # while the game runs press key 's' to toggle screenshot mechanism on/off
    # initially screen saving is set to False

    RUN_SIM     = 1
    RUN_FARN    = 0
    RUN_LK      = 0

    if RUN_SIM:
        # start game simulation
        run_simulation()

        # create the video from saved screenshots
        make_video('vid_out_car.avi', TEMP_FOLDER)

    if RUN_FARN:
        print("Performing farneback flow computation on saved sequence.")
        # # create farneback output 
        run_farneback('vid_out_car.avi')

        # # create the video file
        make_video('farn_vid_out_car.avi', FARN_TEMP_FOLDER)

    if RUN_LK:
        print("Performing lucas-kanade (pyr) flow computation and tracking on saved sequence.")
        # create lucas-kanade output 
        run_lk('vid_out_car.avi')

        # create the video file
        make_video('lk_vid_out_car.avi', LK_TEMP_FOLDER)