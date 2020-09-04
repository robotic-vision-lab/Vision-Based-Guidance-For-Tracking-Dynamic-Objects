import os
import sys
import cv2 as cv
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
cur_path = os.path.abspath(os.path.join('..'))
if cur_path not in sys.path:
    sys.path.append(cur_path)

from game import Game
from game_utils import _prep_temp_folder
from utils.vid_utils import create_video_from_images
from utils.optical_flow_utils import get_OF_color_encoded
from utils.img_utils import convert_to_grayscale
from algorithms.optical_flow import (compute_optical_flow_farneback, 
                                     compute_optical_flow_HS, 
                                     compute_optical_flow_LK)

def run_simulation():
    """Runs the game simulation. 
    Let's us record the frames into the temp folder set in settings.
    """
    # instantiate game 
    car_sim_game = Game()

    # start new game
    car_sim_game.start_new()

    # run
    car_sim_game.run()

def make_video(video_name, folder_path):
    """Looks for frames in temp folder (set in settings module), 
    writes them into a video, with the given name. 
    Also removes the temp folder after creating the video
    """
    if os.path.isdir(folder_path):
        create_video_from_images(folder_path, 'jpg', video_name, FPS)

        # delete temp folder
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

        cur = nxt

    vid_cap.release()


def run_lk(video_name):
    """uses lucas kanade to computer optical flow from video file,
    Also tracks good features, save results in temp folder

    Args:
        video_name (str): name of input video file. eg: 'vid_out_car.avi'
    """
    

if __name__ == "__main__":
    # note :
    # while the game runs press key 's' to toggle screenshot mechanism on/off
    # initially screen saving is set to False

    # start game simulation
    # run_simulation()

    # create the video from saved screenshots
    # make_video('vid_out_car.avi', TEMP_FOLDER)

    # create farneback output 
    run_farneback('vid_out_car.avi')

    # create the video file
    make_video('farn_vid_out_car.avi', FARN_TEMP_FOLDER)