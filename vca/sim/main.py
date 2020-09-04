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
from utils.vid_utils import create_video_from_images
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

def make_video(video_name):
    """Looks for frames in temp folder (set in settings module), 
    writes them into a video, with the given name. 
    Also removes the temp folder after creating the video
    """
    if os.path.isdir(TEMP_FOLDER):
        create_video_from_images(TEMP_FOLDER, 'jpg', video_name, FPS)

        # delete temp folder
        shutil.rmtree(TEMP_FOLDER)

def run_farneback(video_name):
    """uses farneback to compute optical flow of video, returns out video.

    Args:
        video_name (str): name of video file. eg: 'vid_out_car.avi'
    """


if __name__ == "__main__":
    # note :
    # while the game runs press key 's' to toggle screenshot mechanism on/off
    # initially screen saving is set to False

    # start game simulation
    run_simulation()

    # create the video from saved screenshots
    make_video('vid_out_car.avi')

    # create farneback output 