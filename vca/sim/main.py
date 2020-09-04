import os
import shutil
import pygame 

from pygame.locals import *
from settings import *

from game import Game


import os
import sys

# add vca\ to sys.path
cur_path = os.path.abspath(os.path.join('..'))
if cur_path not in sys.path:
    sys.path.append(cur_path)

from utils.vid_utils import create_video_from_images
if __name__ == "__main__":
    # note :
    # while the game runs press key 's' to toggle screenshot mechanism on/off
    # initially screen saving is set to False

    # instantiate game 
    car_sim_game = Game()

    # start new game
    car_sim_game.start_new()

    # run
    car_sim_game.run()

    if os.path.isdir(TEMP_FOLDER):
        print("Creating video.")
        create_video_from_images(TEMP_FOLDER, 'jpg', 'vid_out_car.avi', FPS)

        # delete temp folder
        shutil.rmtree(TEMP_FOLDER)

