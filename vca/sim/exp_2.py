import os
import sys
import cv2 as cv
import numpy as np
import shutil
import pygame 
import threading

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

from game_utils import load_image, screen_saver
from utils.vid_utils import create_video_from_images
from utils.optical_flow_utils import (get_OF_color_encoded, 
                                      draw_sparse_optical_flow_arrows,
                                      draw_tracks)
from utils.img_utils import convert_to_grayscale
from algorithms.optical_flow import (compute_optical_flow_farneback, 
                                     compute_optical_flow_HS, 
                                     compute_optical_flow_LK)

from car import Car
from block import Block
from drone_camera import DroneCamera

""" 

There is a game simulation.
Has a car and a green alien (drone with camera).
Car starts from some fixed set position and constant velocity.
Camera (on alien) placed at the center and can be moved by giving acceleration commands.

Idea is to have the car move with certain velocity.
The Car will be drawn on the screen. 
These screens can be captured and stored.
From these screenshot images we figure out the position and velocity of the car.
Give the position and velocity information to the controlling agent.

Agent generates an acceleration command for the alien drone (with the camera).
Game loop takes this into account and accordingly renders updated screens.

In doing so, the goal is for the drone to try and hover over the car
or in other words have the car right in the center of it's view.


"""

class Simulator:
    def __init__(self, manager):
        self.manager = manager
        # initialize screen
        pygame.init()
        self.screen_surface = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption(SCREEN_DISPLAY_TITLE)

        # create clock 
        self.clock = pygame.time.Clock()

        # load sprite images
        self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True, scale=CAR_SCALE)
        self.drone_img = load_image(DRONE_IMG, colorkey=BLACK, alpha=True, scale=DRONE_SCALE)

        # set screen saving to False
        self.save_screen = False

        self.cam_accel_command = pygame.Vector2(0,0)


    def start_new(self):
        # initiate screen shot generator
        self.screen_shot = screen_saver(screen=self.screen_surface, path=TEMP_FOLDER)

        # spawn blocks
        self.blocks = []
        for i in range(NUM_BLOCKS):
            self.blocks.append(Block(self))

        # spawn car
        self.car = Car(self, *CAR_INITIAL_POSITION, *CAR_INITIAL_VELOCITY, *CAR_ACCELERATION)

        # spawn drone camera
        self.camera = DroneCamera(self)
        self.cam_accel_command = pygame.Vector2(0,0)

    
    def run(self):
        self.running = True
        while self.running:
            # make clock tick and measure time elapsed
            self.dt = self.clock.tick(FPS) / 1000.0

            # handle events
            self.handle_events()
            if not self.running:
                break

            # update game objects
            self.update()

            # draw stuffs
            self.draw()

            # save screen
            if self.save_screen:
                next(self.screen_shot)


        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or     \
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.save_screen = not self.save_screen
                    if self.save_screen:
                        print("Screen recording started.")
                    else:
                        print("Screen recording stopped.")

            key_state = pygame.key.get_pressed()
            if key_state[pygame.K_LEFT]:
                pass
            if key_state[pygame.K_RIGHT]:
                pass
            if key_state[pygame.K_UP]:
                pass
            if key_state[pygame.K_DOWN]:
                pass


    def update(self):
        pass


    def draw(self):
        pass

    
    def quit(self):
        self.running = False
        pygame.quit()
        

class Tracker:
    def __init__(self, manager):
        self.manager = manager

class Controller:
    def __init__(self, manager):
        self.manager = manager

class ExperimentManager:
    """
    Experiment:

    - Run the game simulator with car. 
    - Let user select a bounding box for the car to be tracked.
        - 
    The manager is responsible for running the simulator and controller in separate threads.
    The manager can start and stop both applications.
    """
    def __init__(self):
        simulator = Simulator(self)
        tracker = Tracker(self)
        controller = Controller(self)


    def run_simulator(self):
        """
        this method keeps the simulator running 
        """
        pass


    def run_controller(self):
        """
        this method keeps the controller running
        """
        pass