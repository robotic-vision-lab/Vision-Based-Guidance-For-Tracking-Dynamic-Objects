import os
import sys
from copy import deepcopy
from datetime import timedelta

import cv2 as cv
import pygame
import pygame.locals as GAME_GLOBALS
import pygame.event as GAME_EVENTS
import numpy as np

from .high_precision_clock import HighPrecisionClock
from .block import Block
from .bar import Bar
from .car import Car
from .drone_camera import DroneCamera
from .target import Target
from .settings import *


from .my_imports import (load_image_rect,
                        _prep_temp_folder,
                        vec_str,
                        images_assemble,)


class Simulator:
    """
    
    A class to represent a Pygame simulation object
    ...
        

        Attributes:
        -----------
        manager : ExperimentManager
        clock : HighPrecisionClock


        Methods:
        --------

    
    Enables image data capture through simulation using Pygame.
    Simulates orthogonally projected image data capture of a simulated scene, from 
    a (dynamic) camera mounted on a drone. 
    Additionally, simulates kinematics of scene along with appropriate response 
    to control acceleration commands for drone.
    """

    def __init__(self, manager):

        self.manager = manager

        # initialize screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = "2,30"
        pygame.init()

        # set screen size, bg color and display title
        self.SCREEN_SURFACE = pygame.display.set_mode(SCREEN_SIZE, flags=pygame.DOUBLEBUF)
        self.SCREEN_SURFACE.fill(SCREEN_BG_COLOR)
        pygame.display.set_caption(SCREEN_DISPLAY_TITLE)
        pygame.event.set_allowed([GAME_GLOBALS.QUIT, pygame.KEYDOWN])
        self.SCREEN_SURFACE.set_alpha(None)

        # initialize clock
        self.clock = HighPrecisionClock()

        # load image and rect for each car and drone sprite
        self.car_img_rect = load_image_rect(CAR_IMG, colorkey=BLACK, alpha=True, scale=CAR_SCALE)
        self.car_img_rect_2 = load_image_rect(CAR_IMG_2, colorkey=BLACK, alpha=True, scale=CAR_SCALE)
        self.car_img_rect_3 = load_image_rect(CAR_IMG_3, colorkey=BLACK, alpha=True, scale=CAR_SCALE)
        self.drone_img_rect = load_image_rect(DRONE_IMG, colorkey=BLACK, alpha=True, scale=DRONE_SCALE)

        # set screen saving to False
        self.save_screen = False

        self.pause = True
        self.time_font = pygame.font.SysFont(TIME_FONT, 16, False, False)

        self.running = True
        self.tracker_ready = True

        self.alt_change_fac = 1.0
        self.pxm_fac = PIXEL_TO_METERS_FACTOR

        self.time = 0.0
        self.dt = 0.0

    def start_new(self):
        """Initializes simulation components.
        """
        self.time = 0.0

        # initiate screen shot generator
        self.screen_shot = self.screen_saver(path=SIMULATOR_TEMP_FOLDER)

        # create a Group for each type of sprite
        self.all_sprites = pygame.sprite.Group()
        self.drone_sprites = pygame.sprite.Group()
        self.car_sprites = pygame.sprite.Group()
        self.block_sprites = pygame.sprite.Group()
        self.bar_sprites = pygame.sprite.Group()

        # spawn blocks
        self.blocks = []
        for _ in range(NUM_BLOCKS):
            self.blocks.append(Block(self))

        # spawn car
        self.car = Car(self, *CAR_INITIAL_POSITION, *CAR_INITIAL_VELOCITY, *CAR_ACCELERATION, loaded_image_rect=self.car_img_rect)
        self.car_2 = Car(self, *CAR_INITIAL_POSITION_2, *CAR_INITIAL_VELOCITY_2, *CAR_ACCELERATION, loaded_image_rect=self.car_img_rect_2)
        self.car_3 = Car(self, *CAR_INITIAL_POSITION_3, *CAR_INITIAL_VELOCITY_3, *CAR_ACCELERATION, loaded_image_rect=self.car_img_rect_3,traj=LANE_CHANGE_TRAJECTORY)

        #spawn bar
        self.bars = []
        for _ in range(NUM_BARS):
            self.bars.append(Bar(self))

        # spawn drone camera
        self.camera = DroneCamera(self)

        # compensate camera motion on all sprites
        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)

        

    def show_drawing(self):
        """Flip the drawing board to show drawings.
        """
        pygame.display.flip()

    def handle_events(self):
        """Handles captured events.
        """
        # respond to all events posted in the event queue
        for event in GAME_EVENTS.get():
            # QUIT event
            if event.type == GAME_GLOBALS.QUIT or     \
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit()
                break
            
            # KEYDOWN events
            if event.type == pygame.KEYDOWN:
                # K_s => toggle save_screen
                if event.key == pygame.K_s:
                    self.save_screen = not self.save_screen
                    if self.save_screen:
                        print("\nScreen recording started.")
                    else:
                        print("\nScreen recording stopped.")
                # K_SPACE => toggle play/pause
                if event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    if self.pause:
                        print("\nSimulation paused.")
                    else:
                        print("\nSimulation running.")
                # K_i => drone_up
                if event.key == pygame.K_i:
                    self.drone_up()
                # K_k => drone_down
                if event.key == pygame.K_k:
                    self.drone_down()
            


            GAME_EVENTS.pump()

    def update(self):
        """Update positions of components.
        """
        # update Group. (All sprites in it will get updated)
        self.all_sprites.update()

        # compensate camera motion for all sprites
        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)

    def draw(self):
        """Draws components on screen. 
        Note: drone_img is drawn after screen capture for tracking is performed.
        """
        # fill background
        self.SCREEN_SURFACE.fill(SCREEN_BG_COLOR)

        # make and set the window title
        sim_fps = 'NA' if self.dt == 0 else f'{1/self.dt:.2f}'
        pygame.display.set_caption(f'  FPS {sim_fps}')

        # draw only blocks and cars (in that order). Do not draw drone crosshair yet
        self.block_sprites.draw(self.SCREEN_SURFACE)
        self.car_sprites.draw(self.SCREEN_SURFACE)

        # update car image and rect in response to orientation change
        for car_sprite in self.car_sprites:
            car_sprite.update_image_rect()

        # draw bars (after blocks and cars)
        if self.manager.draw_occlusion_bars:
            self.bar_sprites.draw(self.SCREEN_SURFACE)

    def draw_extra(self):
        """Components to be drawn after tracker captures screen, are drawn here.
        """
        # draw simulation time
        time_str = f'Simulation Time - {str(timedelta(seconds=self.time))}'
        time_surf = self.time_font.render(time_str, True, TIME_COLOR)
        time_rect = time_surf.get_rect()
        self.SCREEN_SURFACE.blit(time_surf, (WIDTH - 12 - time_rect.width, HEIGHT - 25))

        # draw bounding box
        if self.pause:
            pygame.draw.rect(self.SCREEN_SURFACE, BB_COLOR, pygame.rect.Rect(*self.manager.targets[0].get_updated_true_bounding_box()), 1)
            pygame.draw.rect(self.SCREEN_SURFACE, BB_COLOR, pygame.rect.Rect(*self.manager.targets[1].get_updated_true_bounding_box()), 1)
            pygame.draw.rect(self.SCREEN_SURFACE, BB_COLOR, pygame.rect.Rect(*self.manager.targets[2].get_updated_true_bounding_box()), 1)


        # draw drone altitude info
        if not CLEAR_TOP:
            alt_str = f'car loc - {self.car.rect.center}, Alt - {self.camera.altitude:0.2f}m, fac - {self.alt_change_fac:0.4f}, pxm - {self.pxm_fac:0.4f}'
            alt_surf = self.time_font.render(alt_str, True, TIME_COLOR)
            self.SCREEN_SURFACE.blit(alt_surf, (15, 15))
            alt_str = f'drone loc - {self.camera.rect.center}, FOV - {WIDTH * self.pxm_fac:0.2f}m x {HEIGHT * self.pxm_fac:0.2f}m'
            alt_surf = self.time_font.render(alt_str, True, TIME_COLOR)
            self.SCREEN_SURFACE.blit(alt_surf, (15, 35))
        
        # draw drone cross hair
        self.drone_sprites.draw(self.SCREEN_SURFACE)


    def screen_saver(self, path):
        """Creates a generator to perform screen saving.

        Args:
            path (str): Path where screen captured frames are to be stored.
        """
        # make sure the folder is there and empty
        _prep_temp_folder(path)

        frame_num = 0
        while True:
            # construct full path string with incremental next image name
            frame_num += 1
            image_name = f'frame_{str(frame_num).zfill(4)}.png'
            file_path = os.path.join(path, image_name)

            # collect screen capture from simulator
            img_sim = self.get_screen_capture(save_mode=True)

            # collect tracker output image
            img_track = self.manager.multi_tracker.cur_img
            if img_track is None:
                img_track = np.ones_like(img_sim, dtype='uint8') * TRACKER_BLANK

            # assemble simulator and tracker images in a grid
            img = images_assemble([img_sim, img_track], (1, 2))
            # img = img_sim

            # write image
            cv.imwrite(file_path, img)
            yield

    def get_screen_capture(self, save_mode=False):
        """Get screen capture from pygame and convert it to return opencv compatible images.
        Args:
            save_mode (bool): Indicates if it's save_mode. In save mode scaling up/down or snr noise won't be done.
        Returns:
            [np.ndarray]: Captured and converted opencv compatible image.
        """
        data = pygame.image.tostring(self.SCREEN_SURFACE, 'RGB')
        img = np.frombuffer(data, np.uint8).reshape(HEIGHT, WIDTH, 3)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if not save_mode:
            if not OPTION == 0:
                img = cv_scale_img(img, SCALE_1)
                img = cv_scale_img(img, SCALE_2)
            if not SNR == 1.0:
                img = add_salt_pepper(img, SNR)
                img = cv.GaussianBlur(img, (5, 5), 0)

        return img

    def drone_up(self):
        """Helper function to implement drone altitude increments.
        Calls camera.fly_higher() to update altitude using fixed alt_change and to update simulator.alt_change_fac.
        load() gets called for all sprties except camera. For car additionally image and rect and loaded again with udpated car_scale
        """
        self.camera.fly_higher()
        self.pxm_fac = ((self.camera.altitude * PIXEL_SIZE) / FOCAL_LENGTH)
        car_scale = (CAR_LENGTH / (CAR_LENGTH_PX * self.pxm_fac)) / self.alt_change_fac

        self.car_img_rect = load_image_rect(CAR_IMG, colorkey=BLACK, alpha=True, scale=car_scale)
        self.car.load()
        
        for block in self.blocks:
            block.load()

        for bar in self.bars:
            bar.load()

    def drone_down(self):
        """Helper function to implement drone altitude decrements.
        Calls camera.fly_higher() to update altitude using fixed alt_change and to update simulator.alt_change_fac.
        load() gets called for all sprties except camera. For car additionally image and rect and loaded again with udpated car_scale
        """
        self.camera.fly_lower()
        self.pxm_fac = ((self.camera.altitude * PIXEL_SIZE) / FOCAL_LENGTH)
        car_scale = (CAR_LENGTH / (CAR_LENGTH_PX * self.pxm_fac)) / self.alt_change_fac

        self.car_img_rect = load_image_rect(CAR_IMG, colorkey=BLACK, alpha=True, scale=car_scale)
        self.car.load()

        for block in self.blocks:
            block.load()
        for bar in self.bars:
            bar.load()

    def get_drone_position(self):
        """Returns drone(UAS) true position

        Returns:
            pygame.Vector2: UAS true position
        """
        return self.camera.position

    def get_camera_fov(self):
        """Helper function, returns drone camera field of view in meters.

        Returns:
            tuple(float32, float32): Drone camera field of view
        """
        return (WIDTH * self.pxm_fac, HEIGHT * self.pxm_fac)

    def can_begin_tracking(self):
        """Indicates the green signal from Simulator for proceeding with tracking
        Simulator gives green signal if
            1. simulator is not paused
            2. bounding box is selected

        Returns:
            bool: Can tracker begin tracking
        """
        return self.tracker_ready

    def quit(self):
        """Helper function, sets running flag to False and quits pygame.
        """
        self.running = False
        pygame.quit()
