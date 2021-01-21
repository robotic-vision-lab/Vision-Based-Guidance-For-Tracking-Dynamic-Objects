import os
import sys
import ctypes
import random
import shutil
import time
import threading as th
from queue import deque
from copy import deepcopy
from random import randrange
from datetime import timedelta
from math import atan2, degrees, cos, sin, pi, isnan

import numpy as np
import cv2 as cv
import pygame


from pygame.locals import *                                 #pylint: disable=unused-wildcard-import
from settings import *                                      #pylint: disable=unused-wildcard-import
from optical_flow_config import (FARNEBACK_PARAMS,          #pylint: disable=unused-import
                                 FARN_TEMP_FOLDER,
                                 FEATURE_PARAMS,
                                 LK_PARAMS,
                                 LK_TEMP_FOLDER,
                                 MAX_NUM_CORNERS)


# add vca\ to sys.path
vca_path = os.path.abspath(os.path.join('..'))
if vca_path not in sys.path:
    sys.path.append(vca_path)

from utils.vid_utils import create_video_from_images
from utils.optical_flow_utils \
                import (get_OF_color_encoded,               #pylint: disable=unused-import
                        draw_sparse_optical_flow_arrows,
                        draw_tracks)
from utils.img_utils import (convert_to_grayscale,          #pylint: disable=unused-import
                             convert_grayscale_to_BGR,
                             put_text,
                             draw_point,
                             images_assemble,
                             add_salt_pepper)
from utils.img_utils import scale_image as cv_scale_img
from game_utils import (load_image,                         #pylint: disable=unused-import
                        _prep_temp_folder,
                        vec_str,
                        scale_img,
                        ImageDumper)
from algorithms.optical_flow \
                import (compute_optical_flow_farneback,     #pylint: disable=unused-import
                        compute_optical_flow_HS,
                        compute_optical_flow_LK)

from algorithms.feature_detection \
                import (Sift,)

from algorithms.feature_match \
                import (BruteL2,)

from algorithms.template_match \
                import (CorrelationCoeffNormed,
                        TemplateMatcher)


""" Summary:
    Experiment 9:
    In this module we try to complete implementation of the Occlusion Bars.

    Pygame runs a simulation.
    In the simulation, we have 3 kinds of sprites:
        - Blocks
        - Car
        - DroneCamera
    There are multiple Blocks, 1 Car and 1 DroneCamera.
    All sprites have a position, velocity and acceleration.
    The Simulator object can call the update() method on all sprites at a specified FPS clock rate.
    Simulator runs in the main thread, and it's clock runs in parallel to all other processes.

    The Manager object instantiates Simulator, Tracker and Controller.
    Manager can call Simulator's update().

    In this module, Manager runs experiment, calls methods from Simulator, Tracker and Controller.

"""


class Block(pygame.sprite.Sprite):
    """Defines a Block sprite.
    """

    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.car_block_sprites]

        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        self.simulator = simulator
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.w = BLOCK_WIDTH / PIXEL_TO_METERS_FACTOR
        self.h = BLOCK_HEIGHT / PIXEL_TO_METERS_FACTOR
        self.image = pygame.Surface((int(self.w), int(self.h)))
        self.fill_image()

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()

        # self.reset_kinematics()
        _x = randrange(-(WIDTH - self.rect.width), (WIDTH - self.rect.width))
        _y = randrange(-(HEIGHT - self.rect.height), (HEIGHT - self.rect.height))
        self.position = pygame.Vector2(_x, _y) * self.simulator.pxm_fac
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)

        # self.rect.center = self.position
        self.update_rect()

    def reset_kinematics(self):
        """resets the kinematics of block
        """
        # set vectors representing the position, velocity and acceleration
        # note the velocity we assign below will be interpreted as pixels/sec
        fov = self.simulator.get_camera_fov()
        drone_pos = self.simulator.get_drone_position()

        _x = random.uniform(drone_pos[0] - fov[0] / 2, drone_pos[0] + fov[0])
        _y = random.uniform(drone_pos[1] - fov[1] / 2, drone_pos[1] + fov[1])
        self.position = pygame.Vector2(_x, _y)
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position
        self.velocity += self.acceleration * self.simulator.dt
        self.position += self.velocity * self.simulator.dt #+ 0.5 * \
            #self.acceleration * self.simulator.dt**2  # pylint: disable=line-too-long

        # re-spawn in view
        if self.rect.centerx > WIDTH or \
                self.rect.centerx < 0 - self.rect.width or \
                self.rect.centery > HEIGHT or \
                self.rect.centery < 0 - self.rect.height:
            self.reset_kinematics()

    def update_rect(self):
        """Position information is in bottom-left reference frame.
        This method transforms it to top-left reference frame and update the sprite's rect.
        This is for rendering purposes only, to decide where to draw the sprite.
        """

        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)

    def update(self):
        """Overwrites Sprite.update()
            When we call update() on a group this methods gets called.
            Every next frame while running the game loop this will get called
        """
        # for example if we want the sprite to move 5 pixels to the right
        self.update_kinematics()
        # self.update_rect()
        # self.rect.center = self.position

    def fill_image(self):
        """Helper function fills block image
        """
        r, g, b = BLOCK_COLOR
        d = BLOCK_COLOR_DELTA
        r += random.randint(-d, d)
        g += random.randint(-d, d)
        b += random.randint(-d, d)
        self.image.fill((r, g, b))

    def load(self):
        """Helper function updates width and height of image and fills image.
        Also, updates rect.
        """
        self.w /= self.simulator.alt_change_fac
        self.h /= self.simulator.alt_change_fac

        if self.w >= 2 and self.h >= 2:
            self.image = pygame.Surface((int(self.w), int(self.h)))
            self.fill_image()

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()


class Car(pygame.sprite.Sprite):
    """Defines a car sprite.
    """

    def __init__(self, simulator, x, y, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
        # assign itself to the all_sprites group
        self.groups = [simulator.all_sprites, simulator.car_block_sprites]

        # call Sprite initializer with group info
        pygame.sprite.Sprite.__init__(self, self.groups)

        # assign Sprite.image and Sprite.rect attributes for this Sprite
        self.image, self.rect = simulator.car_img

        # set kinematics
        # note the velocity and acceleration we assign below
        # will be interpreted as pixels/sec
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(vx, vy)
        self.acceleration = pygame.Vector2(ax, ay)

        # hold onto the game/simulator reference
        self.simulator = simulator

        # set initial rect location to position
        self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position
        self.velocity += self.acceleration * self.simulator.dt
        self.position += self.velocity * self.simulator.dt #+ 0.5 * \
            # self.acceleration * self.simulator.dt**2  # pylint: disable=line-too-long

    def update_rect(self):
        """update car sprite's rect.
        """
        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)

    def update(self):
        """ update sprite attributes.
            This will get called in game loop for every frame
        """
        self.update_kinematics()
        # self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER

    def load(self):
        """Helper function called when altitude is changed. Updates image and rect.
        """
        self.image, self.rect = self.simulator.car_img
        self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER


class DroneCamera(pygame.sprite.Sprite):
    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.drone_sprite]

        # call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        self.image, self.rect = simulator.drone_img
        self.image.fill((255, 255, 255, DRONE_IMG_ALPHA), None, pygame.BLEND_RGBA_MULT)
        self.reset_kinematics()
        self.origin = self.position
        self.altitude = ALTITUDE
        self.alt_change = 1.0

        # self.rect.center = self.position + SCREEN_CENTER
        self.simulator = simulator
        self.update_rect()

        self.vel_limit = DRONE_VELOCITY_LIMIT
        self.acc_limit = DRONE_ACCELERATION_LIMIT

    def update(self):
        """helper function update kinematics
        """
        self.update_kinematics()
        # self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER

    def update_rect(self):
        """update drone sprite's rect.
        """
        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)

    def reset_kinematics(self):
        """helper function to reset kinematics
        """
        self.position = pygame.Vector2(DRONE_POSITION)
        self.velocity = pygame.Vector2(DRONE_INITIAL_VELOCITY)
        self.acceleration = pygame.Vector2(0, 0)

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # set a drag coefficient
        # COEFF = 0.1
        # print(f'a {self.acceleration}, v {self.velocity}, dt {self.game.dt}')
        # self.acceleration -= self.acceleration * COEFF
        # print(f'a {self.acceleration}, v {self.velocity}')

        # update velocity and position
        self.velocity += self.acceleration * self.simulator.dt
        if abs(self.velocity.length()) > self.vel_limit:
            self.velocity -= self.acceleration * self.simulator.dt

        delta_pos = self.velocity * self.simulator.dt #+ 0.5 * self.acceleration * \
            # self.simulator.dt**2      # i know how this looks like but,   pylint: disable=line-too-long
        self.position = self.velocity * self.simulator.dt #+ 0.5 * self.acceleration * \
            # self.simulator.dt**2  # donot touch ☠                    pylint: disable=line-too-long
        self.origin += delta_pos

    def compensate_camera_motion(self, sprite_obj):
        """Compensates camera motion by updating position of sprite object.

        Args:
            sprite_obj (pygame.sprite.Sprite): Sprite object whose motion needs compensation.
        """
        sprite_obj.position -= self.position
        # sprite_obj.velocity -= self.velocity    # consider investigating for correctness
        sprite_obj.update_rect()

    def change_acceleration(self, command_vec):
        """Changes the drone acceleration appropriately in reponse to given command vector.

        Args:
            command_vec (tuple(float, float)): Command vector tuple. Indicates acceleration change to be made.
        """
        # update acceleration
        COMMAND_SENSITIVITY = 0.1
        command_vec *= COMMAND_SENSITIVITY
        self.acceleration += command_vec

        # counter floating point arithmetic noise
        if abs(self.acceleration[0]) < COMMAND_SENSITIVITY:
            self.acceleration[0] = 0.0
        if abs(self.acceleration[1]) < COMMAND_SENSITIVITY:
            self.acceleration[1] = 0.0

        # make sure acceleration magnitude stays within a set limit
        if abs(self.acceleration.length()) > self.acc_limit:
            self.acceleration -= command_vec

    def convert_px_to_m(self, p):
        """Convert pixels to meters

        Args:
            p (float): Value in pixel units

        Returns:
            float: Value in SI units
        """
        return p * ((self.altitude * PIXEL_SIZE) / FOCAL_LENGTH)

    def convert_m_to_px(self, x):
        """Convert meters to pixel units

        Args:
            x (float): Value in SI units

        Returns:
            float: Value in pixels
        """
        return x / ((self.altitude * PIXEL_SIZE) / FOCAL_LENGTH)

    def fly_higher(self):
        """Helper function to implement drone raise altitude
        """
        self.simulator.alt_change_fac = 1.0 + self.alt_change / self.altitude
        self.altitude += self.alt_change

    def fly_lower(self):
        """Helper function to implement drone lower altitude
        """
        self.simulator.alt_change_fac = 1.0 - self.alt_change / self.altitude
        self.altitude -= self.alt_change


class Bar(pygame.sprite.Sprite):
    """Defines an Occlusion Bar sprite.
    """

    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.bar_sprites]

        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        self.simulator = simulator
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.w = BAR_WIDTH / PIXEL_TO_METERS_FACTOR
        self.h = BAR_HEIGHT / PIXEL_TO_METERS_FACTOR
        self.image = pygame.Surface((int(self.w), int(self.h)))
        self.fill_image()

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()

        # self.reset_kinematics()
        _x = WIDTH + self.rect.centerx
        _x = randrange(-(WIDTH - self.rect.width), (WIDTH - self.rect.width))
        _y = randrange(-(HEIGHT - self.rect.height), (HEIGHT - self.rect.height))
        self.position = pygame.Vector2(_x, _y) * self.simulator.pxm_fac
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)

        # self.rect.center = self.position
        self.update_rect()

    def reset_kinematics(self):
        """resets the kinematics of occlusion bar
        """
        # set vectors representing the position, velocity and acceleration
        # note the velocity we assign below will be interpreted as pixels/sec
        fov = self.simulator.get_camera_fov()
        drone_pos = self.simulator.get_drone_position()

        _x = drone_pos[0] + fov[0] * random.uniform(0.5, 1.0)
        _y = random.uniform(drone_pos[1] - fov[1] / 2, drone_pos[1] + fov[1])
        self.position = pygame.Vector2(_x, _y)
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position
        self.velocity += self.acceleration * self.simulator.dt
        self.position += self.velocity * self.simulator.dt #+ 0.5 * \
            #self.acceleration * self.simulator.dt**2  # pylint: disable=line-too-long

        # re-spawn in view
        if self.rect.centerx > WIDTH or \
                self.rect.centerx < 0 - self.rect.width or \
                self.rect.centery > HEIGHT or \
                self.rect.centery < 0 - self.rect.height:
            self.reset_kinematics()

    def update_rect(self):
        """Position information is in bottom-left reference frame.
        This method transforms it to top-left reference frame and update the sprite's rect.
        This is for rendering purposes only, to decide where to draw the sprite.
        """

        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)

    def update(self):
        """Overwrites Sprite.update()
            When we call update() on a group this methods gets called.
            Every next frame while running the game loop this will get called
        """
        # for example if we want the sprite to move 5 pixels to the right
        self.update_kinematics()
        # self.update_rect()
        # self.rect.center = self.position

    def fill_image(self):
        """Helper function fills block image
        """
        r, g, b = BAR_COLOR
        d = BAR_COLOR_DELTA
        r += random.randint(-d[0], d[0])
        g += random.randint(-d[1], d[1])
        b += random.randint(-d[2], d[2])
        self.image.fill((r, g, b))

    def load(self):
        """Helper function updates width and height of image and fills image.
        Also, updates rect.
        """
        self.w /= self.simulator.alt_change_fac
        self.h /= self.simulator.alt_change_fac

        if self.w >= 2 and self.h >= 2:
            self.image = pygame.Surface((int(self.w), int(self.h)))
            self.fill_image()

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()


class HighPrecisionClock:
    """High precision clock for time resolution in microseconds
    """

    def __init__(self):
        self.micro_timestamp = self.micros()

    def tick(self, framerate):
        """Implements appropriate delay given a framerate.

        Args:
            framerate (float): Desired framerate

        Returns:
            float: time elapsed
        """
        self.delay_microseconds(1000000 // framerate )

        _new_micro_ts = self.micros()
        self.time_diff = _new_micro_ts - self.micro_timestamp
        self.micro_timestamp = _new_micro_ts

        return self.time_diff

    @staticmethod
    def micros():
        """return timestamp in microseconds"""
        tics = ctypes.c_int64()
        freq = ctypes.c_int64()

        # get ticks on the internal ~3.2GHz QPC clock
        ctypes.windll.Kernel32.QueryPerformanceCounter(ctypes.byref(tics))
        # get the actual freq. of the internal ~3.2GHz QPC clock
        ctypes.windll.Kernel32.QueryPerformanceFrequency(ctypes.byref(freq))

        t_us = tics.value * 1e6 / freq.value
        return t_us

    def delay_microseconds(self, delay_us):
        """delay for delay_us microseconds (us)"""
        t_start = self.micros()
        while (self.micros() - t_start < delay_us):
            pass    # do nothing
        return


class Simulator:
    """Simulator object creates the simulation game.
    Responds to keypresses 'SPACE' to toggle play/pause, 's' to save screen mode, ESC to quit.
    While running simulation, it also dumps the screens to a shared memory location.
    Designed to work with an ExperimentManager object.
    Computer Graphics techniques are employed here.
    """

    def __init__(self, manager):

        self.manager = manager

        # initialize screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = "2,30"
        pygame.init()
        self.screen_surface = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption(SCREEN_DISPLAY_TITLE)

        # create clock
        # self.clock = pygame.time.Clock()
        self.clock = HighPrecisionClock()

        # load sprite images
        self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True, scale=CAR_SCALE)
        self.drone_img = load_image(DRONE_IMG, colorkey=BLACK, alpha=True, scale=DRONE_SCALE)

        # set screen saving to False
        self.save_screen = False

        self.cam_accel_command = pygame.Vector2(0, 0)
        self.euc_factor = 1.0
        self.pause = True
        self.time_font = pygame.font.SysFont(TIME_FONT, 16, False, False)
        self.bb_start = None
        self.bb_end = None
        self.bb_drag = False
        self.bounding_box_drawn = False
        self.running = True
        self.tracker_ready = False

        self.alt_change_fac = 1.0
        self.pxm_fac = PIXEL_TO_METERS_FACTOR

        self.car_rect_center_bb_offset = [0,0]

    def start_new(self):
        """Initializes simulation components.
        """
        self.time = 0.0

        # initiate screen shot generator
        self.screen_shot = self.screen_saver(path=SIMULATOR_TEMP_FOLDER)

        # create default Group for all sprites, but drone
        self.all_sprites = pygame.sprite.Group()
        self.drone_sprite = pygame.sprite.Group()
        self.car_block_sprites = pygame.sprite.Group()
        self.bar_sprites = pygame.sprite.Group()

        # spawn blocks
        self.blocks = []
        for _ in range(NUM_BLOCKS):
            self.blocks.append(Block(self))

        # spawn car
        self.car = Car(self, *CAR_INITIAL_POSITION, *CAR_INITIAL_VELOCITY, *CAR_ACCELERATION)

        #spawn bar
        self.bars = []
        for _ in range(NUM_BARS):
            self.bars.append(Bar(self))

        # spawn drone camera
        self.camera = DroneCamera(self)
        self.cam_accel_command = pygame.Vector2(0, 0)
        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)

    def run(self):
        """Keeps simulation game running until quit.
        """
        self.running = True
        while self.running:
            # make clock tick and measure time elapsed
            self.dt = self.clock.tick(FPS) / 1000.0
            if not self.manager.use_real_clock:
                self.dt = DELTA_TIME
            if self.pause:          # DO NOT TOUCH! CLOCK MUST TICK REGARDLESS!
                self.dt = 0.0
            self.time += self.dt

            # handle events
            self.handle_events()
            if not self.running:
                break

            if not self.pause:
                # update game objects
                self.update()
                # print stuffs
                # print(f'SSSS >> {str(timedelta(seconds=self.time))} >> DRONE - x:{vec_str(self.camera.rect.center)} | v:{vec_str(self.camera.velocity)} | a:{vec_str(self.camera.acceleration)} | a_comm:{vec_str(self.cam_accel_command)} | CAR - x:{vec_str(self.car.rect.center)}, v: {vec_str(self.car.velocity)},  v_c-v_d: {vec_str(self.car.velocity - self.camera.velocity)}              ', end='\n')
                if not CLEAN_CONSOLE:
                    print(f'SSSS >> {str(timedelta(seconds=self.time))} >> DRONE - x:{vec_str(self.camera.position)} | v:{vec_str(self.camera.velocity)} | CAR - x:{vec_str(self.car.position)}, v: {vec_str(self.car.velocity)} | COMMANDED a:{vec_str(self.camera.acceleration)} | a_comm:{vec_str(self.cam_accel_command)} | rel_car_pos: {vec_str(self.car.position - self.camera.position)}', end='\n')
                # self.manager.true_rel_vel = self.car.velocity - self.camera.velocity

            # draw stuffs
            self.draw()

            if not self.pause and self.manager.tracker_on:
                # put the screen capture into image_deque
                self.put_image()

            # draw extra parts like drone cross hair, simulation time, bounding box etc
            self.draw_extra()

            # show drawing board
            self.show_drawing()
            # pygame.display.flip()

            # save screen
            if self.save_screen:
                next(self.screen_shot)

    def show_drawing(self):
        """Flip the drawing board to show drawings.
        """
        pygame.display.flip()

    def handle_events(self):
        """Handles captured events.
        """
        # respond to all events posted in the event queue
        for event in pygame.event.get():
            # QUIT event
            if event.type == pygame.QUIT or     \
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
                        self.bb_start = self.bb_end = None
                        print("\nSimulation paused.")
                    else:
                        print("\nSimulation running.")
                # K_i => drone_up
                if event.key == pygame.K_i:
                    self.drone_up()
                # K_k => drone_down
                if event.key == pygame.K_k:
                    self.drone_down()
            
            # mutliple keypresses for manual acceleration control
            key_state = pygame.key.get_pressed()
            if key_state[pygame.K_LEFT]:
                self.cam_accel_command.x = -1
            if key_state[pygame.K_RIGHT]:
                self.cam_accel_command.x = 1
            if key_state[pygame.K_UP]:
                self.cam_accel_command.y = -1
            if key_state[pygame.K_DOWN]:
                self.cam_accel_command.y = 1

            self.euc_factor = 0.7071 if abs(self.cam_accel_command.elementwise()) == (1, 1) else 1.0

            # capture bounding box input 
            # Bounding box will be input in the following manner
            #   1. Simulator will be paused
            #   2. MOUSEBUTTONDOWN event (triggers bb corner (start and end) points collection)
            #   3. MOUSEMOTION event (triggers instantaneous end point update while mouse drag)
            #   4. MOUSEBUTTONUP event (triggers end of drag along with final end point update)
            # At step 4, we have a bounding box
            # Bounding Box input event handling will be statefully managed
            if self.pause and event.type == pygame.MOUSEBUTTONDOWN:
                self.bb_start = self.bb_end = pygame.mouse.get_pos()
                self.bb_drag = True
            if event.type == pygame.MOUSEMOTION and self.bb_drag:
                self.bb_end = pygame.mouse.get_pos()

            if event.type == pygame.MOUSEBUTTONUP:
                self.bb_end = pygame.mouse.get_pos()
                self.bb_drag = False
                # at this point bounding box is assumed to be drawn
                self.bounding_box_drawn = True
                # assume appropriate bounding box was inputted and indicate green flag for tracker
                self.tracker_ready = True
                # set car rect center offset from bounding box topleft
                self.car_rect_center_bb_offset[0] = self.bb_start[0] - self.car.rect.centerx 
                self.car_rect_center_bb_offset[1] = self.bb_start[1] - self.car.rect.centery

            pygame.event.pump()

        # respond to the command posted by controller
        # if len(self.manager.command_deque) > 0:
        #     self.camera.acceleration = self.manager.get_from_command_deque()

    def update(self):
        """Update positions of components.
        """
        # update drone acceleration using acceleration command (force)
        if not self.manager.control_on:
            self.camera.change_acceleration(deepcopy(self.euc_factor * self.cam_accel_command))
            self.cam_accel_command = pygame.Vector2(0, 0)

        # update Group. (All sprites in it will get updated)
        self.all_sprites.update()

        # compensate camera motion for all sprites
        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)

    def draw(self):
        """Draws components on screen. Note: drone_img is drawn after screen capture for tracking is performed.
        """
        # fill background
        self.screen_surface.fill(SCREEN_BG_COLOR)

        # make title
        sim_fps = 'NA' if self.dt == 0 else f'{1/self.dt:.2f}'
        pygame.display.set_caption(
            f'  FPS {sim_fps} | car: x-{vec_str(self.car.position)} v-{vec_str(self.car.velocity)} a-{vec_str(self.car.acceleration)} | cam x-{vec_str(self.camera.position)} v-{vec_str(self.camera.velocity)} a-{vec_str(self.camera.acceleration)} ')

        # draw only car and blocks (not drone)
        self.car_block_sprites.draw(self.screen_surface)

        # draw bars
        if self.manager.draw_occlusion_bars:
            self.bar_sprites.draw(self.screen_surface)

    def draw_extra(self):
        """Components to be drawn after screen capture for tracking/controllers is performed.
        """
        # draw drone cross hair
        self.drone_sprite.draw(self.screen_surface)

        # draw simulation time
        time_str = f'Simulation Time - {str(timedelta(seconds=self.time))}'
        time_surf = self.time_font.render(time_str, True, TIME_COLOR)
        time_rect = time_surf.get_rect()
        self.screen_surface.blit(time_surf, (WIDTH - 12 - time_rect.width, HEIGHT - 25))

        # draw bounding box
        if self.bb_start and self.bb_end and self.pause:
            x = min(self.bb_start[0], self.bb_end[0])
            y = min(self.bb_start[1], self.bb_end[1])
            w = abs(self.bb_start[0] - self.bb_end[0])
            h = abs(self.bb_start[1] - self.bb_end[1])
            self.bounding_box = (x, y, w, h)
            pygame.draw.rect(self.screen_surface, BB_COLOR, pygame.rect.Rect(x, y, w, h), 2)

        if not CLEAR_TOP:
            # draw drone altitude info
            alt_str = f'car loc - {self.car.rect.center}, Alt - {self.camera.altitude:0.2f}m, fac - {self.alt_change_fac:0.4f}, pxm - {self.pxm_fac:0.4f}'
            alt_surf = self.time_font.render(alt_str, True, TIME_COLOR)
            alt_rect = alt_surf.get_rect()
            self.screen_surface.blit(alt_surf, (15, 15))
            alt_str = f'drone loc - {self.camera.rect.center}, FOV - {WIDTH * self.pxm_fac:0.2f}m x {HEIGHT * self.pxm_fac:0.2f}m'
            alt_surf = self.time_font.render(alt_str, True, TIME_COLOR)
            alt_rect = alt_surf.get_rect()
            self.screen_surface.blit(alt_surf, (15, 35))

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
            img_track = self.manager.tracker.cur_img
            if img_track is None:
                img_track = np.ones_like(img_sim, dtype='uint8') * TRACKER_BLANK

            # assemble simulator and tracker images in a grid
            img = images_assemble([img_sim, img_track], (1, 2))

            # write image
            cv.imwrite(file_path, img)
            yield

    def get_screen_capture(self, save_mode=False):
        """Get screen capture from pygame and convert it to return opencv compatible images.

        Returns:
            [np.ndarray]: Captured and converted opencv compatible image.
        """
        data = pygame.image.tostring(self.screen_surface, 'RGB')
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

    def put_image(self):
        """Helper function, captures screen and adds to manager's image deque.
        """
        img = self.get_screen_capture()
        self.manager.add_to_image_deque(img)

    def drone_up(self):
        """Helper function to implement drone altitude increments.
        """
        self.camera.fly_higher()
        self.pxm_fac = ((self.camera.altitude * PIXEL_SIZE) / FOCAL_LENGTH)
        car_scale = (CAR_LENGTH / (CAR_LENGTH_PX * self.pxm_fac)) / self.alt_change_fac
        self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True, scale=car_scale)
        self.car.load()
        for block in self.blocks:
            block.load()

    def drone_down(self):
        """Helper function to implement drone altitude decrements.
        """
        self.camera.fly_lower()
        self.pxm_fac = ((self.camera.altitude * PIXEL_SIZE) / FOCAL_LENGTH)
        car_scale = (CAR_LENGTH / (CAR_LENGTH_PX * self.pxm_fac)) / self.alt_change_fac
        self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True, scale=car_scale)
        self.car.load()
        for block in self.blocks:
            block.load()

    def get_drone_position(self):
        """Returns drone(UAS) true position

        Returns:
            pygame.Vector2: UAS true position
        """
        return self.camera.position

    def get_camera_fov(self):
        """Helper function, returns drone camera field of view.

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
        # ready = True

        # not ready if bb not selected or if simulated is still paused
        if (self.bb_start and self.bb_end) or not self.pause:
            # self.manager.image_deque.clear()
            ready = True

        return self.tracker_ready

    def quit(self):
        """Helper function, sets running flag to False and quits pygame.
        """
        self.running = False
        pygame.quit()


class Tracker:
    """Tracker object is designed to work with and ExperimentManager object.
    It can be used to process screen captures and produce tracking information for feature points.
    Computer Vision techniques employed here.
    """

    def __init__(self, manager):

        self.manager = manager

        self.frame_old_gray = None
        self.frame_old_color = None
        self.frame_new_gray = None
        self.frame_new_color = None
        
        self.keypoints_old = None
        self.keypoints_old_good = None
        self.keypoints_old_bad = None
        self.keypoints_new = None
        self.keypoints_new_good = None
        self.keypoints_new_bad = None
        self.rel_keypoints = None

        self.centroid_old = None
        self.centroid_new = None
        self.centroid_adjustment = None

        self.initial_keypoints = None
        self.initial_centroid = None

        self.frame_new_color_edited = None
        self.img_tracker_display = None
        
        self.feature_found_statuses = None
        self.cross_feature_errors = None

        # desc are computed at keypoints, detected ones are all desc inside bounding box
        self.initial_target_descriptors = None
        self.initial_detected_target_descriptors = None
        self.initial_target_template_gray = None
        self.initial_target_template_color = None
        self.target_bounding_box = None
        self.patch_size = 15

        self.detector = Sift()
        self.descriptor_matcher = BruteL2()
        self.template_matcher = CorrelationCoeffNormed()

        self.true_old_pt = None
        self.true_new_pt = None

        self.cur_img = None

        self._can_begin_control_flag = False    # will be modified in process_image
        self.kin = None
        self.window_size = 5
        # self.prev_car_pos = None
        # self.count = 0
        # self._target_old_occluded_flag = False
        # self._target_new_occluded_flag = False

        self._NO_OCC = 0
        self._PARTIAL_OCC = 1
        self._TOTAL_OCC = 2
        self.display_arrow_color = {self._NO_OCC:GREEN_CV, self._PARTIAL_OCC:ORANGE_PEEL_BGR, self._TOTAL_OCC:TOMATO_BGR}

        self.target_occlusion_case_old = None
        self.target_occlusion_case_new = self._NO_OCC   # assumption: start with no_occ

        self._frame_num = 0
        self.track_length = 10
        self.tracker_info_mask = None
        self.target_bounding_box_mask = None
        self.win_name = 'Tracking in progress'
        self.img_dumper = ImageDumper(TRACKER_TEMP_FOLDER)
        self.DES_MATCH_DISTANCE_THRESH = 250 #450
        self.DES_MATCH_DEV_THRESH = 0.50 # float('inf') to get every match
        self.TEMP_MATCH_THRESH = 0.9849

        self._FAILURE = False, None
        self._SUCCESS = True, self.kin
        self.MAX_ERR = 15

    def is_first_time(self):
        """Indicates if tracker never received a frame for the first time

        Returns:
            bool: Boolean indicating first time
        """
        # this function is called in process_image in the beginning. 
        # it indicates if this the first time process_image received a frame
        return self.frame_old_gray is None

    def can_begin_control(self):
        """Returns boolean check indicating if controller can be used post tracking.

        Returns:
            bool: Indicator for controller begin doing it's thing
        """
        return self._can_begin_control_flag  # and self.prev_car_pos is not None

    def save_initial_target_descriptors(self):
        """Helper function used after feature keypoints and centroid computation. Saves initial target descriptors.
        """
        # use keypoints from new frame, 
        # save descriptors of new keypoints(good)
        keyPoints = [cv.KeyPoint(*kp.ravel(), 15) for kp in self.initial_keypoints]
        self.initial_kps, self.initial_target_descriptors = self.detector.get_descriptors_at_keypoints(
                                                                self.frame_new_gray, 
                                                                keyPoints,
                                                                self.target_bounding_box)

    def save_initial_target_template(self):
        """Helper function used after feature keypoints and centroid computation. Saves initial target template.
        """
        # use the bounding box location to save the target template
        # x, y, w, h = bb = self.manager.get_target_bounding_box()
        # center = tuple(map(int, (x+w/2, y+h/2)))

        self.initial_target_template_color = self.get_bb_patch_from_image(self.frame_new_color, self.target_bounding_box)
        self.initial_target_template_gray = self.get_bb_patch_from_image(self.frame_new_gray, self.target_bounding_box)

    def save_initial_patches(self):
        """Helper function used after feature keypoints and centroid computation. Saves initial patches around keypoints.
        Also, initializes dedicated template matchers
        """
        self.initial_patches_color = [self.get_neighborhood_patch(self.frame_new_color, tuple(map(int,kp.flatten())), self.patch_size) for kp in self.initial_keypoints]
        self.initial_patches_gray = [self.get_neighborhood_patch(self.frame_new_gray, tuple(map(int,kp.flatten())), self.patch_size) for kp in self.initial_keypoints]
        
        # initialize template matcher object for each patch
        self.template_matchers = [TemplateMatcher(patch, self.template_matcher) for patch in self.initial_patches_gray]

    def augment_old_frame(self):
        # keypoints that were not found in new frame would get discounted by the next iteration
        # these bad points from old frame can be reconstructed in new frame
        # and then corresponding patches cna be drawn in new frame after the flow computation
        # then save to old
        pass

    def find_saved_patches_in_img_bb(self, img, bb):
        """Uses patch template matchers to locate patches in given image inside given bounding box.

        Args:
            img (numpy.ndarray): Image in which patches are to be found.
            bb (tuple): Bounding box inside which template matching is to be performed.

        Returns:
            tuple: Best matched template locations, best match values
        """
        self.template_points = np.array([
            temp_matcher.find_template_center_in_image_bb(img, bb)
            for temp_matcher in self.template_matchers
            ]).reshape(-1, 1, 2)

        self.template_scores = np.array([
            temp_matcher.get_best_match_score()
            for temp_matcher in self.template_matchers
            ]).reshape(-1, 1)

        return self.template_points, self.template_scores

    def get_relative_associated_patch(self, keypoint, centroid):
        # make sure shape is consistent
        keypoint = keypoint.reshape(-1, 1, 2)
        centroid = centroid.reshape(-1, 1, 2)
        rel = keypoint - centroid

        # find index of closest relative keypoints
        index = ((self.rel_keypoints - rel)**2).sum(axis=2).argmin()

        # return corresponding patch
        return self.initial_patches_gray[index]

    def put_patch_at_point(self, img, patch, point):
        """Stick a given patch onto given image at given point

        Args:
            img (numpy.ndarray): Image onto which we want to put a patch
            patch (numpy.ndarray): Patch that we want to stick on the image 
            point (tuple): Point at which patch center would go

        Returns:
            numpy.ndarray: Image after patch is put with it's center aligned with the given point
        """
        # assumption: patch size is fixed by tracker
        x_1 = point[0] - self.patch_size//2
        y_1 = point[1] - self.patch_size//2
        x_2 = x_1 + self.patch_size
        y_2 = y_1 + self.patch_size
        img[y_1:y_2, x_1:x_2] = patch

        return img

    @staticmethod
    def get_bb_patch_from_image(img, bounding_box):
        """Returns a patch from image using bounding box

        Args:
            img (numpy.ndarray): Image from which patch is a to be drawn
            bounding_box (tuple): Bounding box surrounding the image patch of interest

        Returns:
            numpy.ndarray: The image patch 
        """
        x, y, w, h = bounding_box
        return img[y:y+h, x:x+w] # same for color or gray

    @staticmethod
    def get_neighborhood_patch(img, center, size):
        """Returns a patch from image using a center point and size

        Args:
            img (numpy.ndarray): Image from which a patch is to be drawn
            center (tuple): Point in image at which patch center would align
            size (int): Size of patch 

        Returns:
            numpy.ndaray: Patch
        """
        size = (size, size) if isinstance(size, int) else size
        x = center[0] - size[0]//2
        y = center[1] - size[1]//2
        w, h = size

        return img[y:y+h, x:x+w] # same for color or gray

    @staticmethod
    def get_patch_mask(img, patch_center, patch_size):
        """Returns a mask, given a patch center and size

        Args:
            img (numpy.ndarray): Image using which mask is to be created
            patch_center (tuple): Center location of patch
            patch_size (tuple): Size of patch

        Returns:
            numpy.ndarray: Mask
        """
        x = patch_center[0] - patch_size //2
        y = patch_center[1] - patch_size //2
        mask = np.zeros_like(img)
        mask[y:y+patch_size[1], x:x+patch_size[0]] = 255
        return mask

    @staticmethod
    def get_bounding_box_mask(img, x, y, width, height):
        """Returns mask, using bounding box

        Args:
            img (numpy.ndarray): Image using which mask is to be created
            x (int): x coord of top left of bounding box
            y (int): y coord of top left of bounding box
            width (int): width of bounding box
            height (int): height of bounding box

        Returns:
            numpt.ndarray: mask
        """
        # assume image is grayscale
        mask = np.zeros_like(img)
        mask[y:y+height, x:x+width] = 255
        return mask

    @staticmethod
    def get_centroid(points):
        """Returns centroid of given list of points

        Args:
            points (np.ndarray): Centroid point. [shape: (1,2)]
        """
        points_ = np.array(points).reshape(-1, 1, 2)
        return np.mean(points_, axis=0)

    def get_feature_keypoints_from_mask(self, img, mask, bb=None):
        """Returns feature keypoints compute in given image using given mask

        Args:
            img (numpy.ndarray): Image in which feature keypoints are to be computed
            mask (numpy.ndarray): Mask indicating selected region

        Returns:
            numpy.ndarray: Feature keypoints
        """
        shi_tomasi_kpts = cv.goodFeaturesToTrack(img, mask=mask, **FEATURE_PARAMS)
        detector_kpts = self.detector.get_keypoints(img, mask, bb)
        detector_kpts = np.array([pt.pt for pt in detector_kpts]).astype(np.float32).reshape(-1, 1, 2)
        if shi_tomasi_kpts is None and detector_kpts is None:
            return None

        if shi_tomasi_kpts is None and detector_kpts is not None:
            return detector_kpts
        
        if shi_tomasi_kpts is not None and detector_kpts is None:
            return shi_tomasi_kpts
        
        comb_kpts = np.concatenate((shi_tomasi_kpts, detector_kpts), axis=0)
        return comb_kpts

    def get_descriptors_at_keypoints(self, img, keypoints, bb=None):
        """Returns computed descriptors at keypoints

        Args:
            img (numpy.ndarray): Image
            keypoints (numpy.ndarray): Keypoints of interest

        Returns:
            list: List of descriptors corresponding to given keypoints.
        """
        kps = [cv.KeyPoint(*kp.ravel(), 15) for kp in keypoints]
        kps, descriptors = self.detector.get_descriptors_at_keypoints(self.frame_new_gray, kps, bb)
        return kps, descriptors

    def get_true_bb_from_oracle(self):
        """Helper function to get the true target bounding box.

        Returns:
            tuple: True target bounding box
        """
        return self.manager.get_target_bounding_box_from_offset()

    def _get_kin_from_manager(self):
        """Helper function to fetch appropriate kinematics from manager

        Returns:
            tuple: Kinematics tuple
        """
        #TODO switch based true or est 
        return self.manager.get_true_kinematics()

    def _get_target_image_location(self):
        """Helper function returns true target location measured in pixels

        Returns:
            [type]: [description]
        """
        kin = self._get_kin_from_manager()
        x,y = kin[2].elementwise()* (1,-1) / self.manager.simulator.pxm_fac
        target_location = (int(x), int(y) + HEIGHT)
        target_location += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)
        return target_location

    def process_image_complete(self, new_frame):
        """Processes new frame and performs and delegated various tracking based tasks.
            1. Extracts target attributes and stores them
            2. Processes each next frame and tracks target
            3. Delegates kinematics computation
            4. Handles occlusions

        Args:
            new_frame (numpy.ndarray): Next frame to be processed

        Returns:
            tuple: Sentinel tuple consisting indicator of process success/failure and kinematics if computed successfully.
        """
        # save new frame, compute grayscale
        self.frame_new_color = new_frame
        self.frame_new_gray = convert_to_grayscale(self.frame_new_color)
        self.true_new_pt = self._get_target_image_location()
        cv.imshow('nxt_frame', self.frame_new_gray); cv.waitKey(1)
        if self.is_first_time():
            # compute bb
            self.target_bounding_box = self.manager.get_target_bounding_box()
            self.target_bounding_box_mask = self.get_bounding_box_mask(self.frame_new_gray, *self.target_bounding_box)

            # compute initial feature keypoints and centroid
            self.initial_keypoints = cv.goodFeaturesToTrack(self.frame_new_gray, mask=self.target_bounding_box_mask, **FEATURE_PARAMS)
            self.initial_centroid = self.get_centroid(self.initial_keypoints)
            self.rel_keypoints = self.initial_keypoints - self.initial_centroid

            # compute and save descriptors at keypoints, save target template
            self.save_initial_target_descriptors()
            self.save_initial_target_template()
            self.save_initial_patches()

            # posterity - save frames, keypoints, centroid, occ_case, centroid location relative to rect center
            self.frame_old_gray = self.frame_new_gray
            self.frame_old_color = self.frame_new_color
            self.keypoints_old = self.keypoints_new = self.initial_keypoints
            # self.keypoints_old_good = self.keypoints_new_good # not needed, since next iter will have from_no_occ
            self.centroid_old = self.centroid_new = self.initial_centroid
            self.target_occlusion_case_old = self.target_occlusion_case_new
            self.manager.set_target_centroid_offset()
            self.true_old_pt = self.true_new_pt
            return self._FAILURE

        cv.imshow('cur_frame', self.frame_old_gray); cv.waitKey(1)
        self._can_begin_control_flag = True
        
        # ################################################################################
        # CASE |NO_OCC, _>
        if self.target_occlusion_case_old == self._NO_OCC:
            # (a priori) older could have been start or no_occ, or partial_occ or total_occ
            # we should have all keypoints as good ones, and old centroid exists, if old now has no_occ

            # try to compute flow at keypoints and infer next occlusion case
            self.compute_flow()

            # amplify bad errors
            self.cross_feature_errors[(self.cross_feature_errors > 0.75 * self.MAX_ERR) & (self.cross_feature_errors > 100*self.cross_feature_errors.min())] *= 10

            # ---------------------------------------------------------------------
            # |NO_OCC, NO_OCC>
            if (self.feature_found_statuses.all() and self.feature_found_statuses.shape[0] == MAX_NUM_CORNERS and 
                    self.cross_feature_errors.max() < self.MAX_ERR):
                self.target_occlusion_case_new = self._NO_OCC

                # set good points (since no keypoints were occluded all are good, no need to compute)
                self.keypoints_new_good = self.keypoints_new
                self.keypoints_old_good = self.keypoints_old

                # compute centroid
                self.centroid_new = self.get_centroid(self.keypoints_new_good)

                # compute kinematics measurements using centroid
                self.kin = self.compute_kinematics_by_centroid(self.centroid_old, self.centroid_new)

                # update tracker display
                self.display()

                # posterity - save frames, keypoints, centroid
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.keypoints_old = self.keypoints_new
                self.keypoints_old_good = self.keypoints_new_good
                self.centroid_adjustment = None
                self.centroid_old = self.centroid_new
                self.target_occlusion_case_old = self.target_occlusion_case_new
                return self._SUCCESS

            # ---------------------------------------------------------------------
            # |NO_OCC, TOTAL_OCC>
            elif not self.feature_found_statuses.all() or self.cross_feature_errors.min() >= self.MAX_ERR:
                self.target_occlusion_case_new = self._TOTAL_OCC

                # cannot compute kinematics
                self.kin = None

                # update tracker display
                self.display()

                # posterity
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.centroid_adjustment = None
                self.target_occlusion_case_old = self.target_occlusion_case_new
                return self._FAILURE

            # ---------------------------------------------------------------------
            # |NO_OCC, PARTIAL_OCC>
            else:
                self.target_occlusion_case_new = self._PARTIAL_OCC

                # in this case of from no_occ to partial_occ, no more keypoints are needed to be found
                # good keypoints need to be computed for kinematics computation as well as posterity
                self.keypoints_new_good = self.keypoints_new[(self.feature_found_statuses==1) & (self.cross_feature_errors < self.MAX_ERR)].reshape(-1, 1, 2)
                self.keypoints_old_good = self.keypoints_old[(self.feature_found_statuses==1) & (self.cross_feature_errors < self.MAX_ERR)].reshape(-1, 1, 2)

                # compute adjusted centroid
                centroid_old_good = self.get_centroid(self.keypoints_old_good)
                centroid_new_good = self.get_centroid(self.keypoints_new_good)
                self.centroid_adjustment = self.centroid_old - centroid_old_good
                self.centroid_new = centroid_new_good + self.centroid_adjustment

                # compute kinematics measurements
                self.kin = self.compute_kinematics_by_centroid(self.centroid_old, self.centroid_new)

                # adjust missing old keypoints (no need to check recovery)
                keypoints_missing = self.keypoints_old[(self.feature_found_statuses==0) | (self.cross_feature_errors >= self.MAX_ERR)]
                self.keypoints_new_bad = keypoints_missing - self.centroid_old + self.centroid_new

                # put patches over bad points in new frame
                for kp in self.keypoints_new_bad:
                    # fetch appropriate patch
                    patch = self.get_relative_associated_patch(kp, self.centroid_new)
                    # paste patch at appropriate location
                    self.put_patch_at_point(self.frame_new_gray, patch, tuple(map(int,kp.flatten())))

                # update tracker display
                self.display()

                # add revived bad points to good points
                self.keypoints_new_good = np.concatenate((self.keypoints_new_good, self.keypoints_new_bad.reshape(-1, 1, 2)), axis=0)

                # posterity
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.keypoints_old = self.keypoints_new
                self.keypoints_old_good = self.keypoints_new_good.reshape(-1, 1, 2)
                self.keypoints_old_bad = self.keypoints_new_bad.reshape(-1, 1, 2)
                self.centroid_old = self.centroid_new
                self.centroid_old_true = self.manager.get_target_centroid()
                self.target_occlusion_case_old = self.target_occlusion_case_new
                return self._SUCCESS

        # ################################################################################
        # CASE |PARTIAL_OCC, _>
        if self.target_occlusion_case_old == self._PARTIAL_OCC:
            # (a priori) older could have been no_occ, partial_occ or total_occ
            # we should have some keypoints that are good, if old now has partial_occ
            
            # use good keypoints, to compute flow (keypoints_old_good used as input, and outputs into keypoints_new)
            self.compute_flow(use_good=True)

            # amplify bad errors 
            if self.cross_feature_errors.shape[0] > 0.5 * MAX_NUM_CORNERS:
                self.cross_feature_errors[(self.cross_feature_errors > 0.75 * self.MAX_ERR) & (self.cross_feature_errors > 100*self.cross_feature_errors.min())] *= 10
            
            # update good old and new
            # good keypoints are used for kinematics computation as well as posterity
            self.keypoints_new_good = self.keypoints_new[(self.feature_found_statuses==1) & (self.cross_feature_errors < self.MAX_ERR)].reshape(-1, 1, 2)
            self.keypoints_old_good = self.keypoints_old[(self.feature_found_statuses==1) & (self.cross_feature_errors < self.MAX_ERR)].reshape(-1, 1, 2)

            # these good keypoints may not be sufficient for precise partial occlusion detection
            # for precision, we will need to check if any other keypoints can be recovered or reconstructed
            # we should still have old centroid
            self.target_bounding_box = self.get_true_bb_from_oracle()
            self.target_bounding_box_mask = self.get_bounding_box_mask(self.frame_new_gray, *self.target_bounding_box)

            # perform template matching for patches to update template points and scores
            self.find_saved_patches_in_img_bb(self.frame_new_gray, self.target_bounding_box)

            # compute good feature keypoints in the new frame (shi-tomasi + SIFT)
            good_keypoints_new = self.get_feature_keypoints_from_mask(self.frame_new_gray, mask=self.target_bounding_box_mask, bb=self.target_bounding_box)

            if good_keypoints_new is None or good_keypoints_new.shape[0] == 0:
                good_distances = []
            else:
                # compute descriptors at the new keypoints
                kps, descriptors = self.get_descriptors_at_keypoints(self.frame_new_gray, good_keypoints_new, bb=self.target_bounding_box)

                # match descriptors 
                matches = self.descriptor_matcher.compute_matches(self.initial_target_descriptors, 
                                                                descriptors, 
                                                                threshold=-1)

                distances = np.array([m.distance for m in matches]).reshape(-1, 1)
                good_distances = distances[distances < self.DES_MATCH_DISTANCE_THRESH]

            # ---------------------------------------------------------------------
            # |PARTIAL_OCC, TOTAL_OCC>
            if ((not self.feature_found_statuses.all() or 
                    self.cross_feature_errors.min() >= self.MAX_ERR) and 
                    len(good_distances) == 0):
                self.target_occlusion_case_new = self._TOTAL_OCC

                # cannot compute kinematics
                self.kin = None

                # update tracker display
                self.display()

                # posterity
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.target_occlusion_case_old = self.target_occlusion_case_new
                return self._FAILURE

            # ---------------------------------------------------------------------
            # |PARTIAL_OCC, NO_OCC>
            elif ((self.keypoints_new_good.shape[0] > 0) and
                    len(good_distances) == MAX_NUM_CORNERS and
                    (self.template_scores > self.TEMP_MATCH_THRESH).sum()==MAX_NUM_CORNERS):
                self.target_occlusion_case_new = self._NO_OCC

                # compute centroid 
                self.centroid_new = self.get_centroid(self.keypoints_new_good)

                # update keypoints
                if len(good_distances) == MAX_NUM_CORNERS:
                    good_matches = np.array(matches).reshape(-1, 1)[distances < self.DES_MATCH_DISTANCE_THRESH]
                    self.keypoints_new_good = np.array([list(good_keypoints_new[gm.trainIdx]) for gm in good_matches.flatten()]).reshape(-1,1,2)
                    self.keypoints_new = self.keypoints_new_good
                    self.centroid_new = self.get_centroid(self.keypoints_new_good)
                    self.rel_keypoints = self.keypoints_new - self.centroid_new

                    # also adjust old centroid, since 
                    # no dont .. old centroid would have been adjusted


                # compute kinematics
                self.kin = self.compute_kinematics_by_centroid(self.centroid_old, self.centroid_new)

                # adjust centroid
                if len(good_distances) == MAX_NUM_CORNERS:
                   self.centroid_adjustment = None
                else: 
                    centroid_new_good = self.get_centroid(good_keypoints_new)
                    self.centroid_adjustment = centroid_new_good - self.centroid_new
                    self.centroid_new = centroid_new_good

                    # update keypoints
                    self.keypoints_new_good = self.keypoints_new = self.centroid_new + self.rel_keypoints         

                # update tracker display
                self.display()

                # posterity
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.keypoints_old = self.keypoints_new_good
                self.keypoints_old_good = self.keypoints_new_good
                self.centroid_adjustment = None
                self.centroid_old = self.centroid_new
                self.target_occlusion_case_old = self.target_occlusion_case_new
                return self._SUCCESS

            # ---------------------------------------------------------------------
            # |PARTIAL_OCC, PARTIAL_OCC>
            else:
                # if we come to this, it means at least something can be salvaged
                self.target_occlusion_case_new = self._PARTIAL_OCC


                if self.keypoints_new_good.shape[0] == 0 and len(good_distances) > 0: 
                    # flow failed, matching succeeded (feature or template)
                    # compute new good keypoints using matching
                    good_matches = np.array(matches).reshape(-1, 1)[distances < self.DES_MATCH_DISTANCE_THRESH]
                    self.keypoints_new_good = np.array([list(good_keypoints_new[gm.trainIdx]) for gm in good_matches.flatten()]).reshape(-1,1,2)
                    self.keypoints_new = self.keypoints_new_good
                    self.centroid_old = self.centroid_old_true
                    self.centroid_new = self.manager.get_target_centroid()
                
                elif self.keypoints_new_good.shape[0] > 0:
                    # flow succeeded, (at least one new keypoint was found)
                    # compute adjusted centroid and compute kinematics
                    centroid_old_good = self.get_centroid(self.keypoints_old_good)
                    centroid_new_good = self.get_centroid(self.keypoints_new_good)
                    self.centroid_adjustment = self.centroid_old - centroid_old_good
                    self.centroid_new = centroid_new_good + self.centroid_adjustment
                else:
                    # flow failed, matching also failed
                    # recover before adjusting, remember we assume we still have old centroid
                    pass

                self.kin = self.compute_kinematics_by_centroid(self.centroid_old, self.centroid_new)

                # treat keypoints that were lost during flow
                if (self.keypoints_new_good.shape[0] > 0 and 
                        ((self.feature_found_statuses==0) | (self.cross_feature_errors >= self.MAX_ERR)).sum() > 0):
                    # adjust missing old keypoints (need to check recovery)
                    keypoints_missing = self.keypoints_old[(self.feature_found_statuses==0) | (self.cross_feature_errors >= self.MAX_ERR)]
                    self.keypoints_new_bad = keypoints_missing - self.centroid_old + self.centroid_new

                    # put patches over bad points in new frame
                    for kp in self.keypoints_new_bad:
                        # fetch appropriate patch
                        patch = self.get_relative_associated_patch(kp, self.centroid_new)
                        # paste patch at appropriate location
                        self.put_patch_at_point(self.frame_new_gray, patch, tuple(map(int,kp.flatten())))
                else:
                    self.keypoints_new_bad = None


                # update tracker display
                self.display()

                # add bad points to good 
                if self.keypoints_new_bad is not None:
                    self.keypoints_new_good = np.concatenate((self.keypoints_new_good, self.keypoints_new_bad.reshape(-1, 1, 2)), axis=0)

                # posterity
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.keypoints_old = self.keypoints_new
                self.keypoints_old_good = self.keypoints_new_good.reshape(-1, 1, 2)
                self.centroid_old = self.centroid_new
                self.centroid_old_true = self.manager.get_target_centroid()
                self.target_occlusion_case_old = self.target_occlusion_case_new
                return self._SUCCESS


        # ################################################################################
        # CASE FROM_TOTAL_OCC
        if self.target_occlusion_case_old == self._TOTAL_OCC:
            # (a priori) older could have been no_occ, partial_occ or total_occ
            # we should have no good keypoints, if old now has total_occ
            # here we are looking for the target again, see if we can spot it again
            # purpose being redetecting target to recover from occlusion
            
            # where do we start, nothing in the old frame to work off of
            # no flow computations, so ask help from oracle or estimator (KF or EKF)
            self.target_bounding_box = self.get_true_bb_from_oracle()
            self.target_bounding_box_mask = self.get_bounding_box_mask(self.frame_new_gray, *self.target_bounding_box)

            # perform template matching for patches to update template points and scores
            self.find_saved_patches_in_img_bb(self.frame_new_gray, self.target_bounding_box)

            # compute good feature keypoints in the new frame
            # good_keypoints_new = cv.goodFeaturesToTrack(self.frame_new_gray, mask=self.target_bounding_box_mask, **FEATURE_PARAMS)
            good_keypoints_new = self.get_feature_keypoints_from_mask(self.frame_new_gray, mask=self.target_bounding_box_mask, bb=self.target_bounding_box)

            if good_keypoints_new is None or good_keypoints_new.shape[0] == 0:
                good_distances = []
            else:
                # compute descriptors at the new keypoints
                kps, descriptors = self.get_descriptors_at_keypoints(self.frame_new_gray, good_keypoints_new, bb=self.target_bounding_box)

                # match descriptors 
                # note, matching only finds best matching/pairing, 
                # no guarantees of quality of match
                matches = self.descriptor_matcher.compute_matches(self.initial_target_descriptors, 
                                                                descriptors, 
                                                                threshold=-1)

                distances = np.array([m.distance for m in matches]).reshape(-1, 1)  # redundant TODO clean it

                # good distances indicate good matches
                good_distances = distances[distances < self.DES_MATCH_DISTANCE_THRESH]
                # if (distances < self.DES_MATCH_DISTANCE_THRESH).sum()


            # ---------------------------------------------------------------------
            # |TOTAL_OCC, NO_OCC>
            if len(good_distances) == MAX_NUM_CORNERS:
                self.target_occlusion_case_new = self._NO_OCC
                good_matches = np.array(matches).reshape(-1, 1)[distances < self.DES_MATCH_DISTANCE_THRESH]
                self.keypoints_new = np.array([list(good_keypoints_new[gm.trainIdx]) for gm in good_matches.flatten()]).reshape(-1,1,2)
                self.keypoints_new_good = self.keypoints_new
                self.centroid_new = self.get_centroid(self.keypoints_new)
                self.rel_keypoints = self.keypoints_new - self.centroid_new

                # update tracker display
                self.display()

                # posterity
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.keypoints_old = self.keypoints_new
                self.initial_keypoints = self.keypoints_old = self.keypoints_new
                self.initial_centroid = self.centroid_old = self.centroid_new
                self.centroid_adjustment = None
                self.centroid_old = self.centroid_new
                self.target_occlusion_case_old = self.target_occlusion_case_new
                self.manager.set_target_centroid_offset()
                return self._FAILURE

            # ---------------------------------------------------------------------
            # |TOTAL_OCC, PARTIAL_OCC>
            if len(good_distances) < MAX_NUM_CORNERS and len(good_distances) > 0:
                self.target_occlusion_case_new = self._PARTIAL_OCC

                # compute good matches
                good_matches = np.array(matches).reshape(-1, 1)[distances < self.DES_MATCH_DISTANCE_THRESH]
                
                # compute good points, centroid adjustments
                self.keypoints_new_good = np.array([list(good_keypoints_new[gm.trainIdx]) for gm in good_matches.flatten()]).reshape(-1,1,2)    #NOTE changed queryIdx to trainIdx .. double check later
                self.centroid_new = self.manager.get_target_centroid()
                
                # update tracker display
                self.display()

                # posterity
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.keypoints_old = self.keypoints_new  = self.keypoints_new_good
                self.keypoints_old_good = self.keypoints_new_good
                self.centroid_old = self.centroid_new
                self.centroid_old_true = self.manager.get_target_centroid()
                self.target_occlusion_case_old = self.target_occlusion_case_new
                self.manager.set_target_centroid_offset()
                return self._FAILURE

            # ---------------------------------------------------------------------
            # |TOTAL_OCC, TOTAL_OCC>
            if len(good_distances) == 0:
                self.target_occlusion_case_new = self._TOTAL_OCC

                # cannot compute kinematics
                self.kin = None

                # update tracker display
                self.display()

                # posterity
                self.frame_old_gray = self.frame_new_gray
                self.frame_old_color = self.frame_new_color
                self.target_occlusion_case_old = self.target_occlusion_case_new
                return self._FAILURE

    def compute_flow(self, use_good=False):
        # it's main purpose is to compute new points
        # looks at 2 frames, uses flow, tells where old points went
        # make clever use of this function, we want to use good 
        # for from_partial_occ case

        if use_good:
            flow_output = compute_optical_flow_LK(self.frame_old_gray,
                                                self.frame_new_gray,
                                                self.keypoints_old_good, # good from previous frame
                                                LK_PARAMS)
            # self.keypoints_old_good = flow_output[0]
        else:
            flow_output = compute_optical_flow_LK(self.frame_old_gray,
                                                self.frame_new_gray,
                                                self.keypoints_old, # good from previous frame
                                                LK_PARAMS)

        # note that new keypoints are going to be of cardinality at most that of old keypoints
        self.keypoints_old = flow_output[0]
        self.keypoints_new = flow_output[1]
        self.feature_found_statuses = flow_output[2]
        self.cross_feature_errors  = flow_output[3]

    def compute_kinematics_by_centroid(self, old_centroid, new_centroid):

        # assumptions:
        # - centroids are computed using get_centroid, therefore, centroid shape (1,2)
        # - centroids represent the target location in old and new frames

        # form pygame.Vector2 objects representing measured car_position and car_velocity 
        # in corner image coord frame in spatial units of *pixels* 
        measured_car_pos = pygame.Vector2(list(new_centroid.flatten()))
        dt = self.manager.get_sim_dt()
        measured_car_vel = pygame.Vector2(list( ((new_centroid-old_centroid)/dt).flatten() ))

        # collect fov and true drone position and velocity from simulator
        fov = self.manager.get_drone_cam_field_of_view()
        true_drone_pos = self.manager.get_true_drone_position()
        true_drone_vel = self.manager.get_true_drone_velocity()

        # transform measured car kinematics from topleft img coord frame to centered world coord frame
        # also, convert spatial units from image pixels to meters
        measured_car_pos_cam_frame_meters = self.manager.transform_pos_corner_img_pixels_to_center_cam_meters(measured_car_pos)
        measured_car_vel_cam_frame_meters = self.manager.transform_vel_img_pixels_to_cam_meters(measured_car_vel)

        #TODO consider handling the filtering part in a separate function
        # filter tracked measurements
        if USE_TRACKER_FILTER:
            if USE_MA:
                if not self.manager.MAF.ready:
                    self.manager.MAF.init_filter(measured_car_pos_cam_frame_meters, measured_car_vel_cam_frame_meters)    
                else:
                    self.manager.MAF.add_pos(measured_car_pos_cam_frame_meters)
                    maf_est_car_pos = self.manager.MAF.get_pos()
                    if dt == 0:
                        maf_est_car_vel = self.manager.MAF.get_vel()
                    else:
                        maf_est_car_vel = (self.manager.MAF.new_pos - self.manager.MAF.old_pos) / dt

                    self.manager.MAF.add_vel(measured_car_vel_cam_frame_meters)
            else:
                maf_est_car_pos = NAN
                maf_est_car_vel = NAN


            if USE_KALMAN:
                if not self.manager.KF.ready:
                    self.manager.KF.init_filter(measured_car_pos_cam_frame_meters, measured_car_vel_cam_frame_meters)
                else:
                    self.manager.KF.add(measured_car_pos_cam_frame_meters, measured_car_vel_cam_frame_meters)
                    kf_est_car_pos = self.manager.KF.get_pos()
                    kf_est_car_vel = self.manager.KF.get_vel()
            else:
                kf_est_car_pos = NAN
                kf_est_car_vel = NAN
        else:
            maf_est_car_pos = NAN
            maf_est_car_vel = NAN
            kf_est_car_pos = NAN
            kf_est_car_vel = NAN


        # return kinematics in camera frame in spatial units of meters
        ret_maf_est_car_vel = NAN if isnan(maf_est_car_vel) else maf_est_car_vel + true_drone_vel
        ret_kf_est_car_vel = NAN if isnan(kf_est_car_vel) else kf_est_car_vel + true_drone_vel

        return (
            true_drone_pos,
            true_drone_vel,
            maf_est_car_pos,
            ret_maf_est_car_vel,
            kf_est_car_pos,
            ret_kf_est_car_vel,
            measured_car_pos_cam_frame_meters,
            measured_car_vel_cam_frame_meters
        )

    def display(self):
        if self.manager.tracker_display_on:
            # add cosmetics to frame_2 for display purpose
            self.frame_color_edited, self.tracker_info_mask = self.add_cosmetics(self.frame_new_color.copy(), 
                                                                                 self.tracker_info_mask,
                                                                                 self.keypoints_old_good,
                                                                                 self.keypoints_new_good,
                                                                                 self.kin)

            # set cur_img; to be used for saving # TODO investigated it's need, used in Simulator, fix it
            self.cur_img = self.frame_color_edited

            # show resultant img
            cv.imshow(self.win_name, self.frame_color_edited)
            cv.imshow("cur_frame", self.frame_old_gray)

        # dump frames for analysis
        assembled_img = images_assemble([self.frame_old_gray.copy(), self.frame_new_gray.copy(), self.frame_color_edited.copy()], (1,3))
        self.img_dumper.dump(assembled_img)

        # ready for next iteration. set cur frame and points to next frame and points
        # self.frame_cur_gray = self.frame_nxt_gray.copy()
        # self.key_point_set_cur = self.key_point_set_nxt_good.reshape(-1, 1, 2)  # -1 indicates to infer that dim size

        cv.waitKey(1)

    def add_cosmetics(self, frame, mask, good_cur, good_nxt, kin):
        img = frame
        old_pt = np.array(self.true_old_pt).astype(np.int).reshape(-1,1,2)
        new_pt = np.array(self.true_new_pt).astype(np.int).reshape(-1,1,2)
        self.true_old_pt = self.true_new_pt

        _ARROW_COLOR = self.display_arrow_color[self.target_occlusion_case_new]
        # draw tracks on the mask, apply mask to frame, save mask for future use
        if kin is None:
            # its to and from TOTAL_OCC cases, use true old and new points
            img, mask = draw_tracks(frame, old_pt, new_pt, [TRACK_COLOR], mask, track_thickness=2, radius=7, circle_thickness=2)
            img = draw_sparse_optical_flow_arrows(img,
                                                  old_pt,
                                                  new_pt,
                                                  thickness=2,
                                                  arrow_scale=ARROW_SCALE,
                                                  color=_ARROW_COLOR)
            if good_nxt is not None:
                # from TOTAL_OCC
                for nxt in good_nxt:
                    img = cv.circle(img, tuple(map(int,nxt.flatten())), 7, TURQUOISE_GREEN_BGR, 1)
        else:
            # if self.centroid_adjustment is not None:
            #     cent_old = self.get_centroid(good_cur) + self.centroid_adjustment
            #     cent_new = self.get_centroid(good_nxt) + self.centroid_adjustment
            # else:
            #     cent_old = self.get_centroid(good_cur)
            #     cent_new = self.get_centroid(good_nxt)

            # cent_old = None if np.isnan(np.sum(cent_old)) else cent_old.astype(np.int)
            # cent_new = None if np.isnan(np.sum(cent_new)) else cent_new.astype(np.int)

            # img, mask = draw_tracks(frame, cent_old, cent_new, [TRACK_COLOR], mask, track_thickness=2, radius=7, circle_thickness=2)
            
            # draw circle and tracks between old and new centroids
            img, mask = draw_tracks(frame, self.centroid_old, self.centroid_new, [TRACK_COLOR], mask, track_thickness=2, radius=7, circle_thickness=2)
            
            # draw tracks between old and new keypoints
            for cur, nxt in zip(good_cur, good_nxt):
                img, mask = draw_tracks(frame, [cur], [nxt], [TURQUOISE_GREEN_BGR], mask, track_thickness=1, radius=7, circle_thickness=1)
            # draw circle for new keypoints
            for nxt in good_nxt:
                img, mask = draw_tracks(frame, None, [nxt], [TURQUOISE_GREEN_BGR], mask, track_thickness=1, radius=7, circle_thickness=1)
                

            # add optical flow arrows
            img = draw_sparse_optical_flow_arrows(
                img,
                self.centroid_old, # self.get_centroid(good_cur),
                self.centroid_new, # self.get_centroid(good_nxt),
                thickness=2,
                arrow_scale=ARROW_SCALE,
                color=_ARROW_COLOR)

        # add a drone center
        img = cv.circle(img, SCREEN_CENTER, radius=1, color=DOT_COLOR, thickness=2)

        # add axes in the bottom corner
        img = cv.arrowedLine(img, (16, HEIGHT - 15), (41, HEIGHT - 15), (51, 51, 255), 2)
        img = cv.arrowedLine(img, (15, HEIGHT - 16), (15, HEIGHT - 41), (51, 255, 51), 2)

        # put metrics text
        img = self.put_metrics(img, kin)

        return img, mask

    def put_metrics(self, img, k):
        """Helper function, put metrics and stuffs on opencv image.

        Args:
            k (tuple): drone_position, drone_velocity, car_position, car_velocity

        Returns:
            [np.ndarray]: Image after putting all kinds of crap
        """
        if ADD_ALTITUDE_INFO:
            img = put_text(
                img,
                f'Altitude = {self.manager.simulator.camera.altitude:0.2f} m',
                (WIDTH - 175,
                 HEIGHT - 15),
                font_scale=0.5,
                color=METRICS_COLOR,
                thickness=1)
            img = put_text(
                img,
                f'1 pixel = {self.manager.simulator.pxm_fac:0.4f} m',
                (WIDTH - 175,
                 HEIGHT - 40),
                font_scale=0.5,
                color=METRICS_COLOR,
                thickness=1)

        if ADD_METRICS:
            if k is None:
                dpos = self.manager.simulator.camera.position
                dvel = self.manager.simulator.camera.velocity
            kin_str_1 = f'car_pos (m) : '      .rjust(20)
            kin_str_2 = '--' if k is None else f'<{k[6][0]:6.2f}, {k[6][1]:6.2f}>'
            kin_str_3 = f'car_vel (m/s) : '    .rjust(20)
            kin_str_4 = '--' if k is None else f'<{k[7][0]:6.2f}, {k[7][1]:6.2f}>'
            kin_str_5 = f'drone_pos (m) : '    .rjust(20)
            kin_str_6 = f'<{dpos[0]:6.2f}, {dpos[1]:6.2f}>' if k is None else f'<{k[0][0]:6.2f}, {k[0][1]:6.2f}>'
            kin_str_7 = f'drone_vel (m/s) : '  .rjust(20)
            kin_str_8 = f'<{dvel[0]:6.2f}, {dvel[1]:6.2f}>' if k is None else f'<{k[1][0]:6.2f}, {k[1][1]*-1:6.2f}>'
            kin_str_9 = f'drone_acc (m/s^2) : '.rjust(20)
            kin_str_0 = f'<{self.manager.simulator.camera.acceleration[0]:6.2f}, {self.manager.simulator.camera.acceleration[1]:6.2f}>'
            kin_str_11 = f'r (m) : '       .rjust(20)
            kin_str_12 = f'{self.manager.simulator.camera.position.distance_to(self.manager.simulator.car.position):0.4f}'
            kin_str_13 = f'theta (degrees) : '  .rjust(20)
            kin_str_14 = f'{(self.manager.simulator.car.position - self.manager.simulator.camera.position).as_polar()[1]:0.4f}'
            kin_str_15 = f'cam origin : <{self.manager.simulator.camera.origin[0]:6.2f}, {self.manager.simulator.camera.origin[1]:6.2f}>'

            img = put_text(img, kin_str_1, (WIDTH - (330 + 25), 25),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_2, (WIDTH - (155 + 25), 25),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_3, (WIDTH - (328 + 25), 50),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_4, (WIDTH - (155 + 25), 50),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_5, (WIDTH - (332 + 25), 75),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_6, (WIDTH - (155 + 25), 75),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_7, (WIDTH - (330 + 25), 100),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_8, (WIDTH - (155 + 25), 100),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_9, (WIDTH - (340 + 25), 125),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_0, (WIDTH - (155 + 25), 125),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_11, (WIDTH - (323 + 25), 150),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_12, (WIDTH - (155 + 25), 150),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_13, (WIDTH - (323 + 25), 175),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_14, (WIDTH - (155 + 25), 175),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_15, (50, HEIGHT - 15),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)

        occ_str_dict = {self._NO_OCC:'NO OCCLUSION', self._PARTIAL_OCC:'PARTIAL OCCLUSION', self._TOTAL_OCC:'TOTAL OCCLUSION'}
        occ_color_dict = {self._NO_OCC:EMERALD_BGR, self._PARTIAL_OCC:MARIGOLD_BGR, self._TOTAL_OCC:VERMILION_BGR}
        occ_str = occ_str_dict[self.target_occlusion_case_new]
        occ_color = occ_color_dict[self.target_occlusion_case_new]
        img = put_text(img, occ_str, (WIDTH//2 - 50, HEIGHT - 40),
                           font_scale=0.55, color=occ_color, thickness=1)
        return img

    def print_to_console(self):
        if not CLEAN_CONSOLE:
            # NOTE: 
            # drone kinematics are assumed to be known (IMU and/or FPGA optical flow)
            # here, the drone position and velocity is known from Simulator
            # only the car kinematics are tracked/measured by tracker
            drone_position, drone_velocity, car_position, car_velocity, cp_, cv_ = self.kin
            print(f'TTTT >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:{vec_str(drone_position)} | v:{vec_str(drone_velocity)} | CAR - x:{vec_str(car_position)} | v:{vec_str(car_velocity)}')

    def show_me_something(self):
        """Worker function to aid debugging
        """
        # 1 for shi-tomasi, 2 for SIFT, 3 for combination, 4 for template match, 5 for new good flow
        self.nf1 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf2 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf3 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf4 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf5 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)

        # set some colors
        colors = [(16,16,16), (16,16,255), (16,255,16), (255, 16, 16)]

        # compute bounding box
        self.target_bounding_box = self.get_true_bb_from_oracle()
        self.target_bounding_box_mask = self.get_bounding_box_mask(self.frame_new_gray, *self.target_bounding_box)

        # compute shi-tomasi good feature points 
        good_keypoints_new = cv.goodFeaturesToTrack(self.frame_new_gray, 
                                                        mask=self.target_bounding_box_mask, 
                                                        **FEATURE_PARAMS)
        # draw shi-tomasi good feature points
        if good_keypoints_new is not None:
            for i, pt in enumerate(good_keypoints_new): self.nf1 = draw_point(self.nf1, tuple(pt.flatten()), colors[i])
            cv.imshow('nxt_frame', self.nf1); cv.waitKey(1)

        # compute SIFT feature keypoints and draw points and matches
        gkn = self.detector.get_keypoints(self.frame_new_gray, mask=self.target_bounding_box_mask)
        if gkn is not None:
            self.gpn = np.array([list(gp.pt) for gp in gkn]).reshape(-1,1,2)
            for i, pt in enumerate(self.gpn): self.nf2 = draw_point(self.nf2, tuple(map(int,pt.flatten())))
            self.k,self.d = self.get_descriptors_at_keypoints(self.frame_new_gray, self.gpn, bb=self.target_bounding_box)
            self.mat = self.descriptor_matcher.compute_matches(self.initial_target_descriptors, 
                                                                    self.d, 
                                                                    threshold=-1)
            self.dist = np.array([m.distance for m in self.mat]).reshape(-1, 1)
            for i,m in enumerate(self.mat):
                pt = tuple(map(int,self.gpn[m.trainIdx].flatten()))
                self.nf2 = cv.circle(self.nf2, pt, 5, colors[i], 2)

        # use combination of both shi-tomasi and SIFT keypoints, and draw points and matches
        self.cmb_pts = self.get_feature_keypoints_from_mask(self.frame_new_gray, mask=self.target_bounding_box_mask, bb=self.target_bounding_box)
        if self.cmb_pts is not None:
            for i, pt in enumerate(self.cmb_pts): self.nf3 = draw_point(self.nf3, tuple(map(int,pt.flatten())))
            self.kc,self.dc = self.get_descriptors_at_keypoints(self.frame_new_gray, self.cmb_pts, bb=self.target_bounding_box)
            self.matc = self.descriptor_matcher.compute_matches(self.initial_target_descriptors, 
                                                                    self.dc, 
                                                                    threshold=-1)
            self.distc = np.array([m.distance for m in self.matc]).reshape(-1, 1)
            for i,m in enumerate(self.matc):
                pt = tuple(map(int,self.cmb_pts[m.trainIdx].flatten()))
                self.nf3 = cv.circle(self.nf3, pt, 5, colors[i], 2)
                if m.distance < self.DES_MATCH_DISTANCE_THRESH:
                    self.nf3 = cv.circle(self.nf3, pt, 9, colors[i], 1)
        
        # find patch templates, and draw location points and matches
        self.find_saved_patches_in_img_bb(convert_to_grayscale(self.nf4), self.target_bounding_box)
        for i, t_pt in enumerate(self.template_points):
            pt = tuple(t_pt.flatten())
            self.nf4 = draw_point(self.nf4, pt, colors[i])
            if self.template_scores.flatten()[i] > self.TEMP_MATCH_THRESH:
                self.nf4 = cv.circle(self.nf4, pt, 7, colors[i], 2)

        # draw new good flow points
        if self.keypoints_new is not None:
            for i, pt in enumerate(self.keypoints_new):
                pt = tuple(map(int, pt.flatten()))
                self.nf5 = draw_point(self.nf5, pt, colors[i])
        if self.keypoints_new_good is not None:
            for i, pt in enumerate(self.keypoints_new_good):
                pt = tuple(map(int, pt.flatten()))
                self.nf5 = cv.circle(self.nf4, pt, 7, colors[i], 2)


class Controller:
    def __init__(self, manager):
        self.manager = manager
        self.plot_info_file = 'plot_info.txt'
        self.R = CAR_RADIUS
        self.f = None
        self.a_ln = 0.0
        self.a_lt = 0.0
        self.est_def = False


    @staticmethod
    def sat(x, bound):
        return min(max(x, -bound), bound)

    def generate_acceleration(self, kin):
        X, Y = kin[0]
        Vx, Vy = kin[1]
        car_x, car_y = kin[2]
        car_speed, cvy = kin[3]

        if USE_WORLD_FRAME:
            # add camera origin to positions
            orig = self.manager.get_cam_origin()
            X += orig[0]
            Y += orig[1]
            car_x += orig[0]
            car_y += orig[1]

        # speed of drone
        S = (Vx**2 + Vy**2) ** 0.5

        # heading angle of drone wrt x axis
        alpha = atan2(Vy, Vx)

        # heading angle of car
        beta = 0

        # distance between the drone and car
        r = ((car_x - X)**2 + (car_y - Y)**2)**0.5

        # angle of LOS from drone to car
        theta = atan2(car_y - Y, car_x - X)

        # compute Vr and Vθ
        Vr = car_speed * cos(beta - theta) - S * cos(alpha - theta)
        Vtheta = car_speed * sin(beta - theta) - S * sin(alpha - theta)

        # save measured r, θ, Vr, Vθ
        r_ = r
        theta_ = theta
        Vr_ = Vr
        Vtheta_ = Vtheta

        # at this point r, theta, Vr, Vtheta are computed
        # we can consider EKF filtering [r, theta, Vr, Vtheta]
        if not USE_TRUE_KINEMATICS and USE_EXTENDED_KALMAN:
            self.manager.EKF.add(r, theta, Vr, Vtheta, alpha, self.a_lt, self.a_ln)
            r, theta, Vr, Vtheta = self.manager.EKF.get_estimated_state()

        # calculate y from drone to car
        y2 = Vtheta**2 + Vr**2
        y1 = r**2 * Vtheta**2 - y2 * self.R**2
        # y1 = Vtheta**2 * (r**2 - self.R**2) - self.R**2 * Vr**2

        # time to collision from drone to car
        # tm = -vr * r / (vtheta**2 + vr**2)

        # compute desired acceleration
        w = w_
        K1 = K_1 * np.sign(-Vr)    # lat
        K2 = K_2                   # long

        # compute lat and long accelerations
        _D = 2 * Vr * Vtheta * r**2

        if abs(_D) < 0.01:
            a_lat = 0.0
            a_long = 0.0
        else:
            a_lat = (K1 * Vr * y1 * cos(alpha - theta) - K1 * Vr * w * cos(alpha - theta) - K1 * Vtheta * w * sin(alpha - theta) + K1 * Vtheta * y1 * sin(alpha - theta) +
                     K2 * self.R**2 * Vr * y2 * cos(alpha - theta) + K2 * self.R**2 * Vtheta * y2 * sin(alpha - theta) - K2 * Vtheta * r**2 * y2 * sin(alpha - theta)) / _D
            a_long = (K1 * Vtheta * w * cos(alpha - theta) - K1 * Vtheta * y1 * cos(alpha - theta) - K1 * Vr * w * sin(alpha - theta) + K1 * Vr * y1 * sin(alpha - theta) -
                      K2 * self.R**2 * Vtheta * y2 * cos(alpha - theta) + K2 * self.R**2 * Vr * y2 * sin(alpha - theta) + K2 * Vtheta * r**2 * y2 * cos(alpha - theta)) / _D

        a_long_bound = 5
        a_lat_bound = 5

        a_long = self.sat(a_long, a_long_bound)
        a_lat = self.sat(a_lat, a_lat_bound)

        self.a_ln = a_long
        self.a_lt = a_lat

        # compute acceleration command
        delta = alpha + pi / 2
        ax = a_lat * cos(delta) + a_long * cos(alpha)
        ay = a_lat * sin(delta) + a_long * sin(alpha)

        if not CLEAN_CONSOLE:
            print(f'CCC0 >> r:{r:0.2f} | theta:{theta:0.2f} | alpha:{alpha:0.2f} | car_speed:{car_speed:0.2f} | S:{S:0.2f} | Vr:{Vr:0.2f} | Vtheta:{Vtheta:0.2f} | y1:{y1:0.2f} | y2:{y2:0.2f} | a_lat:{a_lat:0.2f} | a_long:{a_long:0.2f} | _D:{_D:0.2f}')

        tru_kin = self.manager.get_true_kinematics()
        tX, tY = tru_kin[0]
        tVx, tVy = tru_kin[1]
        tcar_x, tcar_y = tru_kin[2]
        tcar_speed, tcvy = tru_kin[3]
        tS = (tVx**2 + tVy**2) ** 0.5
        tr = ((tcar_x - tX)**2 + (tcar_y - tY)**2)**0.5
        ttheta = atan2(tcar_y - tY, tcar_x - tX)
        tVr = tcar_speed * cos(beta - ttheta) - tS * cos(alpha - ttheta)
        tVtheta = tcar_speed * sin(beta - ttheta) - tS * sin(alpha - ttheta)

        tra_kin = self.manager.get_tracked_kinematics()
        vel = self.manager.simulator.camera.velocity
        if not CLEAN_CONSOLE:
            print(
                f'CCCC >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:[{X:0.2f}, {Y:0.2f}] | v:[{Vx:0.2f}, {Vy:0.2f}] | CAR - x:[{car_x:0.2f}, {car_y:0.2f}] | v:[{car_speed:0.2f}, {cvy:0.2f}] | COMMANDED a:[{ax:0.2f}, {ay:0.2f}] | TRACKED x:[{tra_kin[2][0]:0.2f},{tra_kin[2][1]:0.2f}] | v:[{tra_kin[3][0]:0.2f},{tra_kin[3][1]:0.2f}]')
        if self.manager.write_plot:
            self.f.write(f'\
                {self.manager.simulator.time},\
                {r},\
                {degrees(theta)},\
                {degrees(Vtheta)},\
                {Vr},\
                {tru_kin[0][0]},\
                {tru_kin[0][1]},\
                {tru_kin[2][0]},\
                {tru_kin[2][1]},\
                {ax},\
                {ay},\
                {a_lat},\
                {a_long},\
                {tru_kin[3][0]},\
                {tru_kin[3][1]},\
                {tra_kin[2][0]},\
                {tra_kin[2][1]},\
                {tra_kin[3][0]},\
                {tra_kin[3][1]},\
                {self.manager.simulator.camera.origin[0]},\
                {self.manager.simulator.camera.origin[1]},\
                {S},\
                {degrees(alpha)},\
                {tru_kin[1][0]},\
                {tru_kin[1][1]},\
                {tra_kin[4][0]},\
                {tra_kin[4][1]},\
                {tra_kin[5][0]},\
                {tra_kin[5][1]},\
                {self.manager.simulator.camera.altitude},\
                {abs(_D)},\
                {r_},\
                {degrees(theta_)},\
                {Vr_},\
                {degrees(Vtheta_)},\
                {tr},\
                {degrees(ttheta)},\
                {tVr},\
                {degrees(tVtheta)},\
                {self.manager.simulator.dt},\
                {y1},\
                {y2}\n')\

        if not self.manager.control_on:
            ax, ay = pygame.Vector2((0.0, 0.0))

        return ax, ay


class ExperimentManager:
    """
    Experiment:

    - Run the game Simulator with a car and static blocks to aid motion perception for biological creatures with visual perception.
    - Let user select a bounding box for the car to be tracked.
        - Simulator needs to be paused and played back again after bounding box selection.
    - Simulator keeps simulating and rendering images on screen.
    - Also, dumps screen captures for Tracker or Controller to consume.
    - Additionally, concatenate screen captures and tracker produced images with tracking information, into one image and save it.
    - Tracker consumes these images and produces tracking information at each frame.
    - Controller consumes tracking information and produces acceleration commands for Simulator
    - Simulator consumes acceleration command and updates simulated components appropriately.
    - All information relay across the Simulator, Tracker and Controller can be mediated through the ExperimentManager

    The manager is responsible for running Simulator, Tracker and controller in separate threads and manage shared memory.
    The manager can start and stop Simulator, Tracker and Controller.
    """

    def __init__(
            self,
            save_on=False,
            write_plot=False,
            control_on=False,
            tracker_on=True,
            tracker_display_on=False,
            use_true_kin=True,
            use_real_clock=True,
            draw_occlusion_bars=False):

        self.save_on = save_on
        self.write_plot = write_plot
        self.control_on = control_on
        self.tracker_on = tracker_on
        self.tracker_display_on = tracker_display_on
        self.use_true_kin = use_true_kin
        self.use_real_clock = use_real_clock
        self.draw_occlusion_bars = draw_occlusion_bars

        self.simulator = Simulator(self)
        self.tracker = Tracker(self)
        self.controller = Controller(self)

        self.MAF = MA(window_size=10)
        self.KF = Kalman(self)
        self.EKF = ExtendedKalman(self)

        # self.image_deque = deque(maxlen=2)
        # self.command_deque = deque(maxlen=2)
        # self.kinematics_deque = deque(maxlen=2)

        self.sim_dt = 0
        self.true_rel_vel = None
        self.car_rect_center_centroid_offset = [0, 0]

        if self.save_on:
            self.simulator.save_screen = True

    def get_drone_cam_field_of_view(self):
        return self.simulator.get_camera_fov()

    def get_true_drone_position(self):
        return self.simulator.camera.position

    def get_true_drone_velocity(self):
        return self.simulator.camera.velocity

    def get_target_bounding_box(self):
        return self.simulator.bounding_box

    def transform_pos_corner_img_pixels_to_center_cam_meters(self, pos):
        pos = pos.elementwise() * (1, -1) + (0, HEIGHT)
        pos *= self.simulator.pxm_fac
        pos += -pygame.Vector2(self.get_drone_cam_field_of_view()) / 2

        return pos

    def transform_vel_img_pixels_to_cam_meters(self, vel):
        vel = vel.elementwise() * (1, -1)
        vel *= self.simulator.pxm_fac

        return vel

    def set_target_centroid_offset(self):
        # this will be called from tracker at the first run after first centroid calculation
        # uses tracked new centroid to compute it's relative position from car center
        self.car_rect_center_centroid_offset[0] = self.tracker.centroid_new.flatten()[0] - self.simulator.car.rect.centerx
        self.car_rect_center_centroid_offset[1] = self.tracker.centroid_new.flatten()[1] - self.simulator.car.rect.centery

    def get_target_centroid(self):
        target_cent = self.car_rect_center_centroid_offset.copy()
        target_cent[0] += self.simulator.car.rect.centerx
        target_cent[1] += self.simulator.car.rect.centery
        return np.array(target_cent).reshape(1, 2)

    def get_target_centroid_offset(self):
        return self.car_rect_center_centroid_offset

    def get_bounding_box_offset(self):
        return self.simulator.car_rect_center_bb_offset

    def get_target_bounding_box_from_offset(self):
        x, y, w, h = self.simulator.bounding_box
        bb_offset = self.get_bounding_box_offset()
        x = self.simulator.car.rect.center[0] + bb_offset[0]
        y = self.simulator.car.rect.center[1] + bb_offset[1]
        return x, y, w, h

    def filters_ready(self):
        ready = True
        if USE_TRACKER_FILTER:
            if USE_KALMAN:
                ready = ready and self.KF.done_waiting()
            if USE_MA:
                ready = ready and self.MAF.done_waiting()
        return ready


    def run(self):
        # initialize simulator
        self.simulator.start_new()

        # open plot file if write_plot is indicated
        if self.write_plot:
            self.controller.f = open(self.controller.plot_info_file, '+w')

        # run experiment
        while self.simulator.running:
            # get delta time between ticks (0 when paused), update elapsed simulated time
            self.simulator.dt = self.simulator.clock.tick(FPS) / 1000000.0
            if not self.use_real_clock:
                self.simulator.dt = DELTA_TIME
            if self.simulator.pause:
                self.simulator.dt = 0.0
            self.simulator.time += self.simulator.dt

            # handle events on simulator
            self.simulator.handle_events()

            # if quit event occurs, running will be updated; respond to it
            if not self.simulator.running:
                break
            
            # update rects and images for all sprites (not when paused)
            if not self.simulator.pause:
                self.simulator.update()
                if not CLEAN_CONSOLE:
                    print(f'SSSS >> {str(timedelta(seconds=self.simulator.time))} >> DRONE - x:{vec_str(self.simulator.camera.position)} | v:{vec_str(self.simulator.camera.velocity)} | CAR - x:{vec_str(self.simulator.car.position)}, v: {vec_str(self.simulator.car.velocity)} | COMMANDED a:{vec_str(self.simulator.camera.acceleration)} | a_comm:{vec_str(self.simulator.cam_accel_command)} | rel_car_pos: {vec_str(self.simulator.car.position - self.simulator.camera.position)}', end='\n')

            # draw updated car, blocks and bars (drone will be drawn later)
            self.simulator.draw()

            # process screen capture *PARTY IS HERE*
            if not self.simulator.pause:

                # let tracker process image, when simulator indicates ok
                if self.simulator.can_begin_tracking():
                    screen_capture = self.simulator.get_screen_capture()
                    status = self.tracker.process_image_complete(screen_capture)
                    self.tracker.print_to_console()

                    # let controller generate acceleration, when tracker indicates ok
                    if self.tracker.can_begin_control() and (
                            self.use_true_kin or self.tracker.kin is not None) and (
                            # self.filter.done_waiting() or not USE_TRACKER_FILTER):
                            self.filters_ready()):
                        # collect kinematics tuple
                        # kin = self.tracker.kin if status[0] else self.get_true_kinematics()
                        kin = self.get_true_kinematics() if (self.use_true_kin or not status[0]) else self.get_tracked_kinematics()
                        # let controller process kinematics
                        ax, ay = self.controller.generate_acceleration(kin)
                        # feed controller generated acceleration commands to simulator
                        self.simulator.camera.acceleration = pygame.Vector2((ax, ay))

            self.simulator.draw_extra()
            self.simulator.show_drawing()

            if self.simulator.save_screen:
                next(self.simulator.screen_shot)

        cv.destroyAllWindows()
        if self.write_plot:
            self.controller.f.close()

    @staticmethod
    def make_video(video_name, folder_path):
        """Helper function, looks for frames in given folder,
        writes them into a video file, with the given name.
        Also removes the folder after creating the video.
        """
        if os.path.isdir(folder_path):
            create_video_from_images(folder_path, 'png', video_name, FPS)

            # delete folder
            shutil.rmtree(folder_path)

    def get_sim_dt(self):
        return self.simulator.dt

    def get_true_kinematics(self):
        """Helper function returns drone and car position and velocity

        Returns:
            tuple: drone_pos, drone_vel, car_pos, car_vel
        """
        kin = (self.simulator.camera.position,
               self.simulator.camera.velocity,
               self.simulator.car.position,
               self.simulator.car.velocity)

        return kin

    def get_tracked_kinematics(self):
        return (
            self.tracker.kin[0],    # true drone position    
            self.tracker.kin[1],    # true drone velocity
            self.tracker.kin[6],    # measured car position in camera frame (meters)
            self.tracker.kin[7],    # measured car velocity in camera frame (meters)
            self.tracker.kin[2],    # kalman estimated car position
            self.tracker.kin[3],    # kalman estimated car velocity
            self.tracker.kin[4],    # moving averaged car position
            self.tracker.kin[5],    # moving averaged car velocity
        ) if self.tracker.kin is not None else None

    def get_cam_origin(self):
        return self.simulator.camera.origin


class MA:
    """Filters statefully using a moving average technique
    """

    def __init__(self, window_size=10):
        self.car_x = deque(maxlen=window_size)
        self.car_y = deque(maxlen=window_size)
        self.car_vx = deque(maxlen=window_size)
        self.car_vy = deque(maxlen=window_size)

        self.ready = False

        # self.old_pos = self.avg_pos()
        # self.old_vel = self.avg_vel()

    def done_waiting(self):
        """Indicates readiness of filter

        Returns:
            bool: Ready or not
        """
        return len(self.car_vx) > 5

    def init_filter(self, pos, vel):
        """Initializes filter. Meant to be run for the first time only.

        Args:
            pos (pygame.Vector2): Car position measurement
            vel (pygame.Vector2): Car velocity measurement
        """
        self.new_pos = pygame.Vector2(pos)
        self.new_vel = pygame.Vector2(vel)
        self.add_pos(pos)
        self.add_vel(vel)
        self.ready = True

    def add(self, pos, vel):
        """Add a measurement

        Args:
            pos (pygame.Vector2): Car position measurement
            vel (pygame.Vector2): Car velocity measurement
        """
        # remember the last new average before adding to deque
        self.old_pos = self.new_pos
        self.old_vel = self.new_vel

        # add to deque
        self.car_x.append(pos[0])
        self.car_y.append(pos[1])
        self.car_vx.append(vel[0])
        self.car_vy.append(vel[1])

        # compute new average
        self.new_pos = self.avg_pos()
        self.new_vel = self.avg_vel()

    def add_pos(self, pos):
        """Add position measurement

        Args:
            pos (pygame.Vector2): Car position measurement
        """
        # remember the last new average before adding to deque
        self.old_pos = self.new_pos

        # add to deque
        self.car_x.append(pos[0])
        self.car_y.append(pos[1])

        # compute new average
        self.new_pos = self.avg_pos()

    def add_vel(self, vel):
        """Add velocity measurement

        Args:
            vel (pygame.Vector2): Car velocity measurement
        """
        # remember the last new average before adding to deque
        self.old_vel = self.new_vel

        # add to deque
        self.car_vx.append(vel[0])
        self.car_vy.append(vel[1])

        # compute new average
        self.new_vel = self.avg_vel()

    def get_pos(self):
        """Fetch estimated position

        Returns:
            pygame.Vector2: Car estimate position
        """
        return self.new_pos

    def get_vel(self):
        """Get estimated velocity

        Returns:
            pygame.Vector2: Car estimated velocity
        """
        return self.new_vel

    def avg_pos(self):
        """Helper function to average position measurements

        Returns:
            pygame.Vector2: Averaged car position
        """
        x = sum(self.car_x) / len(self.car_x)
        y = sum(self.car_y) / len(self.car_y)
        return pygame.Vector2(x, y)

    def avg_vel(self):
        """Helper function to average velocity measurements

        Returns:
            pygame.Vector2: Averaged car velocity
        """
        vx = sum(self.car_vx) / len(self.car_vx)
        vy = sum(self.car_vy) / len(self.car_vy)
        return pygame.Vector2(vx, vy)


class Kalman:
    """Implements Discrete-time Kalman filtering in a stateful fashion
    """

    def __init__(self, manager):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.sig = 0.1
        self.sig_r = 0.1
        self.sig_q = 1.0
        self.manager = manager

        # process noise
        self.Er = np.array([[0.01], [0.01], [0.01], [0.01]])

        # measurement noise
        self.Eq = np.array([[0.01], [0.01], [0.01], [0.01]])

        # initialize belief state and covariance
        self.Mu = np.array([[self.x], [self.y], [self.vx], [self.vy]])
        self.var_S = np.array([10**-4, 10**-4, 10**-4, 10**-4])
        self.S = np.diag(self.var_S.flatten())

        # noiseless connection between state vector and measurement vector
        self.C = np.identity(4)

        # covariance of process noise model
        self.var_R = np.array([10**-6, 10**-6, 10**-5, 10**-5])
        self.R = np.diag(self.var_R.flatten())

        # covariance of measurement noise model
        self.var_Q = np.array([0.0156 * 10**-3, 0.0155 * 10**-3, 7.3811 * 10**-3, 6.5040 * 10**-3])
        self.Q = np.diag(self.var_Q.flatten())

        self.ready = False

    def done_waiting(self):
        """Indicates filter readiness

        Returns:
            bool: Ready or not
        """
        return self.ready

    def init_filter(self, pos, vel):
        """Initializes filter. Meant to be run only at first.

        Args:
            pos (pygame.Vector2): Car position measurement
            vel (pygame.Vector2): Car velocity measurement
        """
        self.x = pos[0]
        self.y = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.X = np.array([[self.x], [self.y], [self.vx], [self.vy]])
        self.Mu = self.X
        self.ready = True

    def add(self, pos, vel):
        """Add a measurement.

        Args:
            pos (pygame.Vector2): Car position measurement
            vel (pygame.Vector2): Car velocity measurement
        """
        # pos and vel are the measured values. (remember x_bar)
        self.x = pos[0]
        self.y = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.X = np.array([[self.x], [self.y], [self.vx], [self.vy]])

        self.predict()
        self.correct()

    def predict(self):
        """Implement discrete-time Kalman filter prediction/forecast step
        """
        # collect params
        dt = self.manager.get_sim_dt()
        dt2 = dt**2
        # motion model
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # control model
        B = np.array([[0.5 * dt2, 0], [0, 0.5 * dt2], [dt, 0], [0, dt]])
        # B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

        # process noise covariance
        R = self.R

        command = self.manager.simulator.camera.acceleration
        U = np.array([[command[0]], [command[1]]])

        # predict
        self.Mu = np.matmul(A, self.Mu) + np.matmul(B, U)
        self.S = np.matmul(np.matmul(A, self.S), np.transpose(A)) + R

    def correct(self):
        """Implement discrete-time Kalman filter correction/update step
        """
        Z = self.X
        K = np.matmul(
            np.matmul(
                self.S, self.C), np.linalg.pinv(
                np.matmul(
                    np.matmul(
                        self.C, self.S), np.transpose(
                        self.C)) + self.Q))

        self.Mu = self.Mu + np.matmul(K, (Z - np.matmul(self.C, self.Mu)))
        self.S = np.matmul((np.identity(4) - np.matmul(K, self.C)), self.S)

    def add_pos(self, pos):
        """Add position measurement

        Args:
            pos (pygame.Vector2): Car position measurement
        """
        self.add(pos, (self.vx, self.vy))

    def add_vel(self, vel):
        """Add velocity measurement

        Args:
            vel (pygame.Vector2): Car velocity measurement
        """
        self.add((self.x, self.y), vel)

    def get_pos(self):
        """Get estimated car position

        Returns:
            pygame.Vector2: Car estimated position
        """
        return pygame.Vector2(self.Mu.flatten()[0], self.Mu.flatten()[1])

    def get_vel(self):
        """Get estimated car velocity

        Returns:
            pygame.Vector2: Car estimated velocity
        """
        return pygame.Vector2(self.Mu.flatten()[2], self.Mu.flatten()[3])


class ExtendedKalman:
    """Implement continuous-continuous EKF for the UAS and Vehicle system in stateful fashion
    """

    def __init__(self, manager):
        self.manager = manager

        self.prev_r = None
        self.prev_theta = None
        self.prev_Vr = None
        self.prev_Vtheta = None
        self.alpha = None
        self.a_lat = None
        self.a_long = None
        self.filter_initialized_flag = False

        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])

        self.P = np.diag([0.1, 0.1, 0.1, 0.1])
        self.R = np.diag([0.1, 0.1])
        self.Q = np.diag([0.1, 0.1, 1, 0.1])

        self.ready = False

    def is_initialized(self):
        """Indicates if EKF is initialized

        Returns:
            bool: EKF initalized or not
        """
        return self.filter_initialized_flag

    def initialize_filter(self, r, theta, Vr, Vtheta, alpha, a_lat, a_long):
        """Initializes EKF. Meant to run only once at first.

        Args:
            r (float32): Euclidean distance between vehicle and UAS (m)
            theta (float32): Angle (atan2) of LOS from UAS to vehicle (rad)
            Vr (float32): Relative LOS velocity of vehicle w.r.t UAS (m/s)
            Vtheta (float32): Relative LOS angular velocity of vehicle w.r.t UAS (rad/s)
            alpha (float32): Angle of UAS velocity vector w.r.t to inertial frame
            a_lat (float32): Lateral acceleration control command for UAS
            a_long (float32): Longitudinal acceleration control command for UAS
        """
        self.prev_r = r
        self.prev_theta = theta
        self.prev_Vr = -5
        self.prev_Vtheta = 5
        self.alpha = alpha
        self.a_lat = a_lat
        self.a_long = a_long
        self.filter_initialized_flag = True

    def add(self, r, theta, Vr, Vtheta, alpha, a_lat, a_long):
        """Add measurements and auxiliary data for filtering

        Args:
            r (float32): Euclidean distance between vehicle and UAS (m)
            theta (float32): Angle (atan2) of LOS from UAS to vehicle (rad)
            Vr (float32): Relative LOS velocity of vehicle w.r.t UAS (m/s)
            Vtheta (float32): Relative LOS angular velocity of vehicle w.r.t UAS (rad/s)
            alpha (float32): Angle of UAS velocity vector w.r.t to inertial frame
            a_lat (float32): Lateral acceleration control command for UAS
            a_long (float32): Longitudinal acceleration control command for UAS
        """
        # make sure filter is initialized
        if not self.is_initialized():
            self.initialize_filter(r, theta, Vr, Vtheta, alpha, a_lat, a_long)
            return

        # filter is initialized; set ready to true
        self.ready = True

        if (np.sign(self.prev_theta) != np.sign(theta)):
            self.prev_theta = theta

        # store measurement
        self.r = r
        self.theta = theta
        self.Vr = Vr
        self.Vtheta = Vtheta
        self.alpha = alpha
        self.a_lat = a_lat
        self.a_long = a_long

        # perform predictor and filter step
        self.predict()
        self.correct()

        # remember previous state
        self.prev_r = self.r
        self.prev_theta = self.theta
        self.prev_Vr = self.Vr
        self.prev_Vtheta = self.Vtheta

    def predict(self):
        """Implement continuous-continuous EKF prediction (implicit) step.
        """
        # perform predictor step
        self.A = np.array([[0.0, 0.0, 0.0, 1.0],
                           [-self.prev_Vtheta / self.prev_r**2, 0.0, 1 / self.prev_r, 0.0],
                           [self.prev_Vtheta * self.prev_Vr / self.prev_r**2, 0.0, -self.prev_Vr / self.prev_r, -self.prev_Vtheta / self.prev_r],
                           [-self.prev_Vtheta**2 / self.prev_r**2, 0.0, 2 * self.prev_Vtheta / self.prev_r, 0.0]])

        self.B = np.array([[0.0, 0.0],
                           [0.0, 0.0],
                           [-sin(self.alpha + pi / 2 - self.prev_theta), -sin(self.alpha - self.prev_theta)],
                           [-cos(self.alpha + pi / 2 - self.prev_theta), -cos(self.alpha - self.prev_theta)]])

    def correct(self):
        """Implement continuous-continuous EKF correction (implicit) step.
        """
        self.Z = np.array([[self.r], [self.theta]])
        self.K = np.matmul(np.matmul(self.P, np.transpose(self.H)), np.linalg.pinv(self.R))

        U = np.array([[self.a_lat], [self.a_long]])
        state = np.array([[self.prev_r], [self.prev_theta], [self.prev_Vtheta], [self.prev_Vr]])
        dyn = np.array([[self.prev_Vr],
                        [self.prev_Vtheta / self.prev_r],
                        [-self.prev_Vtheta * self.prev_Vr / self.prev_r],
                        [self.prev_Vtheta**2 / self.prev_r]])

        state_dot = dyn + np.matmul(self.B, U) + np.matmul(self.K,
                                                           (self.Z - np.matmul(self.H, state)))
        P_dot = np.matmul(self.A, self.P) + np.matmul(self.P, np.transpose(self.A)
                                                      ) - np.matmul(np.matmul(self.K, self.H), self.P) + self.Q

        dt = self.manager.get_sim_dt()
        state = state + state_dot * dt
        self.P = self.P + P_dot * dt

        self.r = state.flatten()[0]
        self.theta = state.flatten()[1]
        self.Vtheta = state.flatten()[2]
        self.Vr = state.flatten()[3]

    def get_estimated_state(self):
        """Get estimated state information.

        Returns:
            tuple(float32, float32, float, float32): (r, theta, V_r, V_theta)
        """
        if self.ready:
            return (self.r, self.theta, self.Vr, self.Vtheta)
        else:
            return (self.prev_r, self.prev_theta, self.prev_Vr, self.prev_Vtheta)


# dummy moving average for testing (not used)
def compute_moving_average(sequence, window_size):
    """Generate moving average sequence of given sequence using given window size.

    Args:
        sequence (list): Sequence to be averaged
        window_size (int): Window size

    Returns:
        list: Averaged sequence
    """

    mov_avg_seq = []

    for i in range(len(sequence)):
        start = max(0, i - window_size + 1)
        stop = i + 1
        seq_window = sequence[start:stop]
        mov_avg_seq.append(sum(seq_window) / len(seq_window))

    return mov_avg_seq


if __name__ == '__main__':

    EXPERIMENT_SAVE_MODE_ON = 0  # pylint: disable=bad-whitespace
    WRITE_PLOT = 0  # pylint: disable=bad-whitespace
    CONTROL_ON = 0  # pylint: disable=bad-whitespace
    TRACKER_ON = 1  # pylint: disable=bad-whitespace
    TRACKER_DISPLAY_ON = 1  # pylint: disable=bad-whitespace
    USE_TRUE_KINEMATICS = 1  # pylint: disable=bad-whitespace
    USE_REAL_CLOCK = 0  # pylint: disable=bad-whitespace
    DRAW_OCCLUSION_BARS = 1  # pylint: disable=bad-whitespace

    RUN_EXPERIMENT = 1  # pylint: disable=bad-whitespace
    RUN_TRACK_PLOT = 0  # pylint: disable=bad-whitespace

    RUN_VIDEO_WRITER = 0  # pylint: disable=bad-whitespace

    if RUN_EXPERIMENT:
        EXPERIMENT_MANAGER = ExperimentManager(save_on=EXPERIMENT_SAVE_MODE_ON,
                                               write_plot=WRITE_PLOT,
                                               control_on=CONTROL_ON,
                                               tracker_on=TRACKER_ON,
                                               tracker_display_on=TRACKER_DISPLAY_ON,
                                               use_true_kin=USE_TRUE_KINEMATICS,
                                               use_real_clock=USE_REAL_CLOCK,
                                               draw_occlusion_bars=DRAW_OCCLUSION_BARS)
        print(f'\nExperiment started. [{time.strftime("%H:%M:%S")}]\n')
        EXPERIMENT_MANAGER.run()

        print(f'\n\nExperiment finished. [{time.strftime("%H:%M:%S")}]\n')

    if RUN_TRACK_PLOT:
        FILE = open('plot_info.txt', 'r')
        
        # plot switches
        SHOW_ALL = 1    # set to 1 to show all plots 

        SHOW_CARTESIAN_PLOTS = 0
        SHOW_LOS_KIN_1 = 1
        SHOW_LOS_KIN_2 = 1
        SHOW_ACCELERATIONS = 1
        SHOW_TRAJECTORIES = 1
        SHOW_SPEED_HEADING = 1
        SHOW_ALTITUDE_PROFILE = 0
        SHOW_3D_TRAJECTORIES = 0
        SHOW_DELTA_TIME_PROFILE = 0

        _TIME = []
        _R = []
        _THETA = []
        _V_THETA = []
        _V_R = []
        _DRONE_POS_X = []
        _DRONE_POS_Y = []
        _CAR_POS_X = []
        _CAR_POS_Y = []
        _DRONE_ACC_X = []
        _DRONE_ACC_Y = []
        _DRONE_ACC_LAT = []
        _DRONE_ACC_LNG = []
        _CAR_VEL_X = []
        _CAR_VEL_Y = []
        _TRACKED_CAR_POS_X = []
        _TRACKED_CAR_POS_Y = []
        _TRACKED_CAR_VEL_X = []
        _TRACKED_CAR_VEL_Y = []
        _CAM_ORIGIN_X = []
        _CAM_ORIGIN_Y = []
        _DRONE_SPEED = []
        _DRONE_ALPHA = []
        _DRONE_VEL_X = []
        _DRONE_VEL_Y = []
        _MEASURED_CAR_POS_X = []
        _MEASURED_CAR_POS_Y = []
        _MEASURED_CAR_VEL_X = []
        _MEASURED_CAR_VEL_Y = []
        _DRONE_ALTITUDE = []
        _ABS_DEN = []
        _MEASURED_R = []
        _MEASURED_THETA = []
        _MEASURED_V_R = []
        _MEASURED_V_THETA = []
        _TRUE_R = []
        _TRUE_THETA = []
        _TRUE_V_R = []
        _TRUE_V_THETA = []
        _DELTA_TIME = []
        _Y1 = []
        _Y2 = []

        # get all the data in memory
        for line in FILE.readlines():
            data = tuple(map(float, list(map(str.strip, line.strip().split(',')))))
            _TIME.append(data[0])
            _R.append(data[1])
            _THETA.append(data[2])      # degrees
            _V_THETA.append(data[3])    # degrees
            _V_R.append(data[4])
            _DRONE_POS_X.append(data[5])    # true
            _DRONE_POS_Y.append(data[6])    # true
            _CAR_POS_X.append(data[7])      # true
            _CAR_POS_Y.append(data[8])      # true
            _DRONE_ACC_X.append(data[9])
            _DRONE_ACC_Y.append(data[10])
            _DRONE_ACC_LAT.append(data[11])
            _DRONE_ACC_LNG.append(data[12])
            _CAR_VEL_X.append(data[13])
            _CAR_VEL_Y.append(data[14])
            _TRACKED_CAR_POS_X.append(data[15])
            _TRACKED_CAR_POS_Y.append(data[16])
            _TRACKED_CAR_VEL_X.append(data[17])
            _TRACKED_CAR_VEL_Y.append(data[18])
            _CAM_ORIGIN_X.append(data[19])
            _CAM_ORIGIN_Y.append(data[20])
            _DRONE_SPEED.append(data[21])
            _DRONE_ALPHA.append(data[22])
            _DRONE_VEL_X.append(data[23])
            _DRONE_VEL_Y.append(data[24])
            _MEASURED_CAR_POS_X.append(data[25])
            _MEASURED_CAR_POS_Y.append(data[26])
            _MEASURED_CAR_VEL_X.append(data[27])
            _MEASURED_CAR_VEL_Y.append(data[28])
            _DRONE_ALTITUDE.append(data[29])
            _ABS_DEN.append(data[30])
            _MEASURED_R.append(data[31])
            _MEASURED_THETA.append(data[32])
            _MEASURED_V_R.append(data[33])
            _MEASURED_V_THETA.append(data[34])
            _TRUE_R.append(data[35])
            _TRUE_THETA.append(data[36])
            _TRUE_V_R.append(data[37])
            _TRUE_V_THETA.append(data[38])
            _DELTA_TIME.append(data[39])
            _Y1.append(data[40])
            _Y2.append(data[41])

        FILE.close()

        # plot
        import matplotlib.pyplot as plt
        import scipy.stats as st

        _PATH = f'./sim_outputs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        _prep_temp_folder(os.path.realpath(_PATH))

        # copy the plot_info file to the where plots figured will be saved
        shutil.copyfile('plot_info.txt', f'{_PATH}/plot_info.txt')
        plt.style.use('seaborn-whitegrid')

        # -------------------------------------------------------------------------------- figure 1
        # line of sight kinematics 1
        if SHOW_ALL or SHOW_LOS_KIN_1:
            f0, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.25})
            if SUPTITLE_ON:
                f0.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)

            # t vs r
            axs[0].plot(
                _TIME,
                _MEASURED_R,
                color='goldenrod',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$measured\ r$',
                alpha=0.9)
            axs[0].plot(
                _TIME,
                _R,
                color='royalblue',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$estimated\ r$',
                alpha=0.9)
            axs[0].plot(
                _TIME,
                _TRUE_R,
                color='red',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$true\ r$',
                alpha=0.9)

            axs[0].legend(loc='upper right')
            axs[0].set(ylabel=r'$r\ (m)$')
            axs[0].set_title(r'$\mathbf{r}$', fontsize=SUB_TITLE_FONT_SIZE)

            # t vs θ
            axs[1].plot(
                _TIME,
                _MEASURED_THETA,
                color='goldenrod',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$measured\ \theta$',
                alpha=0.9)
            axs[1].plot(
                _TIME,
                _THETA,
                color='royalblue',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$estimated\ \theta$',
                alpha=0.9)
            axs[1].plot(
                _TIME,
                _TRUE_THETA,
                color='red',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$true\ \theta$',
                alpha=0.9)

            axs[1].legend(loc='upper right')
            axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\theta\ (^{\circ})$')
            axs[1].set_title(r'$\mathbf{\theta}$', fontsize=SUB_TITLE_FONT_SIZE)

            f0.savefig(f'{_PATH}/1_los1.png', dpi=300)
            f0.show()

        # -------------------------------------------------------------------------------- figure 2
        # line of sight kinematics 2
        if SHOW_ALL or SHOW_LOS_KIN_2:
            f1, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.25})
            if SUPTITLE_ON:
                f1.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ II}$', fontsize=TITLE_FONT_SIZE)

            # t vs vr
            axs[0].plot(
                _TIME,
                _MEASURED_V_R,
                color='palegoldenrod',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$measured\ V_{r}$',
                alpha=0.9)
            axs[0].plot(
                _TIME,
                _V_R,
                color='royalblue',
                linestyle='-',
                linewidth=LINE_WIDTH_2,
                label=r'$estimated\ V_{r}$',
                alpha=0.9)
            axs[0].plot(
                _TIME,
                _TRUE_V_R,
                color='red',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$true\ V_{r}$',
                alpha=0.9)

            axs[0].legend(loc='upper right')
            axs[0].set(ylabel=r'$V_{r}\ (\frac{m}{s})$')
            axs[0].set_title(r'$\mathbf{V_{r}}$', fontsize=SUB_TITLE_FONT_SIZE)

            # t vs vtheta
            axs[1].plot(
                _TIME,
                _MEASURED_V_THETA,
                color='palegoldenrod',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$measured\ V_{\theta}$',
                alpha=0.9)
            axs[1].plot(
                _TIME,
                _V_THETA,
                color='royalblue',
                linestyle='-',
                linewidth=LINE_WIDTH_2,
                label=r'$estimated\ V_{\theta}$',
                alpha=0.9)
            axs[1].plot(
                _TIME,
                _TRUE_V_THETA,
                color='red',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$true\ V_{\theta}$',
                alpha=0.9)

            axs[1].legend(loc='upper right')
            axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$V_{\theta}\ (\frac{^{\circ}}{s})$')
            axs[1].set_title(r'$\mathbf{V_{\theta}}$', fontsize=SUB_TITLE_FONT_SIZE)

            f1.savefig(f'{_PATH}/1_los2.png', dpi=300)
            f1.show()

        # -------------------------------------------------------------------------------- figure 2
        # acceleration commands
        if SHOW_ALL or SHOW_ACCELERATIONS:
            f2, axs = plt.subplots()
            if SUPTITLE_ON:
                f2.suptitle(r'$\mathbf{Acceleration\ commands}$', fontsize=TITLE_FONT_SIZE)

            axs.plot(
                _TIME,
                _DRONE_ACC_LAT,
                color='forestgreen',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$a_{lat}$',
                alpha=0.9)
            axs.plot(
                _TIME,
                _DRONE_ACC_LNG,
                color='deeppink',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$a_{long}$',
                alpha=0.9)
            axs.legend()
            axs.set(xlabel=r'$time\ (s)$', ylabel=r'$acceleration\ (\frac{m}{s_{2}})$')

            f2.savefig(f'{_PATH}/2_accel.png', dpi=300)
            f2.show()

        # -------------------------------------------------------------------------------- figure 3
        # trajectories
        if SHOW_ALL or SHOW_TRAJECTORIES:
            f3, axs = plt.subplots(2, 1, gridspec_kw={'hspace': 0.4})
            if SUPTITLE_ON:
                f3.suptitle(
                    r'$\mathbf{Vehicle\ and\ UAS\ True\ Trajectories}$',
                    fontsize=TITLE_FONT_SIZE)

            ndx = np.array(_DRONE_POS_X) + np.array(_CAM_ORIGIN_X)
            ncx = np.array(_CAR_POS_X) + np.array(_CAM_ORIGIN_X)
            ndy = np.array(_DRONE_POS_Y) + np.array(_CAM_ORIGIN_Y)
            ncy = np.array(_CAR_POS_Y) + np.array(_CAM_ORIGIN_Y)

            axs[0].plot(
                ndx,
                ndy,
                color='darkslategray',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$UAS$',
                alpha=0.9)
            axs[0].plot(
                ncx,
                ncy,
                color='limegreen',
                linestyle='-',
                linewidth=LINE_WIDTH_2,
                label=r'$Vehicle$',
                alpha=0.9)
            axs[0].set(ylabel=r'$y\ (m)$')
            axs[0].set_title(r'$\mathbf{World\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[0].legend()

            ndx = np.array(_DRONE_POS_X)
            ncx = np.array(_CAR_POS_X)
            ndy = np.array(_DRONE_POS_Y)
            ncy = np.array(_CAR_POS_Y)

            x_pad = (max(ncx) - min(ncx)) * 0.05
            y_pad = (max(ncy) - min(ncy)) * 0.05
            xl = max(abs(max(ncx)), abs(min(ncx))) + x_pad
            yl = max(abs(max(ncy)), abs(min(ncy))) + y_pad
            axs[1].plot(
                ndx,
                ndy,
                color='darkslategray',
                marker='+',
                markersize=10,
                label=r'$UAS$',
                alpha=0.7)
            axs[1].plot(
                ncx,
                ncy,
                color='limegreen',
                linestyle='-',
                linewidth=LINE_WIDTH_2,
                label=r'$Vehicle$',
                alpha=0.9)
            axs[1].set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$')
            axs[1].set_title(r'$\mathbf{Camera\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[1].legend(loc='lower right')
            axs[1].set_xlim(-xl, xl)
            axs[1].set_ylim(-yl, yl)
            f3.savefig(f'{_PATH}/3_traj.png', dpi=300)
            f3.show()

        # -------------------------------------------------------------------------------- figure 4
        # true and estimated trajectories
        if SHOW_ALL or SHOW_CARTESIAN_PLOTS:
            f4, axs = plt.subplots()
            if SUPTITLE_ON:
                f4.suptitle(
                    r'$\mathbf{Vehicle\ True\ and\ Estimated\ Trajectories}$',
                    fontsize=TITLE_FONT_SIZE)

            axs.plot(
                _TRACKED_CAR_POS_X,
                _TRACKED_CAR_POS_Y,
                color='darkturquoise',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$estimated\ trajectory$',
                alpha=0.9)
            axs.plot(
                _CAR_POS_X,
                _CAR_POS_Y,
                color='crimson',
                linestyle=':',
                linewidth=LINE_WIDTH_1,
                label=r'$true\ trajectory$',
                alpha=0.9)
            axs.set_title(r'$\mathbf{camera\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs.legend()
            axs.axis('equal')
            axs.set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$')
            f4.savefig(f'{_PATH}/4_traj_comp.png', dpi=300)
            f4.show()

        # -------------------------------------------------------------------------------- figure 5
        # true and tracked pos
        if SHOW_ALL or SHOW_CARTESIAN_PLOTS:
            f4, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.4})
            if SUPTITLE_ON:
                f4.suptitle(
                    r'$\mathbf{Vehicle\ True\ and\ Estimated\ Positions}$',
                    fontsize=TITLE_FONT_SIZE)

            axs[0].plot(
                _TIME,
                _TRACKED_CAR_POS_X,
                color='rosybrown',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$estimated\ x$',
                alpha=0.9)
            axs[0].plot(
                _TIME,
                _CAR_POS_X,
                color='red',
                linestyle=':',
                linewidth=LINE_WIDTH_1,
                label=r'$true\ x$',
                alpha=0.9)
            axs[0].set(ylabel=r'$x\ (m)$')
            axs[0].set_title(r'$\mathbf{x}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[0].legend()
            axs[1].plot(
                _TIME,
                _TRACKED_CAR_POS_Y,
                color='mediumseagreen',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$estimated\ y$',
                alpha=0.9)
            axs[1].plot(
                _TIME,
                _CAR_POS_Y,
                color='green',
                linestyle=':',
                linewidth=LINE_WIDTH_1,
                label=r'$true\ y$',
                alpha=0.9)
            axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$y\ (m)$')
            axs[1].set_title(r'$\mathbf{y}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[1].legend()
            f4.savefig(f'{_PATH}/5_pos_comp.png', dpi=300)
            f4.show()

        # -------------------------------------------------------------------------------- figure 6
        # true and tracked velocities
        if SHOW_ALL or SHOW_CARTESIAN_PLOTS:
            f5, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.4})
            if SUPTITLE_ON:
                f5.suptitle(
                    r'$\mathbf{True,\ Measured\ and\ Estimated\ Vehicle\ Velocities}$',
                    fontsize=TITLE_FONT_SIZE)

            axs[0].plot(
                _TIME,
                _MEASURED_CAR_VEL_X,
                color='paleturquoise',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$measured\ V_x$',
                alpha=0.9)
            axs[0].plot(
                _TIME,
                _TRACKED_CAR_VEL_X,
                color='darkturquoise',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$estimated\ V_x$',
                alpha=0.9)
            axs[0].plot(
                _TIME,
                _CAR_VEL_X,
                color='crimson',
                linestyle='-',
                linewidth=LINE_WIDTH_2,
                label=r'$true\ V_x$',
                alpha=0.7)
            axs[0].set(ylabel=r'$V_x\ (\frac{m}{s})$')
            axs[0].set_title(r'$\mathbf{V_x}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[0].legend(loc='upper right')

            axs[1].plot(
                _TIME,
                _MEASURED_CAR_VEL_Y,
                color='paleturquoise',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$measured\ V_y$',
                alpha=0.9)
            axs[1].plot(
                _TIME,
                _TRACKED_CAR_VEL_Y,
                color='darkturquoise',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$estimated\ V_y$',
                alpha=0.9)
            axs[1].plot(
                _TIME,
                _CAR_VEL_Y,
                color='crimson',
                linestyle='-',
                linewidth=LINE_WIDTH_2,
                label=r'$true\ V_y$',
                alpha=0.7)
            axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$V_y\ (\frac{m}{s})$')
            axs[1].set_title(r'$\mathbf{V_y}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[1].legend(loc='upper right')

            f5.savefig(f'{_PATH}/6_vel_comp.png', dpi=300)
            f5.show()

        # -------------------------------------------------------------------------------- figure 7
        # speed and heading
        if SHOW_ALL or SHOW_SPEED_HEADING:
            f6, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.4})
            if SUPTITLE_ON:
                f6.suptitle(
                    r'$\mathbf{Vehicle\ and\ UAS,\ Speed\ and\ Heading}$',
                    fontsize=TITLE_FONT_SIZE)
            c_speed = (CAR_INITIAL_VELOCITY[0]**2 + CAR_INITIAL_VELOCITY[1]**2)**0.5
            c_heading = degrees(atan2(CAR_INITIAL_VELOCITY[1], CAR_INITIAL_VELOCITY[0]))

            axs[0].plot(_TIME,
                        [c_speed for i in _DRONE_SPEED],
                        color='lightblue',
                        linestyle='-',
                        linewidth=LINE_WIDTH_1,
                        label=r'$|V_{vehicle}|$',
                        alpha=0.9)
            axs[0].plot(
                _TIME,
                _DRONE_SPEED,
                color='blue',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$|V_{UAS}|$',
                alpha=0.9)
            axs[0].set(ylabel=r'$|V|\ (\frac{m}{s})$')
            axs[0].set_title(r'$\mathbf{speed}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[0].legend()

            axs[1].plot(_TIME, [c_heading for i in _DRONE_ALPHA], color='lightgreen',
                        linestyle='-', linewidth=LINE_WIDTH_2, label=r'$\angle V_{vehicle}$', alpha=0.9)
            axs[1].plot(
                _TIME,
                _DRONE_ALPHA,
                color='green',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$\angle V_{UAS}$',
                alpha=0.9)
            axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\angle V\ (^{\circ})$')
            axs[1].set_title(r'$\mathbf{heading}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[1].legend()

            f6.savefig(f'{_PATH}/7_speed_head.png', dpi=300)
            f6.show()

        # -------------------------------------------------------------------------------- figure 7
        # altitude profile
        if SHOW_ALL or SHOW_ALTITUDE_PROFILE:
            f7, axs = plt.subplots()
            if SUPTITLE_ON:
                f7.suptitle(r'$\mathbf{Altitude\ profile}$', fontsize=TITLE_FONT_SIZE)
            axs.plot(
                _TIME,
                _DRONE_ALTITUDE,
                color='darkgoldenrod',
                linestyle='-',
                linewidth=2,
                label=r'$altitude$',
                alpha=0.9)
            axs.set(xlabel=r'$time\ (s)$', ylabel=r'$z\ (m)$')

            f7.savefig(f'{_PATH}/8_alt_profile.png', dpi=300)
            f7.show()

        # -------------------------------------------------------------------------------- figure 7
        # 3D Trajectories
        ndx = np.array(_DRONE_POS_X) + np.array(_CAM_ORIGIN_X)
        ncx = np.array(_CAR_POS_X) + np.array(_CAM_ORIGIN_X)
        ndy = np.array(_DRONE_POS_Y) + np.array(_CAM_ORIGIN_Y)
        ncy = np.array(_CAR_POS_Y) + np.array(_CAM_ORIGIN_Y)

        if SHOW_ALL or SHOW_3D_TRAJECTORIES:
            f8 = plt.figure()
            if SUPTITLE_ON:
                f8.suptitle(r'$\mathbf{3D\ Trajectories}$', fontsize=TITLE_FONT_SIZE)
            axs = f8.add_subplot(111, projection='3d')
            axs.plot3D(
                ncx,
                ncy,
                0,
                color='limegreen',
                linestyle='-',
                linewidth=2,
                label=r'$Vehicle$',
                alpha=0.9)
            axs.plot3D(
                ndx,
                ndy,
                _DRONE_ALTITUDE,
                color='darkslategray',
                linestyle='-',
                linewidth=LINE_WIDTH_1,
                label=r'$UAS$',
                alpha=0.9)

            for point in zip(ndx, ndy, _DRONE_ALTITUDE):
                x = [point[0], point[0]]
                y = [point[1], point[1]]
                z = [point[2], 0]
                axs.plot3D(x, y, z, color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.1)
            axs.plot3D(ndx, ndy, 0, color='silver', linestyle='-', linewidth=1, alpha=0.9)
            axs.scatter3D(ndx, ndy, _DRONE_ALTITUDE, c=_DRONE_ALTITUDE, cmap='plasma', alpha=0.3)

            axs.set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$', zlabel=r'$z\ (m)$')
            axs.view_init(elev=41, azim=-105)
            # axs.view_init(elev=47, azim=-47)
            axs.set_title(r'$\mathbf{World\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs.legend()

            f8.savefig(f'{_PATH}/9_3D_traj.png', dpi=300)
            f8.show()

        # -------------------------------------------------------------------------------- figure 7
        # delta time
        if SHOW_ALL or SHOW_DELTA_TIME_PROFILE:
            f9, axs = plt.subplots(2, 1, gridspec_kw={'hspace': 0.4})
            if SUPTITLE_ON:
                f9.suptitle(r'$\mathbf{Time\ Delay\ profile}$', fontsize=TITLE_FONT_SIZE)
            axs[0].plot(
                _TIME,
                _DELTA_TIME,
                color='darksalmon',
                linestyle='-',
                linewidth=2,
                label=r'$\Delta\ t$',
                alpha=0.9)
            axs[0].set(xlabel=r'$time\ (s)$', ylabel=r'$\Delta t\ (s)$')

            _NUM_BINS = 300
            _DIFF = max(_DELTA_TIME) - min(_DELTA_TIME)
            _BAR_WIDTH = _DIFF/_NUM_BINS if USE_REAL_CLOCK else DELTA_TIME * 0.1
            _RANGE = (min(_DELTA_TIME), max(_DELTA_TIME)) if USE_REAL_CLOCK else (-2*abs(DELTA_TIME), 4*abs(DELTA_TIME))
            _HIST = np.histogram(_DELTA_TIME, bins=_NUM_BINS, range=_RANGE, density=1) if USE_REAL_CLOCK else np.histogram(_DELTA_TIME, bins=_NUM_BINS, density=1)
            axs[1].bar(_HIST[1][:-1], _HIST[0]/sum(_HIST[0]), width=_BAR_WIDTH*0.9, 
                        color='lightsteelblue', label=r'$Frequentist\ PMF\ distribution$', alpha=0.9)
            if not USE_REAL_CLOCK:
                axs[1].set_xlim(-2*abs(DELTA_TIME), 4*abs(DELTA_TIME))
            
            if USE_REAL_CLOCK:
                _MIN, _MAX = axs[1].get_xlim()
                axs[1].set_xlim(_MIN, _MAX)
                _KDE_X = np.linspace(_MIN, _MAX, 301)
                _GAUSS_KER = st.gaussian_kde(_DELTA_TIME)
                _PDF_DELTA_T = _GAUSS_KER.pdf(_KDE_X)
                axs[1].plot(_KDE_X, _PDF_DELTA_T/sum(_PDF_DELTA_T), color='royalblue', linestyle='-',
                            linewidth=2, label=r'$Gaussian\ Kernel\ Estimate\ PDF$', alpha=0.8)
            axs[1].set(ylabel=r'$Probabilities$', xlabel=r'$\Delta t\ values$')
            axs[1].legend(loc='upper left')

            f9.savefig(f'{_PATH}/9_delta_time.png', dpi=300)

            f9.show()
        plt.show()

    if RUN_VIDEO_WRITER:
        EXPERIMENT_MANAGER = ExperimentManager()
        # create folder path inside ./sim_outputs
        _PATH = f'./sim_outputs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        _prep_temp_folder(os.path.realpath(_PATH))
        VID_PATH = f'{_PATH}/sim_track_control.avi'
        print('Making video.')
        EXPERIMENT_MANAGER.make_video(VID_PATH, SIMULATOR_TEMP_FOLDER)
