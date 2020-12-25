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
from math import atan2, degrees, cos, sin, pi

import numpy as np
import cv2 as cv
import pygame


from pygame.locals import *                                 #pylint: disable=unused-wildcard-import
from settings import *                                      #pylint: disable=unused-wildcard-import
from optical_flow_config import (FARNEBACK_PARAMS,          #pylint: disable=unused-import
                                 FARN_TEMP_FOLDER,
                                 FEATURE_PARAMS,
                                 LK_PARAMS,
                                 LK_TEMP_FOLDER)


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
                             put_text,
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
                import CorrelationCoeffNormed


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
        r, g, b = BLOCK_COLOR
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

    def start_new(self):
        """Initializes simulation components.
        """
        self.time = 0.0

        # initiate screen shot generator
        self.screen_shot = self.screen_saver(path=TEMP_FOLDER)

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

        Returns:
            bool: Can tracker begin tracking
        """
        # ready = True

        # not ready if bb not selected or if simulated is still paused
        if (self.bb_start and self.bb_end) or not self.pause:
            self.manager.image_deque.clear()
            ready = True

        return self.tracker_ready

    def quit(self):
        """Helper function, sets running flag to False and quits pygame.
        """
        self.running = False
        pygame.quit()


class Tracker:
    """Tracker object is desgined to work with and ExperimentManager object.
    It can be used to process screen captures and produce tracking information for feature points.
    Computer Vision techniques employed here.
    """

    def __init__(self, manager):

        self.manager = manager

        self.frame_old_gray = None
        self.frame_old_color = None
        self.frame_new_color_edited = None
        self.frame_new_gray = None
        self.frame_new_color = None

        self.keypoints_old = None
        self.keypoints_new = None
        self.keypoints_old_good = None
        self.keypoints_new_good = None

        self.feature_found_statuses = None
        self.cross_feature_errors = None

        self.target_descriptors = None
        self.target_template_gray = None
        self.target_template_color = None

        self.detector = Sift()
        self.descriptor_matcher = BruteL2()
        self.template_matcher = CorrelationCoeffNormed()

        self.frame_1 = None
        self.cur_frame = None
        self.cur_img = None
        self.nxt_frame = None
        self.cur_points = None
        self._can_begin_control_flag = False    # will be modified in process_image
        self.kin = None
        self.window_size = 5
        self.prev_car_pos = None
        self.count = 0
        self._target_old_occluded_flag = False
        self._target_new_occluded_flag = False

        self._frame_num = 0
        self.track_length = 10
        self.tracker_info_mask = None
        self.target_feature_mask = None
        self.win_name = 'Tracking in progress'
        self.img_dumper = ImageDumper(TRACKER_TEMP_FOLDER)

        self._FAILURE = False, None
        self._SUCCESS = True, self.kin

    def is_first_time(self):
        # this function is called in process_image in the beginning. 
        # it indicates if this the first time process_image received a frame
        return self.frame_old_gray is None

    def is_target_occluded_in_old_frame(self):
        # Given the context and mechanism, it indicates if target was occluded in old frame;
        # the function call would semantically equate to a question asked about occlusion 
        # in the previous frame
        # syntactically, we could return the occlusion flag to serve the purpose
        return self._target_old_occluded_flag

    def is_target_occluded_in_new_frame(self):
        # Given the context and mechanism, it indicates if target is occluded in new frame;
        # the function call would semantically equate to a question asked about occlusion in 
        # the new frame, which would entail inferring from flow computations and descriptor matching
        
        # occlusion detection from flow criterion
        # 1. Rise in flow computation error residual.
        # occlusion detection from feature matching
        # 1. Drop in similarity of descriptor at old and new points
        # if (not self.feature_found_statuses.all() or 
        #         self.cross_feature_errors.max() > 15 or
        #         not self.is_target_descriptor_matching()):
        #     self._target_new_occluded_flag = True
        return self._target_new_occluded_flag

    def detect_occlusion_in_new_frame(self):
        # Infers from flow computations and descriptor and template matching algorithms
        # if target is occluded (partial/total)
        
        # occlusion detection from flow criterion
        # 1. Rise in flow computation error residual.
        # occlusion detection from feature matching
        # 1. Drop in similarity of descriptor pairs at old and new points
        if (not self.feature_found_statuses.all() or 
                self.cross_feature_errors.max() > 15 or
                not self.is_target_descriptor_matching()):
            self._target_new_occluded_flag = True

    def is_target_descriptor_matching(self):
        # check if target descriptors are matching
        keyPoints = [cv.KeyPoint(*kp.ravel(), 15) for kp in self.keypoints_new_good]
        descriptors_new = self.detector.get_descriptors_at_keypoints(self.frame_new_gray, 
                                                                       keyPoints) #self.keypoints_new_good)
        
        matches = self.descriptor_matcher.compute_matches(self.target_descriptors, descriptors_new)
        distances = [m.distance for m in matches]
        mxd = max(distances)
        is_match = True if mxd < 70 else False
        return is_match

    def is_target_template_matching(self):
        # check if target template is matching
        new_location = self._get_target_image_location()
        new_location_patch = self.get_neighborhood_patch(self.frame_new_gray, new_location, (25,25))
        match_result = self.template_matcher.compute_match(new_location_patch, self.target_template_gray)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_result)

        is_match = True if match_result[max_loc] < 0.95 else False

    def target_recovered(self):
        return False

    @staticmethod
    def get_bb_patch_from_image(img, bounding_box):
        x, y, w, h = bounding_box
        return img[y:y+h, x:x+w] # same for color or gray

    @staticmethod
    def get_neighborhood_patch(img, center, size):
        x = center[0] - size[0]/2
        y = center[1] - size[1]/2

        return get_bb_patch_from_image(img, (x, y, *size))

    def save_target_descriptors(self):
        # use keypoints from old frame, 
        # save descriptors of good keypoints
        keyPoints = [cv.KeyPoint(*kp.ravel(), 15) for kp in self.keypoints_old_good]
        self.target_descriptors = self.detector.get_descriptors_at_keypoints(self.frame_old_gray, 
                                                                             keyPoints) #self.keypoints_old_good)

    def save_target_template(self):
        # use the bounding box location to save the template
        # compute bounding box center
        x, y, w, h = bb = self.manager.get_target_bounding_box()
        center = tuple(map(int, (x+w/2, y+h/2)))

        self.target_template_color = self.get_bb_patch_from_image(self.frame_old_color, bb)
        self.target_template_gray = self.get_bb_patch_from_image(self.frame_old_gray, bb)

    def _get_kin_from_manager(self):
        #TODO switch based true or est 
        return self.manager.get_true_kinematics()

    def _get_target_image_location(self):
        kin = self._get_kin_from_manager()
        x,y = kin[2].elementwise()* (1,-1) / self.manager.simulator.pxm_fac
        target_location = (int(x), int(y))
        return target_location

    def target_redetected(self):
        return self.is_target_template_matching()

    def compute_flow(self):
        # it's main purpose is to compute new points
        # looks at 2 frames, uses flow, tells where old points went
        flow_output = compute_optical_flow_LK(self.frame_old_gray,
                                              self.frame_new_gray,
                                              self.keypoints_old, # good from previous frame
                                              LK_PARAMS)
        
        self.keypoints_old = flow_output[0]
        self.keypoints_new = flow_output[1]
        self.feature_found_statuses = flow_output[2]
        self.cross_feature_errors  = flow_output[3]

    def update_measured_kinematics(self):
        self.kin = self.compute_kinematics(self.keypoints_old_good.reshape(-1,2),
                                           self.keypoints_new_good.reshape(-1,2))
    
    @staticmethod
    def get_bounding_box_mask(img, x, y, width, height):
        # assume image is grayscale
        mask = np.zeros_like(img)
        mask[y:y+height, x:x+width] = 1
        return mask

    @staticmethod
    def get_patch_mask(img, patch_center, patch_size):
        x = patch_center[0] - patch_size //2
        y = patch_center[1] - patch_size //2
        mask = np.zeros_like(img)
        mask[y:y+patch_size[1], x:x+patch_size[0]] = 1
        return mask
        
    def copy_new_to_old(self):
        self.frame_old_gray = self.frame_new_gray.copy()
        self.frame_old_color = self.frame_new_color.copy()
        self.keypoints_old = self.keypoints_new.copy()
        self.keypoints_old_good = self.keypoints_new_good.copy()
        self._target_old_occluded_flag = self._target_new_occluded_flag

    def add_cosmetics(self, frame, mask, good_cur, good_nxt, kin):
        # draw tracks on the mask, apply mask to frame, save mask for future use
        img, mask = draw_tracks(frame, self.get_centroid(good_cur), self.get_centroid(
            good_nxt), [TRACK_COLOR], mask, track_thickness=2, radius=5, circle_thickness=1)
        for cur, nxt in zip(good_cur, good_nxt):
            img, mask = draw_tracks(frame, [cur], [nxt], [(204, 204, 204)], mask, track_thickness=1, radius=5, circle_thickness=1)
            

        # add optical flow arrows
        img = draw_sparse_optical_flow_arrows(
            img,
            self.get_centroid(good_cur),
            self.get_centroid(good_nxt),
            thickness=2,
            arrow_scale=ARROW_SCALE,
            color=RED_CV)

        # add a center
        img = cv.circle(img, SCREEN_CENTER, radius=1, color=DOT_COLOR, thickness=2)

        # draw axes
        img = cv.arrowedLine(img, (16, HEIGHT - 15), (41, HEIGHT - 15), (51, 51, 255), 2)
        img = cv.arrowedLine(img, (15, HEIGHT - 16), (15, HEIGHT - 41), (51, 255, 51), 2)

        # put velocity text
        img = self.put_metrics(img, kin)

        return img, mask

    def can_begin_control(self):
        return self._can_begin_control_flag  # and self.prev_car_pos is not None

    def compute_kinematics(self, cur_pts, nxt_pts):
        """Helper function, takes in current and next points (corresponding to an object) and computes the average velocity using elapsed simulation time from it's ExperimentManager.

        Args:
            cur_pts (np.ndarray): feature points in frame_1 or current frame (prev frame)
            nxt_pts (np.ndarray): feature points in frame_2 or next frame

        Returns:
            tuple(float, float), tuple(float, float): mean of positions and velocities computed from each point pair. Transformed to world coordinates.
        """
        # # check non-zero number of points
        num_pts = len(cur_pts)
        # if num_pts == 0:
        # return self.manager.simulator.camera.position,
        # self.manager.simulator.camera.velocity, self.car_position,
        # self.car_velocity

        # sum over all pairs and deltas between them
        car_x = 0
        car_y = 0
        car_vx = 0
        car_vy = 0

        for cur_pt, nxt_pt in zip(cur_pts, nxt_pts):
            car_x += nxt_pt[0]
            car_y += nxt_pt[1]
            car_vx += nxt_pt[0] - cur_pt[0]
            car_vy += nxt_pt[1] - cur_pt[1]

        # converting from px/frame to px/secs. Averaging
        d = self.manager.get_sim_dt() * num_pts
        car_x /= num_pts
        car_y /= num_pts
        car_vx /= d
        car_vy /= d

        # form (MEASURED, camera frame) car_position and car_velocity vectors (in
        # PIXELS and PIXELS/secs)
        car_position = pygame.Vector2((car_x, car_y))
        car_velocity = pygame.Vector2((car_vx, car_vy))

        # collect drone position, drone velocity and fov from simulator
        drone_position = self.manager.simulator.camera.position
        drone_velocity = self.manager.simulator.camera.velocity
        fov = self.manager.simulator.get_camera_fov()

        # transform (MEASURED) car position and car velocity to world reference
        # frame (also from PIXELS to METERS)
        cp = car_position.elementwise() * (1, -1) + (0, HEIGHT)
        cp *= self.manager.simulator.pxm_fac
        cp += - pygame.Vector2(fov) / 2

        cv = car_velocity.elementwise() * (1, -1)
        cv *= self.manager.simulator.pxm_fac

        # filter car kin
        if USE_FILTER:
            if not self.manager.filter.ready:
                self.manager.filter.init_filter(car_position, car_velocity)
            else:
                if not USE_KALMAN:
                    self.manager.filter.add_pos(car_position)
                    car_position = self.manager.filter.get_pos()
                    # car_velocity = self.manager.filter.get_vel()
                    if self.manager.get_sim_dt() == 0:
                        car_velocity = self.manager.filter.get_vel()
                    else:
                        car_velocity = (self.manager.filter.new_pos -
                                        self.manager.filter.old_pos) / self.manager.get_sim_dt()
                    # car_velocity = pygame.Vector2(car_velocity).elementwise() * (1, -1)
                    # car_velocity *= self.manager.simulator.pxm_fac
                    self.manager.filter.add_vel(car_velocity)
                else:  # KALMAN CASE
                    if self.count > 0:
                        self.count -= 1
                        car_position = self.manager.simulator.car.position
                        car_velocity = self.manager.simulator.car.velocity - self.manager.simulator.camera.velocity
                        self.manager.filter.add(car_position, car_velocity)
                    else:
                        # car_velocity = self.manager.simulator.car.velocity - self.manager.simulator.camera.velocity
                        self.manager.filter.add(car_position, car_velocity)
                        car_position = self.manager.filter.get_pos()
                        car_velocity = self.manager.filter.get_vel()
                    # if self.prev_car_pos is None:
                    #     car_velocity = self.manager.filter.get_vel()
                    # else:
                    #     car_velocity = (car_position - self.prev_car_pos ) / self.manager.get_sim_dt()
                    self.prev_car_pos = car_position

        # transform (ESTIMATED) car position and car velocity to world reference
        # frame (also from PIXELS to METERS)
        car_position = car_position.elementwise() * (1, -1) + (0, HEIGHT)
        car_position *= self.manager.simulator.pxm_fac
        car_position += - pygame.Vector2(fov) / 2

        car_velocity = car_velocity.elementwise() * (1, -1)
        car_velocity *= self.manager.simulator.pxm_fac

        # return kinematics in world reference frame
        return (
            drone_position,
            drone_velocity,
            car_position,
            car_velocity +
            drone_velocity,
            cp,
            cv +
            drone_velocity)
        # return (drone_position, drone_velocity, car_position, car_velocity, cp, cv)

    @staticmethod
    def get_centroid(points):
        """Returns centroid of given list of points

        Args:
            points (np.ndarray): List of points
        """
        centroid_x, centroid_y = 0, 0
        for point in points:
            centroid_x += point[0]
            centroid_y += point[1]

        return np.array([[int(centroid_x / len(points)), int(centroid_y / len(points))]])


    def process_image_new(self, nxt_frame):
        # save as new frame 
        self.frame_new_color = nxt_frame
        self.frame_new_gray = convert_to_grayscale(self.frame_new_color)

        if self.is_first_time():
            # comply with the selected bounding box, create target feature mask
            bounding_box = self.manager.get_target_bounding_box()
            self.target_feature_mask = self.get_bounding_box_mask(self.frame_new_gray, *bounding_box)

            # compute good features within the selected bounding box
            self.keypoints_new = cv.goodFeaturesToTrack(self.frame_new_gray, mask=self.target_feature_mask, **FEATURE_PARAMS)
            self.keypoints_new_good = self.keypoints_new.copy()

            # store new in old for next iteration
            self.copy_new_to_old()

            # save descriptors and template, first frame will always have no occlusion
            self.save_target_descriptors()
            self.save_target_template()
            return self._FAILURE
            

        # target was occluded
        if self.is_target_occluded_in_old_frame():  
            '''
            Assume target was not occluded in first frame.
            Target was occluded, meaning old frame had occlusion.
            In the new frame we will need to conclude if we still have occlusion
            Presumption is that upon occlusion, the descriptor from old image was 
            saved earlier. Positions (old points) would have to be either estimated 
            or known magically without using tracker. 
            Check if we find the features in vicinity of currently estimated position.
            Note: This is return failure, regardless of whether or not the target is found
            in the new frame. Since success is returned only when kin was computed successfully,
            which can only be done if target was no occluded in both old and new frames.
            '''
            if self.target_redetected():      # target re-detected
                '''
                if we find it then, save it and set occlusion to false, return failure
                so that in next iteration, tracker will assume no occlusion and use 
                the saved positions as old points to then compute new points.
                '''
                self.save_target_descriptors()
                self._target_occluded_flag = False

                return self._FAILURE
            else:                           # target not re-detected
                '''
                if we don't find it then, let the occlusion flag be,
                don't save new frame into old frame and return failure
                '''
                return self._FAILURE

        # target was not occluded
        '''
        We reach this point, means target is not occluded in old frame.
        no guarantees that it won't be occluded in new frame, so caution is necessary.
        All we can guarantee is that old points are available and probably correspond to our target.
        Also old frame is available. 
        So we compute new key points, corresponding to the target
        '''
        # compute optical flow
        # track current points in next frame, update new points, status and error
        self.compute_flow()

        # if target is occluded, it implies that the position tracking is far from accurate
        # therefore to sustain reliability in computed measurements, they will not be computed 
        # when occlusion is detected. We do not save new frame into old frame and return failure. 
        if self.is_target_occluded_in_new_frame():
            if self.target_recovered():  # can recover from occlusion (if partial)
                # compute kin
                self.update_measured_kinematics()
                # save keypoints new to old, but not descriptor
                self.copy_new_to_old()
                
                return self._SUCCESS
            else:                   # cannot recover from occlusion 
                # do not save new to old, do not save target descriptor
                return self._FAILURE
        
        # NOTE:
        # selecting good points does not make sense if we are not interested in partial occlusion
        # however, if we were, we would use "centroid adjustment" to recover bad keypoints using good keypoints

        # at this point we can assume target was detected successfully and old key points 
        # indeed correspond to the target in old frame, the trivial case!
        # Also, we can assume that each set keypoints_old_good and keypoints_new_good is not empty.
        # We can now use measurements (good points) to compute and update kinematics
        self.update_measured_kinematics()
        self.copy_new_to_old()

        
        return self._SUCCESS

    def print_to_console(self):
        if not CLEAN_CONSOLE:
            # NOTE: 
            # drone kinematics are assumed to be known (IMU and/or FPGA optical flow)
            # here, the drone position and velocity is known from Simulator
            # only the car kinematics are tracked/measured by tracker
            drone_position, drone_velocity, car_position, car_velocity, cp_, cv_ = self.kin
            print(f'TTTT >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:{vec_str(drone_position)} | v:{vec_str(drone_velocity)} | CAR - x:{vec_str(car_position)} | v:{vec_str(car_velocity)}')

    def display(self):
        if self.manager.tracker_display_on:
            # add cosmetics to frame_2 for display purpose
            self.frame_color_edited, self.tracker_info_mask = self.add_cosmetics(self.frame_new_color, 
                                                                                 self.tracker_info_mask,
                                                                                 self.keypoints_old_good,
                                                                                 self.keypoints_new_good,
                                                                                 self.kin)

            # set cur_img; to be used for saving # TODO investigated it's need, used in Simulator, fix it
            self.cur_img = self.frame_color_edited

            # show resultant img
            cv.imshow(self.win_name, self.frame_color_edited)
            cv.imshow("cur_frame", self.frame_cur_gray)
            cv.imshow("nxt_frame", self.frame_nxt_gray)

        # dump frames for analysis
        assembled_img = images_assemble([self.frame_cur_gray.copy(), self.frame_nxt_gray.copy(), self.frame_color_edited.copy()], (1,3))
        self.img_dumper.dump(assembled_img)

        # ready for next iteration. set cur frame and points to next frame and points
        # self.frame_cur_gray = self.frame_nxt_gray.copy()
        # self.key_point_set_cur = self.key_point_set_nxt_good.reshape(-1, 1, 2)  # -1 indicates to infer that dim size

        cv.waitKey(1)



            

        # difference between pts and good_pts is that good_pts correspond to stronger feature matches

        # -------------------------------------------------------------------------------
        # prepare for next iteration (cur <-- next) points, frames (gray only) 
        # old color frames are not important 
        # self.frame_cur_gray = self.frame_nxt_gray.copy()
        # self.frame_cur_color = self.frame_nxt_color.copy()
        # self.key_point_set_cur = self.key_point_set_nxt_good.reshape(-1, 1, 2)

    def process_image(self, img):
        """Processes given image and generates tracking information


        Args:
            img (np.ndarray): Rendered image capture from Simulator. (grayscale not guaranteed)

        Returns:
            [type]: [description]
        """
        # first frame will get different treatment
        if self.frame_cur_gray is None: # TODO this condition needs to be polished
                                        # this should determine if it's the first frame
            self.frame_cur_color = img

            # all processing to be done on grayscale of image
            self.frame_cur_gray = convert_to_grayscale(self.frame_cur_color)

            # initialize and compute target feature mask from bounding box
            self.target_feature_mask = np.zeros_like(self.frame_cur_gray)
            x, y, w, h = self.manager.simulator.bounding_box
            self.target_feature_mask[y : y+h+1, x : x+w+1] = 1

            # compute good features within the selected bounding box
            self.key_point_set_cur = cv.goodFeaturesToTrack(
                self.frame_cur_gray, mask= self.target_feature_mask, **FEATURE_PARAMS)

            # create mask for adding tracker information
            # tracker information mask will be applied to 3 channel RGB image, 
            # so mask will have 3 channels too
            self.tracker_info_mask = np.zeros_like(self.frame_cur_color)

            # if display is on, set window location
            if self.manager.tracker_display_on:
                from win32api import GetSystemMetrics
                cv.namedWindow(self.win_name)
                cv.moveWindow(self.win_name, GetSystemMetrics(0) - self.frame_cur_gray.shape[1] - 10, 0)
        else:
            self._can_begin_control_flag = True
            self.frame_nxt_color = img
            self.frame_nxt_gray = convert_to_grayscale(self.frame_nxt_color)

            # track current points in next frame, compute optical flow
            self.key_point_set_cur, self.key_point_set_nxt, stdev, err = compute_optical_flow_LK(self.frame_cur_gray,
                                                                                                 self.frame_nxt_gray,
                                                                                                 self.key_point_set_cur,
                                                                                                 LK_PARAMS)

            # select good points, with standard deviation 1. use numpy index trick
            # note: keypointset shape: (n,1,2); keypoint_good shape: (n,2)
            self.key_point_set_cur_good = self.key_point_set_cur[stdev == 1]
            self.key_point_set_nxt_good = self.key_point_set_nxt[stdev == 1]

            # print(f'TTTT0>> \nOCCLUDED: {self.occluded} \nstdev: \n{stdev.all()} \nmax err: {err.max()}\n')
            # detect occlusion
            if not stdev.all() or err.max() > 15:
                self.occluded = True

            # compute and create kinematics tuple # TODO this part needs to change
            if len(self.key_point_set_cur_good) == 0 or len(self.key_point_set_nxt_good) == 0:
                self.frame_cur_gray = self.frame_nxt_gray.copy()
                self.key_point_set_cur = self.key_point_set_nxt_good.reshape(-1, 1, 2)
                return False, None

            self.kin = self.compute_kinematics(self.key_point_set_cur_good.copy(),  # TODO can not pass any arguments instead
                                               self.key_point_set_nxt_good.copy())

            # note: the drone position and velocity is taken from simulator
            # drone kinematic are assumed to be known (IMU and/or FPGA optical flow)
            # only the car kinematics are tracked/measured by tracker,
            if not CLEAN_CONSOLE:
                drone_position, drone_velocity, car_position, car_velocity, cp_, cv_ = self.kin
                print(f'TTTT >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:{vec_str(drone_position)} | v:{vec_str(drone_velocity)} | CAR - x:{vec_str(car_position)} | v:{vec_str(car_velocity)}')

            if self.manager.tracker_display_on:
                # add cosmetics to frame_2 for display purpose
                self.frame_color_edited, self.tracker_info_mask = self.add_cosmetics(
                    self.frame_nxt_color, self.tracker_info_mask, self.key_point_set_cur_good, self.key_point_set_nxt_good, self.kin)

                # set cur_img; to be used for saving # TODO investigate need and fix
                self.cur_img = self.frame_color_edited

                # show resultant img
                cv.imshow(self.win_name, self.frame_color_edited)
                cv.imshow("cur_frame", self.frame_cur_gray)
                cv.imshow("nxt_frame", self.frame_nxt_gray)

            # dump frames for analysis
            assembled_img = images_assemble([self.frame_cur_gray.copy(), self.frame_nxt_gray.copy(), self.frame_color_edited.copy()], (1,3))
            self.img_dumper.dump(assembled_img)

            # ready for next iteration. set cur frame and points to next frame and points
            self.frame_cur_gray = self.frame_nxt_gray.copy()
            self.key_point_set_cur = self.key_point_set_nxt_good.reshape(-1, 1, 2)  # -1 indicates to infer that dim size

            cv.waitKey(1)

            return True, self.kin

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
            kin_str_1 = f'car_pos (m) : '      .rjust(20)
            kin_str_2 = f'<{k[2][0]:6.2f}, {k[2][1]:6.2f}>'
            kin_str_3 = f'car_vel (m/s) : '    .rjust(20)
            kin_str_4 = f'<{k[3][0]:6.2f}, {k[3][1]:6.2f}>'
            kin_str_5 = f'drone_pos (m) : '    .rjust(20)
            kin_str_6 = f'<{k[0][0]:6.2f}, {k[0][1]:6.2f}>'
            kin_str_7 = f'drone_vel (m/s) : '  .rjust(20)
            kin_str_8 = f'<{k[1][0]:6.2f}, {k[1][1]*-1:6.2f}>'
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

        occ_str = "TARGET OCCLUDED" if self._target_occluded_flag else "TARGET TRACKED"
        occ_color = RED_CV if self.occluded else METRICS_COLOR
        img = put_text(img, occ_str, (WIDTH//2 - 50, HEIGHT - 40),
                           font_scale=0.55, color=occ_color, thickness=1)
        return img


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
            orig = self.manager.get_cam_origin()
            X += orig[0]
            Y += orig[1]
            car_x += orig[0]
            car_y += orig[1]

        # speed of drone
        S = (Vx**2 + Vy**2) ** 0.5

        # distance between the drone and car
        r = ((car_x - X)**2 + (car_y - Y)**2)**0.5

        # heading angle of drone wrt x axis
        alpha = atan2(Vy, Vx)

        # angle of LOS from drone to car
        theta = atan2(car_y - Y, car_x - X)

        # heading angle of car
        beta = 0

        # compute vr and vtheta
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
            self.f.write(f'{self.manager.simulator.time},{r},{degrees(theta)},{degrees(Vtheta)},{Vr},{tru_kin[0][0]},{tru_kin[0][1]},{tru_kin[2][0]},{tru_kin[2][1]},{ax},{ay},{a_lat},{a_long},{tru_kin[3][0]},{tru_kin[3][1]},{tra_kin[2][0]},{tra_kin[2][1]},{tra_kin[3][0]},{tra_kin[3][1]},{self.manager.simulator.camera.origin[0]},{self.manager.simulator.camera.origin[1]},{S},{degrees(alpha)},{tru_kin[1][0]},{tru_kin[1][1]},{tra_kin[4][0]},{tra_kin[4][1]},{tra_kin[5][0]},{tra_kin[5][1]},{self.manager.simulator.camera.altitude},{abs(_D)},{r_},{degrees(theta_)},{Vr_},{degrees(Vtheta_)},{tr},{degrees(ttheta)},{tVr},{degrees(tVtheta)},{self.manager.simulator.dt},{y1},{y2}\n')

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

        self.filter = Kalman(self) if USE_KALMAN else MA(window_size=10)
        self.EKF = ExtendedKalman(self)

        self.image_deque = deque(maxlen=2)
        self.command_deque = deque(maxlen=2)
        self.kinematics_deque = deque(maxlen=2)

        self.sim_dt = 0
        self.true_rel_vel = None

        if self.save_on:
            self.simulator.save_screen = True

    def add_to_image_deque(self, img):
        """Helper function, adds given image to manager's image deque

        Args:
            img (np.ndarray): Image to be added to manager's image deque
        """
        self.image_deque.append(img)

    def add_to_command_deque(self, command):
        """Helper function, adds given command to manager's command deque

        Args:
            command (tuple(float, float)): Command to be added to manager's command deque
        """
        self.command_deque.append(command)

    def add_to_kinematics_deque(self, kinematics):
        """Helper function, adds given kinematics to manager's kinematics deque

        Args:
            kinematics (tuple(Vector2, Vector2, Vector2, Vector2, float)): Kinematics to be added to manager's kinematics deque
        """
        self.kinematics_deque.append(kinematics)

    def get_from_image_deque(self):
        """Helper function, gets image from manager's image deque
        """
        return self.image_deque.popleft()

    def get_from_command_deque(self):
        """Helper function, gets command from manager's command deque
        """
        return self.command_deque.popleft()

    def get_from_kinematics_deque(self):
        """Helper function, gets kinematics from manager's kinematics deque
        """
        return self.kinematics_deque.popleft()

    def get_target_bounding_box(self):
        return self.simulator.bounding_box

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
                    status = self.tracker.process_image_new(screen_capture)
                    self.tracker.print_to_console()
                    self.tracker.display
                    # kin = self.tracker.kin if status[0] else 

                    # let controller generate acceleration, when tracker indicates ok
                    if self.tracker.can_begin_control() and (
                            self.use_true_kin or self.tracker.kin is not None) and (
                            self.filter.done_waiting() or not USE_FILTER):
                        # collect kinematics tuple
                        kin = self.get_true_kinematics() if self.use_true_kin else self.tracker.kin
                        kin = self.tracker.kin if status[0] else self.get_true_kinematics()
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
        return self.tracker.kin

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
    CONTROL_ON = 1  # pylint: disable=bad-whitespace
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
        EXPERIMENT_MANAGER.make_video(VID_PATH, TEMP_FOLDER)
