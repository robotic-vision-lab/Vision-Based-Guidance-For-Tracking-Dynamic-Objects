import os
import sys
import random
import cv2 as cv
import numpy as np
import shutil
import time
import pygame 
import threading as th

from queue import deque
from PIL import Image
from copy import deepcopy
from random import randrange
from datetime import timedelta
from math import atan2, degrees, cos, sin, pi, copysign

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

from utils.vid_utils import create_video_from_images
from utils.optical_flow_utils import (get_OF_color_encoded, 
                                      draw_sparse_optical_flow_arrows,
                                      draw_tracks)
from utils.img_utils import convert_to_grayscale, put_text, images_assemble, add_salt_pepper
from utils.img_utils import scale_image as cv_scale_img
from game_utils import load_image, _prep_temp_folder, vec_str, scale_img
from algorithms.optical_flow import (compute_optical_flow_farneback, 
                                     compute_optical_flow_HS, 
                                     compute_optical_flow_LK)


""" Summary:
    Experiment 7:
    In this module we try to experiment with EKF.

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

    In this module the Manager runs the experiment, by calling methods from Simulator, Tracker and Controller.

"""



class Block(pygame.sprite.Sprite):
    """Defines a Block sprite.
    """

    # Constructor. Pass in the color of the block,
    # and its x and y position
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
        self.position = pygame.Vector2(randrange(-(WIDTH - self.rect.width),(WIDTH - self.rect.width))*self.simulator.pxm_fac, randrange(-(HEIGHT - self.rect.height),(HEIGHT - self.rect.height))*self.simulator.pxm_fac)
        self.velocity = pygame.Vector2(0.0,0.0)#(randrange(-50, 50), randrange(-50, 50))
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
        self.position = pygame.Vector2(random.uniform(drone_pos[0]-fov[0]/2,drone_pos[0]+fov[0]), random.uniform(drone_pos[1]-fov[1]/2,drone_pos[1]+fov[1]))
        # self.position = pygame.Vector2(randrange(-(WIDTH - self.rect.width),(WIDTH - self.rect.width))*self.simulator.pxm_fac, randrange(-(HEIGHT - self.rect.height),(HEIGHT - self.rect.height))*self.simulator.pxm_fac)
        self.velocity = pygame.Vector2(0.0,0.0)#(randrange(-50, 50), randrange(-50, 50))
        self.acceleration = pygame.Vector2(0.0, 0.0)


    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position
        self.velocity += self.acceleration * self.simulator.dt
        self.position += self.velocity * self.simulator.dt + 0.5 * self.acceleration * self.simulator.dt**2

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
        r,g,b = BLOCK_COLOR
        d = BLOCK_COLOR_DELTA
        r += random.randint(-d, d)
        g += random.randint(-d, d)
        b += random.randint(-d, d)
        self.image.fill((r,g,b))
        # self.image.fill(BLOCK_COLOR)


    def load(self):
        self.w /= self.simulator.alt_change_fac
        self.h /= self.simulator.alt_change_fac

        if self.w >= 2 and self.h >= 2:
            self.image = pygame.Surface((int(self.w), int(self.h)))
            self.fill_image()

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()

        # self.rect.center = self.position



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
        self.position += self.velocity * self.simulator.dt + 0.5 * self.acceleration * self.simulator.dt**2


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
        self.image, self.rect = self.simulator.car_img
        self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER



class DroneCamera(pygame.sprite.Sprite):
    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.drone_sprite]

        # call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        # self.drone = pygame.Rect(0, 0, WIDTH, HEIGHT)
        # self.image = pygame.Surface((20, 20))
        # self.image.fill(BLUE)
        # self.rect = self.image.get_rect()
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
        """[summary]
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
        """[summary]
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

        delta_pos = self.velocity * self.simulator.dt + 0.5 * self.acceleration * self.simulator.dt**2      # i know how this looks like but,
        self.position = self.velocity * self.simulator.dt + 0.5 * self.acceleration * self.simulator.dt**2  # donot touch ☠
        self.origin += delta_pos


    def compensate_camera_motion(self, sprite_obj):
        """[summary]

        Args:
            sprite_obj ([type]): [description]
        """
        sprite_obj.position -= self.position #self.velocity * self.game.dt + 0.5 * self.acceleration * self.game.dt**2
        sprite_obj.update_rect()
        # sprite_obj.rect.centerx = sprite_obj.rect.centerx - self.rect.centerx + WIDTH//2
        # sprite_obj.rect.centery = sprite_obj.rect.centery - self.rect.centery + HEIGHT//2
        

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
        self.simulator.alt_change_fac = 1.0 + self.alt_change/self.altitude
        self.altitude += self.alt_change
        

    def fly_lower(self):
        self.simulator.alt_change_fac = 1.0 - self.alt_change/self.altitude
        self.altitude -= self.alt_change



class Simulator:
    """Simulator object creates the simulation game. 
    Responds to keypresses 'SPACE' to toggle play/pause, 's' to save screen mode, ESC to quit.
    While running simulation, it also dumps the screens to a shared memory location.
    Designed to work with an ExperimentManager object.
    """
    def __init__(self, manager):

        self.manager = manager

        # initialize screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = "2,30"        
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
        self.euc_factor = 1.0
        self.pause = False
        self.time_font = pygame.font.SysFont(TIME_FONT, 16, False, False)
        self.bb_start = None
        self.bb_end = None
        self.bb_drag = False
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
        
        # spawn blocks
        self.blocks = []
        for i in range(NUM_BLOCKS):
            self.blocks.append(Block(self))

        # spawn car
        self.car = Car(self, *CAR_INITIAL_POSITION, *CAR_INITIAL_VELOCITY, *CAR_ACCELERATION)

        # spawn drone camera
        self.camera = DroneCamera(self)
        self.cam_accel_command = pygame.Vector2(0,0)
        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)

    
    def run(self):
        """Keeps simulation game running until quit.
        """
        self.running = True
        while self.running:
            # make clock tick and measure time elapsed
            self.dt = self.clock.tick(FPS) / 1000.0 
            if self.pause:          # DO NOT TOUCH! clock needs to tick regardless
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
        pygame.display.flip()          
        

    def handle_events(self):
        """Handles captured events.
        """
        # respond to all events posted in the event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT or     \
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit()
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.save_screen = not self.save_screen
                    if self.save_screen:
                        print("\nScreen recording started.")
                    else:
                        print("\nScreen recording stopped.")
                if event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    if self.pause:
                        self.bb_start = self.bb_end = None
                        print("\nSimulation paused.")
                    else:
                        print("\nSimulation running.")
                if event.key == pygame.K_i:
                    self.drone_up()
                if event.key == pygame.K_k:
                    self.drone_down()

            key_state = pygame.key.get_pressed()
            if key_state[pygame.K_LEFT]:
                self.cam_accel_command.x = -1
            if key_state[pygame.K_RIGHT]:
                self.cam_accel_command.x = 1
            if key_state[pygame.K_UP]:
                self.cam_accel_command.y = -1
            if key_state[pygame.K_DOWN]:
                self.cam_accel_command.y = 1

            self.euc_factor = 0.7071 if self.cam_accel_command == (1, 1) else 1.0

            if event.type == pygame.MOUSEBUTTONDOWN and self.pause:
                self.bb_start = self.bb_end = pygame.mouse.get_pos()
                self.bb_drag = True
            if self.bb_drag and event.type == pygame.MOUSEMOTION:
                self.bb_end = pygame.mouse.get_pos()

            if event.type == pygame.MOUSEBUTTONUP:
                self.bb_end = pygame.mouse.get_pos()
                self.bb_drag = False
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
        pygame.display.set_caption(f'  FPS {sim_fps} | car: x-{vec_str(self.car.position)} v-{vec_str(self.car.velocity)} a-{vec_str(self.car.acceleration)} | cam x-{vec_str(self.camera.position)} v-{vec_str(self.camera.velocity)} a-{vec_str(self.camera.acceleration)} ')


        # draw only car and blocks (not drone)
        self.car_block_sprites.draw(self.screen_surface)


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
            self.bounding_box = (x,y,w,h)
            pygame.draw.rect(self.screen_surface, BB_COLOR, pygame.rect.Rect(x, y, w, h), 2)
        
        if not CLEAR_TOP:
            # draw drone altitude info
            alt_str = f'car location - {self.car.rect.center}, Drone Altitude - {self.camera.altitude:0.2f}m, fac - {self.alt_change_fac:0.4f}, pxm - {self.pxm_fac:0.4f}'
            alt_surf = self.time_font.render(alt_str, True, TIME_COLOR)
            alt_rect = alt_surf.get_rect()
            self.screen_surface.blit(alt_surf, (15, 15))
            alt_str = f'drone location - {self.camera.rect.center}, FOV - {WIDTH * self.pxm_fac:0.2f}m x {HEIGHT * self.pxm_fac:0.2f}m'
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
            # make full file path string
            frame_num += 1
            image_name = f'frame_{str(frame_num).zfill(4)}.png'
            file_path = os.path.join(path, image_name)

            # get capture from simulator
            img_sim = self.get_screen_capture(save_mode=True)

            # get tracker image
            img_track = self.manager.tracker.cur_img
            if img_track is None:
                img_track = np.ones_like(img_sim, dtype='uint8') * TRACKER_BLANK
            
            # assemble images
            img = images_assemble([img_sim, img_track], (1,2))
            
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
            if not OPTION==0:
                img = cv_scale_img(img, SCALE_1)
                img = cv_scale_img(img, SCALE_2)
            if not SNR==1.0:
                img = add_salt_pepper(img, SNR)
                img = cv.GaussianBlur(img, (5,5), 0)

        return img


    def put_image(self):
        """Helper function, captures screen and adds to manager's image deque.
        """
        img = self.get_screen_capture()
        self.manager.add_to_image_deque(img)


    def drone_up(self):
        self.camera.fly_higher()
        self.pxm_fac = ((self.camera.altitude * PIXEL_SIZE) / FOCAL_LENGTH)
        car_scale = (CAR_LENGTH / (CAR_LENGTH_PX * self.pxm_fac)) / self.alt_change_fac
        self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True, scale=car_scale)
        self.car.load()
        for block in self.blocks:
            block.load()


    def drone_down(self):
        self.camera.fly_lower()
        self.pxm_fac = ((self.camera.altitude * PIXEL_SIZE) / FOCAL_LENGTH)
        car_scale = (CAR_LENGTH / (CAR_LENGTH_PX * self.pxm_fac)) / self.alt_change_fac
        self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True, scale=car_scale)
        self.car.load()
        for block in self.blocks:
            block.load()


    def get_drone_position(self):
        return self.camera.position

    
    def get_camera_fov(self):
        return (WIDTH * self.pxm_fac, HEIGHT * self.pxm_fac)


    def can_begin_tracking(self):
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
    """
    def __init__(self, manager):
        
        self.manager = manager
        self.frame_1 = None
        self.cur_frame = None
        self.cur_img = None
        self.cur_points = None
        self.win_name = 'Tracking in progress'
        self._can_begin_control_flag = False
        self.kin = None
        self.window_size = 5
        self.prev_car_pos = None
        self.count = 0

        # self.nxt_points = None
        # self.car_position = None
        # self.car_velocity = None


    def run(self):
        """Keeps running the tracker main functions.
        Reads bounding box from it's ExperimentManager and computed features to be tracked.
        """
        # get first frame once bounding box is selected 
        frame_1, cur_frame = self.get_first_frame()      
        
        # compute feature mask from selected bounding box
        feature_mask = np.zeros_like(cur_frame)
        x,y,w,h = self.manager.simulator.bounding_box
        feature_mask[y:y+h+1, x:x+w+1] = 1

        # compute good features in the selected bounding box
        cur_points = cv.goodFeaturesToTrack(cur_frame, mask=feature_mask, **FEATURE_PARAMS)

        # create mask for drawing tracks
        mask = np.zeros_like(frame_1)

        # set tracker window location 
        if self.manager.tracker_display_on:
            from win32api import GetSystemMetrics
            win_name = 'Tracking in progress'
            cv.namedWindow(win_name)
            cv.moveWindow(win_name, GetSystemMetrics(0)-frame_1.shape[1] -10, 0)

        # begin tracking
        while self.manager.simulator.running:

            # make sure there is more than 0 frames in the deque
            if len(self.manager.image_deque) < 0 or self.manager.get_sim_dt() == 0:
                continue

            # get next frame from image deque and convert to grayscale
            frame_2 = self.manager.get_from_image_deque()
            nxt_frame = convert_to_grayscale(frame_2)

            # compute optical flow between current and next frame
            cur_points, nxt_points, stdev, err = compute_optical_flow_LK( cur_frame, 
                                                                          nxt_frame, 
                                                                          cur_points, 
                                                                          LK_PARAMS )
            
            # select good points, with standard deviation 1. use numpy index trick
            good_cur = cur_points[stdev==1]
            good_nxt = nxt_points[stdev==1]

            # compute and create kinematics tuple
            if len(good_cur)==0 or len(good_nxt)==0:
                continue
            
            kin = self.compute_kinematics( good_cur.copy(), 
                                           good_nxt.copy() )
            drone_position, drone_velocity, car_position, car_velocity = kin
            if not CLEAN_CONSOLE:
                print(f'TTTT >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:{vec_str(drone_position)} | v:{vec_str(drone_velocity)} | CAR - x:{vec_str(car_position)} | v:{vec_str(car_velocity)}')
            
            # add kinematics tuple to manager's kinematics deque
            self.manager.add_to_kinematics_deque(kin)

            # cosmetics/visual aids if the display is on
            # create img with added tracks for all point pairs on next frame
            # give car positions in current and next frame
            # self.cur_img = nxt_frame
            if self.manager.tracker_display_on:
                # add cosmetics to frame_2 for display purpose
                img, mask = self.add_cosmetics(frame_2, mask, good_cur, good_nxt, kin)

                # set cur_img; to be used for saving 
                self.cur_img = img

                # show resultant img
                cv.imshow(win_name, img)            

            # ready for next iteration. set cur frame and points to next frame and points
            cur_frame = nxt_frame.copy()
            cur_points = good_nxt.reshape(-1, 1, 2) # -1 indicates to infer that dim size

            # every n seconds (n*FPS frames), get good points
            # num_seconds = 1
            # if frame_num % (num_seconds*FPS) == 0:
            #     pass#cur_points = cv.goodFeaturesToTrack(cur_frame, mask=None, **FEATURE_PARAMS)
                # for every point in good point if its not there in cur points, add , update color too
                
            cv.waitKey(1)

        cv.destroyAllWindows()


    def add_cosmetics(self, frame, mask, good_cur, good_nxt, kin):
        # draw tracks on the mask, add mask to frame, save mask for future use
        img, mask = draw_tracks(frame, self.get_centroid(good_cur), self.get_centroid(good_nxt), [TRACK_COLOR], mask, track_thickness=2)

        # add optical flow arrows 
        img = draw_sparse_optical_flow_arrows(img, self.get_centroid(good_cur), self.get_centroid(good_nxt), thickness=2, arrow_scale=ARROW_SCALE, color=RED_CV)

        # add a center
        img = cv.circle(img, SCREEN_CENTER, radius=1, color=DOT_COLOR, thickness=2)

        # draw axes
        img = cv.arrowedLine(img, (16,HEIGHT-15), (41, HEIGHT-15), (51,51,255), 2)
        img = cv.arrowedLine(img, (15,HEIGHT-16), (15, HEIGHT-41), (51,255,51), 2)

        # put velocity text 
        img = self.put_metrics(img, kin)

        return img, mask


    def get_first_frame(self):
        while True:
            # do nothing until bounding box is selected
            if not self.manager.simulator.can_begin_tracking():
                continue

            # get a frame from the image deque and break
            if len(self.manager.image_deque) > 0:
                frame_1 = self.manager.image_deque.popleft()
                cur_frame = convert_to_grayscale(frame_1)
                break
        
        self._can_begin_control_flag = True

        return frame_1, cur_frame


    def can_begin_control(self):
        return self._can_begin_control_flag #and self.prev_car_pos is not None


    def compute_kinematics(self, cur_pts, nxt_pts):
        """Helper function, takes in current and next points (corresponding to an object) and 
        computes the average velocity using elapsed simulation time from it's ExperimentManager.

        Args:
            cur_pts (np.ndarray): feature points in frame_1 or current frame (prev frame)
            nxt_pts (np.ndarray): feature points in frame_2 or next frame 

        Returns:
            tuple(float, float), tuple(float, float): mean of positions and velocities computed from each point pair. Transformed to world coordinates.
        """
        # # check non-zero number of points
        num_pts = len(cur_pts)
        # if num_pts == 0:
        #     return self.manager.simulator.camera.position, self.manager.simulator.camera.velocity, self.car_position, self.car_velocity
        
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
        
        # form (MEASURED, camera frame) car_position and car_velocity vectors (in PIXELS and PIXELS/secs)
        car_position = pygame.Vector2((car_x , car_y))
        car_velocity = pygame.Vector2((car_vx, car_vy))

        # collect drone position, drone velocity and fov from simulator
        drone_position = self.manager.simulator.camera.position
        drone_velocity = self.manager.simulator.camera.velocity
        fov = self.manager.simulator.get_camera_fov()

        # transform (MEASURED) car position and car velocity to world reference frame (also from PIXELS to METERS)
        cp = car_position.elementwise() * (1, -1) + (0, HEIGHT)
        cp *= self.manager.simulator.pxm_fac
        cp += - pygame.Vector2(fov)/2

        cv = car_velocity.elementwise() * (1, -1)
        cv *= self.manager.simulator.pxm_fac
        # cp = car_position
        # cv = car_velocity

        # filter car kin 
        if USE_FILTER:
            if not self.manager.filter.ready:
                self.manager.filter.init_filter(car_position, car_velocity)
            else:            
                if not USE_KALMAN:
                    self.manager.filter.add_pos(car_position)
                    car_position = self.manager.filter.get_pos()
                    # car_velocity = self.manager.filter.get_vel()
                    if self.manager.get_sim_dt()==0:
                        car_velocity = self.manager.filter.get_vel()
                    else:
                        car_velocity = (self.manager.filter.new_pos - self.manager.filter.old_pos) / self.manager.get_sim_dt()
                    # car_velocity = pygame.Vector2(car_velocity).elementwise() * (1, -1)
                    # car_velocity *= self.manager.simulator.pxm_fac
                    self.manager.filter.add_vel(car_velocity)
                else: # KALMAN CASE
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


        # transform (ESTIMATED) car position and car velocity to world reference frame (also from PIXELS to METERS)
        car_position = car_position.elementwise() * (1, -1) + (0, HEIGHT)
        car_position *= self.manager.simulator.pxm_fac
        car_position += - pygame.Vector2(fov)/2

        car_velocity = car_velocity.elementwise() * (1, -1)
        car_velocity *= self.manager.simulator.pxm_fac

        # return kinematics in world reference frame
        return (drone_position, drone_velocity, car_position, car_velocity+drone_velocity, cp, cv+drone_velocity)
        # return (drone_position, drone_velocity, car_position, car_velocity, cp, cv)

    
    def get_centroid(self, points):
        """Returns centroid of given list of points

        Args:
            points (np.ndarray): List of points
        """
        cx, cy = 0, 0
        for x, y in points:
            cx += x
            cy += y

        return np.array([[int(cx/len(points)), int(cy/len(points))]])


    def process_image(self, img):
        if self.cur_frame is None:
            # print('tracker 1st frame')
            self.frame_1 = img
            self.cur_frame = convert_to_grayscale(self.frame_1)

            # compute feature mask from selected bounding box
            feature_mask = np.zeros_like(self.cur_frame)
            x,y,w,h = self.manager.simulator.bounding_box
            feature_mask[y:y+h+1, x:x+w+1] = 1

            # compute good features in the selected bounding box
            self.cur_points = cv.goodFeaturesToTrack(self.cur_frame, mask=feature_mask, **FEATURE_PARAMS)

            # create mask for drawing tracks
            self.mask = np.zeros_like(self.frame_1)

            # set tracker window location 
            if self.manager.tracker_display_on:
                from win32api import GetSystemMetrics                
                cv.namedWindow(self.win_name)
                cv.moveWindow(self.win_name, GetSystemMetrics(0)-self.frame_1.shape[1] -10, 0)
        else:
            self._can_begin_control_flag = True
            self.frame_2 = img
            self.nxt_frame = convert_to_grayscale(self.frame_2)

            # track current points in next frame, compute optical flow
            self.cur_points, self.nxt_points, stdev, err = compute_optical_flow_LK( self.cur_frame, 
                                                                                    self.nxt_frame, 
                                                                                    self.cur_points, 
                                                                                    LK_PARAMS )
            
            # select good points, with standard deviation 1. use numpy index trick
            good_cur = self.cur_points[stdev==1]
            good_nxt = self.nxt_points[stdev==1]

            # compute and create kinematics tuple
            if len(good_cur)==0 or len(good_nxt)==0:
                return False, None

            self.kin = self.compute_kinematics( good_cur.copy(), 
                                                good_nxt.copy() )

            drone_position, drone_velocity, car_position, car_velocity, cp_, cv_ = self.kin
            if not CLEAN_CONSOLE:
                print(f'TTTT >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:{vec_str(drone_position)} | v:{vec_str(drone_velocity)} | CAR - x:{vec_str(car_position)} | v:{vec_str(car_velocity)}')

            if self.manager.tracker_display_on:
                # add cosmetics to frame_2 for display purpose
                img, self.mask = self.add_cosmetics(self.frame_2, self.mask, good_cur, good_nxt, self.kin)

                # set cur_img; to be used for saving 
                self.cur_img = img

                # show resultant img
                cv.imshow(self.win_name, img)

            # ready for next iteration. set cur frame and points to next frame and points
            self.cur_frame = self.nxt_frame.copy()
            self.cur_points = good_nxt.reshape(-1, 1, 2) # -1 indicates to infer that dim size

            cv.waitKey(1)

            return True, self.kin


    def put_velocity_text(self, img, velocity):
        """Helper function, put computed velocity in text form on (opencv) image.

        Args:
            img (np.ndarray): Image on which text is to be put.
            velocity (tuple(float, float)): velocity to be put in text form on image.

        Returns:
            np.ndarray: Image with velotcity text.
        """
        img = put_text(img, f'computed velocity: ', (WIDTH - 180, 25), font_scale=0.5, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, f'vx = {velocity[0]:.2f} ', (WIDTH - 130, 50), font_scale=0.5, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, f'vy = {velocity[1]:.2f} ', (WIDTH - 130, 75), font_scale=0.5, color=LIGHT_GRAY_2, thickness=1)

        return img


    def put_metrics(self, img, k):
        """Helper function, put metrics and stuffs on opencv image.

        Args:
            k (tuple): drone_position, drone_velocity, car_position, car_velocity

        Returns:
            [np.ndarray]: Image after putting all kinds of crap
        """
        if ADD_ALTITUDE_INFO:
            img = put_text(img, f'Altitude = {self.manager.simulator.camera.altitude:0.2f} m', (WIDTH-175, HEIGHT-15), font_scale=0.5, color=METRICS_COLOR, thickness=1)
            img = put_text(img, f'1 pixel = {self.manager.simulator.pxm_fac:0.4f} m', (WIDTH-175, HEIGHT-40), font_scale=0.5, color=METRICS_COLOR, thickness=1)

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

            img = put_text(img, kin_str_1,  (WIDTH - (330 + 25), 25),   font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_2,  (WIDTH - (155 + 25), 25),   font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_3,  (WIDTH - (328 + 25), 50),   font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_4,  (WIDTH - (155 + 25), 50),   font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_5,  (WIDTH - (332 + 25), 75),   font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_6,  (WIDTH - (155 + 25), 75),   font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_7,  (WIDTH - (330 + 25), 100),  font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_8,  (WIDTH - (155 + 25), 100),  font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_9,  (WIDTH - (340 + 25), 125),  font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_0,  (WIDTH - (155 + 25), 125),  font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_11, (WIDTH - (323 + 25), 150),  font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_12, (WIDTH - (155 + 25), 150),  font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_13, (WIDTH - (323 + 25), 175),  font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_14, (WIDTH - (155 + 25), 175),  font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_15, (50, HEIGHT - 15),          font_scale=0.45, color=METRICS_COLOR, thickness=1)

        return img



class Controller:
    def __init__(self, manager):
        self.manager = manager
        self.plot_info_file = 'plot_info.txt'
        self.R = CAR_RADIUS
        self.f = None
        self.a_ln = 0.0
        self.a_lt = 0.0


    def run(self):
        print('Controller running')
        if self.manager.write_plot:
            f = open(self.plot_info_file, '+w')
        R = CAR_RADIUS
        while True:
            # if self.manager.tracker_on and len(self.manager.kinematics_deque) == 0:
            #     continue
            if not self.manager.simulator.running:
                break
            if self.manager.simulator.pause:
                continue
            
            # kin = self.manager.get_from_kinematics_deque()
            kin = self.manager.get_true_kinematics() if self.manager.use_true_kin else self.manager.get_from_kinematics_deque()
            

            mpx_fac = 1/self.manager.simulator.pxm_fac
            
            x, y            = kin[0] #.elementwise() * (1, -1) * mpx_fac + SCREEN_CENTER#(0, HEIGHT)
            vx, vy          = kin[1] #.elementwise() * (1, -1) * mpx_fac 
            car_x, car_y    = kin[2] #.elementwise() * (1, -1) * mpx_fac + SCREEN_CENTER#(0, HEIGHT)
            car_speed, cvy  = kin[3] #.elementwise() * (1, -1) * mpx_fac

            if not CLEAN_CONSOLE:
                print(f'CCCC >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:[{x:0.2f}, {y:0.2f}] | v:[{vx:0.2f}, {vy:0.2f}] | CAR - x:[{car_x:0.2f}, {car_y:0.2f}] | v:[{car_speed:0.2f}, {cvy:0.2f}]')
            
            # speed of drone
            s = (vx**2 + vy**2) **0.5

            # distance between the drone and car
            r = ((car_x - x)**2 + (car_y - y)**2)**0.5

            # heading angle of drone wrt x axis
            # alpha = kin[4]
            alpha = atan2(vy,vx)

            # angle of LOS from drone to car
            theta = atan2(car_y-y, car_x-x)

            # heading angle of car
            beta = 0

            # compute vr and vtheta
            vr = car_speed * cos(beta - theta) - s * cos(alpha - theta)
            vtheta = car_speed * sin(beta - theta) - s * sin(alpha - theta)
            # print(car_speed, theta, vr, vtheta)
            # calculate y from drone to car
            y2 = vtheta**2 + vr**2
            y1 = r**2 * vtheta**2 - y2 * R**2

            # time to collision from drone to car
            # tm = -vr * r / (vtheta**2 + vr**2)

            # compute desired acceleration
            w = -0.1
            K1 = 0.1 * np.sign(-vr)
            K2 = 0.05

            # c = cos(alpha - theta)
            # s = sin(alpha - theta)
            # K1y1 = K1*y1
            # K2y2Vrr2 = K2*y2*vr*r**2
            # d = 2*vr*vtheta*r**2

            # a_lat = (K1y1 * (vr*c + vtheta*s) + K2y2Vrr2*c) / d
            # a_long = (K1y1 * (vr*s - vtheta*c) + K2y2Vrr2*s) / d

            # a_lat = (K1*vr*y1*cos(alpha - theta) + K1*vtheta*y1*sin(alpha - theta) + K2*R**2*vr*y2*cos(alpha - theta) + K2*R**2*vtheta*y2*sin(alpha - theta) - K2*vtheta*r**2*y2*sin(alpha - theta))/(2*(vr*vtheta*r**2*cos(alpha - theta)**2 + vr*vtheta*r**2*sin(alpha - theta)**2))
            # a_long = (K1*vr*y1*sin(alpha - theta) - K1*vtheta*y1*cos(alpha - theta) - K2*R**2*vtheta*y2*cos(alpha - theta) + K2*R**2*vr*y2*sin(alpha - theta) + K2*vtheta*r**2*y2*cos(alpha - theta))/(2*(vr*vtheta*r**2*cos(alpha - theta)**2 + vr*vtheta*r**2*sin(alpha - theta)**2))

            
            a_lat = (K1*vr*y1*cos(alpha - theta) + K1*vtheta*y1*sin(alpha - theta) + K2*R**2*vr*y2*cos(alpha - theta) + K2*R**2*vtheta*y2*sin(alpha - theta) - K2*vtheta*r**2*y2*sin(alpha - theta))/(2*(vr*vtheta*r**2*cos(alpha - theta)**2 + vr*vtheta*r**2*sin(alpha - theta)**2))
            a_long = (K1*vr*y1*sin(alpha - theta) - K1*vtheta*y1*cos(alpha - theta) - K2*R**2*vtheta*y2*cos(alpha - theta) + K2*R**2*vr*y2*sin(alpha - theta) + K2*vtheta*r**2*y2*cos(alpha - theta))/(2*(vr*vtheta*r**2*cos(alpha - theta)**2 + vr*vtheta*r**2*sin(alpha - theta)**2))

            a_long_bound = 8
            a_lat_bound = 8
            
            # a_long = self.sat(a_long, a_long_bound)
            # a_lat = self.sat(a_lat, a_lat_bound)

            delta = alpha + pi/2
            ax = a_lat * cos(delta) + a_long * cos(alpha)
            ay = a_lat * sin(delta) + a_long * sin(alpha)

            # self.manager.add_to_command_deque((ax, ay))
            self.manager.simulator.camera.acceleration = pygame.Vector2((ax, ay))

            if self.manager.write_plot:
                f.write(f'{self.manager.simulator.time},{r},{theta},{vtheta},{vr},{x},{y},{car_x},{car_y},{ax},{ay},{a_lat},{a_long}\n')

        if self.manager.write_plot:
            f.close()


    def sat(self, x, bound):
        return min(max(x, -bound), bound)


    def generate_acceleration(self, kin):
        X, Y            = kin[0]
        Vx, Vy          = kin[1]
        car_x, car_y    = kin[2]
        car_speed, cvy  = kin[3]

        if USE_WORLD_FRAME:
            orig = self.manager.get_cam_origin()
            X += orig[0]
            Y += orig[1]
            car_x += orig[0]
            car_y += orig[1]
        
        # speed of drone
        S = (Vx**2 + Vy**2) **0.5

        # distance between the drone and car
        r = ((car_x - X)**2 + (car_y - Y)**2)**0.5

        # heading angle of drone wrt x axis
        alpha = atan2(Vy,Vx)

        # angle of LOS from drone to car
        theta = atan2(car_y-Y, car_x-X)

        # heading angle of car
        beta = 0

        # compute vr and vtheta
        Vr = car_speed * cos(beta - theta) - S * cos(alpha - theta)
        Vtheta = car_speed * sin(beta - theta) - S * sin(alpha - theta)

        r_ = r
        theta_ = theta
        Vr_ = Vr
        Vtheta_ = Vtheta
        # at this point r, theta, Vr, Vtheta are computed 
        # we can consider EKF filtering [r, theta, Vr, Vtheta]
        if USE_EXTENDED_KALMAN:
            self.manager.EKF.add(r, theta, Vr, Vtheta, alpha, self.a_lt, self.a_ln)
            r, theta, Vr, Vtheta = self.manager.EKF.get_estimated_state()

        # calculate y from drone to car
        y2 = Vtheta**2 + Vr**2
        y1 = r**2 * Vtheta**2 - y2 * self.R**2

        # time to collision from drone to car
        # tm = -vr * r / (vtheta**2 + vr**2)

        # compute desired acceleration
        w = w_
        K1 = K_1 * np.sign(-Vr)    # lat 
        K2 = K_2                   # long

        # compute lat and long accelerations
        _D = 2*Vr*Vtheta*r**2

        if abs(_D) < 0.01:
            a_lat = 0.0
            a_long = 0.0
        else:
            _c = cos(alpha - theta)
            _s = sin(alpha - theta)
            _A = K2 * y2 / (2*Vr)
            _B = K2 * y2 * self.R**2 - K1*w + K1*y1

            # 2
            # a_lat = ((Vr * _c + Vtheta * _s) * (_B / _D)) - _A * _s
            # a_long = - ((Vtheta * _c + Vr * _s) * (_B / _D)) + _A * _s

            # 1
            # a_lat = (K1*Vr*y1*cos(alpha - theta) + K1*Vtheta*y1*sin(alpha - theta) + K2*self.R**2*Vr*y2*cos(alpha - theta) + K2*self.R**2*Vtheta*y2*sin(alpha - theta) - K2*Vtheta*r**2*y2*sin(alpha - theta))/(2*(Vr*Vtheta*r**2*cos(alpha - theta)**2 + Vr*Vtheta*r**2*sin(alpha - theta)**2))
            # a_long = (K1*Vr*y1*sin(alpha - theta) - K1*Vtheta*y1*cos(alpha - theta) - K2*self.R**2*Vtheta*y2*cos(alpha - theta) + K2*self.R**2*Vr*y2*sin(alpha - theta) + K2*Vtheta*r**2*y2*cos(alpha - theta))/(2*(Vr*Vtheta*r**2*cos(alpha - theta)**2 + Vr*Vtheta*r**2*sin(alpha - theta)**2))

            #3
            a_lat = (K1*Vr*y1*cos(alpha - theta) - K1*Vr*w*cos(alpha - theta) - K1*Vtheta*w*sin(alpha - theta) + K1*Vtheta*y1*sin(alpha - theta) + K2*self.R**2*Vr*y2*cos(alpha - theta) + K2*self.R**2*Vtheta*y2*sin(alpha - theta) - K2*Vtheta*r**2*y2*sin(alpha - theta))/_D
            a_long = (K1*Vtheta*w*cos(alpha - theta) - K1*Vtheta*y1*cos(alpha - theta) - K1*Vr*w*sin(alpha - theta) + K1*Vr*y1*sin(alpha - theta) - K2*self.R**2*Vtheta*y2*cos(alpha - theta) + K2*self.R**2*Vr*y2*sin(alpha - theta) + K2*Vtheta*r**2*y2*cos(alpha - theta))/_D



        a_long_bound = 5
        a_lat_bound = 5
        
        a_long = self.sat(a_long, a_long_bound)
        a_lat = self.sat(a_lat, a_lat_bound)

        self.a_ln = a_long
        self.a_lt = a_lat

        # compute acceleration command
        delta = alpha + pi/2
        ax = a_lat * cos(delta) + a_long * cos(alpha)
        ay = a_lat * sin(delta) + a_long * sin(alpha)
        
        if not CLEAN_CONSOLE:
            print(f'CCC0 >> r:{r:0.2f} | theta:{theta:0.2f} | alpha:{alpha:0.2f} | car_speed:{car_speed:0.2f} | S:{S:0.2f} | Vr:{Vr:0.2f} | Vtheta:{Vtheta:0.2f} | y1:{y1:0.2f} | y2:{y2:0.2f} | a_lat:{a_lat:0.2f} | a_long:{a_long:0.2f}')
        
        tru_kin = self.manager.get_true_kinematics()
        tX, tY            = tru_kin[0]
        tVx, tVy          = tru_kin[1]
        tcar_x, tcar_y    = tru_kin[2]
        tcar_speed, tcvy  = tru_kin[3]
        tS = (tVx**2 + tVy**2) **0.5
        tr = ((tcar_x - tX)**2 + (tcar_y - tY)**2)**0.5
        ttheta = atan2(tcar_y-tY, tcar_x-tX)
        tVr = tcar_speed * cos(beta - ttheta) - tS * cos(alpha - ttheta)
        tVtheta = tcar_speed * sin(beta - ttheta) - tS * sin(alpha - ttheta)

        
        tra_kin = self.manager.get_tracked_kinematics()
        vel = self.manager.simulator.camera.velocity
        if not CLEAN_CONSOLE:
            print(f'CCCC >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:[{X:0.2f}, {Y:0.2f}] | v:[{Vx:0.2f}, {Vy:0.2f}] | CAR - x:[{car_x:0.2f}, {car_y:0.2f}] | v:[{car_speed:0.2f}, {cvy:0.2f}] | COMMANDED a:[{ax:0.2f}, {ay:0.2f}] | TRACKED x:[{tra_kin[2][0]:0.2f},{tra_kin[2][1]:0.2f}] | v:[{tra_kin[3][0]:0.2f},{tra_kin[3][1]:0.2f}]')
        if self.manager.write_plot:
            self.f.write(f'{self.manager.simulator.time},{r},{degrees(theta)},{degrees(Vtheta)},{Vr},{tru_kin[0][0]},{tru_kin[0][1]},{tru_kin[2][0]},{tru_kin[2][1]},{ax},{ay},{a_lat},{a_long},{tru_kin[3][0]},{tru_kin[3][1]},{tra_kin[2][0]},{tra_kin[2][1]},{tra_kin[3][0]},{tra_kin[3][1]},{self.manager.simulator.camera.origin[0]},{self.manager.simulator.camera.origin[1]},{S},{degrees(alpha)},{tru_kin[1][0]},{tru_kin[1][1]},{tra_kin[4][0]},{tra_kin[4][1]},{tra_kin[5][0]},{tra_kin[5][1]},{self.manager.simulator.camera.altitude},{abs(_D)},{r_},{degrees(theta_)},{Vr_},{degrees(Vtheta_)},{tr},{degrees(ttheta)},{tVr},{degrees(tVtheta)}\n')

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
    def __init__(self, save_on=False, write_plot=False, control_on=False, tracker_on=True, tracker_display_on=False, use_true_kin=True):

        self.save_on = save_on
        self.write_plot = write_plot
        self.control_on = control_on
        self.tracker_on = tracker_on
        self.tracker_display_on = tracker_display_on
        self.use_true_kin = use_true_kin

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


    def run_simulator(self):
        """Run Simulator
        """
        self.simulator.start_new()
        self.simulator.run()


    def run_controller(self):
        """Run Controller
        """
        self.controller.run()
    

    def run_tracker(self):
        """Run Tracker
        """
        self.tracker.run()
        

    def run_experiment(self):
        """Run Experiment by running Simulator, Tracker and Controller.
        """

        if self.tracker_on:
            self.tracker_thread = th.Thread(target=self.run_tracker, daemon=True)
            self.tracker_thread.start()

        if self.control_on:
            self.controller_thread = th.Thread(target=self.run_controller, daemon=True)
            self.controller_thread.start()

        self.run_simulator()

        if self.save_on:
            # create folder path inside ./sim_outputs
            _path = f'./sim_outputs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
            _prep_temp_folder(os.path.realpath(_path))
            vid_path = f'{_path}/sim_track_control.avi'
            print('Making video.')
            self.make_video(vid_path, TEMP_FOLDER)


    def run(self):
        self.simulator.start_new()
        if self.write_plot:
            self.controller.f = open(self.controller.plot_info_file, '+w')

        # start the experiment
        while self.simulator.running:
            self.simulator.dt = self.simulator.clock.tick(FPS) / 1000.0
            if self.simulator.pause:
                self.simulator.dt = 0.0
            self.simulator.time += self.simulator.dt

            self.simulator.handle_events()
            if not self.simulator.running:
                break

            if not self.simulator.pause:
                self.simulator.update()
                if not CLEAN_CONSOLE:
                    print(f'SSSS >> {str(timedelta(seconds=self.simulator.time))} >> DRONE - x:{vec_str(self.simulator.camera.position)} | v:{vec_str(self.simulator.camera.velocity)} | CAR - x:{vec_str(self.simulator.car.position)}, v: {vec_str(self.simulator.car.velocity)} | COMMANDED a:{vec_str(self.simulator.camera.acceleration)} | a_comm:{vec_str(self.simulator.cam_accel_command)} | rel_car_pos: {vec_str(self.simulator.car.position - self.simulator.camera.position)}', end='\n')
            
            # draw stuffs
            self.simulator.draw()

            # process screen capture *PARTY IS HERE*
            if not self.simulator.pause:
                # let tracker process image, when simulator says so
                if self.simulator.can_begin_tracking():
                    screen_capture = self.simulator.get_screen_capture()
                    self.tracker.process_image(screen_capture)
                    # let controller generate acceleration, when tracker says so
                    if self.tracker.can_begin_control() and (self.use_true_kin or self.tracker.kin is not None) and (self.filter.done_waiting() or not USE_FILTER):
                        # collect kinematics tuple
                        kin = self.get_true_kinematics() if self.use_true_kin else self.tracker.kin
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


    def make_video(self, video_name, folder_path):
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
    def __init__(self, window_size=10):
        self.car_x  = deque(maxlen=window_size)
        self.car_y  = deque(maxlen=window_size)
        self.car_vx = deque(maxlen=window_size)
        self.car_vy = deque(maxlen=window_size)

        self.ready = False

        
        # self.old_pos = self.avg_pos()
        # self.old_vel = self.avg_vel()


    def done_waiting(self):
        return len(self.car_vx) > 5


    def init_filter(self, pos, vel):
        self.new_pos = pygame.Vector2(pos)
        self.new_vel = pygame.Vector2(vel)
        self.add_pos(pos)
        self.add_vel(vel)
        self.ready = True


    def add(self, pos, vel):
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
        # remember the last new average before adding to deque
        self.old_pos = self.new_pos

        # add to deque
        self.car_x.append(pos[0])
        self.car_y.append(pos[1])

        # compute new average
        self.new_pos = self.avg_pos()


    def add_vel(self, vel):
        # remember the last new average before adding to deque
        self.old_vel = self.new_vel

        # add to deque
        self.car_vx.append(vel[0])
        self.car_vy.append(vel[1])

        # compute new average
        self.new_vel = self.avg_vel()


    def get_pos(self):
        return self.new_pos


    def get_vel(self):
        return self.new_vel


    def avg_pos(self):
        x = sum(self.car_x) / len(self.car_x)
        y = sum(self.car_y) / len(self.car_y)
        return pygame.Vector2(x,y)


    def avg_vel(self):
        vx = sum(self.car_vx) / len(self.car_vx)
        vy = sum(self.car_vy) / len(self.car_vy)
        return pygame.Vector2(vx,vy)



class Kalman:
    def __init__(self, manager):
        self.x  = 0.0
        self.y  = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.sig = 0.1
        self.sig_r = 0.1
        self.sig_q = 1.0
        self.manager = manager
        
        # process noise
        # self.Er = np.array([[0.01], [0.01], [0.01], [0.01]])
        self.Er = np.array([[0.01], [0.01], [0.01], [0.01]])

        # measurement noise
        # self.Eq = np.array([[0.01], [0.01], [0.01], [0.01]])
        self.Eq = np.array([[0.01], [0.01], [0.01], [0.01]])

        # predicted belief state
        self.Mu = np.array([[self.x], [self.y], [self.vx], [self.vy]])
        # self.S = np.array([ [0.00001, 0, 0, 0],    \
        #                     [0, 0.00001, 0, 0],    \
        #                     [0, 0, 1, 0],     \
        #                     [0, 0, 0, 1]  ])
        # self.S = np.array([ [1, 0, 0, 0],    \
        #                     [0, 1, 0, 0],    \
        #                     [0, 0, 1, 0],     \
        #                     [0, 0, 0, 1]  ])
        # self.S = np.array([ [0.0001, 0, 0, 0],    \
        #                     [0, 0.0001, 0, 0],    \
        #                     [0, 0, 0.0001, 0],     \
        #                     [0, 0, 0, 0.0001]  ])
        self.var_S = np.array([10**-4, 10**-4, 10**-4, 10**-4])
        self.S = np.diag(self.var_S.flatten())

        # noiseless connection between state vector and measurement vector
        # self.C = np.identity(4)
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        # covariance of process noise model
        # self.R = np.matmul(self.Er, np.transpose(self.Er))
        # self.R = np.array([ [0.01, 0, 0.1, 0],    \
        #                     [0, 0.01, 0, 0.1],    \
        #                     [0, 0, .01, 0],     \
        #                     [0, 0, 0, .01]  ])
        # self.R = np.array([ [0.000001, 0, 0, 0],    \
        #                     [0, 0.000001, 0, 0],    \
        #                     [0, 0, .000001, 0],     \
        #                     [0, 0, 0, .000001]  ])
        self.var_R = np.array([10**-6, 10**-6, 10**-5, 10**-5])
        self.R = np.diag(self.var_R.flatten())
        # self.R = np.array([ [0.000001, 0, 0, 0],    \
        #                     [0, 0.000001, 0, 0],    \
        #                     [0, 0, .000001, 0],     \
        #                     [0, 0, 0, .000001]  ])
        

        # covariance of measurement noise model
        # self.Q = np.matmul(self.Eq, np.transpose(self.Eq))
        # self.Q = np.array([[0.01, 0.01, 0.01, 0.0], [0.01, 0.01, 0, 0.01], [0.01, 0, 0.01, 0.01], [0, 0.01, 0.01, 0.01]])
        # self.Q = np.array([ [0.00001, 0, 0, 0],    \
        #                     [0, 0.00001, 0, 0],    \
        #                     [0, 0, 0.00001, 0],       \
        #                     [0, 0, 0, 0.00001]    ])
        # self.var_Q = np.array([10**-5, 10**-5, 10**-5, 10**-5])
        self.var_Q = np.array([0.0156*10**-3, 0.0155*10**-3, 7.3811*10**-3, 6.5040*10**-3])
        # self.var_Q = np.array([0.0156, 0.0155, 7.3811, 6.5040])
        self.Q = np.diag(self.var_Q.flatten())
        # self.Q = np.array([ [0.00001, 0, 0.0, 0],    \
        #                     [0, 0.00001, 0, 0.0],    \
        #                     [0, 0, 0.0001, 0],       \
        #                     [0, 0, 0, 0.0001]    ])
        # self.Q = np.array([ [0.00001, 0, 0.0, 0],    \
        #                     [0, 0.00001, 0, 0.0],    \
        #                     [0, 0, 0.00001, 0],       \
        #                     [0, 0, 0, 0.00001]    ])
        # self.Q = np.array([ [0.0154, -0.0008, 0.0555, -0.0010],    \
        #                     [-0.0008, 0.0144, -0.0010, 0.0532],    \
        #                     [0.0555, -0.0010, 1.8173, -0.0791],       \
        #                     [-0.0010, 0.0532, -0.0791, 1.7557]    ])
        # self.Q = np.array([ [0.0158, 0.000, 0.0666, -0.0001],    \
        #                     [0.000, 0.0151, -0.0006, 0.0650],    \
        #                     [0.0666, -0.0010, 1.8173, -0.0119],       \
        #                     [-0.0001, 0.0650, -0.0119, 3.7924]    ])
        # self.Q = np.matmul(self.Eq, np.transpose(self.Eq))

        self.ready = False
  

    def done_waiting(self):
        return self.ready


    def init_filter(self, pos, vel):
        self.x  = pos[0]
        self.y  = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.X = np.array([[self.x], [self.y], [self.vx], [self.vy]])
        self.Mu = self.X
        self.ready = True


    def add(self, pos, vel):
        # pos and vel are the measured values. (remember x_bar)
        self.x  = pos[0]
        self.y  = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.X = np.array([[self.x], [self.y], [self.vx], [self.vy]])

        self.predict()
        self.correct()
        # self.simple_predict()
        # self.simple_correct()


    def predict(self):
        # collect params
        dt = self.manager.get_sim_dt()
        dt2 = dt**2
        # motion model
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # control model
        B = np.array([[0.5*dt2, 0], [0, 0.5*dt2], [dt, 0], [0, dt]])
        # B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

        # process noise covariance
        R = self.R

        command = self.manager.simulator.camera.acceleration
        U = np.array([[command[0]],[command[1]]])
        
        # predict
        self.Mu = np.matmul(A, self.Mu) + np.matmul(B, U)
        self.S = np.matmul(np.matmul(A, self.S), np.transpose(A)) + R


    def correct(self):
        # Z = np.matmul(self.C, self.X) + self.Eq
        # Z = np.matmul(self.C, self.X) 
        Z = self.X
        K = np.matmul( np.matmul(self.S, self.C), np.linalg.pinv( np.matmul(np.matmul(self.C, self.S), np.transpose(self.C)) + self.Q ))
        # K = np.matmul( self.S, np.linalg.pinv( self.S + self.Q ))

        self.Mu = self.Mu + np.matmul(K, (Z - np.matmul(self.C, self.Mu)))
        self.S = np.matmul((np.identity(4) - np.matmul(K, self.C)), self.S)
        # self.Mu = self.Mu + np.matmul(K, (Z - self.Mu))
        # self.S = np.matmul((np.identity(4) - K), self.S)
        
 
    def simple_predict(self):
        # collect params
        dt = self.manager.get_sim_dt()
        dt2 = dt**2
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[0.5*dt2, 0], [0, 0.5*dt2], [dt, 0], [0, dt]])
        # B = np.array([[0.5*dt2, 0], [0, 0.5*dt2], [0, 0], [0, 0]])
        commmand = self.manager.simulator.camera.acceleration
        U = np.array([[commmand[0]],[commmand[1]]])
        
        # predict
        self.Mu = np.matmul(A,self.Mu) + np.matmul(B, U)
        self.S = np.matmul(np.matmul(A, self.S), np.transpose(A))


    def simple_correct(self):
        Z = self.X + self.Eq
        K = np.matmul( self.S, np.linalg.pinv( self.S + self.Q ))

        self.Mu = self.Mu + np.matmul(K, (Z - self.Mu))
        self.S = np.matmul((np.identity(4) - K), self.S)


    def add_pos(self, pos):
        self.add(pos, (self.vx, self.vy))


    def add_vel(self, vel):
        self.add((self.x, self.y), vel)


    def get_pos(self):
        return pygame.Vector2(self.Mu.flatten()[0], self.Mu.flatten()[1])


    def get_vel(self):
        return pygame.Vector2(self.Mu.flatten()[2], self.Mu.flatten()[3])



class ExtendedKalman:
    """Implement continuous-continuous EKF for the UAS anf Vehicle system
    """
    def __init__(self, manager):
        self.manager = manager
        self.prev_r = None
        self.prev_theta = None
        self.prev_Vr = None
        self.prev_Vtheta = None
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], 
                           [0.0, 1.0, 0.0, 0.0]])

        self.P = np.diag([0.1, 0.1, 0.1, 0.1])
        self.R = np.diag([0.1, 0.1])
        # self.Q = np.diag([0.001, 0.001, 0.1, 0.1])
        self.Q = np.diag([0.1, 0.1, 1, 0.1])

        self.filter_initialized_flag = False
        self.ready = False


    def is_initialized(self):
        return self.filter_initialized_flag


    def initialize_filter(self, r, theta, Vr, Vtheta, alpha, a_lat, a_long):
        self.prev_r = r
        self.prev_theta = theta
        self.prev_Vr = -5
        self.prev_Vtheta = 5
        self.alpha = alpha
        self.a_lat = a_lat
        self.a_long = a_long
        self.filter_initialized_flag = True
        

    def add(self, r, theta, Vr, Vtheta, alpha, a_lat, a_long):
        # make sure filter is initialized
        if not self.is_initialized():
            self.initialize_filter(r, theta, Vr, Vtheta, alpha, a_lat, a_long)
            return
        else:
            self.ready = True

        # next part executes only when filter is initialized and ready
        if not (np.sign(self.prev_theta) == np.sign(theta)):
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
        # perform predictor step
        self.A = np.array([[0.0, 0.0, 0.0, 1.0],
                           [-self.prev_Vtheta/self.prev_r**2, 0.0, 1/self.prev_r, 0.0],
                           [self.prev_Vtheta*self.prev_Vr/self.prev_r**2, 0.0, -self.prev_Vr/self.prev_r, -self.prev_Vtheta/self.prev_r],
                           [-self.prev_Vtheta**2/self.prev_r**2, 0.0, 2*self.prev_Vtheta/self.prev_r, 0.0]])

        self.B = np.array([[0.0, 0.0],
                           [0.0, 0.0],
                           [-sin(self.alpha + pi/2 - self.prev_theta), -sin(self.alpha - self.prev_theta)],
                           [-cos(self.alpha + pi/2 - self.prev_theta), -cos(self.alpha - self.prev_theta)]])


    def correct(self):
        self.Z = np.array([[self.r], [self.theta]])
        self.K = np.matmul(np.matmul(self.P, np.transpose(self.H)) , np.linalg.pinv(self.R))

        U = np.array([[self.a_lat], [self.a_long]])
        state = np.array([[self.prev_r], [self.prev_theta], [self.prev_Vtheta], [self.prev_Vr]])
        dyn = np.array([[self.prev_Vr], [self.prev_Vtheta/self.prev_r], [-self.prev_Vtheta*self.prev_Vr/self.prev_r], [self.prev_Vtheta**2/self.prev_r]])

        state_dot = dyn + np.matmul(self.B, U) + np.matmul(self.K, (self.Z - np.matmul(self.H, state)))
        P_dot = np.matmul(self.A, self.P) + np.matmul(self.P, np.transpose(self.A)) - np.matmul(np.matmul(self.K, self.H),self.P) + self.Q

        dt = self.manager.get_sim_dt()
        state = state + state_dot * dt
        self.P = self.P + P_dot * dt

        self.r = state.flatten()[0]
        self.theta = state.flatten()[1]
        self.Vtheta = state.flatten()[2]
        self.Vr = state.flatten()[3]


    def get_estimated_state(self):
        if self.ready:
            return (self.r, self.theta, self.Vr, self.Vtheta)
        else:
            return (self.prev_r, self.prev_theta, self.prev_Vr, self.prev_Vtheta)



        
# dummy moving average for testing (not used)
def get_moving_average(a, w):
    ret = []
    for i in range(len(a)):
        start = max(0,i - w+1)
        stop = i+1
        b = a[start:stop]
        ret.append(sum(b) / len(b))

    return ret



if __name__ == "__main__":

    EXPERIMENT_SAVE_MODE_ON = 0
    WRITE_PLOT              = 1
    CONTROL_ON              = 1
    TRACKER_ON              = 1
    TRACKER_DISPLAY_ON      = 1
    USE_TRUE_KINEMATICS     = 0
    
    RUN_EXPERIMENT          = 0
    RUN_TRACK_PLOT          = 1

    RUN_VIDEO_WRITER        = 0

    if RUN_EXPERIMENT:
        experiment_manager = ExperimentManager(EXPERIMENT_SAVE_MODE_ON, WRITE_PLOT, CONTROL_ON, TRACKER_ON, TRACKER_DISPLAY_ON, USE_TRUE_KINEMATICS)
        print("\nExperiment started.\n")
        # experiment_manager.run_experiment()
        experiment_manager.run()
        print("\n\nExperiment finished.\n")


    if RUN_TRACK_PLOT:
        f = open('plot_info.txt', 'r')
        t = []
        r = []
        theta = []
        vtheta = []
        vr = []
        dx = []
        dy = []
        cx = []
        cy = []
        ax = []
        ay = []
        a_lat = []
        a_long = []
        cvx = []
        cvy = []
        tcx = []
        tcy = []
        tcvx = []
        tcvy = []
        dox = []
        doy = []
        S = []
        alpha = []
        dvx = []
        dvy=[]
        mcx=[]
        mcy=[]
        mcvx=[]
        mcvy=[]
        alt=[]
        d=[]
        mr = []
        mtheta = []
        mvr = []
        mvtheta = []
        tr = []
        ttheta = []
        tvr = []
        tvtheta = []



        # get all the data in memory
        for line in f.readlines():
            data = tuple(map(float, list(map(str.strip, line.strip().split(',')))))
            t.append(data[0])            
            r.append(data[1])
            theta.append(data[2])
            vtheta.append(data[3])
            vr.append(data[4])
            dx.append(data[5])
            dy.append(data[6])
            cx.append(data[7])
            cy.append(data[8])
            ax.append(data[9])
            ay.append(data[10])            
            a_lat.append(data[11])            
            a_long.append(data[12])            
            cvx.append(data[13])            
            cvy.append(data[14])            
            tcx.append(data[15])            
            tcy.append(data[16])            
            tcvx.append(data[17])            
            tcvy.append(data[18])            
            dox.append(data[19])            
            doy.append(data[20])            
            S.append(data[21])            
            alpha.append(data[22])            
            dvx.append(data[23])            
            dvy.append(data[24])            
            mcx.append(data[25])            
            mcy.append(data[26])            
            mcvx.append(data[27])            
            mcvy.append(data[28])            
            alt.append(data[29])            
            d.append(data[30])
            mr.append(data[31])
            mtheta.append(data[32])
            mvr.append(data[33])
            mvtheta.append(data[34])
            tr.append(data[35])
            ttheta.append(data[36])
            tvr.append(data[37])
            tvtheta.append(data[38])

        f.close()

        # plot
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.colors import ListedColormap

        
        _path = f'./sim_outputs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        _prep_temp_folder(os.path.realpath(_path))

        # copy the plot_info file to the where plots figured will be saved
        shutil.copyfile('plot_info.txt', f'{_path}/plot_info.txt')
        plt.style.use('seaborn-whitegrid')

        
        # ----------------------------------------------------------------------------------------- figure 1
        # line of sight kinematics 1
        f0, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace':0.25})
        if SUPTITLE_ON:
            f0.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
 
        # t vs r
        axs[0].plot(t, mr, color='goldenrod', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$measured\ r$',alpha=0.9)
        axs[0].plot(t, r, color='royalblue', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$estimated\ r$',alpha=0.9)
        axs[0].plot(t, tr, color='red', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$true\ r$',alpha=0.9)

        axs[0].legend(loc='upper right')
        axs[0].set(ylabel=r'$r\ (m)$')
        axs[0].set_title(r'$\mathbf{r}$', fontsize=SUB_TITLE_FONT_SIZE)

        # t vs θ
        axs[1].plot(t, mtheta, color='goldenrod', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$measured\ \theta$',alpha=0.9)
        axs[1].plot(t, theta, color='royalblue', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$estimated\ \theta$',alpha=0.9)
        axs[1].plot(t, ttheta, color='red', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$true\ \theta$',alpha=0.9)

        axs[1].legend(loc='upper right')
        axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\theta\ (^{\circ})$')
        axs[1].set_title(r'$\mathbf{\theta}$', fontsize=SUB_TITLE_FONT_SIZE)

        f0.savefig(f'{_path}/1_los1.png', dpi=300)
        f0.show()


        # ----------------------------------------------------------------------------------------- figure 2
        # line of sight kinematics 2
        f1, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace':0.25})
        if SUPTITLE_ON:
            f1.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ II}$', fontsize=TITLE_FONT_SIZE)

        # t vs vr
        axs[0].plot(t, mvr, color='palegoldenrod', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$measured\ V_{r}$',alpha=0.9)
        axs[0].plot(t, vr, color='royalblue', linestyle='-', linewidth=LINE_WIDTH_2, label=r'$estimated\ V_{r}$',alpha=0.9)
        axs[0].plot(t, tvr, color='red', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$true\ V_{r}$',alpha=0.9)

        axs[0].legend(loc='upper right')
        axs[0].set(ylabel=r'$V_{r}\ (\frac{m}{s})$')
        axs[0].set_title(r'$\mathbf{V_{r}}$', fontsize=SUB_TITLE_FONT_SIZE)

        # t vs vtheta
        axs[1].plot(t, mvtheta, color='palegoldenrod', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$measured\ V_{\theta}$',alpha=0.9)
        axs[1].plot(t, vtheta, color='royalblue', linestyle='-', linewidth=LINE_WIDTH_2, label=r'$estimated\ V_{\theta}$',alpha=0.9)
        axs[1].plot(t, tvtheta, color='red', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$true\ V_{\theta}$',alpha=0.9)

        axs[1].legend(loc='upper right')
        axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$V_{\theta}\ (\frac{^{\circ}}{s})$')
        axs[1].set_title(r'$\mathbf{V_{\theta}}$', fontsize=SUB_TITLE_FONT_SIZE)

        f1.savefig(f'{_path}/1_los2.png', dpi=300)
        f1.show()
        
        # ----------------------------------------------------------------------------------------- figure 2
        # acceleration commands
        f2, axs = plt.subplots()
        if SUPTITLE_ON:
            f2.suptitle(r'$\mathbf{Acceleration\ commands}$', fontsize=TITLE_FONT_SIZE)

        axs.plot(t, a_lat, color='forestgreen', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$a_{lat}$', alpha=0.9)
        axs.plot(t, a_long, color='deeppink', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$a_{long}$', alpha=0.9)
        axs.legend()
        axs.set(xlabel=r'$time\ (s)$', ylabel=r'$acceleration\ (\frac{m}{s_{2}})$')

        f2.savefig(f'{_path}/2_accel.png', dpi=300)
        f2.show()

        # ----------------------------------------------------------------------------------------- figure 3
        # trajectories
        f3, axs = plt.subplots(2, 1, gridspec_kw={'hspace':0.4})
        if SUPTITLE_ON:
            f3.suptitle(r'$\mathbf{Vehicle\ and\ UAS\ True\ Trajectories}$', fontsize=TITLE_FONT_SIZE)

        ndx = np.array(dx) + np.array(dox)
        ncx = np.array(cx) + np.array(dox)
        ndy = np.array(dy) + np.array(doy)
        ncy = np.array(cy) + np.array(doy)

        axs[0].plot(ndx, ndy, color='darkslategray', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$UAS$', alpha=0.9)
        axs[0].plot(ncx, ncy, color='limegreen', linestyle='-', linewidth=LINE_WIDTH_2, label=r'$Vehicle$', alpha=0.9)
        axs[0].set(ylabel=r'$y\ (m)$')
        axs[0].set_title(r'$\mathbf{World\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
        axs[0].legend()

        ndx = np.array(dx)
        ncx = np.array(cx)
        ndy = np.array(dy)
        ncy = np.array(cy)

        x_pad = (max(ncx) - min(ncx)) * 0.05
        y_pad = (max(ncy) - min(ncy)) * 0.05
        xl = max(abs(max(ncx)), abs(min(ncx))) + x_pad
        yl = max(abs(max(ncy)), abs(min(ncy))) + y_pad
        axs[1].plot(ndx, ndy, color='darkslategray', marker='+', markersize=10, label=r'$UAS$',alpha=0.7)
        axs[1].plot(ncx, ncy, color='limegreen', linestyle='-', linewidth=LINE_WIDTH_2, label=r'$Vehicle$',alpha=0.9)
        axs[1].set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$')
        axs[1].set_title(r'$\mathbf{Camera\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
        axs[1].legend(loc='lower right')
        axs[1].set_xlim(-xl,xl)
        axs[1].set_ylim(-yl,yl)
        f3.savefig(f'{_path}/3_traj.png',dpi=300)
        f3.show()


        # ----------------------------------------------------------------------------------------- figure 4
        # true and estimated trajectories
        if 0:
            f4, axs = plt.subplots()
            if SUPTITLE_ON:
                f4.suptitle(r'$\mathbf{Vehicle\ True\ and\ Estimated\ Trajectories}$', fontsize=TITLE_FONT_SIZE)

            axs.plot(tcx, tcy, color='darkturquoise', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$estimated\ trajectory$',alpha=0.9)
            axs.plot(cx, cy, color='crimson', linestyle=':', linewidth=LINE_WIDTH_1, label=r'$true\ trajectory$',alpha=0.9)
            axs.set_title(r'$\mathbf{camera\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs.legend()
            axs.axis('equal')
            axs.set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$')
            f4.savefig(f'{_path}/4_traj_comp.png', dpi=300)
            f4.show()


        # ----------------------------------------------------------------------------------------- figure 5
        # true and tracked pos
        if 0:
            f4, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace':0.4})
            if SUPTITLE_ON:
                f4.suptitle(r'$\mathbf{Vehicle\ True\ and\ Estimated\ Positions}$', fontsize=TITLE_FONT_SIZE)

            axs[0].plot(t, tcx, color='rosybrown', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$estimated\ x$',alpha=0.9)
            axs[0].plot(t, cx, color='red', linestyle=':', linewidth=LINE_WIDTH_1, label=r'$true\ x$',alpha=0.9)
            axs[0].set(ylabel=r'$x\ (m)$')
            axs[0].set_title(r'$\mathbf{x}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[0].legend()
            axs[1].plot(t, tcy, color='mediumseagreen', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$estimated\ y$',alpha=0.9)
            axs[1].plot(t, cy, color='green', linestyle=':', linewidth=LINE_WIDTH_1, label=r'$true\ y$',alpha=0.9)
            axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$y\ (m)$')
            axs[1].set_title(r'$\mathbf{y}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[1].legend()
            f4.savefig(f'{_path}/5_pos_comp.png', dpi=300)
            f4.show()


        # ----------------------------------------------------------------------------------------- figure 6
        # true and tracked velocities
        if 0:
            f5, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace':0.4})
            if SUPTITLE_ON:
                f5.suptitle(r'$\mathbf{True,\ Measured\ and\ Estimated\ Vehicle\ Velocities}$', fontsize=TITLE_FONT_SIZE)
            


            axs[0].plot(t, mcvx, color='paleturquoise', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$measured\ V_x$',alpha=0.9)
            axs[0].plot(t, tcvx, color='darkturquoise', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$estimated\ V_x$',alpha=0.9)
            axs[0].plot(t, cvx, color='crimson', linestyle='-', linewidth=LINE_WIDTH_2, label=r'$true\ V_x$',alpha=0.7)
            axs[0].set(ylabel=r'$V_x\ (\frac{m}{s})$')
            axs[0].set_title(r'$\mathbf{V_x}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[0].legend(loc='upper right')

            axs[1].plot(t, mcvy, color='paleturquoise', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$measured\ V_y$',alpha=0.9)
            axs[1].plot(t, tcvy, color='darkturquoise', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$estimated\ V_y$',alpha=0.9)
            axs[1].plot(t, cvy, color='crimson', linestyle='-', linewidth=LINE_WIDTH_2, label=r'$true\ V_y$',alpha=0.7)
            axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$V_y\ (\frac{m}{s})$')
            axs[1].set_title(r'$\mathbf{V_y}$', fontsize=SUB_TITLE_FONT_SIZE)
            axs[1].legend(loc='upper right')

            f5.savefig(f'{_path}/6_vel_comp.png', dpi=300)
            f5.show()

        # ----------------------------------------------------------------------------------------- figure 7
        # speed and heading
        f6, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace':0.4})
        if SUPTITLE_ON:
            f6.suptitle(r'$\mathbf{Vehicle\ and\ UAS,\ Speed\ and\ Heading}$', fontsize=TITLE_FONT_SIZE)
        c_speed = (CAR_INITIAL_VELOCITY[0]**2 + CAR_INITIAL_VELOCITY[1]**2)**0.5
        c_heading = degrees(atan2(CAR_INITIAL_VELOCITY[1], CAR_INITIAL_VELOCITY[0]))

        axs[0].plot(t, [c_speed for i in S], color='lightblue', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$|V_{vehicle}|$',alpha=0.9)
        axs[0].plot(t, S, color='blue', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$|V_{UAS}|$',alpha=0.9)
        axs[0].set(ylabel=r'$|V|\ (\frac{m}{s})$')
        axs[0].set_title(r'$\mathbf{speed}$', fontsize=SUB_TITLE_FONT_SIZE)
        axs[0].legend()

        axs[1].plot(t, [c_heading for i in alpha], color='lightgreen', linestyle='-', linewidth=LINE_WIDTH_2, label=r'$\angle V_{vehicle}$',alpha=0.9)
        axs[1].plot(t, alpha, color='green', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$\angle V_{UAS}$',alpha=0.9)
        axs[1].set(xlabel=r'$time\ (s)$',ylabel=r'$\angle V\ (^{\circ})$')
        axs[1].set_title(r'$\mathbf{heading}$', fontsize=SUB_TITLE_FONT_SIZE)
        axs[1].legend()

        f6.savefig(f'{_path}/7_speed_head.png', dpi=300)
        f6.show()
        
        # ----------------------------------------------------------------------------------------- figure 7
        # altitude profile
        f7, axs = plt.subplots()
        if SUPTITLE_ON:
            f7.suptitle(r'$\mathbf{Altitude\ profile}$', fontsize=TITLE_FONT_SIZE)
        axs.plot(t, alt, color='darkgoldenrod', linestyle='-', linewidth=2, label=r'$altitude$',alpha=0.9)
        axs.set(xlabel=r'$time\ (s)$', ylabel=r'$z\ (m)$')

        f7.savefig(f'{_path}/8_alt_profile.png', dpi=300)
        f7.show()

        # ----------------------------------------------------------------------------------------- figure 7
        # 3D Trajectories
        ndx = np.array(dx) + np.array(dox)
        ncx = np.array(cx) + np.array(dox)
        ndy = np.array(dy) + np.array(doy)
        ncy = np.array(cy) + np.array(doy)

        f8 = plt.figure()
        if SUPTITLE_ON:
            f8.suptitle(r'$\mathbf{3D\ Trajectories}$', fontsize=TITLE_FONT_SIZE)
        axs = f8.add_subplot(111, projection='3d')
        axs.plot3D(ncx, ncy, 0,color='limegreen', linestyle='-', linewidth=2, label=r'$Vehicle$',alpha=0.9)
        axs.plot3D(ndx, ndy, alt, color='darkslategray', linestyle='-', linewidth=LINE_WIDTH_1, label=r'$UAS$',alpha=0.9)
        # viridis = cm.get_map('viridis', 512)

        for point in zip(ndx,ndy,alt):
            x = [point[0], point[0]]
            y = [point[1], point[1]]
            z = [point[2], 0]
            axs.plot3D(x,y,z,color='gainsboro', linestyle='-', linewidth=0.5,alpha=0.1)
        axs.plot3D(ndx, ndy, 0,color='silver', linestyle='-', linewidth=1, alpha=0.9)
        axs.scatter3D(ndx, ndy, alt, c=alt, cmap='plasma',alpha=0.3)

        axs.set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$', zlabel=r'$z\ (m)$')
        axs.view_init(elev=41, azim=-105)
        # axs.view_init(elev=47, azim=-47)
        axs.set_title(r'$\mathbf{World\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
        axs.legend()

        f8.savefig(f'{_path}/9_3D_traj.png', dpi=300)
        f8.show()
        plt.show()
        

    if RUN_VIDEO_WRITER:
        experiment_manager = ExperimentManager()
        # create folder path inside ./sim_outputs
        _path = f'./sim_outputs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        _prep_temp_folder(os.path.realpath(_path))
        vid_path = f'{_path}/sim_track_control.avi'
        print('Making video.')
        experiment_manager.make_video(vid_path, TEMP_FOLDER)
