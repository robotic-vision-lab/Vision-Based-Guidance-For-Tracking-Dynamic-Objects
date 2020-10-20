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
from utils.img_utils import convert_to_grayscale, put_text, images_assemble
from game_utils import load_image, _prep_temp_folder, vec_str, scale_img
from algorithms.optical_flow import (compute_optical_flow_farneback, 
                                     compute_optical_flow_HS, 
                                     compute_optical_flow_LK)


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
        self.update_rect()
        # self.rect.center = self.position


    def fill_image(self):
        r,g,b = BLOCK_COLOR
        r += random.randint(-8, 8)
        g += random.randint(-8, 8)
        b += random.randint(-8, 8)
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
        self.update_rect()
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
        self.image.fill((255, 255, 255, 204), None, pygame.BLEND_RGBA_MULT)
        self.reset_kinematics()
        self.origin = self.position
        self.altitude = ALTITUDE
        self.alt_change = 20.0
        
        # self.rect.center = self.position + SCREEN_CENTER
        self.simulator = simulator
        self.update_rect()
        
        self.vel_limit = DRONE_VELOCITY_LIMIT
        self.acc_limit = DRONE_ACCELERATION_LIMIT


    def update(self):
        """[summary]
        """
        self.update_kinematics()
        self.update_rect()
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
        self.position = self.velocity * self.simulator.dt + 0.5 * self.acceleration * self.simulator.dt**2  # donot touch
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
        # print(f'SSSS1 >> {str(timedelta(seconds=self.time))} >> DRONE - x:{vec_str(self.camera.rect.center)} | v:{vec_str(self.camera.velocity)} | a:{vec_str(self.camera.acceleration)} | a_comm:{vec_str(self.cam_accel_command)} | CAR - x:{vec_str(self.car.rect.center)}, v: {vec_str(self.car.velocity)},  v_c-v_d: {vec_str(self.car.velocity - self.camera.velocity)}              ', end='\n')
        # print(self.camera.position)
        # update Group. (All sprites in it will get updated)
        self.all_sprites.update()

        # print(f'SSSS2 >> {str(timedelta(seconds=self.time))} >> DRONE - x:{vec_str(self.camera.rect.center)} | v:{vec_str(self.camera.velocity)} | a:{vec_str(self.camera.acceleration)} | a_comm:{vec_str(self.cam_accel_command)} | CAR - x:{vec_str(self.car.rect.center)}, v: {vec_str(self.car.velocity)},  v_c-v_d: {vec_str(self.car.velocity - self.camera.velocity)}              ', end='\n')
        # print(self.camera.position)
        # compensate camera motion for all sprites
        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)

        # print(f'SSSS3 >> {str(timedelta(seconds=self.time))} >> DRONE - x:{vec_str(self.camera.rect.center)} | v:{vec_str(self.camera.velocity)} | a:{vec_str(self.camera.acceleration)} | a_comm:{vec_str(self.cam_accel_command)} | CAR - x:{vec_str(self.car.rect.center)}, v: {vec_str(self.car.velocity)},  v_c-v_d: {vec_str(self.car.velocity - self.camera.velocity)}              ', end='\n')
        # print(self.camera.position)


    def draw(self):
        """Draws components on screen. Note: drone_img is drawn after screen capture for tracking is performed.
        """
        # fill background
        self.screen_surface.fill(SCREEN_BG_COLOR)

        # make title
        sim_fps = 'NA' if self.dt == 0 else f'{1/self.dt:.2f}'
        pygame.display.set_caption(f'  FPS {sim_fps} | car: x-{vec_str(self.car.position)} v-{vec_str(self.car.velocity)} a-{vec_str(self.car.acceleration)} | cam x-{vec_str(self.camera.position)} v-{vec_str(self.camera.velocity)} a-{vec_str(self.camera.acceleration)} ')

        # # compensate camera motion for all sprites
        # for sprite in self.all_sprites:
        #     self.camera.compensate_camera_motion(sprite)

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
            
        # draw drone altitude
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
            image_name = f'frame_{str(frame_num).zfill(4)}.jpg'
            file_path = os.path.join(path, image_name)

            # get capture from simulator
            img_sim = self.get_screen_capture()

            # get tracker image
            img_track = self.manager.tracker.cur_img
            if img_track is None:
                img_track = np.ones_like(img_sim, dtype='uint8') * 31
            
            # assemble images
            img = images_assemble([img_sim, img_track], (1,2))
            
            # write image
            cv.imwrite(file_path, img)
            yield
        

    def get_screen_capture(self):
        """Get screen capture from pygame and convert it to return opencv compatible images.

        Returns:
            [np.ndarray]: Captured and converted opencv compatible image.
        """
        data = pygame.image.tostring(self.screen_surface, 'RGB')
        img = np.frombuffer(data, np.uint8).reshape(HEIGHT, WIDTH, 3)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
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
        img, mask = draw_tracks(frame, self.get_centroid(good_cur), self.get_centroid(good_nxt), None, mask, track_thickness=1)

        # add optical flow arrows 
        img = draw_sparse_optical_flow_arrows(img, self.get_centroid(good_cur), self.get_centroid(good_nxt), thickness=2, arrow_scale=10.0, color=RED_CV)

        # add a center
        img = cv.circle(img, SCREEN_CENTER, radius=1, color=WHITE, thickness=2)

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
        return self._can_begin_control_flag


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
        
        # collect drone position, drone velocity and fov from simulator
        drone_position = self.manager.simulator.camera.position
        drone_velocity = self.manager.simulator.camera.velocity
        fov = self.manager.simulator.get_camera_fov()

        # transform car position and car velocity to world reference frame
        car_position = pygame.Vector2((car_x , car_y)).elementwise() * (1, -1) + (0, HEIGHT)
        car_position *= self.manager.simulator.pxm_fac
        cam_origin = drone_position - pygame.Vector2(fov)/2
        car_position += cam_origin
        car_velocity = pygame.Vector2((car_vx, car_vy)).elementwise() * (1, -1)
        car_velocity *= self.manager.simulator.pxm_fac

        # return kinematics in world reference frame
        return (drone_position, drone_velocity, car_position, car_velocity)

    
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
            print('tracker 1st frame')
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

            drone_position, drone_velocity, car_position, car_velocity = self.kin
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
        img = put_text(img, f'Altitude = {self.manager.simulator.camera.altitude:0.2f} m', (WIDTH-175, HEIGHT-15), font_scale=0.5, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, f'1 pixel = {self.manager.simulator.pxm_fac:0.4f} m', (WIDTH-175, HEIGHT-40), font_scale=0.5, color=LIGHT_GRAY_2, thickness=1)
        
        # fac = self.manager.simulator.pxm_fac
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

        img = put_text(img, kin_str_1,  (WIDTH - (330 + 25), 25),   font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_2,  (WIDTH - (155 + 25), 25),   font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_3,  (WIDTH - (328 + 25), 50),   font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_4,  (WIDTH - (155 + 25), 50),   font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_5,  (WIDTH - (332 + 25), 75),   font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_6,  (WIDTH - (155 + 25), 75),   font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_7,  (WIDTH - (330 + 25), 100),  font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_8,  (WIDTH - (155 + 25), 100),  font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_9,  (WIDTH - (340 + 25), 125),  font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_0,  (WIDTH - (155 + 25), 125),  font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_11, (WIDTH - (323 + 25), 150),  font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_12, (WIDTH - (155 + 25), 150),  font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_13, (WIDTH - (323 + 25), 175),  font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_14, (WIDTH - (155 + 25), 175),  font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)
        img = put_text(img, kin_str_15, (50, HEIGHT - 15),          font_scale=0.45, color=LIGHT_GRAY_2, thickness=1)

        return img



class Controller:
    def __init__(self, manager):
        self.manager = manager
        self.plot_info_file = 'plot_info.txt'
        self.R = CAR_RADIUS
        self.f = None

    def run(self):
        print('Controller run')
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
            # print(kin)

            # print(len(self.manager.kinematics_deque), ', ', len(self.manager.kinematics_deque) > 0)
            # l = len(self.manager.kinematics_deque)
            # if l > 0:
            #     # print(len(self.manager.kinematics_deque))
            #     self.manager.get_from_kinematics_deque()
            # else:
            #     continue

            mpx_fac = 1/self.manager.simulator.pxm_fac
            
            x, y            = kin[0]#.elementwise() * (1, -1) * mpx_fac + SCREEN_CENTER#(0, HEIGHT)
            vx, vy          = kin[1]#.elementwise() * (1, -1) * mpx_fac 
            car_x, car_y    = kin[2]#.elementwise() * (1, -1) * mpx_fac + SCREEN_CENTER#(0, HEIGHT)
            car_speed, cvy  = kin[3]#.elementwise() * (1, -1) * mpx_fac

            # print(f'CCCC1 >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:[{x:0.2f},{y:0.2f}] | v:[{vx:0.2f},{vy:0.2f}] | CAR - x:[{car_x:0.2f},{car_y:0.2f}] | v:[{car_speed:0.2f},0.00]')
            # x, y            = kin[0].elementwise() * (1, -1) * mpx_fac + SCREEN_CENTER#(0, HEIGHT)
            # vx, vy          = kin[1].elementwise() * (1, -1) * mpx_fac 
            # car_x, car_y    = kin[2].elementwise() * (1, -1) * mpx_fac + SCREEN_CENTER#(0, HEIGHT)
            # car_speed, _    = kin[3].elementwise() * (1, -1) * mpx_fac

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
            
            a_long = self.sat(a_long, a_long_bound)
            a_lat = self.sat(a_lat, a_lat_bound)

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

        # calculate y from drone to car
        y2 = Vtheta**2 + Vr**2
        y1 = r**2 * Vtheta**2 - y2 * self.R**2

        # time to collision from drone to car
        # tm = -vr * r / (vtheta**2 + vr**2)

        # compute desired acceleration
        w = -0.1
        K1 = 0.15 * np.sign(-Vr)    # lat 
        K2 = 0.02                   # long

        # compute lat and long accelerations
        a_lat = (K1*Vr*y1*cos(alpha - theta) + K1*Vtheta*y1*sin(alpha - theta) + K2*self.R**2*Vr*y2*cos(alpha - theta) + K2*self.R**2*Vtheta*y2*sin(alpha - theta) - K2*Vtheta*r**2*y2*sin(alpha - theta))/(2*(Vr*Vtheta*r**2*cos(alpha - theta)**2 + Vr*Vtheta*r**2*sin(alpha - theta)**2))
        a_long = (K1*Vr*y1*sin(alpha - theta) - K1*Vtheta*y1*cos(alpha - theta) - K2*self.R**2*Vtheta*y2*cos(alpha - theta) + K2*self.R**2*Vr*y2*sin(alpha - theta) + K2*Vtheta*r**2*y2*cos(alpha - theta))/(2*(Vr*Vtheta*r**2*cos(alpha - theta)**2 + Vr*Vtheta*r**2*sin(alpha - theta)**2))

        a_long_bound = 3
        a_lat_bound = 3
        
        a_long = self.sat(a_long, a_long_bound)
        a_lat = self.sat(a_lat, a_lat_bound)

        # compute acceleration command
        delta = alpha + pi/2
        ax = a_lat * cos(delta) + a_long * cos(alpha)
        ay = a_lat * sin(delta) + a_long * sin(alpha)

        print(f'CCCC >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:[{X:0.2f}, {Y:0.2f}] | v:[{Vx:0.2f}, {Vy:0.2f}] | CAR - x:[{car_x:0.2f}, {car_y:0.2f}] | v:[{car_speed:0.2f}, {cvy:0.2f}] | COMMANDED a:[{ax:0.2f}, {ay:0.2f}] | r:{r:0.4f} | theta:{degrees(theta):0.4f}')
        if self.manager.write_plot:
            t_kin = self.manager.tracker.kin
            self.f.write(f'{self.manager.simulator.time},{r},{theta},{Vtheta},{Vr},{X},{Y},{car_x},{car_y},{ax},{ay},{a_lat},{a_long},{car_speed},{cvy},{t_kin[2][0]},{t_kin[2][1]},{t_kin[3][0]},{t_kin[3][1]},{self.manager.simulator.camera.origin[0]},{self.manager.simulator.camera.origin[1]},{S},{alpha}\n')

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
                print(f'SSSS >> {str(timedelta(seconds=self.simulator.time))} >> DRONE - x:{vec_str(self.simulator.camera.position)} | v:{vec_str(self.simulator.camera.velocity)} | CAR - x:{vec_str(self.simulator.car.position)}, v: {vec_str(self.simulator.car.velocity)} | COMMANDED a:{vec_str(self.simulator.camera.acceleration)} | a_comm:{vec_str(self.simulator.cam_accel_command)} | rel_car_pos: {vec_str(self.simulator.car.position - self.simulator.camera.position)}', end='\n')
            
            # draw stuffs
            self.simulator.draw()

            # process screen capture
            if not self.simulator.pause:
                # let tracker process image, when simulator says so
                if self.simulator.can_begin_tracking():
                    screen_capture = self.simulator.get_screen_capture()
                    self.tracker.process_image(screen_capture)
                    # let controller generate acceleration, when tracker says so
                    if self.tracker.can_begin_control() and not self.tracker.kin is None:
                        kin = self.get_true_kinematics() if self.use_true_kin else self.tracker.kin
                        ax, ay = self.controller.generate_acceleration(kin)
                        self.simulator.camera.acceleration = pygame.Vector2((ax, ay))
                    # ax, ay = self.controller.generate_acceleration(self.get_true_kinematics())
                    # self.simulator.camera.acceleration = pygame.Vector2((ax, ay))

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
            create_video_from_images(folder_path, 'jpg', video_name, FPS)

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




if __name__ == "__main__":

    EXPERIMENT_SAVE_MODE_ON = 0
    WRITE_PLOT              = 1
    CONTROL_ON              = 1
    TRACKER_ON              = 1
    TRACKER_DISPLAY_ON      = 1
    USE_TRUE_KINEMATICS     = 1
    
    RUN_EXPERIMENT          = 0
    RUN_TRACK_PLOT          = 1

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

        import matplotlib.pyplot as plt
        _path = f'./sim_outputs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        _prep_temp_folder(os.path.realpath(_path))

        plt.style.use('seaborn-whitegrid')
        # t vs r
        plt.plot(t, r, color='teal', linestyle='-', linewidth=1, label='r')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('r')
        plt.savefig(f'{_path}/r.png')
        plt.show()

        # t vs r
        plt.plot(t, theta, color='teal', linestyle='-', linewidth=1, label='theta')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('theta')
        plt.savefig(f'{_path}/theta.png')
        plt.show()

        # t vs vtheta
        plt.plot(t, vtheta, color='teal', linestyle='-', linewidth=1, label='vtheta')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('vtheta')
        plt.savefig(f'{_path}/vtheta.png')
        plt.show()

        # t vs vr
        plt.plot(t, vtheta, color='teal', linestyle='-', linewidth=1, label='vr')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('vr')
        plt.savefig(f'{_path}/vr.png')
        plt.show()

        # trajectories
        ndx = np.array(dx) + np.array(dox)
        ncx = np.array(cx) + np.array(dox)
        ndy = np.array(dy) + np.array(doy)
        ncy = np.array(cy) + np.array(doy)
        plt.plot(ncx, ncy, color='green', linestyle='-', linewidth=1, label='car')
        plt.plot(ndx, ndy, color='gray', linestyle='-', linewidth=2, label='drone')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{_path}/traj.png')
        plt.show()

        plt.plot(cx, cy, color='green', linestyle='-', linewidth=1, label='car')
        plt.plot(dx, dy, color='gray', marker='+', markersize=20, label='drone')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        x_pad = (max(cx) - min(cx)) * 0.05
        y_pad = (max(cy) - min(cy)) * 0.05
        xl = max(abs(max(cx)), abs(min(cx))) + x_pad
        yl = max(abs(max(cy)), abs(min(cy))) + y_pad
        # plt.xlim(min(cx)-x_pad, max(cx)+x_pad)
        # plt.ylim(min(cy)-y_pad, max(cy)+y_pad)
        plt.xlim(-xl, xl)
        plt.ylim(-yl, yl)
        plt.savefig(f'{_path}/traj2.png')
        plt.show()

        # accelerations
        plt.plot(t, ax, color='teal', linestyle='-', linewidth=1, label='ax')
        plt.plot(t, ay, color='green', linestyle='-', linewidth=1, label='ay')
        # plt.plot(t, a_lat, color='blue', linestyle='-', linewidth=1, label='a_lat')
        # plt.plot(t, a_long, color='red', linestyle='-', linewidth=1, label='a_long')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('acceleration')
        plt.savefig(f'{_path}/accel.png')
        plt.show()

        # tracked pos vs true pos
        plt.plot(t, cx, color='red', linestyle='-', linewidth=1, label='cx')
        plt.plot(t, cy, color='blue', linestyle='-', linewidth=1, label='cy')
        plt.plot(t, tcx, color='green', linestyle=':', linewidth=2, label='track_cx')
        plt.plot(t, tcy, color='orange', linestyle=':', linewidth=2, label='track_cy')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('pos and tracked pos')
        plt.savefig(f'{_path}/pos_comp.png')
        plt.show()

        # tracked vel vs true vel
        plt.plot(t, tcvx, color='tan', linestyle='-', linewidth=1, label='track_cvx')
        # plt.plot(t, tcvy, color='orange', linestyle='-', linewidth=1, label='track_cvy')
        plt.plot(t, cvx, color='red', linestyle='-', linewidth=2, label='cvx')
        # plt.plot(t, cvy, color='blue', linestyle='-', linewidth=1, label='cvy')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('vel and tracked vel')
        plt.savefig(f'{_path}/vel_comp.png')
        plt.show()

        # speed and heading
        plt.plot(t, S, color='blue', linestyle='-', linewidth=1, label='drone speed')
        c_speed = (CAR_INITIAL_VELOCITY[0]**2 + CAR_INITIAL_VELOCITY[1]**2)**0.5
        plt.plot(t, [c_speed for i in S], color='lightblue', linestyle='-', linewidth=1, label='car speed')
        c_heading = degrees(atan2(CAR_INITIAL_VELOCITY[1], CAR_INITIAL_VELOCITY[0]))
        plt.plot(t, [degrees(i) for i in alpha], color='green', linestyle='-', linewidth=1, label='drone heading')
        plt.plot(t, [c_heading for i in alpha], color='lightgreen', linestyle='-', linewidth=1, label='car speed')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('heading and alpha')
        plt.savefig(f'{_path}/speed_head.png')
        plt.show()



