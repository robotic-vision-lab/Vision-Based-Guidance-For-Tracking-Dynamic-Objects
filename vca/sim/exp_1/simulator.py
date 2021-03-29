import os
import sys
from copy import deepcopy
from datetime import timedelta

import cv2 as cv
import pygame
import pygame.locals as GAME_GLOBALS
import pygame.event as GAME_EVENTS
import numpy as np

from high_precision_clock import HighPrecisionClock
print(HighPrecisionClock)
from block import Block
from bar import Bar
from car import Car
from drone_camera import DroneCamera
from settings import *


from my_imports import (load_image,
                        _prep_temp_folder,
                        vec_str,
                        images_assemble,)
print(f's -- {_prep_temp_folder}')
print(f'inside simulator.py. {os.listdir()}')

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
        self.screen_surface.fill(SCREEN_BG_COLOR)
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

        self.time = 0.0
        self.dt = 0.0

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

            GAME_EVENTS.pump()

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

        if (USE_TRAJECTORY == ONE_HOLE_TRAJECTORY or
                USE_TRAJECTORY == TWO_HOLE_TRAJECTORY or 
                USE_TRAJECTORY == SQUIRCLE_TRAJECTORY):
            self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True, scale=CAR_SCALE)
            prev_center = self.car_img[0].get_rect(center = self.car_img[0].get_rect().center).center
            rot_img = pygame.transform.rotate(self.car_img[0], degrees(self.car.angle))
            rot_img = rot_img.convert_alpha()
            rot_rect = rot_img.get_rect(center = prev_center)
            self.car_img = (rot_img, rot_rect)
            self.car.load()

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
            # img = img_sim

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

        for bar in self.bars:
            bar.load()

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
        for bar in self.bars:
            bar.load()

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
