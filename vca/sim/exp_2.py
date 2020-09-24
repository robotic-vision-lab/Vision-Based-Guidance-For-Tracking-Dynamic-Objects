import os
import sys
import cv2 as cv
import numpy as np
import shutil
import pygame 
import threading as th
from queue import deque
from PIL import Image
from copy import deepcopy
from datetime import timedelta

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
        self.euc_factor = 1.0
        self.pause = False
        self.time_font = pygame.font.SysFont(TIME_FONT, 16, False, False)
        self.bb_start = None
        self.bb_end = None
        self.bb_drag = False


    def start_new(self):
        self.time = 0.0

        # initiate screen shot generator
        self.screen_shot = screen_saver(screen=self.screen_surface, path=TEMP_FOLDER)

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
        self.running = True
        while self.running:
            # make clock tick and measure time elapsed
            self.dt = self.clock.tick(FPS) / 1000.0 
            if self.pause:
                self.dt = 0
            self.time += self.dt

            # handle events
            self.handle_events()
            if not self.running:
                break

            # if self.pause:
            #     self.dt = 0.0
            #     continue
            # else:
            if not self.pause:
                # update game objects
                self.update()

            # draw stuffs
            self.draw()

            if not self.pause:
                # put the screen capture into image_deque
                self.put_image()

            # draw extra parts like drone cross hair, simulation time, bounding box etc
            self.draw_extra()

            # show drawing board
            pygame.display.flip()

            # save screen
            if self.save_screen:
                next(self.screen_shot)
                
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or     \
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit()
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.save_screen = not self.save_screen
                    if self.save_screen:
                        print("Screen recording started.")
                    else:
                        print("Screen recording stopped.")
                if event.key == pygame.K_p:
                    self.pause = not self.pause
                    if self.pause:
                        print("Simulation paused.")
                    else:
                        print("Simulation running.")

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

            if event.type == pygame.MOUSEBUTTONDOWN:
                self.bb_start = self.bb_end = pygame.mouse.get_pos()
                self.bb_drag = True
            if self.bb_drag and event.type == pygame.MOUSEMOTION:
                self.bb_end = pygame.mouse.get_pos()

            if event.type == pygame.MOUSEBUTTONUP:
                self.bb_end = pygame.mouse.get_pos()
                self.bb_drag = False

            pygame.event.pump()


    def update(self):
        # update Group. (All sprites in it will get updated)
        self.all_sprites.update()
        self.camera.move(deepcopy(self.euc_factor * self.cam_accel_command))
        self.cam_accel_command = pygame.Vector2(0, 0)


    def draw(self):
        # fill background
        self.screen_surface.fill(SCREEN_BG_COLOR)
        sim_fps = 'NA' if self.dt == 0 else f'{1/self.dt:.2f}'
        
        pygame.display.set_caption(f'car position {self.car.position} | cam velocity {self.camera.velocity} | FPS {sim_fps}')

        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)
        self.car_block_sprites.draw(self.screen_surface)


    def draw_extra(self):
        # draw drone cross hair
        self.drone_sprite.draw(self.screen_surface)

        # draw simulation time
        time_str = f'Simulation Time - {str(timedelta(seconds=self.time))}'
        time_surf = self.time_font.render(time_str, True, TIME_COLOR)
        time_rect = time_surf.get_rect()
        self.screen_surface.blit(time_surf, (WIDTH - 12 - time_rect.width, HEIGHT - 25))


        if self.bb_start and self.bb_end and self.pause:
            x = min(self.bb_start[0], self.bb_end[0])
            y = min(self.bb_start[1], self.bb_end[1])
            w = abs(self.bb_start[0] - self.bb_end[0])
            h = abs(self.bb_start[1] - self.bb_end[1])
            self.bounding_box = (x,y,w,h)
            pygame.draw.rect(self.screen_surface, BB_COLOR, pygame.rect.Rect(x, y, w, h), 2)


    def put_image(self):
        data = pygame.image.tostring(self.screen_surface, 'RGB')
        img = np.frombuffer(data, np.uint8).reshape(HEIGHT, WIDTH, 3)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        self.manager.add_to_image_deque(img)

    
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
        self.simulator = Simulator(self)
        self.tracker = Tracker(self)
        self.controller = Controller(self)

        self.image_deque = deque(maxlen=100)
        self.command_deque = deque(maxlen=100)


    def add_to_image_deque(self, img):
        self.image_deque.append(img)



    def run_simulator(self):
        """
        this method keeps the simulator running 
        """
        self.simulator.start_new()
        self.simulator.run()


    def run_controller(self):
        """
        this method keeps the controller running
        """
        # get first frame
        while True:# len(self.image_deque) <= 1:
            print("inside")
            if not (self.simulator.bb_start and self.simulator.bb_end) or self.simulator.pause:
                continue
            if len(self.image_deque) > 0:
                frame_1 = self.image_deque.popleft()
                cur_frame = convert_to_grayscale(frame_1)
                print("First frame")
                break
        
        # compute good feature points to track
        feature_mask = np.zeros_like(cur_frame)
        x,y,w,h = self.simulator.bounding_box
        feature_mask[y:y+h+1, x:x+w+1] = 1
        cur_points = cv.goodFeaturesToTrack(cur_frame, mask=feature_mask, **FEATURE_PARAMS)

        # create mask for drawing tracks
        mask = np.zeros_like(frame_1)

        frame_num = 0
        while True:
            if len(self.image_deque) < 1:
                continue
            print("Here")
            frame_2 = self.image_deque.popleft()
            nxt_frame = convert_to_grayscale(frame_2)

            # compute optical flow between current and next frame
            cur_points, nxt_points, stdev, err = compute_optical_flow_LK(cur_frame, nxt_frame, cur_points, LK_PARAMS)

            # select good points, with standard deviation 1. use numpy index trick
            good_cur = cur_points[stdev==1]
            good_nxt = nxt_points[stdev==1]

            # create img with added tracks for all point pairs on next frame
            img, mask = draw_tracks(frame_2, good_cur, good_nxt, None, mask, track_thickness=2)

            # add optical flow arrows 
            img = draw_sparse_optical_flow_arrows(img, good_cur, good_nxt, thickness=2, arrow_scale=10.0, color=RED_CV)

            cv.imshow('Controllers sees', img)
            # save image
            frame_num += 1
            # img_name = f'frame_{str(_frame_num).zfill(4)}.jpg'
            # img_path = os.path.join(LK_TEMP_FOLDER, img_name)
            # cv.imwrite(img_path, img)

            # ready for next iteration
            cur_frame = nxt_frame.copy()
            cur_points = good_nxt.reshape(-1, 1, 2) # -1 indicates to infer that dim size

            # every n seconds (n*FPS frames), get good points
            num_seconds = 1
            if frame_num % (num_seconds*FPS) == 0:
                pass#cur_points = cv.goodFeaturesToTrack(cur_frame, mask=None, **FEATURE_PARAMS)
                # for every point in good point if its not there in cur points, add , update color too
                


            

            cv.waitKey(1)

        cv.destroyAllWindows()


    def run_experiment(self):
        self.controller_thread = th.Thread(target=self.run_controller, daemon=True)
        self.controller_thread.start()
        self.run_simulator()




if __name__ == "__main__":
    experiment_manager = ExperimentManager()
    experiment_manager.run_experiment()