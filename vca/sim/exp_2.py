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

from game_utils import load_image, _prep_temp_folder
from utils.vid_utils import create_video_from_images
from utils.optical_flow_utils import (get_OF_color_encoded, 
                                      draw_sparse_optical_flow_arrows,
                                      draw_tracks)
from utils.img_utils import convert_to_grayscale, put_text, images_assemble
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



if __name__ == "__main__":

    RUN_EXPERIMENT = 0
    EXPERIMENT_SAVE_MODE_ON = 0
    WRITE_TRACK = 0
    RUN_TRACK_PLOT = 1
    if RUN_EXPERIMENT:
        experiment_manager = ExperimentManager(EXPERIMENT_SAVE_MODE_ON, WRITE_TRACK)
        experiment_manager.run_experiment()

    if RUN_TRACK_PLOT:
        f = open('track.txt', 'r')
        time = []
        true_vel_x = []
        true_vel_y = []
        comp_vel_x = []
        comp_vel_y = []

        for line in f.readlines():
            t, tvx, tvy, cvx, cvy = tuple(map(float, list(map(str.strip, line.strip().split()))))
            time.append(t)
            true_vel_x.append(tvx)
            true_vel_y.append(tvy)
            comp_vel_x.append(cvx)
            comp_vel_y.append(cvy)

        import matplotlib.pyplot as plt
        plt.style.use('seaborn-whitegrid')
        plt.plot(time, comp_vel_x, color='tan', linestyle='-', linewidth=1, label='computed relative vx')
        plt.plot(time, true_vel_x, color='teal', linestyle=':', linewidth=3, label='true relative vx')
        # plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.legend()
        plt.xlabel('seconds')
        plt.ylabel('pixels/second')
        plt.savefig('relative_vx.png')
        plt.show()




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
    def __init__(self, save_on=False, write_track=False):

        self.save_on = save_on
        self.write_track = write_track

        self.simulator = Simulator(self)
        self.tracker = Tracker(self)
        self.controller = Controller(self)

        self.image_deque = deque(maxlen=100)
        self.command_deque = deque(maxlen=100)

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


    def run_simulator(self):
        """Run Simulator
        """
        self.simulator.start_new()
        self.simulator.run()


    def run_controller(self):
        """Run Controller
        """
        pass
    

    def run_tracker(self):
        """Run Tracker
        """
        self.tracker.run()
        

    def run_experiment(self):
        """Run Experiment by running Simulator, Tracker and Controller.
        """
        self.tracker_thread = th.Thread(target=self.run_tracker, daemon=True)
        self.tracker_thread.start()
        self.run_simulator()
        if self.save_on:
            self.make_video('sim_track.avi', TEMP_FOLDER)


    def make_video(self, video_name, folder_path):
        """Helper function, looks for frames in given folder,
        writes them into a video file, with the given name. 
        Also removes the folder after creating the video.
        """
        if os.path.isdir(folder_path):
            create_video_from_images(folder_path, 'jpg', video_name, FPS)

            # delete folder
            shutil.rmtree(folder_path)



class Simulator:
    """Simulator object creates the simulation game. 
    Responds to keypresses 'p' to toggle play/pause, 's' to save screen mode, ESC to quit.
    While running simulation, it also dumps the screens to a shared memory location.
    Designed to work with an ExperimentManager object.
    """
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
            if self.pause:
                self.dt = 0
            self.time += self.dt
            self.manager.sim_dt = self.dt

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
                self.manager.true_rel_vel = self.car.velocity - self.camera.velocity

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
        """Handles captured events.
        """
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
                        self.bb_start = self.bb_end = None
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
        """Update positions of components.
        """
        # update Group. (All sprites in it will get updated)
        self.all_sprites.update()
        self.camera.move(deepcopy(self.euc_factor * self.cam_accel_command))
        self.cam_accel_command = pygame.Vector2(0, 0)


    def draw(self):
        """Draws components on screen. Note: drone_img is drawn after screen capture for tracking is performed.
        """
        # fill background
        self.screen_surface.fill(SCREEN_BG_COLOR)
        sim_fps = 'NA' if self.dt == 0 else f'{1/self.dt:.2f}'
        
        pygame.display.set_caption(f'car position {self.car.position} | cam velocity {self.camera.velocity} | FPS {sim_fps}')

        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)
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
            

    def screen_saver(self, path):
        """Creates a generator to perform screen saving.

        Args:
            path (str): Path where screen captured frames are to be stored.
        """
        _prep_temp_folder(path)

        frame_num = 0
        while True:
            frame_num += 1
            image_name = f'frame_{str(frame_num).zfill(4)}.jpg'
            file_path = os.path.join(path, image_name)
            img_sim = self.get_screen_capture()
            img_track = self.manager.tracker.cur_img
            if img_track is None:
                img_track = np.ones_like(img_sim, dtype='uint8') * 31
            img = images_assemble([img_sim, img_track], (1,2))
            cv.imwrite(file_path, img)
            # pygame.image.save(screen, file_path)
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
        self.velocities = deque()
        self.vel_file = 'track.txt'
        self.cur_img = None


    def run(self):
        """Keeps running the tracker main functions.
        Reads bounding box from it's ExperimentManager and computed features to be tracked.
        """
        # get first frame
        while True:# len(self.image_deque) <= 1:
            if not (self.manager.simulator.bb_start and self.manager.simulator.bb_end) or self.manager.simulator.pause:
                continue
            if len(self.manager.image_deque) > 0:
                frame_1 = self.manager.image_deque.popleft()
                cur_frame = convert_to_grayscale(frame_1)
                break
        
        # compute good feature points to track
        feature_mask = np.zeros_like(cur_frame)
        x,y,w,h = self.manager.simulator.bounding_box
        feature_mask[y:y+h+1, x:x+w+1] = 1
        cur_points = cv.goodFeaturesToTrack(cur_frame, mask=feature_mask, **FEATURE_PARAMS)

        # create mask for drawing tracks
        mask = np.zeros_like(frame_1)

        f = open(self.vel_file, '+w')

        frame_num = 0
        while True:
            if len(self.manager.image_deque) < 1:
                continue
            frame_2 = self.manager.image_deque.popleft()
            nxt_frame = convert_to_grayscale(frame_2)

            # compute optical flow between current and next frame
            cur_points, nxt_points, stdev, err = compute_optical_flow_LK(cur_frame, nxt_frame, cur_points, LK_PARAMS)

            # select good points, with standard deviation 1. use numpy index trick
            good_cur = cur_points[stdev==1]
            good_nxt = nxt_points[stdev==1]

            # compute velocity
            velocity = self.compute_velocity(good_cur, good_nxt)

            # create img with added tracks for all point pairs on next frame
            img, mask = draw_tracks(frame_2, good_cur, good_nxt, None, mask, track_thickness=1)

            # add optical flow arrows 
            img = draw_sparse_optical_flow_arrows(img, good_cur, good_nxt, thickness=2, arrow_scale=10.0, color=RED_CV)

            # put velocity text 
            img = self.put_velocity_text(img, velocity)
            if self.manager.write_track:
                f.write(f'{self.manager.simulator.time:.2f} {self.manager.true_rel_vel[0]:.2f} {self.manager.true_rel_vel[1]:.2f} {velocity[0]:.2f} {velocity[1]:.2f}\n')
            self.cur_img = img
            cv.imshow('Tracking in progress', img)
            
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
        f.close()


    def compute_velocity(self, cur_pts, nxt_pts):
        """Helper function, takes in current and next points (corresponding to an object) and 
        computes the average velocity using elapsed simulation time from it's ExperimentManager.

        Args:
            cur_pts (np.ndarray): feature points in frame_1 or current frame (prev frame)
            nxt_pts (np.ndarray): feature points in frame_2 or next frame 

        Returns:
            tuple(float, float): mean of velocities computed from each point pair.
        """
        vx = 0
        vy = 0
        for cur_pt, nxt_pt in zip(cur_pts, nxt_pts):
            vx += nxt_pt[0] - cur_pt[0]
            vy += nxt_pt[1] - cur_pt[1]

        num_pts = len(cur_pts)
        # converting from px/frame to px/secs. Averaging
        vx /= self.manager.sim_dt * num_pts
        vy /= self.manager.sim_dt * num_pts

        return (vx, vy)

    
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



class Controller:
    def __init__(self, manager):
        self.manager = manager


