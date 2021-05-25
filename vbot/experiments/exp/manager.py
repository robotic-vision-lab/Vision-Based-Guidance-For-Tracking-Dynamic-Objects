import os
import shutil
from datetime import timedelta

import cv2 as cv
import numpy as np
import pygame
from pygame.locals import *

from .simulator import Simulator
from .tracker import Tracker
from .multi_tracker import MultiTracker
from .controller import Controller
from .ekf import ExtendedKalman
from .target import Target

from .settings import *
from .my_imports import (create_video_from_images,)


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

        # save experiment options
        self.save_on = save_on
        self.write_plot = write_plot
        self.control_on = control_on
        self.tracker_on = tracker_on
        self.tracker_display_on = tracker_display_on
        self.use_true_kin = use_true_kin
        self.use_real_clock = use_real_clock
        self.draw_occlusion_bars = draw_occlusion_bars

        # instantiate simulator, tracker, controller and EKF
        self.simulator = Simulator(self)
        self.tracker = MultiTracker(self)
        self.controller = Controller(self)
        self.EKF = ExtendedKalman(self)

        # initialize simulation delta time
        self.sim_dt = 0

        # initialize offset of centroid from car rect center
        self.car_rect_center_centroid_offset = [0, 0]

        if self.save_on:
            self.simulator.save_screen = True

    def get_drone_cam_field_of_view(self):
        """helper function returns field of view (width, height) in meters

        Returns:
            tuple(float32, float32): Drone camera field of view
        """
        return self.simulator.get_camera_fov()

    def get_true_drone_position(self):
        """helper function returns true drone position fetched from simulator

        Returns:
            pygame.Vector2: True drone position
        """
        return self.simulator.camera.position

    def get_true_drone_velocity(self):
        """helper function returns true drone velocity fetched from simulator

        Returns:
            pygame.Vector2: True drone velocity
        """
        return self.simulator.camera.velocity

    def get_target_bounding_box(self):
        """helper function returns bounding box of the target fetched from simulator

        Returns:
            tuple(int, int, int, int): x, y, w, h defining target bounding box
        """
        return self.simulator.bounding_box

    def transform_pos_corner_img_pixels_to_center_cam_meters(self, pos):
        """helper function transforms frame of reference for position.
        Transformation from cam attached topleft inverted image frame to center upright cam attached frame

        Args:
            pos (pygame.Vector2): Measured position

        Returns:
            pygame.Vector2: Tranformed position
        """
        pos = pos.elementwise() * (1, -1) + (0, HEIGHT)
        pos *= self.simulator.pxm_fac
        pos += -pygame.Vector2(self.get_drone_cam_field_of_view()) / 2

        return pos

    def transform_vel_img_pixels_to_cam_meters(self, vel):
        """helper function transforms frame of reference for velocity.
        Transformation from cam attached topleft inverted image frame to center upright cam attached frame

        Args:
            vel (pygame.Vector2): Measured velocity

        Returns:
            pygame.Vector2: Tranformed velocity
        """
        vel = vel.elementwise() * (1, -1)
        vel *= self.simulator.pxm_fac
        vel += self.get_true_drone_velocity()
        return vel

    def set_target_centroid_offset(self, target):
        """Worker function, to be called from tracker at the first run after first centroid calculation
        uses tracked new centroid to compute it's relative position from car rect center
        """
        self.car_rect_center_centroid_offset[0] = target.centroid_new.flatten()[0] - self.simulator.car.rect.centerx
        self.car_rect_center_centroid_offset[1] = target.centroid_new.flatten()[1] - self.simulator.car.rect.centery

    def get_target_centroid(self):
        """helper function adds centroid offset to car rect center and returns the target centroid 

        Returns:
            [np.ndarray]: Target centroid
        """
        target_cent = self.car_rect_center_centroid_offset.copy()
        target_cent[0] += self.simulator.car.rect.centerx
        target_cent[1] += self.simulator.car.rect.centery
        return np.array(target_cent).reshape(1, 2)

    def get_target_centroid_offset(self):
        """helper function returns target centroid offset

        Returns:
            list: Target centroid offset from target rect center
        """
        return self.car_rect_center_centroid_offset

    def get_bounding_box_offset(self):
        """helper function, returns bounding box toplet corner offset w.r.t car rect center

        Returns:
            [type]: [description]
        """
        return self.simulator.car_rect_center_bb_offset

    def get_target_bounding_box_from_offset(self):
        x, y, w, h = self.simulator.bounding_box
        bb_offset = self.get_bounding_box_offset()
        x = self.simulator.car.rect.center[0] + bb_offset[0]
        y = self.simulator.car.rect.center[1] + bb_offset[1]
        return x, y, w, h

    def get_estimated_centroids(self):
        old_x = self.EKF.old_x if self.EKF.old_x is not None else 0.0
        old_y = self.EKF.old_y if self.EKF.old_y is not None else 0.0
        x = self.EKF.x if self.EKF.x is not None else 0.0
        y = self.EKF.y if self.EKF.y is not None else 0.0
        old_centroid_x = old_x-self.simulator.camera.prev_origin[0]
        old_centroid_y = old_y-self.simulator.camera.prev_origin[1]
        new_centroid_x = x-self.simulator.camera.origin[0]
        new_centroid_y = y-self.simulator.camera.origin[1]

        old_centroid_x = int(old_centroid_x/self.simulator.pxm_fac) + SCREEN_CENTER[0]
        new_centroid_x = int(new_centroid_x/self.simulator.pxm_fac) + SCREEN_CENTER[0]
        old_centroid_y = int(-old_centroid_y/self.simulator.pxm_fac) + HEIGHT - SCREEN_CENTER[1]
        new_centroid_y = int(-new_centroid_y/self.simulator.pxm_fac) + HEIGHT - SCREEN_CENTER[1]
        
        return (old_centroid_x,
                old_centroid_y,
                new_centroid_x,
                new_centroid_y)

    # def filters_ready(self):
    #     ready = True
    #     if USE_TRACKER_FILTER:
    #         if USE_KALMAN:
    #             ready = ready and self.KF.done_waiting()
    #         if USE_MA:
    #             ready = ready and self.MAF.done_waiting()
    #     return ready

    def run(self):
        """Main run function. Running experiment equates to calling this function.
        """
        # initialize simulator
        self.simulator.start_new()

        # set targets 
        self.targets = []
        for sprite in self.simulator.car_sprites:
            self.targets.append(Target(sprite))

        self.tracker.set_targets(self.targets)
        # self.target_1 = Target(self.simulator.car)
        # self.target_2 = Target(self.simulator.car_2)
        # self.target_3 = Target(self.simulator.car_3)

        # self.targets = [self.target_1, self.target_2, self.target_3]


        # open plot file if write_plot is indicated
        if self.write_plot:
            self.controller.f = open(self.controller.plot_info_file, '+w')
            self.controller.f.write(
                f'TIME,'+
                f'R,'+
                f'THETA,'+
                f'V_THETA,'+
                f'V_R,'+
                f'DRONE_POS_X,'+
                f'DRONE_POS_Y,'+
                f'CAR_POS_X,'+
                f'CAR_POS_Y,'+
                f'DRONE_ACC_X,'+
                f'DRONE_ACC_Y,'+
                f'DRONE_ACC_LAT,'+
                f'DRONE_ACC_LNG,'+
                f'CAR_VEL_X,'+
                f'CAR_VEL_Y,'+
                f'TRACKED_CAR_POS_X,'+
                f'TRACKED_CAR_POS_Y,'+
                f'TRACKED_CAR_VEL_X,'+
                f'TRACKED_CAR_VEL_Y,'+
                f'CAM_ORIGIN_X,'+
                f'CAM_ORIGIN_Y,'+
                f'DRONE_SPEED,'+
                f'DRONE_ALPHA,'+
                f'DRONE_VEL_X,'+
                f'DRONE_VEL_Y,'+
                f'MEASURED_CAR_POS_X,'+
                f'MEASURED_CAR_POS_Y,'+
                f'MEASURED_CAR_VEL_X,'+
                f'MEASURED_CAR_VEL_Y,'+
                f'DRONE_ALTITUDE,'+
                f'ABS_DEN,'+
                f'MEASURED_R,'+
                f'MEASURED_THETA,'+
                f'MEASURED_V_R,'+
                f'MEASURED_V_THETA,'+
                f'TRUE_R,'+
                f'TRUE_THETA,'+
                f'TRUE_V_R,'+
                f'TRUE_V_THETA,'+
                f'DELTA_TIME,'+
                f'Y1,'+
                f'Y2,' +
                f'CAR_SPEED,' +
                f'CAR_HEADING,' +
                f'TRUE_Y1,' + 
                f'TRUE_Y2,' +
                f'OCC_CASE\n'
            )

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
                # print stuffs to console
                if not CLEAN_CONSOLE:
                    print(f'SSSS >> {str(timedelta(seconds=self.simulator.time))} >> DRONE - x:{vec_str(self.simulator.camera.position)} | v:{vec_str(self.simulator.camera.velocity)} | CAR - x:{vec_str(self.simulator.car.position)}, v: {vec_str(self.simulator.car.velocity)} | COMMANDED a:{vec_str(self.simulator.camera.acceleration)} | a_comm:{vec_str(self.simulator.cam_accel_command)} | rel_car_pos: {vec_str(self.simulator.car.position - self.simulator.camera.position)}', end='\n')

            # draw updated car, blocks and bars (drone will be drawn later)
            self.simulator.draw()

            # process screen capture *PARTY IS HERE*
            if not self.simulator.pause:

                # let tracker process image, when simulator indicates ok
                if self.simulator.can_begin_tracking():
                    if self.tracker_on:
                        screen_capture = self.simulator.get_screen_capture()
                        # process image and record status
                        status = self.tracker.process_image_complete(screen_capture)
                        self.tracker.print_to_console()

                        # let controller generate acceleration, when tracker indicates ok (which is when first frame is processed)
                        if self.tracker.can_begin_control():
                            # collect kinematics tuple
                            kin = self.get_true_kinematics() if (self.use_true_kin or not status) else self.get_tracked_kinematics()
                            # let controller process kinematics
                            ax, ay = self.controller.generate_acceleration(kin)
                            # feed controller generated acceleration commands to simulator
                            self.simulator.camera.acceleration = pygame.Vector2((ax, ay))

                    else: # tracker is off
                        if self.control_on:
                            kin = self.get_true_kinematics()
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
        # use kinematics from the tracker, but rearrange items before returning
        return (
            self.tracker.kin[0],    # true drone position    
            self.tracker.kin[1],    # true drone velocity
            self.tracker.kin[6],    # measured car position in camera frame (meters)
            self.tracker.kin[7],    # measured car velocity in camera frame (meters)
            self.tracker.kin[2],    # kalman estimated car position
            self.tracker.kin[3],    # kalman estimated car velocity
            self.tracker.kin[4],    # moving averaged car position
            self.tracker.kin[5],    # moving averaged car velocity
        ) if self.tracker.kin is not None else self.get_true_kinematics()

    def get_cam_origin(self):
        return self.simulator.camera.origin
