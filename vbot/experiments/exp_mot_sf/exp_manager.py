import os
import shutil
from datetime import timedelta

import cv2 as cv
import numpy as np
import pygame
from pygame.locals import *
from math import degrees, atan2

from .simulator import Simulator
from .multi_tracker import MultiTracker
from .tracking_manager import TrackingManager
from .controller import Controller
from .ellipse_ekf import EllipseEKF
from .ellipse import Ellipse2D
from .target import Target

from .settings import *
from .my_imports import create_video_from_images
from .plot_manager import PlotManager
import matplotlib.pyplot as plt

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

        # initialize target ID
        self.current_id = 0

        # instantiate simulator, tracker, controller and EKF
        self.simulator = Simulator(self)
        self.multi_tracker = MultiTracker(self)
        self.tracking_manager = TrackingManager(self)
        self.controller = Controller(self)
        self.plot_manager = PlotManager(self)

        # move tracker display next to simulator 
        cv.namedWindow(self.multi_tracker.win_name)
        cv.moveWindow(self.multi_tracker.win_name, WIDTH, 0)

        # initialize simulation delta time
        self.sim_dt = 0

        # initialize offset of centroid from car rect center
        self.car_rect_center_centroid_offset = [0, 0]

        self._prev_pause_flag = False

        if self.save_on:
            self.simulator.save_screen = True

        if self.write_plot:
            self.plot_file = 'plot_info.csv'


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

    def get_target_bounding_box(self, target):
        """helper function returns bounding box of the target fetched from simulator

        Returns:
            tuple(int, int, int, int): x, y, w, h defining target bounding box
        """
        return target.get_updated_bounding_box()

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
        target.centroid_offset[0] = target.centroid_new.flatten()[0] - target.sprite_obj.rect.centerx
        target.centroid_offset[1] = target.centroid_new.flatten()[1] - target.sprite_obj.rect.centery

    def get_target_centroid(self,target=None):
        """helper function adds centroid offset to car rect center and returns the target centroid 

        Returns:
            [np.ndarray]: Target centroid
        """
        # target_cent = self.car_rect_center_centroid_offset.copy()
        # target_cent[0] += self.simulator.car.rect.centerx
        # target_cent[1] += self.simulator.car.rect.centery
        target_cent = target.centroid_offset.copy()
        target_cent[0] += target.sprite_obj.rect.centerx
        target_cent[1] += target.sprite_obj.rect.centery
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

    def get_estimated_centroids(self, target):
        # EKF returns state in inertial frame
        old_x = target.EKF.old_x if target.EKF.old_x is not None else 0.0
        old_y = target.EKF.old_y if target.EKF.old_y is not None else 0.0
        x = target.EKF.x if target.EKF.x is not None else 0.0
        y = target.EKF.y if target.EKF.y is not None else 0.0

        # convert from inertial frame to image frame. 
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

    def generate_target_id(self):
        self.current_id += 1
        return self.current_id 

    def run(self):
        """Main run function. Running experiment equates to calling this function.
        """
        # initialize simulator
        self.simulator.start_new()

        # set targets and tell multi_tracker and tracking_manager
        self.targets = [Target(sprite, self) for sprite in self.simulator.car_sprites]
        self.multi_tracker.set_targets(self.targets)
        self.tracking_manager.set_targets(self.targets)

        # open plot file if write_plot is indicated
        if self.write_plot:
            self.data_file = open(self.plot_file, '+w')
            self.data_file.write(
                f'TIME,' +
                f'FP_1_X,' +
                f'FP_1_Y,' +
                f'FP_1_VX,' +
                f'FP_1_VY,' +
                f'FP_1_AX,' +
                f'FP_1_AY,' +
                f'FP_1_R,' +
                f'FP_1_THETA,' +
                f'FP_1_V_R,' +
                f'FP_1_V_THETA,' +
                f'FP_1_SPEED,' +
                f'FP_1_HEADING,' +
                f'FP_1_ACC,' +
                f'FP_1_DELTA,' +
                f'FP_2_X,' +
                f'FP_2_Y,' +
                f'FP_2_VX,' +
                f'FP_2_VY,' +
                f'FP_2_AX,' +
                f'FP_2_AY,' +
                f'FP_2_R,' +
                f'FP_2_THETA,' +
                f'FP_2_V_R,' +
                f'FP_2_V_THETA,' +
                f'FP_2_SPEED,' +
                f'FP_2_HEADING,' +
                f'FP_2_ACC,' +
                f'FP_2_DELTA,' +
                f'Y_1,' +
                f'Y_2,' +
                f'A_LAT,' +
                f'A_LNG,' +
                f'S,' +
                f'C,' +
                f'Z_W,' +
                f'S_DOT,' +
                f'C_DOT,' +
                f'Z_W_DOT,' +
                f'AZ_S,' +
                f'AZ_C,' +
                f'AZ_Z,' +
                f'AZ,' +
                f'T_1_OCCLUSION_CASE,' +
                f'T_1_X_MEAS,' +
                f'T_1_Y_MEAS,' +
                f'T_1_R_MEAS,' +
                f'T_1_THETA_MEAS,' +
                f'T_1_X_EST,' +
                f'T_1_Y_EST,' +
                f'T_1_VX_EST,' +
                f'T_1_VY_EST,' +
                f'T_1_AX_EST,' +
                f'T_1_AY_EST,' +
                f'T_1_R_EST,' +
                f'T_1_THETA_EST,' +
                f'T_1_V_R_EST,' +
                f'T_1_V_THETA_EST,' +
                f'T_1_SPEED_EST,' +
                f'T_1_BETA_EST,' +
                f'T_1_ACC_EST,' +
                f'T_1_DELTA_EST,' +
                f'T_1_TRUE_R,' +
                f'T_1_TRUE_THETA,' +
                f'T_1_TRUE_V_R,' +
                f'T_1_TRUE_V_THETA,' +
                f'T_2_OCCLUSION_CASE,' +
                f'T_2_X_MEAS,' +
                f'T_2_Y_MEAS,' +
                f'T_2_R_MEAS,' +
                f'T_2_THETA_MEAS,' +
                f'T_2_X_EST,' +
                f'T_2_Y_EST,' +
                f'T_2_VX_EST,' +
                f'T_2_VY_EST,' +
                f'T_2_AX_EST,' +
                f'T_2_AY_EST,' +
                f'T_2_R_EST,' +
                f'T_2_THETA_EST,' +
                f'T_2_V_R_EST,' +
                f'T_2_V_THETA_EST,' +
                f'T_2_SPEED_EST,' +
                f'T_2_BETA_EST,' +
                f'T_2_ACC_EST,' +
                f'T_2_DELTA_EST,' +
                f'T_2_TRUE_R,' +
                f'T_2_TRUE_THETA,' +
                f'T_2_TRUE_V_R,' +
                f'T_2_TRUE_V_THETA,' +
                f'T_3_OCCLUSION_CASE,' +
                f'T_3_X_MEAS,' +
                f'T_3_Y_MEAS,' +
                f'T_3_R_MEAS,' +
                f'T_3_THETA_MEAS,' +
                f'T_3_X_EST,' +
                f'T_3_Y_EST,' +
                f'T_3_VX_EST,' +
                f'T_3_VY_EST,' +
                f'T_3_AX_EST,' +
                f'T_3_AY_EST,' +
                f'T_3_R_EST,' +
                f'T_3_THETA_EST,' +
                f'T_3_V_R_EST,' +
                f'T_3_V_THETA_EST,' +
                f'T_3_SPEED_EST,' +
                f'T_3_BETA_EST,' +
                f'T_3_ACC_EST,' +
                f'T_3_DELTA_EST,' +
                f'T_3_TRUE_R,' +
                f'T_3_TRUE_THETA,' +
                f'T_3_TRUE_V_R,' +
                f'T_3_TRUE_V_THETA,' +
                f'DRONE_POS_X,' +
                f'DRONE_POS_Y,' +
                f'DRONE_VEL_X,' +
                f'DRONE_VEL_Y,' +
                f'CAM_ORIGIN_X,' +
                f'CAM_ORIGIN_Y,' +
                f'DRONE_POS_X_W,' + 
                f'DRONE_POS_Y_W,' +
                f'DRONE_SPEED,' +
                f'DRONE_ALPHA\n'
            )
            self.write_count = 0
            self.write_skip = 7

        # run experiment
        while self.simulator.running:
            # get delta time between ticks (0 when paused), update elapsed simulated time
            self.simulator.dt = self.simulator.clock.tick(FPS) / 1000000.0
            if not self.use_real_clock:
                self.simulator.dt = DELTA_TIME
            if self.simulator.pause:
                self.simulator.dt = 0.0
            self.simulator.time += self.simulator.dt

            # check for final time
            if self.simulator.time > FINAL_TIME:
                self.simulator.running = False

            # handle events on simulator
            self.simulator.handle_events()


            # if quit event occurs, running will be updated; respond to it
            if not self.simulator.running:
                break
            
            # update rects and images for all sprites (not when paused)
            if not self.simulator.pause:
                self.simulator.update()
                # print stuffs to console
                # if not CLEAN_CONSOLE:
                #     print(f'SSSS >> {str(timedelta(seconds=self.simulator.time))} >> DRONE - x:{vec_str(self.simulator.camera.position)} | v:{vec_str(self.simulator.camera.velocity)} | CAR - x:{vec_str(self.simulator.car.position)}, v: {vec_str(self.simulator.car.velocity)} | COMMANDED a:{vec_str(self.simulator.camera.acceleration)} | a_comm:{vec_str(self.simulator.cam_accel_command)} | rel_car_pos: {vec_str(self.simulator.car.position - self.simulator.camera.position)}', end='\n')

            # draw updated car, blocks and bars (drone will be drawn later)
            self.simulator.draw()

            # process screen capture *PARTY IS HERE*
            if not self.simulator.pause:

                # let tracker process image, when simulator indicates ok
                if self.simulator.can_begin_tracking():
                    # get screen capture from simulator
                    screen_capture = self.simulator.get_screen_capture()
                    
                    # process image through multi_tracker; it knows to record information in targets
                    self.multi_tracker.process_image_complete(screen_capture)

                    # let controller generate acceleration, when tracker indicates ok (which is when first frame is processed)
                    if self.multi_tracker.can_begin_control():
                        # collect kinematics, compute ellipse parameters, filter
                        self.tracking_manager.compute_enclosing_ellipse(tolerance=ELLIPSE_TOLERANCE)
                        self.tracking_manager.compute_focal_point_estimations()

                        # display tracked information 
                        self.tracking_manager.display()

                        # generate acceleration command for dronecamera, apply
                        ax, ay, az = self.controller.generate_acceleration(self.tracking_manager.ellipse_params_est,
                                                                           self.tracking_manager.ellipse_params_meas[0])

                        self.simulator.camera.apply_accleration_command(ax, ay, az)

                        # self.plot_manager.uas_focal_points_plotter.collect_data()
                        if self.write_plot:
                            if self.write_count==0:
                                self.write_info()
                            self.write_count += 1
                            if self.write_count == self.write_skip:
                                self.write_count = 0
                        # self.plot_manager.plot()
                        

            self.simulator.draw_extra()
            self.simulator.show_drawing()


            if self.simulator.save_screen:
                next(self.simulator.screen_shot)

        cv.destroyAllWindows()
        if self.write_plot:
            self.data_file.close()


    def write_info(self):
              
        """
        Also quad states need to be written

        just to make a note ..
        Things to plot 

        LOS (with occlusion cases) (true, meas, est)
        ---
        1. t, r1, r2
        2. t, theta1, theta2
        3. t, Vr1, Vr2
        4. t, Vtheta1, Vtheta2

        commands
        --------
        5. t, a_lat, a_long, a_z

        Objective functions (true, est)
        -------------------
        6. t, y1
        7. t, y2

        speeds 
        ------
        8. t, speed_A, speed_f1, speed_f2, speed_B1, .. 
        9. t, diff_speed_A_f1, f2, B1, B2 ..

        headings
        --------
        10. t, heading_A, heading_f1, heading_f2, heading_B1, ..
        11. t, diff_heading_A_f1, f2, B1, B2, ..

        Trajectories
        ------------
        12. (cam) drone_x, drone_y, fpx, fpy, .., Bx, By, ..
        13. (world) drone_x, drone_y, fpx, fpy, .., Bx, By, ..


        """
        controller_data = self.controller.stored_data
        tracker_data = self.tracking_manager.stored_data

        TIME = self.simulator.time

        FP_1_X = controller_data[0]
        FP_1_Y = controller_data[1]
        FP_1_VX = controller_data[2]
        FP_1_VY = controller_data[3]
        FP_1_AX = controller_data[4]
        FP_1_AY = controller_data[5]
        FP_1_R = controller_data[6]
        FP_1_THETA = degrees(controller_data[7])
        FP_1_V_R = controller_data[8]
        FP_1_V_THETA = controller_data[9]
        FP_1_SPEED = controller_data[10]
        FP_1_HEADING = degrees(controller_data[11])
        FP_1_ACC = controller_data[12]
        FP_1_DELTA = controller_data[13]
        FP_2_X = controller_data[14]
        FP_2_Y = controller_data[15]
        FP_2_VX = controller_data[16]
        FP_2_VY = controller_data[17]
        FP_2_AX = controller_data[18]
        FP_2_AY = controller_data[19]
        FP_2_R = controller_data[20]
        FP_2_THETA = degrees(controller_data[21])
        FP_2_V_R = controller_data[22]
        FP_2_V_THETA = controller_data[23]
        FP_2_SPEED = controller_data[24]
        FP_2_HEADING = degrees(controller_data[25])
        FP_2_ACC = controller_data[26]
        FP_2_DELTA = controller_data[27]
        Y_1 = controller_data[28]
        Y_2 = controller_data[29]
        A_LAT = controller_data[30]
        A_LNG = controller_data[31]
        S = controller_data[32]
        C = controller_data[33]
        Z_W = controller_data[34]
        S_DOT = controller_data[35]
        C_DOT = controller_data[36]
        Z_W_DOT = controller_data[37]
        AZ_S = controller_data[38]
        AZ_C = controller_data[39]
        AZ_Z = controller_data[40]
        AZ = controller_data[41]

        T_1_OCCLUSION_CASE = tracker_data[0]
        T_1_X_MEAS = tracker_data[1] if tracker_data[1] is not None else NAN
        T_1_Y_MEAS = tracker_data[2] if tracker_data[2] is not None else NAN
        T_1_R_MEAS = tracker_data[3] if tracker_data[3] is not None else NAN
        T_1_THETA_MEAS = degrees(tracker_data[4]) if tracker_data[4] is not None else NAN
        T_1_X_EST = tracker_data[5]
        T_1_Y_EST = tracker_data[6]
        T_1_VX_EST = tracker_data[7]
        T_1_VY_EST = tracker_data[8]
        T_1_AX_EST = tracker_data[9]
        T_1_AY_EST = tracker_data[10]
        T_1_R_EST = tracker_data[11]
        T_1_THETA_EST = degrees(tracker_data[12])
        T_1_V_R_EST = tracker_data[13]
        T_1_V_THETA_EST = tracker_data[14]
        T_1_SPEED_EST = tracker_data[15]
        T_1_BETA_EST = degrees(tracker_data[16])
        T_1_ACC_EST = tracker_data[17]
        T_1_DELTA_EST = tracker_data[18]
        T_1_TRUE_R = tracker_data[19]
        T_1_TRUE_THETA = degrees(tracker_data[20])
        T_1_TRUE_V_R = tracker_data[21]
        T_1_TRUE_V_THETA = tracker_data[22]

        T_2_OCCLUSION_CASE = tracker_data[23]
        T_2_X_MEAS = tracker_data[24] if tracker_data[24] is not None else NAN
        T_2_Y_MEAS = tracker_data[25] if tracker_data[25] is not None else NAN
        T_2_R_MEAS = tracker_data[26] if tracker_data[26] is not None else NAN
        T_2_THETA_MEAS = degrees(tracker_data[27]) if tracker_data[27] is not None else NAN
        T_2_X_EST = tracker_data[28]
        T_2_Y_EST = tracker_data[29]
        T_2_VX_EST = tracker_data[30]
        T_2_VY_EST = tracker_data[31]
        T_2_AX_EST = tracker_data[32]
        T_2_AY_EST = tracker_data[33]
        T_2_R_EST = tracker_data[34]
        T_2_THETA_EST = degrees(tracker_data[35])
        T_2_V_R_EST = tracker_data[36]
        T_2_V_THETA_EST = tracker_data[37]
        T_2_SPEED_EST = tracker_data[38]
        T_2_BETA_EST = degrees(tracker_data[39])
        T_2_ACC_EST = tracker_data[40]
        T_2_DELTA_EST = tracker_data[41]
        T_2_TRUE_R = tracker_data[42]
        T_2_TRUE_THETA = degrees(tracker_data[43])
        T_2_TRUE_V_R = tracker_data[44]
        T_2_TRUE_V_THETA = tracker_data[45]

        T_3_OCCLUSION_CASE = tracker_data[46]
        T_3_X_MEAS = tracker_data[47] if tracker_data[47] is not None else NAN
        T_3_Y_MEAS = tracker_data[48] if tracker_data[48] is not None else NAN
        T_3_R_MEAS = tracker_data[49] if tracker_data[49] is not None else NAN
        T_3_THETA_MEAS = degrees(tracker_data[50]) if tracker_data[50] is not None else NAN
        T_3_X_EST = tracker_data[51]
        T_3_Y_EST = tracker_data[52]
        T_3_VX_EST = tracker_data[53]
        T_3_VY_EST = tracker_data[54]
        T_3_AX_EST = tracker_data[55]
        T_3_AY_EST = tracker_data[56]
        T_3_R_EST = tracker_data[57]
        T_3_THETA_EST = degrees(tracker_data[58])
        T_3_V_R_EST = tracker_data[59]
        T_3_V_THETA_EST = tracker_data[60]
        T_3_SPEED_EST = tracker_data[61]
        T_3_BETA_EST = degrees(tracker_data[62])
        T_3_ACC_EST = tracker_data[63]
        T_3_DELTA_EST = tracker_data[64]
        T_3_TRUE_R = tracker_data[65]
        T_3_TRUE_THETA = degrees(tracker_data[66])
        T_3_TRUE_V_R = tracker_data[67]
        T_3_TRUE_V_THETA = tracker_data[68]

        DRONE_POS_X, DRONE_POS_Y = self.get_true_drone_position()
        DRONE_VEL_X, DRONE_VEL_Y = self.get_true_drone_velocity()
        CAM_ORIGIN_X, CAM_ORIGIN_Y = self.get_cam_origin()

        DRONE_POS_X_W = DRONE_POS_X + CAM_ORIGIN_X
        DRONE_POS_Y_W = DRONE_POS_Y + CAM_ORIGIN_Y

        DRONE_SPEED = (DRONE_VEL_X**2 + DRONE_VEL_Y**2)**0.5
        DRONE_ALPHA = degrees(atan2(DRONE_VEL_Y, DRONE_VEL_X))

        self.data_file.write(
            f'{TIME},' +
            f'{FP_1_X},' +
            f'{FP_1_Y},' +
            f'{FP_1_VX},' +
            f'{FP_1_VY},' +
            f'{FP_1_AX},' +
            f'{FP_1_AY},' +
            f'{FP_1_R},' +
            f'{FP_1_THETA},' +
            f'{FP_1_V_R},' +
            f'{FP_1_V_THETA},' +
            f'{FP_1_SPEED},' +
            f'{FP_1_HEADING},' +
            f'{FP_1_ACC},' +
            f'{FP_1_DELTA},' +
            f'{FP_2_X},' +
            f'{FP_2_Y},' +
            f'{FP_2_VX},' +
            f'{FP_2_VY},' +
            f'{FP_2_AX},' +
            f'{FP_2_AY},' +
            f'{FP_2_R},' +
            f'{FP_2_THETA},' +
            f'{FP_2_V_R},' +
            f'{FP_2_V_THETA},' +
            f'{FP_2_SPEED},' +
            f'{FP_2_HEADING},' +
            f'{FP_2_ACC},' +
            f'{FP_2_DELTA},' +
            f'{Y_1},' +
            f'{Y_2},' +
            f'{A_LAT},' +
            f'{A_LNG},' +
            f'{S},' +
            f'{C},' +
            f'{Z_W},' +
            f'{S_DOT},' +
            f'{C_DOT},' +
            f'{Z_W_DOT},' +
            f'{AZ_S},' +
            f'{AZ_C},' +
            f'{AZ_Z},' +
            f'{AZ},' +
            f'{T_1_OCCLUSION_CASE},' +
            f'{T_1_X_MEAS},' +
            f'{T_1_Y_MEAS},' +
            f'{T_1_R_MEAS},' +
            f'{T_1_THETA_MEAS},' +
            f'{T_1_X_EST},' +
            f'{T_1_Y_EST},' +
            f'{T_1_VX_EST},' +
            f'{T_1_VY_EST},' +
            f'{T_1_AX_EST},' +
            f'{T_1_AY_EST},' +
            f'{T_1_R_EST},' +
            f'{T_1_THETA_EST},' +
            f'{T_1_V_R_EST},' +
            f'{T_1_V_THETA_EST},' +
            f'{T_1_SPEED_EST},' +
            f'{T_1_BETA_EST},' +
            f'{T_1_ACC_EST},' +
            f'{T_1_DELTA_EST},' +
            f'{T_1_TRUE_R},' +
            f'{T_1_TRUE_THETA},' +
            f'{T_1_TRUE_V_R},' +
            f'{T_1_TRUE_V_THETA},' +
            f'{T_2_OCCLUSION_CASE},' +
            f'{T_2_X_MEAS},' +
            f'{T_2_Y_MEAS},' +
            f'{T_2_R_MEAS},' +
            f'{T_2_THETA_MEAS},' +
            f'{T_2_X_EST},' +
            f'{T_2_Y_EST},' +
            f'{T_2_VX_EST},' +
            f'{T_2_VY_EST},' +
            f'{T_2_AX_EST},' +
            f'{T_2_AY_EST},' +
            f'{T_2_R_EST},' +
            f'{T_2_THETA_EST},' +
            f'{T_2_V_R_EST},' +
            f'{T_2_V_THETA_EST},' +
            f'{T_2_SPEED_EST},' +
            f'{T_2_BETA_EST},' +
            f'{T_2_ACC_EST},' +
            f'{T_2_DELTA_EST},' +
            f'{T_2_TRUE_R},' +
            f'{T_2_TRUE_THETA},' +
            f'{T_2_TRUE_V_R},' +
            f'{T_2_TRUE_V_THETA},' +
            f'{T_3_OCCLUSION_CASE},' +
            f'{T_3_X_MEAS},' +
            f'{T_3_Y_MEAS},' +
            f'{T_3_R_MEAS},' +
            f'{T_3_THETA_MEAS},' +
            f'{T_3_X_EST},' +
            f'{T_3_Y_EST},' +
            f'{T_3_VX_EST},' +
            f'{T_3_VY_EST},' +
            f'{T_3_AX_EST},' +
            f'{T_3_AY_EST},' +
            f'{T_3_R_EST},' +
            f'{T_3_THETA_EST},' +
            f'{T_3_V_R_EST},' +
            f'{T_3_V_THETA_EST},' +
            f'{T_3_SPEED_EST},' +
            f'{T_3_BETA_EST},' +
            f'{T_3_ACC_EST},' +
            f'{T_3_DELTA_EST},' +
            f'{T_3_TRUE_R},' +
            f'{T_3_TRUE_THETA},' +
            f'{T_3_TRUE_V_R},' +
            f'{T_3_TRUE_V_THETA},' +
            f'{DRONE_POS_X},' +
            f'{DRONE_POS_Y},' +
            f'{DRONE_VEL_X},' +
            f'{DRONE_VEL_Y},' +
            f'{CAM_ORIGIN_X},' +
            f'{CAM_ORIGIN_Y},' +
            f'{DRONE_POS_X_W},' + 
            f'{DRONE_POS_Y_W},' +
            f'{DRONE_SPEED},' +
            f'{DRONE_ALPHA}\n'
        )





    @staticmethod
    def make_video(video_name, folder_path, fps=FPS):
        """Helper function, looks for frames in given folder,
        writes them into a video file, with the given name.
        Also removes the folder after creating the video.
        """
        if os.path.isdir(folder_path):
            create_video_from_images(folder_path, 'png', video_name, fps)

            # delete folder
            shutil.rmtree(folder_path)

    def get_sim_dt(self):
        """return simulation delta time

        Returns:
            float: simulation delta time
        """
        return self.simulator.dt

    def get_true_kinematics(self, target):
        """Helper function returns drone and car position and velocity

        Returns:
            tuple: drone_pos, drone_vel, car_pos, car_vel
        """
        kin = (self.simulator.camera.position,
               self.simulator.camera.velocity,
               target.sprite_obj.position,
               target.sprite_obj.velocity)

        return kin

    def get_drone_kinematics(self):
        """Helper function returns drone position and velocity (camera frame meters)

        Returns:
            tuple: drone_pos, drone_vel
        """
        drone_kin = (self.simulator.camera.position,
                     self.simulator.camera.velocity)
        
        return drone_kin


    def get_true_target_kinematics(self, target):
        """Helper function returns true position and velocity (camera frame meters) of given target

        Args:
            target (Target): Target object whose kinematics is requested

        Returns:
            tuple: target_pos, target_vel
        """
        target_kin = (target.sprite_obj.position,
                      target.sprite_obj.velocity)

        return target_kin

    def get_tracked_kinematics(self, target):
        """Helper function returns tracked kinematics obtained from measurements by tracker

        Args:
            target (Target): Target object whose tracked kinematics is requested 

        Returns:
            tuple: true_drone_pos, true_drone_vel, meas_target_pos, meas_target_vel
        """
        # use kinematics from the tracker, but rearrange items before returning
        return (
            target.kinematics[0],    # true drone position    
            target.kinematics[1],    # true drone velocity
            target.kinematics[2],    # measured car position in camera frame (meters)
            target.kinematics[3],    # measured car velocity in camera frame (meters)
        ) if target.kinematics is not None else self.get_true_kinematics(target)

    def get_cam_origin(self):
        """Returns the drone camera origin

        Returns:
            pygame.Vector2: Camera origin (world frame meters)
        """
        return self.simulator.camera.origin
