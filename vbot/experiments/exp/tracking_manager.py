from math import degrees

import numpy as np
import pygame
import cv2 as cv
import matplotlib.pyplot as plt
from pygame.locals import *
from .ellipse import Ellipse2D
from .settings import *



class TrackingManager:
    """[summary]
    """
    def __init__(self, exp_manager):
        self.exp_manager = exp_manager
        self.targets = None
        self.ellipse = Ellipse2D(exp_manager, self)

        self.ellipse_params_meas = None
        self.ellipse_params_est = None

        # # for debugging 
        # self.temp = False
        # self.x = None
        # self.y = None
        # self.x_ = None
        # self.y_ = None
        

    def set_targets(self, targets):
        self.targets = targets

    def get_points_to_be_enclosed(self):
        points = []
        for target in self.targets:
            for point in target.get_4_enclosing_points():
                points.append(point)

        points = np.concatenate(points, axis=0).reshape(-1, 1, 2)

        return points

    def compute_enclosing_ellipse(self, tolerance=None):
        points_to_enclose = self.get_points_to_be_enclosed()
        self.ellipse_params_meas = self.ellipse.enclose_points(points_to_enclose, tolerance)

        # # DEBUGGING
        # self.x = np.array([p[0][0] for p in points_to_enclose]).flatten()
        # self.y = np.array([p[0][1] for p in points_to_enclose]).flatten()

        # fp_x = np.array([self.ellipse_params_meas[5][0], self.ellipse_params_meas[6][0]]).flatten()
        # fp_y = np.array([self.ellipse_params_meas[5][1], self.ellipse_params_meas[6][1]]).flatten()
        # self.x = np.concatenate((self.x, fp_x))
        # self.y = np.concatenate((self.y, fp_y))
        # plt.plot(self.x[:-2], self.y[:-2], 'k*', alpha=0.9)
        # plt.plot(self.x[-2:], self.y[-2:], 'r*', alpha=0.9)
        # if self.temp:
        #     plt.plot(self.x_, self.y_, 'k*', alpha=0.3)
        #     plt.plot(self.x_[-2:], self.y_[-2:], 'r*', alpha=0.3)

        # self.x_ = self.x
        # self.y_ = self.y
        # self.temp = True if not self.temp else False
        # plt.title(f'time - {self.exp_manager.simulator.time}')
        # plt.axis('equal')
        # plt.grid()
        # plt.show()


        return self.ellipse_params_meas

    def convert(self, point):
        px2m = self.exp_manager.simulator.pxm_fac
        point = pygame.Vector2(point) - self.exp_manager.get_cam_origin()
        point = point.elementwise() * (1, -1) / px2m
        point[0] = int(point[0])
        point[1] = int(point[1]) + HEIGHT
        point += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)
        return tuple(point)

    def get_ellipse_params(self, frame_of_reference=WORLD_INERTIAL_REF_FRAME):
        if frame_of_reference == WORLD_INERTIAL_REF_FRAME:
            return self.ellipse.get_params()
        elif frame_of_reference == IMAGE_REF_FRAME:

            # convert params
            ellipse_params = self.ellipse.get_params()

            ellipse_center = self.convert(ellipse_params[2])
            ellipse_focal_point_1 = self.convert(ellipse_params[5])
            ellipse_focal_point_2 = self.convert(ellipse_params[6])
            ellipse_axes = [int(axis/self.exp_manager.simulator.pxm_fac) for axis in ellipse_params[:2]]
            ellipse_rotation_angle = degrees(ellipse_params[3])

            return (ellipse_axes[0], 
                    ellipse_axes[1], 
                    ellipse_center, 
                    ellipse_rotation_angle, 
                    ellipse_focal_point_1,
                    ellipse_focal_point_2)


    def display(self):
        # collect all params
        ellipse_params = self.get_ellipse_params(IMAGE_REF_FRAME)
        ellipse_axes = tuple(map(int, ellipse_params[:2]))
        ellipse_center = tuple(map(int, ellipse_params[2]))
        ellipse_rotation_angle = ellipse_params[3]
        ellipse_focal_point_1 = tuple(map(int, ellipse_params[4]))
        ellipse_focal_point_2 = tuple(map(int, ellipse_params[5]))

        # focal point estimations must be called before calling display
        ellipse_focal_point_1_est = (self.ellipse_params_est[0],
                                    self.ellipse_params_est[3])
        ellipse_focal_point_2_est = (self.ellipse_params_est[6],
                                    self.ellipse_params_est[9])

        ellipse_focal_point_1_est = self.convert(ellipse_focal_point_1_est)
        ellipse_focal_point_2_est = self.convert(ellipse_focal_point_2_est)
        ellipse_focal_point_1_est = tuple(map(int, ellipse_focal_point_1_est))
        ellipse_focal_point_2_est = tuple(map(int, ellipse_focal_point_2_est))



        # test out axis aligned bounding box
        ellipse_center_est = ((ellipse_focal_point_1_est[0] + ellipse_focal_point_2_est[0]) //2,
                        (ellipse_focal_point_1_est[1] + ellipse_focal_point_2_est[1]) //2)

        

        
        








        # draw over color edited frame and show it
        ellipse_img = np.zeros_like(self.exp_manager.multi_tracker.frame_color_edited, np.uint8)

        # draw filled ellipse
        ellipse_img = cv.ellipse(img=ellipse_img,
                                 center=ellipse_center,
                                 axes=ellipse_axes,
                                 angle=ellipse_rotation_angle,
                                 startAngle=0,
                                 endAngle=360,
                                 color=ELLIPSE_COLOR,
                                 thickness=cv.FILLED,
                                 lineType=cv.LINE_8)

        # draw measured focal points
        ellipse_img = cv.circle(ellipse_img,
                                ellipse_focal_point_1,
                                radius=ELLIPSE_MEAS_FP_RADIUS,
                                color=ELLIPSE_MEAS_FP_COLOR,
                                thickness=cv.FILLED,
                                lineType=cv.LINE_AA)
        ellipse_img = cv.circle(ellipse_img,
                                ellipse_focal_point_2,
                                radius=ELLIPSE_MEAS_FP_RADIUS,
                                color=ELLIPSE_MEAS_FP_COLOR,
                                thickness=cv.FILLED,
                                lineType=cv.LINE_AA)
        
        # draw estimated focal points
        ellipse_img = cv.circle(ellipse_img,
                                ellipse_focal_point_1_est,
                                radius=ELLIPSE_ESTD_FP_RADIUS,
                                color=ELLIPSE_ESTD_FP_COLOR,
                                thickness=cv.FILLED,
                                lineType=cv.LINE_AA)
        ellipse_img = cv.circle(ellipse_img,
                                ellipse_focal_point_2_est,
                                radius=ELLIPSE_ESTD_FP_RADIUS,
                                color=ELLIPSE_ESTD_FP_COLOR,
                                thickness=cv.FILLED,
                                lineType=cv.LINE_AA)

        # draw midpoint 
        ellipse_img = cv.circle(ellipse_img,
                                ellipse_center_est,
                                radius=2,
                                color=RED_CV,
                                thickness=cv.FILLED,
                                lineType=cv.LINE_AA)

        # blend this with color edited frame
        blended_img = self.exp_manager.multi_tracker.frame_color_edited.copy()
        mask = ellipse_img.astype(bool)
        blended_img[mask] = cv.addWeighted(self.exp_manager.multi_tracker.frame_color_edited, 
                                           1 - ELLIPSE_OPACITY,
                                           ellipse_img,
                                           ELLIPSE_OPACITY,
                                           0)[mask]

        # save it for screensaver
        self.exp_manager.multi_tracker.cur_img = blended_img

        # show blended image
        cv.imshow(self.exp_manager.multi_tracker.win_name, blended_img);cv.waitKey(1)


    def compute_focal_point_estimations(self):
        '''
        exp_manager will use this API to delegate ellipse focal point estimations
        this should be followed by converting to r and theta of focal points, 
        also speed heading, Vr Vtheta and accelerations
        then compute y1 y2
        then compute a_lat and a_long
        '''
        # collect ellipse focal points computed from measured target points 
        fp1_x, fp1_y = self.ellipse_params_meas[5]
        fp2_x, fp2_y = self.ellipse_params_meas[6]

        # filter focal points
        self.ellipse.EKF.add(fp1_x, fp1_y, fp2_x, fp2_y)

        # collect estimated state of focal points
        self.ellipse_params_est = self.ellipse.EKF.get_estimated_state()



        
        



