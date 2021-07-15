from math import degrees

import numpy as np
import pygame

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

        return self.ellipse_params_meas


    def get_ellipse_params(self, frame_of_reference=WORLD_INERTIAL_REF_FRAME):
        if frame_of_reference == WORLD_INERTIAL_REF_FRAME:
            return self.ellipse.get_ellipse_params()
        elif frame_of_reference == IMAGE_REF_FRAME:
            # convert params
            px2m = self.exp_manager.simulator.pxm_fac

            ellipse_params = self.ellipse.get_ellipse_params()

            ellipse_axes = [int(axis/px2m) for axis in ellipse_params[:2]]
            cent = pygame.Vector2(ellipse_params[2]) - self.exp_manager.get_cam_origin()
            cent = cent.elementwise() * (1, -1) / px2m
            cent[0] = int(cent[0])
            cent[1] = int(cent[1]) + HEIGHT
            cent += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)
            ellipse_center = tuple(cent)
            ellipse_rotation_angle = degrees(ellipse_params[3])

            return (ellipse_axes[0], ellipse_axes[1], ellipse_center, ellipse_rotation_angle)



    def compute_focal_point_estimations(self):
        '''
        exp_manager will use this API to delegate ellipse focal point estimations
        this should be followed by converting to r and theta of focal points, 
        also speed heading, Vr Vtheta and accelerations
        then compute y1 y2
        then compute a_lat and a_long
        '''
        fp1_x, fp1_y = self.ellipse_params_meas[5]
        fp2_x, fp2_y = self.ellipse_params_meas[6]

        self.ellipse.EKF.add(fp1_x, fp1_y, fp2_x, fp2_y)

        
        



