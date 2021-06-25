from math import ceil, atan2, cos, sin
import numpy as np

from .ekf import ExtendedKalman
from .optical_flow_config import MAX_NUM_CORNERS
from .settings import (NO_OCC,
                      PARTIAL_OCC,
                      TOTAL_OCC)

class Target:
    """Encapsulates necessary and sufficient attributes to define a visual object/target
    """
    def __init__(self, 
                 sprite_obj=None,
                 manager=None, 
                 bounding_box=None):
        self.sprite_obj = sprite_obj
        self.manager = manager
        self.bounding_box = bounding_box
        self.bounding_box_mask = None
        self.track_status = False
        self.keypoints_old = None
        self.keypoints_new = None
        self.keypoints_old_good = None
        self.keypoints_new_good = None
        self.keypoints_old_bad = None
        self.keypoints_new_bad = None
        self.rel_keypoints = None
        self.feature_found_statuses = np.array([[1]]*MAX_NUM_CORNERS)
        self.cross_feature_errors_old = np.array([[0]]*MAX_NUM_CORNERS)
        self.cross_feature_errors_new = np.array([[0]]*MAX_NUM_CORNERS)
        self.occlusion_case_old = None
        self.occlusion_case_new = NO_OCC
        self.centroid_old_true = None
        self.centroid_old = None
        self.centroid_new = None
        self.centroid_adjustment = None
        self.initial_centroid = None
        self.initial_target_descriptors = None
        self.initial_target_template_gray = None
        self.initial_target_template_color = None
        self.initial_patches_color = None
        self.initial_patches_gray = None
        self.template_matchers = None
        self.patches_gray = None
        self.template_points = None
        self.template_scores = None
        self.kinematics = None
        self.car_beta = None
        self.r_meas = None
        self.theta_meas = None
        self.Vr_meas = None
        self.Vtheta_meas = None
        self.r_est = None
        self.theta_est = None
        self.Vr_est = None
        self.Vtheta_est = None
        self.deltaB_est = None
        self.acc_est = None
        self.a_lt = 0.0
        self.a_ln = 0.0
        self.centroid_offset = [0,0]
        self.bb_top_left_offset = [0,0]

        self.EKF = ExtendedKalman(self.manager, self)
        self.update_true_bounding_box()

    def update_true_bounding_box(self):
        x = self.sprite_obj.rect.centerx - ceil(self.sprite_obj.rect.width * 0.8)
        y = self.sprite_obj.rect.centery - ceil(self.sprite_obj.rect.height * 0.8)
        w = ceil(self.sprite_obj.rect.width * 1.6)
        h = ceil(self.sprite_obj.rect.height * 1.6)
        self.bounding_box = (x, y, w, h)

    def get_updated_true_bounding_box(self):
        self.update_true_bounding_box()
        return self.bounding_box

    def get_bb_top_left_offset(self):
        self.bb_top_left_offset[0] = self.bounding_box[0] - self.sprite_obj.rect.centerx
        self.bb_top_left_offset[1] = self.bounding_box[1] - self.sprite_obj.rect.centery

    def update_estimated_bounding_box(self):
        d = ceil(5 / self.manager.simulator.pxm_fac)
        x = self.centroid_new.flatten()[0] - d
        y = self.centroid_new.flatten()[1] - d
        w = 2*d
        h = 2*d
        self.bounding_box = (x, y, w, h)

    def get_updated_estimated_bounding_box(self):
        self.update_estimated_bounding_box()
        return self.bounding_box

    @staticmethod
    def sat(x, bound):
        return min(max(x, -bound), bound)

    def filter_measurements(self):
        drone_pos_x, drone_pos_y = self.manager.get_true_drone_position()
        drone_vel_x, drone_vel_y = self.manager.get_true_drone_velocity()
        car_pos_x, car_pos_y = self.kinematics[0]
        car_vel_x, car_vel_y = self.kinematics[1]

        # convert kinematics to inertial frame
        cam_origin_x, cam_origin_y = self.manager.get_cam_origin()
        # positions translated by camera origin
        drone_pos_x = cam_origin_x
        drone_pos_y = cam_origin_y
        car_pos_x += cam_origin_x
        car_pos_y += cam_origin_y

        # compute speeds of drone and car
        drone_speed = (drone_vel_x**2 + drone_vel_y**2)**0.5
        car_speed = (car_vel_x**2 + car_vel_y**2)**0.5

        # heading angle of drone
        drone_alpha = atan2(drone_vel_y, drone_vel_x)

        # heading angle of car
        self.car_beta = atan2(car_vel_y, car_vel_x)

        # distance between the drone and car
        self.r_meas = ((car_pos_x - drone_pos_x)**2 + (car_pos_y - drone_pos_y)**2)**0.5

        # angle of LOS from drone to car
        self.theta_meas = atan2(car_pos_y - drone_pos_y, car_pos_x - drone_pos_x)

        # compute Vr and Vθ
        self.Vr_meas = car_speed*cos(self.car_beta - self.theta_meas) - drone_speed*cos(drone_alpha - self.theta_meas)
        self.Vtheta_meas = car_speed*sin(self.car_beta - self.theta_meas) - drone_speed*sin(drone_alpha - self.theta_meas)

        # use EKF 
        self.EKF.add(self.r_meas, 
                     self.theta_meas, 
                     self.Vr_meas, 
                     self.Vtheta_meas, 
                     drone_alpha, 
                     self.a_lt, 
                     self.a_ln, 
                     car_pos_x, 
                     car_pos_y, 
                     car_vel_x, 
                     car_vel_y)
        self.r_est, self.theta_est, self.Vr_est, self.Vtheta_est, self.deltaB_est, self.acc_est = self.EKF.get_estimated_state()

        # update estimated centroids
        centroids_est = self.manager.get_estimated_centroids(self)
        self.centroid_old = np.array([[centroids_est[0], centroids_est[1]]])
        self.centroid_new = np.array([[centroids_est[2], centroids_est[3]]])


