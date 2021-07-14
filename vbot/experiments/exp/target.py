from math import ceil, atan2, cos, sin
import numpy as np

from .target_ekf import TargetEKF
from .optical_flow_config import MAX_NUM_CORNERS
from .settings import (NO_OCC,
                      PARTIAL_OCC,
                      TOTAL_OCC,
                      NONE_KINEMATICS)

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
        
        self.good_keypoints_new = None
        self.good_distances = None
        self.distances = None
        self.matches = None

        self.occlusion_case_old = None
        self.occlusion_case_new = NO_OCC
        self.centroid_old_true = None
        self.centroid_old = None
        self.centroid_new = None
        self.centroid_old_est = None
        self.centroid_new_est = None
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
        self.r_meas = None
        self.theta_meas = None

        self.x_est = None
        self.vx_est = None
        self.ax_est = None
        self.y_est = None
        self.vy_est = None
        self.ay_est = None
        self.r_est = None
        self.theta_est = None
        self.Vr_est = None
        self.beta_est = None
        self.speed_est = None
        self.Vtheta_est = None
        self.deltaB_est = None
        self.acc_est = None

        self.centroid_offset = [0,0]
        self.bb_top_left_offset = [0,0]

        self.ID = manager.generate_target_id()

        self.EKF = TargetEKF(self.manager, self)
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
        if self.occlusion_case_old == TOTAL_OCC:
            size = int(5 + self.EKF.cov_x.flatten()[0] / 0.13)
        else:
            size = 5
        # size = 12 if self.occlusion_case_old == TOTAL_OCC else 6
        d = ceil(size / self.manager.simulator.pxm_fac)
        x = int(self.centroid_new.flatten()[0]) - d
        y = int(self.centroid_new.flatten()[1]) - d
        w = 2*d
        h = 2*d
        self.bounding_box = (x, y, w, h)

    def get_updated_estimated_bounding_box(self):
        self.update_estimated_bounding_box()
        return self.bounding_box


    def update_measurements_and_estimations(self):
        # convert kinematics to inertial frame
        cam_origin_x, cam_origin_y = self.manager.get_cam_origin()

        # drone (known)
        drone_pos_x, drone_pos_y = self.manager.get_true_drone_position()
        drone_vel_x, drone_vel_y = self.manager.get_true_drone_velocity()
        drone_pos_x += cam_origin_x
        drone_pos_y += cam_origin_y
        # speed, heading angle
        drone_speed = (drone_vel_x**2 + drone_vel_y**2)**0.5
        drone_alpha = atan2(drone_vel_y, drone_vel_x)

        # target (measured)
        target_pos_x, target_pos_y = self.kinematics[0]
        # target_vel_x, target_vel_y = self.kinematics[1]
        if not self.kinematics == NONE_KINEMATICS:
            target_pos_x += cam_origin_x
            target_pos_y += cam_origin_y

        # compute measured r, θ, Vr and Vθ
        if self.kinematics == NONE_KINEMATICS:
            self.r_meas = None
            self.theta_meas = None
        else:
            self.r_meas = ((target_pos_x - drone_pos_x)**2 + (target_pos_y - drone_pos_y)**2)**0.5
            self.theta_meas = atan2(target_pos_y - drone_pos_y, target_pos_x - drone_pos_x)

        # use EKF to filter measurements target_pos_x and target_pos_y
        self.EKF.add(target_pos_x, target_pos_y)
        self.x_est, self.vx_est, self.ax_est, self.y_est, self.vy_est, self.ay_est = self.EKF.get_estimated_state()
        
        self.beta_est = atan2(self.vy_est, self.vx_est)
        self.speed_est = (self.vx_est**2 + self.vy_est**2)**0.5

        self.r_est = ((drone_pos_x - self.x_est)**2 + (drone_pos_y - self.y_est)**2)**0.5
        self.theta_est = atan2((self.y_est - drone_pos_y), (self.x_est - drone_pos_x))
        self.Vr_est = self.speed_est*cos(self.beta_est - self.theta_est) - drone_speed*cos(drone_alpha - self.theta_est)
        self.Vtheta_est = self.speed_est*sin(self.beta_est - self.theta_est) - drone_speed*sin(drone_alpha - self.theta_est)

        self.deltaB_est = atan2(self.ay_est, self.ax_est)
        self.acc_est = (self.ax_est**2 + self.ay_est**2)**0.5


    def get_bb_4_points(self):
        d = ceil(4 / self.manager.simulator.pxm_fac)
        if not self.kinematics == NONE_KINEMATICS:
            cent = self.centroid_new.astype('int')
        else:
            cent = self.centroid_new_est.astype('int')
        points = []
        for delta in [[[-d,-d]], [[d,-d]], [[d,d]], [[-d,d]]]:
            points.append(cent + delta)

        return points


        


