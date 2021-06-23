from math import ceil
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
        self.r = None
        self.theta = None
        self.Vr = None
        self.Vtheta = None
        self.deltaB_est = None
        self.acc_est = None
        self.centroid_offset = [0,0]
        self.bb_top_left_offset = [0,0]

        self.EKF = ExtendedKalman(self.manager, self)
        self.update_bounding_box()

    def update_bounding_box(self):
        x = self.sprite_obj.rect.centerx - ceil(self.sprite_obj.rect.width * 0.8)
        y = self.sprite_obj.rect.centery - ceil(self.sprite_obj.rect.height * 0.8)
        w = ceil(self.sprite_obj.rect.width * 1.6)
        h = ceil(self.sprite_obj.rect.height * 1.6)
        self.bounding_box = (x, y, w, h)

    def get_updated_bounding_box(self):
        self.update_bounding_box()
        return self.bounding_box

    def get_bb_top_left_offset(self):
        self.bb_top_left_offset[0] = self.bounding_box[0] - self.sprite_obj.rect.centerx
        self.bb_top_left_offset[1] = self.bounding_box[1] - self.sprite_obj.rect.centery

