from math import ceil
import numpy as np

from .settings import (NO_OCC,
                      PARTIAL_OCC,
                      TOTAL_OCC)

class Target:
    """Encapsulates necessary and sufficient attributes to define a visual object/target
    """
    def __init__(self, 
                 sprite_obj=None, 
                 bounding_box=None,
                 occlusion_state=None,
                 keypoints_old=None,
                 keypoints_new=None,
                 feature_found_statuses=np.array([[1]]*4),
                 cross_feature_errors=np.array([[0]]*4),
                 occlusion_case_old=None,
                 occlusion_case_new=NO_OCC,
                 centroid_old=None,
                 centroid_new=None,
                 centroid_adjustment=None,
                 centroid_offset=[0,0],
                 bb_top_left_offset=[0,0]):
        self.sprite_obj = sprite_obj
        self.bounding_box = bounding_box
        self.occlusion_state = occlusion_state
        self.keypoints_old = keypoints_old
        self.keypoints_new = keypoints_new
        self.feature_found_statuses = feature_found_statuses
        self.cross_feature_errrors = cross_feature_errors
        self.occlusion_case_old = occlusion_case_old
        self.occlusion_case_new = occlusion_case_new
        self.centroid_offset = centroid_offset
        self.bb_top_left_offset = bb_top_left_offset

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

    