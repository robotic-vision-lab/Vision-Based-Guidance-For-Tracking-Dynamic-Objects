from math import ceil

class Target:
    """Encapsulates necessary and sufficient attributes to define a visual object/target
    """
    def __init__(self, 
                 sprite_obj=None, 
                 bounding_box=None,
                 occlusion_state=None,
                 old_feature_points=None,
                 new_feature_points=None,
                 centroid_offset=[0,0]):
        self.sprite_obj = sprite_obj
        self.bounding_box = bounding_box
        self.occlusion_state = occlusion_state
        self.old_feature_points = old_feature_points
        self.new_feature_points = new_feature_points
        self.centroid_offset = centroid_offset

    def get_bounding_box(self):
        x = self.sprite_obj.rect.centerx - ceil(self.sprite_obj.rect.width * 0.8)
        y = self.sprite_obj.rect.centery - ceil(self.sprite_obj.rect.height * 0.8)
        w = ceil(self.sprite_obj.rect.width * 1.6)
        h = ceil(self.sprite_obj.rect.height * 1.6)
        self.bounding_box = (x, y, w, h)
        return (x, y, w, h)