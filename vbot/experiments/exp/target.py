class Target:
    """Encapsulates necessary and sufficient attributes to define a visual object/target
    """
    def __init__(self, 
                 position=None, 
                 bounding_box=None,
                 occlusion_state=None,
                 old_feature_points=None,
                 new_feature_points=None,
                 centroid_offset=[0,0]):
        self.position = position
        self.bounding_box = bounding_box
        self.occlusion_state = occlusion_state
        self.old_feature_points = old_feature_points
        self.new_feature_points = new_feature_points
        self.centroid_offset = centroid_offset

    def get_bounding_box(self):
        pass