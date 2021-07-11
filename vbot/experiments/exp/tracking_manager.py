import numpy as np
from .ellipse import Ellipse2D

class TrackingManager:
    """[summary]
    """
    def __init__(self, exp_manager):
        self.exp_manager = exp_manager
        self.targets = None
        self.ellipse = Ellipse2D()

    def set_targets(self, targets):
        self.targets = targets

    def get_points_to_be_enclosed(self):
        points = []
        for target in self.targets:
            for point in target.get_bb_4_points():
                points.append(point)

        points = np.concatenate(points, axis=0).reshape(-1, 1, 2)

        return points

    def get_enclosing_ellipse(self, tolerance=None):
        points_to_enclose = self.get_points_to_be_enclosed()
        ellipse_params = self.ellipse.enclose_points(points_to_enclose, tolerance)

        return ellipse_params



