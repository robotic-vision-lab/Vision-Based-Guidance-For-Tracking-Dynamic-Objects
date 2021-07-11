class Ellipse2D:
  def __init__(self, major_axis_len=1, minor_axis_len=1, center_coords=(0,0), rotation_angle=0):
    self.major_axis_len = major_axis_len
    self.minor_axis_len = minor_axis_len
    self.center_coords = center_coords
    self.rotation_angle = rotation_angle

    self._POINT_ENCLOSURE_TOLERANCE = 0.1

  def enclose_points(self, points, tolerance=None):
    if tolerance is None:
      tolerance = self._POINT_ENCLOSURE_TOLERANCE

    

