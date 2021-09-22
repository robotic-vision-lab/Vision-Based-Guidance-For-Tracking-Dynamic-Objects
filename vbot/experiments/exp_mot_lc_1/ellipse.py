from math import cos, sin
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
from .ellipse_ekf import EllipseEKF

class Ellipse2D:
	def __init__(self, exp_manager=None, tracking_manager=None, semi_major_axis=1, semi_minor_axis=1, center_coords=(0,0), rotation_angle=0):
		self.exp_manager = exp_manager
		self.tracking_manager = tracking_manager
		
		self.semi_major_axis = semi_major_axis
		self.semi_minor_axis = semi_minor_axis
		self.center_coords = center_coords
		self.rotation_angle = rotation_angle	# rads

		self.focal_length = 0.0
		self.focal_point_1 = [0.0, 0.0]
		self.focal_point_2 = [0.0, 0.0]

		self.EKF = EllipseEKF(exp_manager, tracking_manager, self)

		self._POINT_ENCLOSURE_TOLERANCE = 0.1

	def get_params(self):
		""" returns ellipse parameters

		Returns:
			tuple: tuple of ellipse params including focal points
		"""
		return (self.semi_major_axis,		#0
				self.semi_minor_axis,		#1
				self.center_coords,			#2
				self.rotation_angle,		#3
				self.focal_length,			#4
				self.focal_point_1,			#5
				self.focal_point_2)			#6

	def update_focal_length(self):
		""" helper function updates focal length
		"""
		self.focal_length = (self.semi_major_axis**2 - self.semi_minor_axis**2)**0.5

	def update_focal_points(self):
		""" helper function updates focal points 
		"""
		self.focal_point_1[0] = self.center_coords[0] + self.focal_length * cos(self.rotation_angle)
		self.focal_point_1[1] = self.center_coords[1] + self.focal_length * sin(self.rotation_angle)
		
		self.focal_point_2[0] = self.center_coords[0] - self.focal_length * cos(self.rotation_angle)
		self.focal_point_2[1] = self.center_coords[1] - self.focal_length * sin(self.rotation_angle)

	def enclose_points(self, points, tolerance=None):
		""" Enclosed points inside an ellipse. Returns computed ellipse params

		Args:
			points (np.ndarray): points to be enclosed
			tolerance (float32, optional): Tolerance for optimization error. Defaults to None.

		Returns:
			tuple: Tuple of ellipse params including focal points
		"""
		if tolerance is None:
			tolerance = self._POINT_ENCLOSURE_TOLERANCE

		# points passed will be of shape (-1, 1, 2)
		# reshape points to 2 x NUM_POINTS
		points = points.reshape(-1,2).T
		NUM_POINTS = points.shape[-1]

		# augment points with ones to have shape 3 x NUM_POINTS
		Q = np.concatenate((points, np.ones((1, NUM_POINTS))), axis=0)


		err = 1
		u  = np.ones((NUM_POINTS, 1)) / NUM_POINTS
		d = 2

		while err > tolerance:
			# X = Q . diag(u) . Q'
			X = LA.multi_dot([Q, np.diag(u.flatten()), Q.T])

			# M = diag(Q' . X^-1 . Q)
			M = np.diag(LA.multi_dot([Q.T, LA.pinv(X), Q]))

			_MAX_M = np.max(M)
			_ARG_MAX_M = np.argmax(M)

			step_size = (_MAX_M - d - 1) / ((d + 1) * (_MAX_M - 1))
			new_u = (1 - step_size) * u
			new_u[_ARG_MAX_M] += step_size

			err = LA.norm(new_u - u)
			u = new_u

		# U = diag(u)
		U = np.diag(u.flatten())

		# A = (1/d) * ( (points . U . points') - (points . u . u.T . points.T) )
		A = (1/d) * LA.pinv(LA.multi_dot([points, U, points.T]) - LA.multi_dot([points, u, u.T, points.T]))

		# compute SVD(A)
		_, Q, V = LA.svd(A)

		# rotate V matrix by 270. augment zeros to the right, then aug 0,0,1 to the bottom
		rot_V = np.rot90(V,3)
		rot_aug2_V = np.concatenate((rot_V, np.zeros((2,1))), axis=1)
		rot_aug3_V = np.concatenate((rot_aug2_V, np.array([[0,0,1]])), axis=0)

		# extract rotation from rotated augmented V
		rotation = R.from_matrix(rot_aug3_V)

		# update ellipse params
		self.semi_minor_axis = 1 / (Q[0])**0.5
		self.semi_major_axis = 1 / (Q[1])**0.5
		self.center_coords = tuple(np.matmul(points, u).flatten())
		self.rotation_angle = rotation.as_euler('ZYX')[0]

		# update focal length and focal points. note: in that order
		self.update_focal_length()
		self.update_focal_points()

		return self.get_params()

	def update_estimations(self):
		"""
		tracking_manager will use this API to have the filtering run and collect estimations of focal_points
		(DEPRECATED)
		"""
		pass

		