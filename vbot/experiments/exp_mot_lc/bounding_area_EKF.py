import numpy as np
import numpy.linalg as LA
from math import e, pow

class BoundingAreaEKF:
    """Implement continuous-continuous EKF for bounding area in stateful fashion
    """

    def __init__(self, manager, target=None):
        self.manager = manager
        # self.target = target

        self.old_width = None
        self.old_height = None
        self.old_dist = None
        self.width = None
        self.height = None
        self.dist = None

        self.H = np.array([[1.0, 0.0, 0.0]])

        self.P_width = np.diag([0.0, 0.0, 0.0])
        self.P_height = np.diag([0.0, 0.0, 0.0])
        self.P_dist = np.diag([0.0, 0.0, 0.0])

        self.cov_width = np.array([[self.P_width[0,0]], [self.P_width[1,1]], [self.P_width[2,2]]])
        self.cov_height = np.array([[self.P_height[0,0]], [self.P_height[1,1]], [self.P_height[2,2]]])
        self.cov_dist = np.array([[self.P_dist[0,0]], [self.P_dist[1,1]], [self.P_dist[2,2]]])
        
        self.alpha_acc = 0.1    # reciprocal of maneuver(acceleration) time constant. 1/60-lazy turn, 1/20-evasive,  1-atmospheric turbulence
        self.sigma_square_width = 0.1 
        self.sigma_square_height = 0.05
        self.sigma_square_dist = 0.1

        self.filter_initialized_flag = False
        self.ready = False

    def is_initialized(self):
        """Indicates if EKF is initialized

        Returns:
            bool: EKF initalized or not
        """
        return self.filter_initialized_flag

    def initialize_filter(self, width, height, dist):
        """Initializes EKF. Meant to run only once at first.

        Args:
            width (float32): width of bounding area
            heigth (float32): height of bounding area
            dist (float32): dist of bounding area
        """
        self.prev_width = width
        self.prev_height = height
        self.prev_dist = dist

        self.prev_width_dot = 0.0
        self.prev_height_dot = 0.0
        self.prev_dist_dot = 0.0

        self.prev_width_ddot = 0.0
        self.prev_height_ddot = 0.0
        self.prev_dist_ddot = 0.0
        
        self.filter_initialized_flag = True

    def add(self, width, height, dist):
        """Add measurements for filtering

        Args:
            width (float32): width of bounding area
            height (float32): height of bounding area
            dist (float32): dist of bounding area
        """
        # make sure filter is initialized
        if not self.is_initialized():
            self.initialize_filter(width, height, dist)
            return

        # filter is initialized; set ready to true
        self.ready = True

        # store measurement
        self.width = width
        self.height = height
        self.dist = dist

        # perform predictor and filter step
        self.preprocess()
        self.estimate_width()
        self.estimate_height()
        self.estimate_dist()

        # remember state estimations
        self.old_width = self.prev_width
        self.old_height = self.prev_height
        self.old_dist = self.prev_dist
        self.prev_width = self.width
        self.prev_height = self.height
        self.prev_dist = self.dist
        self.prev_width_dot = self.width_dot
        self.prev_height_dot = self.height_dot


    def preprocess(self):
        """pre compute transition matrix and process noise"""
        # set variables for better numerical efficiency
        dt = self.manager.get_sim_dt()
        adt = self.alpha_acc * dt   # αΔt
        adt2 = pow(adt, 2)
        adt3 = pow(adt, 3)
        a2 = pow(self.alpha_acc, 2)
        a3 = pow(self.alpha_acc, 3)
        a4 = pow(self.alpha_acc, 4)
        eadt = pow(e, (-adt))
        e2adt = pow(e, (-2*adt))

        # transition matrix
        self.A = np.array([[1.0, dt, (eadt + adt -1) / a2],
                           [0.0, 1.0, (1 - eadt)/(self.alpha_acc)],
                           [0.0, 0.0, eadt]])

        self.q11 = (1 - e2adt + 2*adt + (2/3)*adt3 - 2*adt2 - 4*adt*eadt) / (a4)
        self.q12 = (e2adt + 1 - 2*eadt + 2*adt*eadt - 2*adt + adt**2) / (a3)
        self.q13 = (1 - e2adt - 2*adt*eadt) / (a2)
        self.q22 = (4*eadt - 3 - e2adt + 2*adt) / (a2)
        self.q23 = (e2adt + 1 -2*eadt) / (self.alpha_acc)
        self.q33 = (1 - e2adt)

        # process noise
        self.Q = np.array([[self.q11, self.q12, self.q13],
                           [self.q12, self.q22, self.q23],
                           [self.q13, self.q23, self.q33]])


    def estimate_width(self):
        # set R and width_measured
        self.R_width = 1 #1
        self.width_measured = self.width

        self.Q_width = self.sigma_square_width * self.Q
        
        # form state vector
        state_est = np.array([[self.prev_width], [self.prev_width_dot], [self.prev_width_ddot]])

        # predict
        state_est_pre = self.A @ state_est
        P_pre = self.A @ self.P_width @ self.A.T + self.Q_width
        S = self.H @ P_pre @ self.H.T + self.R_width
        K = P_pre @ self.H.T @ LA.pinv(S)

        # correct
        state_est = state_est_pre + K @ (self.width_measured - self.H @ state_est_pre)
        self.P_width = (np.eye(3) - K @ self.H) @ P_pre
        self.cov_width = np.array([[self.P_width[0,0]], [self.P_width[1,1]], [self.P_width[1,1]]])

        # extract estimations from state vector
        self.width = state_est.flatten()[0]
        self.width_dot = state_est.flatten()[1]
        self.width_ddot = state_est.flatten()[2]


    def estimate_height(self):
        # set R and height
        self.R_height = 1 #10
        self.height_measured = self.height

        self.Q_height = self.sigma_square_height * self.Q

        # form state vector
        state_est = np.array([[self.prev_height], [self.prev_height_dot], [self.prev_height_ddot]])

        # predict
        state_est_pre = self.A @ state_est
        P_pre = self.A @ self.P_height @ self.A.T + self.height
        S = self.H @ P_pre @ self.H.T + self.R_height
        K = P_pre @ self.H.T @ LA.pinv(S)

        # correct
        state_est = state_est_pre + K @ (self.height_measured - self.H @ state_est_pre)
        self.P_height = (np.eye(3) - K @ self.H) @ P_pre
        self.cov_height = np.array([[self.P_height[0,0]], [self.P_height[1,1]], [self.P_height[1,1]]])

        # extract estimations from state vector
        self.height = state_est.flatten()[0]
        self.height_dot = state_est.flatten()[1]
        self.height_ddot = state_est.flatten()[2]


    def estimate_dist(self):
        # set R and dist
        self.R_dist = 1 #10
        self.dist_measured = self.dist

        self.Q_dist = self.sigma_square_dist * self.Q

        # form state vector
        state_est = np.array([[self.prev_dist], [self.prev_dist_dot], [self.prev_dist_ddot]])

        # predict
        state_est_pre = self.A @ state_est
        P_pre = self.A @ self.P_dist @ self.A.T + self.dist
        S = self.H @ P_pre @ self.H.T + self.R_dist
        K = P_pre @ self.H.T @ LA.pinv(S)

        # correct
        state_est = state_est_pre + K @ (self.dist_measured - self.H @ state_est_pre)
        self.P_dist = (np.eye(3) - K @ self.H) @ P_pre
        self.cov_dist = np.array([[self.P_dist[0,0]], [self.P_dist[1,1]], [self.P_dist[1,1]]])

        # extract estimations from state vector
        self.dist = state_est.flatten()[0]
        self.dist_dot = state_est.flatten()[1]
        self.dist_ddot = state_est.flatten()[2]
        

    def get_estimated_state(self):
        """return estimated state information.

        Returns:
            tuple(float32, float32, float32, float32, float32, float32): (width, height, dist, width_dot, height_dot, dist_dot)
        """
        if self.ready:
            return (self.width, self.height, self.dist, self.width_dot, self.height_dot, self.dist_dot)
        else:
            return (self.prev_width, self.prev_height, self.prev_dist, self.prev_width_dot, self.prev_height_dot, self.prev_dist_dot)
