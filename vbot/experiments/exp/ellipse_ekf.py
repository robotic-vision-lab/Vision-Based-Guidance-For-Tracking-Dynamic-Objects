import numpy as np
import numpy.linalg as LA
from math import atan2, sin, cos, e, pi, tau

class EllipseEKF:
    """Implement continuous-continuous EKF for ellipse in stateful fashion
    """

    def __init__(self, exp_manager, tracking_manager, ellipse=None):
        self.manager = exp_manager
        self.tracking_manager = tracking_manager
        self.ellipse = ellipse

        self.old_fp1_x = None
        self.old_fp1_y = None
        self.fp1_x = None
        self.fp1_y = None
        self.old_fp2_x = None
        self.old_fp2_y = None
        self.fp2_x = None
        self.fp2_y = None

        self.H = np.array([[1.0, 0.0, 0.0]])

        self.P_fp1_x = np.diag([0.0, 0.0, 0.0])
        self.P_fp1_y = np.diag([0.0, 0.0, 0.0])
        self.P_fp2_x = np.diag([0.0, 0.0, 0.0])
        self.P_fp2_y = np.diag([0.0, 0.0, 0.0])

        self.cov_fp1_x = np.array([[self.P_fp1_x[0,0]], [self.P_fp1_x[1,1]], [self.P_fp1_x[2,2]]])
        self.cov_fp1_y = np.array([[self.P_fp1_y[0,0]], [self.P_fp1_y[1,1]], [self.P_fp1_y[2,2]]])
        self.cov_fp2_x = np.array([[self.P_fp2_x[0,0]], [self.P_fp2_x[1,1]], [self.P_fp2_x[2,2]]])
        self.cov_fp2_y = np.array([[self.P_fp2_y[0,0]], [self.P_fp2_y[1,1]], [self.P_fp2_y[2,2]]])
        
        self.alpha_acc = 0.1    # reciprocal of maneuver(acceleration) time constant. 1/60-lazy turn, 1/20-evasive,  1-atmospheric turbulence
        self.sigma_square = 0.1

        self.filter_initialized_flag = False
        self.ready = False

    def is_initialized(self):
        """Indicates if EKF is initialized

        Returns:
            bool: EKF initalized or not
        """
        return self.filter_initialized_flag

    def initialize_filter(self, fp1_x, fp1_vx, fp1_y, fp1_vy, fp2_x, fp2_vx, fp2_y, fp2_vy):
        """Initializes ellipse EKF. Meant to run only once at first.

        Args:
            fp1_x (float32): focal point 1 position x component in inertial frame (m)
            fp1_y (float32): focal point 1 position y component in inertial frame (m)
            fp1_vx (float32): focal point 1 velocity vx component in inertial frame (m/s)
            fp1_vy (float32): focal point 1 velocity vy component in inertial frame (m/s)
            fp2_x (float32): focal point 2 position x component in inertial frame (m)
            fp2_y (float32): focal point 2 position y component in inertial frame (m)
            fp2_vx (float32): focal point 2 velocity vx component in inertial frame (m/s)
            fp2_vy (float32): focal point 2 velocity vy component in inertial frame (m/s)
        """
        self.prev_fp1_x = fp1_x
        self.prev_fp1_y = fp1_y
        self.prev_fp1_vx = fp1_vx
        self.prev_fp1_vy = fp1_vy
        self.prev_fp1_ax = 0.0
        self.prev_fp1_ay = 0.0

        self.prev_fp2_x = fp2_x
        self.prev_fp2_y = fp2_y
        self.prev_fp2_vx = fp2_vx
        self.prev_fp2_vy = fp2_vy
        self.prev_fp2_ax = 0.0
        self.prev_fp2_ay = 0.0
        
        self.filter_initialized_flag = True

    def add(self, fp1_x, fp1_y, fp2_x, fp2_y):
        """Add measurements and auxiliary data for filtering

        Args:
            x (float32): target position x component in inertial frame (m)
            y (float32): target position y component in inertial frame (m)
        """
        # make sure filter is initialized
        if not self.is_initialized():
            vx = self.target.sprite_obj.velocity[0]
            vy = self.target.sprite_obj.velocity[1]
            self.initialize_filter(x, y, vx, vy)
            return

        # filter is initialized; set ready to true
        self.ready = True

        # store measurement
        self.fp1_x = fp1_x
        self.fp1_y = fp1_y
        self.fp2_x = fp2_x
        self.fp2_y = fp2_y

        # perform predictor and filter step
        self.preprocess()
        self.estimate_fp1_x()
        self.estimate_fp1_y()
        self.estimate_fp2_x()
        self.estimate_fp2_y()

        # remember state estimations
        self.old_fp1_x = self.prev_fp1_x
        self.old_fp1_y = self.prev_fp1_y
        self.prev_fp1_x = self.fp1_x
        self.prev_fp1_y = self.fp1_y
        self.prev_fp1_vx = self.fp1_vx
        self.prev_fp1_vy = self.fp1_vy
        
        self.old_fp2_x = self.prev_fp2_x
        self.old_fp2_y = self.prev_fp2_y
        self.prev_fp2_x = self.fp2_x
        self.prev_fp2_y = self.fp2_y
        self.prev_fp2_vx = self.fp2_vx
        self.prev_fp2_vy = self.fp2_vy


    def preprocess(self):
        dt = self.manager.get_sim_dt()
        adt = self.alpha_acc * dt   # αΔt
        eadt = e**(-adt)
        e2adt = e**(-2*adt)

        # transition matrix
        self.A = np.array([[1.0, dt, (eadt + adt -1)/(self.alpha_acc**2)],
                               [0.0, 1.0, (1 - eadt)/(self.alpha_acc)],
                               [0.0, 0.0, eadt]])

        self.q11 = (1 - e2adt + 2*adt + (2/3)*adt**3 - 2*adt**2 - 4*adt*eadt) / (self.alpha_acc**4)
        self.q12 = (e2adt + 1 - 2*eadt + 2*adt*eadt - 2*adt + adt**2) / (self.alpha_acc**3)
        self.q13 = (1 - e2adt - 2*adt*eadt) / (self.alpha_acc**2)
        self.q22 = (4*eadt - 3 - e2adt + 2*adt) / (self.alpha_acc**2)
        self.q23 = (e2adt + 1 -2*eadt) / (self.alpha_acc)
        self.q33 = (1 - e2adt)

        # process noise
        self.Q = self.sigma_square * np.array([[self.q11, self.q12, self.q13],
                                               [self.q12, self.q22, self.q23],
                                               [self.q13, self.q23, self.q33]])


    def estimate_fp1_x(self):
        # set R and fp1_x appropriate to occlusion state
        if self.fp1_x is None:
            self.R = 10 #100
            self.fp1_x_measured = self.prev_fp1_x
        else:
            self.R = 1 #1
            self.fp1_x_measured = self.fp1_x

        
        # form state vector
        state_est = np.array([[self.prev_fp1_x], [self.prev_fp1_vx], [self.prev_fp1_ax]])

        # predict
        state_est_pre = np.matmul(self.A, state_est)
        P_pre = LA.multi_dot([self.A, self.P_fp1_x, self.A.T]) + self.Q
        S = LA.multi_dot([self.H, P_pre, self.H.T]) + self.R
        K = LA.multi_dot([P_pre, self.H.T, LA.pinv(S)])

        # correct
        state_est = state_est_pre + np.matmul(K, (self.fp1_x_measured - np.matmul(self.H, state_est_pre)))
        self.P_fp1_x = np.matmul((np.eye(3) - np.matmul(K, self.H)), P_pre)
        self.cov_fp1_x = np.array([[self.P_fp1_x[0,0]], [self.P_fp1_x[1,1]], [self.P_fp1_x[1,1]]])

        # extract estimations from state vector
        self.fp1_x = state_est.flatten()[0]
        self.fp1_vx = state_est.flatten()[1]
        self.fp1_ax = state_est.flatten()[2]
        
    def estimate_fp1_y(self):
        # set R and y appropriate to occlusion state
        if self.fp1_y is None:
            self.R = 10 #100
            self.fp1_y_measured = self.prev_fp1_y
        else:
            self.R = 1 #1
            self.fp1_y_measured = self.fp1_y

        
        # form state vector
        state_est = np.array([[self.prev_fp1_y], [self.prev_fp1_vy], [self.prev_fp1_ay]])

        # predict
        state_est_pre = np.matmul(self.A, state_est)
        P_pre = LA.multi_dot([self.A, self.P_fp1_y, self.A.T]) + self.Q
        S = LA.multi_dot([self.H, P_pre, self.H.T]) + self.R
        K = LA.multi_dot([P_pre, self.H.T, LA.pinv(S)])

        # correct
        state_est = state_est_pre + np.matmul(K, (self.fp1_y_measured - np.matmul(self.H, state_est_pre)))
        self.P_fp1_y = np.matmul((np.eye(3) - np.matmul(K, self.H)), P_pre)
        self.cov_fp1_y = np.array([[self.P_fp1_y[0,0]], [self.P_fp1_y[1,1]], [self.P_fp1_y[1,1]]])

        # extract estimations from state vector
        self.fp1_y = state_est.flatten()[0]
        self.fp1_vy = state_est.flatten()[1]
        self.fp1_ay = state_est.flatten()[2]

    def estimate_fp2_x(self):
        # set R and x appropriate to occlusion state
        if self.fp2_x is None:
            self.R = 10 #100
            self.fp2_x_measured = self.prev_fp2_x
        else:
            self.R = 1 #1
            self.fp2_x_measured = self.fp2_x

        
        # form state vector
        state_est = np.array([[self.prev_fp2_x], [self.prev_fp2_vx], [self.prev_fp2_ax]])

        # predict
        state_est_pre = np.matmul(self.A, state_est)
        P_pre = LA.multi_dot([self.A, self.P_fp2_x, self.A.T]) + self.Q
        S = LA.multi_dot([self.H, P_pre, self.H.T]) + self.R
        K = LA.multi_dot([P_pre, self.H.T, LA.pinv(S)])

        # correct
        state_est = state_est_pre + np.matmul(K, (self.fp2_x_measured - np.matmul(self.H, state_est_pre)))
        self.P_fp2_x = np.matmul((np.eye(3) - np.matmul(K, self.H)), P_pre)
        self.cov_fp2_x = np.array([[self.P_fp2_x[0,0]], [self.P_fp2_x[1,1]], [self.P_fp2_x[1,1]]])

        # extract estimations from state vector
        self.fp2_x = state_est.flatten()[0]
        self.fp2_vx = state_est.flatten()[1]
        self.fp2_ax = state_est.flatten()[2]


    def estimate_fp2_y(self):
        # set R and y appropriate to occlusion state
        if self.fp2_y is None:
            self.R = 10 #100
            self.fp2_y_measured = self.prev_fp2_y
        else:
            self.R = 1 #1
            self.fp2_y_measured = self.fp2_y

        
        # form state vector
        state_est = np.array([[self.prev_fp2_y], [self.prev_fp2_vy], [self.prev_fp2_ay]])

        # predict
        state_est_pre = np.matmul(self.A, state_est)
        P_pre = LA.multi_dot([self.A, self.P_fp2_y, self.A.T]) + self.Q
        S = LA.multi_dot([self.H, P_pre, self.H.T]) + self.R
        K = LA.multi_dot([P_pre, self.H.T, LA.pinv(S)])

        # correct
        state_est = state_est_pre + np.matmul(K, (self.fp2_y_measured - np.matmul(self.H, state_est_pre)))
        self.P_fp2_y = np.matmul((np.eye(3) - np.matmul(K, self.H)), P_pre)
        self.cov_fp2_y = np.array([[self.P_fp2_y[0,0]], [self.P_fp2_y[1,1]], [self.P_fp2_y[1,1]]])

        # extract estimations from state vector
        self.fp2_y = state_est.flatten()[0]
        self.fp2_vy = state_est.flatten()[1]
        self.fp2_ay = state_est.flatten()[2]

        

    def get_estimated_state(self):
        """return estimated state information.

        Returns:
            tuple(float32, float32, float32, float32, float32, float32): (x, y, vx, vy, ax, ay)
        """
        # # return {r_est, theta_est, Vr_est, Vtheta_est, deltab_est, aB_est}
        # if self.ready:
        #     return (self.r, self.theta, self.Vr, self.Vtheta, self.deltaB_est, self.estimated_acceleration)
        # else:
        #     return (self.prev_r, self.prev_theta, self.prev_Vr, self.prev_Vtheta, 0.0, 0.0)
        # return {r_est, theta_est, Vr_est, Vtheta_est, deltab_est, aB_est}
        if self.ready:
            return (self.fp1_x,
                    self.fp1_vx,
                    self.fp1_ax,
                    self.fp1_y,
                    self.fp1_vy,
                    self.fp1_ay,
                    self.fp2_x,
                    self.fp2_vx,
                    self.fp2_ax,
                    self.fp2_y,
                    self.fp2_vy,
                    self.fp2_ay)
        else:
            return (self.prev_fp1_x,
                    self.prev_fp1_vx,
                    self.prev_fp1_ax,
                    self.prev_fp1_y,
                    self.prev_fp1_vy,
                    self.prev_fp1_ay,
                    self.prev_fp2_x,
                    self.prev_fp2_vx,
                    self.prev_fp2_ax,
                    self.prev_fp2_y,
                    self.prev_fp2_vy,
                    self.prev_fp2_ay)