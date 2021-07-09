import numpy as np
from math import atan2, sin, cos, e, pi, tau

class ExtendedKalman:
    """Implement continuous-continuous EKF for the UAS and Vehicle system in stateful fashion
    """

    def __init__(self, manager, target=None):
        self.manager = manager
        self.target = target

        # self.prev_r = None
        # self.prev_theta = None
        # self.prev_Vr = None
        # self.prev_Vtheta = None
        self.old_x = None
        self.old_y = None
        self.x = None
        self.y = None
        self.drone_alpha = None
        self.filter_initialized_flag = False

        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])

        self.P = np.diag([0.0, 0.0, 0.0, 0.0])
        self.R_ = np.diag([1, 0.1])
        self.Q = np.diag([0.001, 0.001, 1, 1])

        self.P_acc = np.diag([0.0, 0.0, 0.0])
        self.P_acc_y = np.diag([0.0, 0.0, 0.0])
        self.cov_acc = np.array([[self.P_acc[0,0]], [self.P_acc[1,1]], [self.P_acc[2,2]]])
        self.cov_acc_y = np.array([[self.P_acc_y[0,0]], [self.P_acc_y[1,1]], [self.P_acc_y[2,2]]])
        self.alpha_acc = 0.1    # reciprocal of maneuver(acceleration) time constant. 1/60-lazy turn, 1/20-evasive,  1-atmospheric turbulence
        self.sigma_square_x = 0.1 
        self.sigma_square_y = 0.05

        self.ready = False

    def is_initialized(self):
        """Indicates if EKF is initialized

        Returns:
            bool: EKF initalized or not
        """
        return self.filter_initialized_flag

    def initialize_filter(self, x, y, vx, vy):
        """Initializes EKF. Meant to run only once at first.

        Args:
            r (float32): Euclidean distance between vehicle and UAS (m)
            theta (float32): Angle (atan2) of LOS from UAS to vehicle (rad)
            Vr (float32): Relative LOS velocity of vehicle w.r.t UAS (m/s)
            Vtheta (float32): Relative LOS angular velocity of vehicle w.r.t UAS (rad/s)
            alpha (float32): Angle of UAS velocity vector w.r.t to inertial frame
        """
        # self.prev_r = r
        # self.prev_theta = theta
        # self.prev_Vr = Vr
        # self.prev_Vtheta = Vtheta
        # self.drone_alpha = drone_alpha
        self.prev_x = x
        self.prev_y = y
        self.prev_vx = vx
        self.prev_vy = vy
        self.prev_ax = 0.0
        self.prev_ay = 0.0
        
        self.filter_initialized_flag = True

    def add(self, x, y):
        """Add measurements and auxiliary data for filtering

        Args:
            r (float32): Euclidean distance between vehicle and UAS (m)
            theta (float32): Angle (atan2) of LOS from UAS to vehicle (rad)
            alpha (float32): Angle of UAS velocity vector w.r.t to inertial frame
        """
        # make sure filter is initialized
        if not self.is_initialized():
            vx = self.target.sprite_obj.velocity[0]
            vy = self.target.sprite_obj.velocity[1]
            self.initialize_filter(x, y, vx, vy)
            return

        # filter is initialized; set ready to true
        self.ready = True

        # # handle theta discontinuity
        # if (np.sign(self.prev_theta) != np.sign(theta)):
        #     # print(f'\n---------prev_theta: {self.prev_theta}, theta: {theta}')
        #     if self.prev_theta > pi/2:
        #         self.prev_theta -= tau
        #     if self.prev_theta < -pi/2:
        #         self.prev_theta += tau
        #     # print(f'\n---------prev_theta: {self.prev_theta}, theta: {theta}\n')

        # store measurement
        # self.r = r
        # self.theta = theta
        # self.drone_alpha = drone_alpha
        self.x = x
        self.y = y
        # self.vx = vx
        # self.vy = vy
        # self.ax = 0.0
        # self.ay = 0.0

        # perform predictor and filter step
        self.estimate_acc_x()
        self.estimate_acc_y()
        # self.update_state()

        # remember state estimations
        # self.prev_r = self.r
        # self.prev_theta = self.theta
        # self.prev_Vr = self.Vr
        # self.prev_Vtheta = self.Vtheta
        self.old_x = self.prev_x
        self.old_y = self.prev_y
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_vx = self.vx
        self.prev_vy = self.vy


    def estimate_acc_x(self):
        # set R and x appropriate to occlusion state
        if self.x is None:
            self.R_acc = 10 #100
            self.x_measured = self.prev_x
        else:
            self.R_acc = 1 #1
            self.x_measured = self.x

        dt = self.manager.get_sim_dt()
        adt = self.alpha_acc * dt   # αΔt
        eadt = e**(-adt)
        e2adt = e**(-2*adt)
        self.A_acc = np.array([[1.0, dt, (eadt + adt -1)/(self.alpha_acc**2)],
                               [0.0, 1.0, (1 - eadt)/(self.alpha_acc)],
                               [0.0, 0.0, eadt]])

        self.q11 = (1 - e2adt + 2*adt + (2/3)*adt**3 - 2*adt**2 - 4*adt*eadt) / (self.alpha_acc**4)
        self.q12 = (e2adt + 1 - 2*eadt + 2*adt*eadt - 2*adt + adt**2) / (self.alpha_acc**3)
        self.q13 = (1 - e2adt - 2*adt*eadt) / (self.alpha_acc**2)
        self.q22 = (4*eadt - 3 - e2adt + 2*adt) / (self.alpha_acc**2)
        self.q23 = (e2adt + 1 -2*eadt) / (self.alpha_acc)
        self.q33 = (1 - e2adt)

        self.Q_acc = self.sigma_square_x * np.array([[self.q11, self.q12, self.q13],
                                         [self.q12, self.q22, self.q23],
                                         [self.q13, self.q23, self.q33]])

        H_acc = np.array([[1.0, 0.0, 0.0]])
        state_est_acc = np.array([[self.prev_x], [self.prev_vx], [self.prev_ax]])
        state_est_pre_acc = np.matmul(self.A_acc, state_est_acc)
        P_pre_acc = np.matmul(np.matmul(self.A_acc, self.P_acc), np.transpose(self.A_acc)) + self.Q_acc
        S_acc = np.matmul(np.matmul(H_acc, P_pre_acc), np.transpose(H_acc)) + self.R_acc
        K_acc = np.matmul(np.matmul(P_pre_acc, np.transpose(H_acc)), np.linalg.pinv(S_acc))

        state_est_acc = state_est_pre_acc + np.matmul(K_acc, (self.x_measured - np.matmul(H_acc,state_est_pre_acc)))
        self.P_acc = np.matmul((np.eye(3) - np.matmul(K_acc,H_acc)), P_pre_acc)
        self.cov_acc = np.array([[self.P_acc[0,0]], [self.P_acc[1,1]], [self.P_acc[1,1]]])

        # collect estimations
        self.x = state_est_acc.flatten()[0]
        self.vx = state_est_acc.flatten()[1]
        self.ax = state_est_acc.flatten()[2]


    def estimate_acc_y(self):
        # set R and y appropriate to occlusion state
        if self.y is None:
            self.R_acc_y = 10 #1000
            self.y_measured = self.prev_y
        else:
            self.R_acc_y = 1 #10
            self.y_measured = self.y

        self.Q_acc_y = self.sigma_square_y * np.array([[self.q11, self.q12, self.q13],
                                         [self.q12, self.q22, self.q23],
                                         [self.q13, self.q23, self.q33]])

        H_acc = np.array([[1.0, 0.0, 0.0]])
        state_est_acc_y = np.array([[self.prev_y], [self.prev_vy], [self.prev_ay]])
        state_est_pre_acc_y = np.matmul(self.A_acc, state_est_acc_y)
        P_pre_acc_y = np.matmul(np.matmul(self.A_acc, self.P_acc_y), np.transpose(self.A_acc)) + self.Q_acc_y
        S_acc_y = np.matmul(np.matmul(H_acc, P_pre_acc_y), np.transpose(H_acc)) + self.R_acc_y
        K_acc_y = np.matmul(np.matmul(P_pre_acc_y, np.transpose(H_acc)), np.linalg.pinv(S_acc_y))

        state_est_acc_y = state_est_pre_acc_y + np.matmul(K_acc_y, (self.y_measured - np.matmul(H_acc,state_est_pre_acc_y)))
        self.P_acc_y = np.matmul((np.eye(3) - np.matmul(K_acc_y,H_acc)), P_pre_acc_y)
        self.cov_acc = np.array([[self.P_acc_y[0,0]], [self.P_acc_y[1,1]], [self.P_acc_y[1,1]]])

        # collect estimations
        self.y = state_est_acc_y.flatten()[0]
        self.vy = state_est_acc_y.flatten()[1]
        self.ay = state_est_acc_y.flatten()[2]
        

    def get_estimated_state(self):
        """return estimated state information.

        Returns:
            tuple(float32, float32, float, float32): (r, theta, V_r, V_theta)
        """
        # # return {r_est, theta_est, Vr_est, Vtheta_est, deltab_est, aB_est}
        # if self.ready:
        #     return (self.r, self.theta, self.Vr, self.Vtheta, self.deltaB_est, self.estimated_acceleration)
        # else:
        #     return (self.prev_r, self.prev_theta, self.prev_Vr, self.prev_Vtheta, 0.0, 0.0)
        # return {r_est, theta_est, Vr_est, Vtheta_est, deltab_est, aB_est}
        if self.ready:
            return (self.x, self.vx, self.ax, self.y, self.vy, self.ay)
        else:
            return (self.prev_x, self.prev_vx, self.prev_ax, self.prev_y, self.prev_vy, self.prev_ay)
