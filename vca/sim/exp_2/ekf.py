import numpy as np
from math import atan2, sin, cos, e, pi, tau

class ExtendedKalman:
    """Implement continuous-continuous EKF for the UAS and Vehicle system in stateful fashion
    """

    def __init__(self, manager):
        self.manager = manager

        self.prev_r = None
        self.prev_theta = None
        self.prev_Vr = None
        self.prev_Vtheta = None
        self.old_x = None
        self.old_y = None
        self.x = None
        self.y = None
        self.alpha = None
        self.a_lat = None
        self.a_long = None
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

    def initialize_filter(self, r, theta, Vr, Vtheta, alpha, a_lat, a_long, x, y, vx, vy):
        """Initializes EKF. Meant to run only once at first.

        Args:
            r (float32): Euclidean distance between vehicle and UAS (m)
            theta (float32): Angle (atan2) of LOS from UAS to vehicle (rad)
            Vr (float32): Relative LOS velocity of vehicle w.r.t UAS (m/s)
            Vtheta (float32): Relative LOS angular velocity of vehicle w.r.t UAS (rad/s)
            alpha (float32): Angle of UAS velocity vector w.r.t to inertial frame
            a_lat (float32): Lateral acceleration control command for UAS
            a_long (float32): Longitudinal acceleration control command for UAS
        """
        self.prev_r = r
        self.prev_theta = theta
        self.prev_Vr = -5.0
        self.prev_Vtheta = -5.0
        self.alpha = alpha
        self.a_lat = a_lat
        self.a_long = a_long
        self.prev_x = x
        self.prev_y = y
        self.prev_vx = vx
        self.prev_vy = vy
        self.prev_ax = 0.0
        self.prev_ay = 0.0
        
        self.filter_initialized_flag = True

    def add(self, r, theta, Vr, Vtheta, alpha, a_lat, a_long, x, y, vx, vy):
        """Add measurements and auxiliary data for filtering

        Args:
            r (float32): Euclidean distance between vehicle and UAS (m)
            theta (float32): Angle (atan2) of LOS from UAS to vehicle (rad)
            Vr (float32): Relative LOS velocity of vehicle w.r.t UAS (m/s)
            Vtheta (float32): Relative LOS angular velocity of vehicle w.r.t UAS (rad/s)
            alpha (float32): Angle of UAS velocity vector w.r.t to inertial frame
            a_lat (float32): Lateral acceleration control command for UAS
            a_long (float32): Longitudinal acceleration control command for UAS
        """
        # make sure filter is initialized
        if not self.is_initialized():
            self.initialize_filter(r, theta, Vr, Vtheta, alpha, a_lat, a_long, x, y, vx, vy)
            return

        # filter is initialized; set ready to true
        self.ready = True

        # handle theta discontinuity
        if (np.sign(self.prev_theta) != np.sign(theta)):
            # print(f'\n---------prev_theta: {self.prev_theta}, theta: {theta}')
            if self.prev_theta > pi/2:
                self.prev_theta -= tau
            if self.prev_theta < -pi/2:
                self.prev_theta += tau
            # print(f'\n---------prev_theta: {self.prev_theta}, theta: {theta}\n')

        # store measurement
        self.r = r
        self.theta = theta
        self.Vr = Vr
        self.Vtheta = Vtheta
        self.alpha = alpha
        self.a_lat = a_lat
        self.a_long = a_long
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = 0.0
        self.ay = 0.0

        # perform predictor and filter step
        self.estimate_acc_x()
        self.estimate_acc_y()
        self.update_state()
        # self.predict()
        # self.correct()

        # remember previous state
        self.prev_r = self.r
        self.prev_theta = self.theta
        self.prev_Vr = self.Vr
        self.prev_Vtheta = self.Vtheta
        self.old_x = self.prev_x
        self.old_y = self.prev_y
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_vx = self.vx
        self.prev_vy = self.vy
        # self.prev_ax = self.ax
        # self.prev_ay = self.ay

    def estimate_acc_x(self):
        # set R and x appropriate to occlusion state
        if self.manager.tracker.is_total_occlusion():
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

        self.x = state_est_acc.flatten()[0]
        self.vx = state_est_acc.flatten()[1]
        self.ax = state_est_acc.flatten()[2]

    def estimate_acc_y(self):
        # set R and x appropriate to occlusion state
        if self.manager.tracker.is_total_occlusion():
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

        self.y = state_est_acc_y.flatten()[0]
        self.vy = state_est_acc_y.flatten()[1]
        self.ay = state_est_acc_y.flatten()[2]


    def update_state(self):
        beta_est = atan2(self.vy, self.vx)
        car_speed_est = (self.vx**2 + self.vy**2)**0.5

        true_kin = self.manager.get_true_kinematics()
        cam_origin_x, cam_origin_y = self.manager.get_cam_origin()
        drone_pos_x, drone_pos_y = true_kin[0]
        drone_vel_x, drone_vel_y = true_kin[1]
        drone_pos_x += cam_origin_x
        drone_pos_y += cam_origin_y
        drone_speed = (drone_vel_x**2 + drone_vel_y**2)**0.5

        self.r = ((drone_pos_x - self.x)**2 + (drone_pos_y - self.y)**2)**0.5
        self.theta = atan2((self.y - drone_pos_y), (self.x - drone_pos_x))
        self.Vr = car_speed_est*cos(beta_est - self.theta) - drone_speed*cos(self.alpha - self.theta)
        self.Vtheta = car_speed_est*sin(beta_est - self.theta) - drone_speed*sin(self.alpha - self.theta)

        self.deltaB_est = atan2(self.ay, self.ax)
        self.estimated_acceleration = (self.ax**2 + self.ay**2)**0.5
        # print(f'\ndeltaB_est: {self.deltaB_est}, est_acc: {self.estimated_acceleration}\n')
        

    def predict(self):
        """Implement continuous-continuous EKF prediction (implicit) step.
        """
        dt = self.manager.get_sim_dt()

        # perform predictor step
        self.A = np.array([[0.0, 0.0, 0.0, 1.0],
                           [-self.prev_Vtheta / self.prev_r**2, 0.0, 1 / self.prev_r, 0.0],
                           [self.prev_Vtheta * self.prev_Vr / self.prev_r**2, 0.0, -self.prev_Vr / self.prev_r, -self.prev_Vtheta / self.prev_r],
                           [-(self.prev_Vtheta / self.prev_r)**2, 0.0, 2 * self.prev_Vtheta / self.prev_r, 0.0]])
  
        self.B1 = np.array([[0.0, 0.0],
                           [0.0, 0.0],
                           [-sin(self.alpha + pi / 2 - self.prev_theta), -sin(self.alpha - self.prev_theta)],
                           [-cos(self.alpha + pi / 2 - self.prev_theta), -cos(self.alpha - self.prev_theta)]])

        self.B2 = np.array([[0.0],
                           [0.0],
                           [sin(self.deltaB_est - self.prev_theta)],
                           [cos(self.deltaB_est - self.prev_theta)]])



    def correct(self):
        """Implement continuous-continuous EKF correction (implicit) step.
        """
        if self.manager.tracker.is_total_occlusion():
            self.R = np.diag([100, 100])
        else:
            self.R = self.R_


        self.Z = np.array([[self.r], [self.theta]])
        self.K = np.matmul(np.matmul(self.P, np.transpose(self.H)), np.linalg.pinv(self.R))

        U = np.array([[self.a_lat], [self.a_long]])
        state = np.array([[self.prev_r], [self.prev_theta], [self.prev_Vtheta], [self.prev_Vr]])
        dyn = np.array([[self.prev_Vr],
                        [self.prev_Vtheta / self.prev_r],
                        [-self.prev_Vtheta * self.prev_Vr / self.prev_r],
                        [self.prev_Vtheta**2 / self.prev_r]])

        state_dot = dyn + np.matmul(self.B1, U) + self.B2 * self.estimated_acceleration + np.matmul(self.K, (self.Z - np.matmul(self.H, state)))
        P_dot = np.matmul(self.A, self.P) + np.matmul(self.P, np.transpose(self.A)) - np.matmul(np.matmul(self.K, self.H), self.P) + self.Q

        dt = self.manager.get_sim_dt()
        state = state + state_dot * dt
        self.P = self.P + P_dot * dt

        self.r = state.flatten()[0]
        self.theta = state.flatten()[1]
        self.Vtheta = state.flatten()[2]
        self.Vr = state.flatten()[3]

    def get_estimated_state(self):
        """Get estimated state information.

        Returns:
            tuple(float32, float32, float, float32): (r, theta, V_r, V_theta)
        """
        if self.ready:
            return (self.r, self.theta, self.Vr, self.Vtheta, self.deltaB_est, self.estimated_acceleration)
        else:
            return (self.prev_r, self.prev_theta, self.prev_Vr, self.prev_Vtheta, 0.0, 0.0)
