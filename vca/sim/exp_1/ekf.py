class ExtendedKalman:
    """Implement continuous-continuous EKF for the UAS and Vehicle system in stateful fashion
    """

    def __init__(self, manager):
        self.manager = manager

        self.prev_r = None
        self.prev_theta = None
        self.prev_Vr = None
        self.prev_Vtheta = None
        self.alpha = None
        self.a_lat = None
        self.a_long = None
        self.filter_initialized_flag = False

        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])

        self.P = np.diag([0.1, 0.1, 0.1, 0.1])
        self.R = np.diag([0.1, 0.1])
        self.Q = np.diag([0.1, 0.1, 1, 0.1])

        self.ready = False

    def is_initialized(self):
        """Indicates if EKF is initialized

        Returns:
            bool: EKF initalized or not
        """
        return self.filter_initialized_flag

    def initialize_filter(self, r, theta, Vr, Vtheta, alpha, a_lat, a_long):
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
        self.prev_Vr = -5
        self.prev_Vtheta = 5
        self.alpha = alpha
        self.a_lat = a_lat
        self.a_long = a_long
        self.filter_initialized_flag = True

    def add(self, r, theta, Vr, Vtheta, alpha, a_lat, a_long):
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
            self.initialize_filter(r, theta, Vr, Vtheta, alpha, a_lat, a_long)
            return

        # filter is initialized; set ready to true
        self.ready = True

        if (np.sign(self.prev_theta) != np.sign(theta)):
            self.prev_theta = theta

        # store measurement
        self.r = r
        self.theta = theta
        self.Vr = Vr
        self.Vtheta = Vtheta
        self.alpha = alpha
        self.a_lat = a_lat
        self.a_long = a_long

        # perform predictor and filter step
        self.predict()
        self.correct()

        # remember previous state
        self.prev_r = self.r
        self.prev_theta = self.theta
        self.prev_Vr = self.Vr
        self.prev_Vtheta = self.Vtheta

    def predict(self):
        """Implement continuous-continuous EKF prediction (implicit) step.
        """
        # perform predictor step
        self.A = np.array([[0.0, 0.0, 0.0, 1.0],
                           [-self.prev_Vtheta / self.prev_r**2, 0.0, 1 / self.prev_r, 0.0],
                           [self.prev_Vtheta * self.prev_Vr / self.prev_r**2, 0.0, -self.prev_Vr / self.prev_r, -self.prev_Vtheta / self.prev_r],
                           [-self.prev_Vtheta**2 / self.prev_r**2, 0.0, 2 * self.prev_Vtheta / self.prev_r, 0.0]])

        self.B = np.array([[0.0, 0.0],
                           [0.0, 0.0],
                           [-sin(self.alpha + pi / 2 - self.prev_theta), -sin(self.alpha - self.prev_theta)],
                           [-cos(self.alpha + pi / 2 - self.prev_theta), -cos(self.alpha - self.prev_theta)]])

    def correct(self):
        """Implement continuous-continuous EKF correction (implicit) step.
        """
        self.Z = np.array([[self.r], [self.theta]])
        self.K = np.matmul(np.matmul(self.P, np.transpose(self.H)), np.linalg.pinv(self.R))

        U = np.array([[self.a_lat], [self.a_long]])
        state = np.array([[self.prev_r], [self.prev_theta], [self.prev_Vtheta], [self.prev_Vr]])
        dyn = np.array([[self.prev_Vr],
                        [self.prev_Vtheta / self.prev_r],
                        [-self.prev_Vtheta * self.prev_Vr / self.prev_r],
                        [self.prev_Vtheta**2 / self.prev_r]])

        state_dot = dyn + np.matmul(self.B, U) + np.matmul(self.K,
                                                           (self.Z - np.matmul(self.H, state)))
        P_dot = np.matmul(self.A, self.P) + np.matmul(self.P, np.transpose(self.A)
                                                      ) - np.matmul(np.matmul(self.K, self.H), self.P) + self.Q

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
            return (self.r, self.theta, self.Vr, self.Vtheta)
        else:
            return (self.prev_r, self.prev_theta, self.prev_Vr, self.prev_Vtheta)

