import numpy as np

import pygame
from pygame.locals import *

class Kalman:
    """Implements Discrete-time Kalman filtering in a stateful fashion
    """

    def __init__(self, manager):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.sig = 0.1
        self.sig_r = 0.1
        self.sig_q = 1.0
        self.manager = manager

        # process noise
        self.Er = np.array([[0.01], [0.01], [0.01], [0.01]])

        # measurement noise
        self.Eq = np.array([[0.01], [0.01], [0.01], [0.01]])

        # initialize belief state and covariance
        self.Mu = np.array([[self.x], [self.y], [self.vx], [self.vy]])
        self.var_S = np.array([10**-4, 10**-4, 10**-4, 10**-4])
        self.S = np.diag(self.var_S.flatten())

        # noiseless connection between state vector and measurement vector
        self.C = np.identity(4)

        # covariance of process noise model
        self.var_R = np.array([10**-6, 10**-6, 10**-5, 10**-5])
        self.R = np.diag(self.var_R.flatten())

        # covariance of measurement noise model
        self.var_Q = np.array([0.0156 * 10**-3, 0.0155 * 10**-3, 7.3811 * 10**-3, 6.5040 * 10**-3])
        self.Q = np.diag(self.var_Q.flatten())

        self.ready = False

    def done_waiting(self):
        """Indicates filter readiness

        Returns:
            bool: Ready or not
        """
        return self.ready

    def init_filter(self, pos, vel):
        """Initializes filter. Meant to be run only at first.

        Args:
            pos (pygame.Vector2): Car position measurement
            vel (pygame.Vector2): Car velocity measurement
        """
        self.x = pos[0]
        self.y = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.X = np.array([[self.x], [self.y], [self.vx], [self.vy]])
        self.Mu = self.X
        self.ready = True

    def add(self, pos, vel):
        """Add a measurement.

        Args:
            pos (pygame.Vector2): Car position measurement
            vel (pygame.Vector2): Car velocity measurement
        """
        # pos and vel are the measured values. (remember x_bar)
        self.x = pos[0]
        self.y = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.X = np.array([[self.x], [self.y], [self.vx], [self.vy]])

        self.predict()
        self.correct()

    def predict(self):
        """Implement discrete-time Kalman filter prediction/forecast step
        """
        # collect params
        dt = self.manager.get_sim_dt()
        dt2 = dt**2
        # motion model
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # control model
        B = np.array([[0.5 * dt2, 0], [0, 0.5 * dt2], [dt, 0], [0, dt]])
        # B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

        # process noise covariance
        R = self.R

        command = self.manager.simulator.camera.acceleration
        U = np.array([[command[0]], [command[1]]])

        # predict
        self.Mu = np.matmul(A, self.Mu) + np.matmul(B, U)
        self.S = np.matmul(np.matmul(A, self.S), np.transpose(A)) + R

    def correct(self):
        """Implement discrete-time Kalman filter correction/update step
        """
        Z = self.X
        K = np.matmul(
            np.matmul(
                self.S, self.C), np.linalg.pinv(
                np.matmul(
                    np.matmul(
                        self.C, self.S), np.transpose(
                        self.C)) + self.Q))

        self.Mu = self.Mu + np.matmul(K, (Z - np.matmul(self.C, self.Mu)))
        self.S = np.matmul((np.identity(4) - np.matmul(K, self.C)), self.S)

    def add_pos(self, pos):
        """Add position measurement

        Args:
            pos (pygame.Vector2): Car position measurement
        """
        self.add(pos, (self.vx, self.vy))

    def add_vel(self, vel):
        """Add velocity measurement

        Args:
            vel (pygame.Vector2): Car velocity measurement
        """
        self.add((self.x, self.y), vel)

    def get_pos(self):
        """Get estimated car position

        Returns:
            pygame.Vector2: Car estimated position
        """
        return pygame.Vector2(self.Mu.flatten()[0], self.Mu.flatten()[1])

    def get_vel(self):
        """Get estimated car velocity

        Returns:
            pygame.Vector2: Car estimated velocity
        """
        return pygame.Vector2(self.Mu.flatten()[2], self.Mu.flatten()[3])
