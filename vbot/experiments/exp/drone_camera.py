from copy import deepcopy
from math import cos, sin, atan, tau
from collections import namedtuple
import pygame
import numpy as np

from .settings import *
from .my_imports import bf, rb, mb, gb, yb, bb, cb, r, m, g, y, b, c, colored, cprint

class DroneCamera(pygame.sprite.Sprite):
    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.drone_sprites]

        # call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        # set simulator reference
        self.simulator = simulator

        # set image and rect
        self.image, self.rect = simulator.drone_img_rect
        self.image.fill((255, 255, 255, DRONE_IMG_ALPHA), None, pygame.BLEND_RGBA_MULT)

        # set inertia parameters
        self.set_inertia_params()

        # set acceleration due to gravity
        self.g = ACC_GRAVITY

        # set gains
        self.set_gains()

        # reset kinematics and dynamics
        self.reset_kinematics()

        # update sprite rect 
        self.update_rect()

        # set accelerations and velocity clips
        self.vel_limit = DRONE_VELOCITY_LIMIT
        self.acc_limit = DRONE_ACCELERATION_LIMIT

    def set_inertia_params(self):
        """Set inertia params - mass and moments of inertia Ixx, Iyy and Izz (diagonal elements of inertia tensor) - for drone. 
        """
        InertiaParams = namedtuple('InertiaParams', 'm Ixx Iyy Izz')
        self.INERTIA = InertiaParams(DRONE_MASS, DRONE_I_XX, DRONE_I_YY, DRONE_I_ZZ)

    def set_gains(self):
        """set proportional and derivative gains for euler angles roll(φ), pitch(θ), yaw(ψ) and altitude
        """
        Gains = namedtuple('Gains', 'KP_phi KD_phi KP_theta KD_theta KP_psi KD_psi KP_z KD_z')
        self.GAINS = Gains(K_P_PHI, K_D_PHI, K_P_THETA, K_D_THETA, K_P_PSI, K_D_PSI, K_P_Z, K_D_Z)

    def reset_kinematics(self):
        """helper function to reset initial kinematic states
        """
        # linear position and velocity
        self.position = pygame.Vector2(DRONE_POSITION)
        self.altitude = ALTITUDE
        self.velocity = pygame.Vector2(DRONE_INITIAL_VELOCITY)
        self.vz = 0.0

        # angular position and velocity
        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        self.p = 0.0
        self.q = 0.0
        self.r = 0.0

        # set initial acceleration
        self.acceleration = pygame.Vector2(0, 0)
        self.w = 0.0
        self.az = 0.0

        # set camera origin 
        self.origin = self.position
        self.origin_z = ALTITUDE
        self.prev_origin = self.position
        self.prev_origin_z = ALTITUDE

        self.delta_pos = pygame.Vector2(0.0,0.0)
        self.delta_pos_z = 0.0


    def get_quadrotor_state(self):
        # construct current state
        state = np.array([self.position[0],       # 0  (pN) inertial (north) position of quadrotor along i_W in world inertial frame W
                          self.position[1],       # 1  (pE) inertial (east) position of quadrotor along j_W in world inertial frame W
                          self.altitude,          # 2  (pH) altitude of quadrotor measured along k_W in world inertial frame W
                          self.velocity[0],       # 3  (u) body frame velocity measured along i_A in body frame A
                          self.velocity[1],       # 4  (v) body frame velocity measured along j_A in body frame A
                          self.vz,                # 5  (w) body frame velocity measured along k_A in body frame A
                          self.phi,               # 6  (φ) roll angle defined with respect to frame N2 (local NED) about i_N2 axis (N2 → A)
                          self.theta,             # 7  (θ) pitch angle defined with respect to frame N1 (local NED) about j_N1 axis (N1 → N2)
                          self.psi,               # 8  (ψ) yaw angle defined with respect to frame N (local NED) about k_N axis (N → N1)
                          self.p,                 # 9  (p) roll rate measured along i_A in body frame A
                          self.q,                 # 10 (q) pitch rate measured along j_A in body frame A
                          self.r])                # 11 (r) yaw rate measured along k_A in body frame A

        return state

    @staticmethod
    def norm_ang(ang):
        return np.sign(ang) * (abs(ang)%tau)


    def update_kinematics(self):
        """helper function to update kinematics of object subject to dynamics
        """
        OLD_STUFFS = 0

        if OLD_STUFFS:
            print(f'{self.acceleration[0]:.2f}, {self.acceleration[1]:.2f}, {self.az:.2f}')
            self.prev_delta_pos = deepcopy(self.delta_pos)
            self.prev_origin = deepcopy(self.origin)

            # update velocity (clip if over limit)
            self.velocity += self.acceleration * self.simulator.dt
            self.vz += self.az * self.simulator.dt
            if abs(self.velocity.length()) > self.vel_limit:
                self.velocity -= self.acceleration * self.simulator.dt
                self.vz -= self.az * self.simulator.dt

            # update position
            self.delta_pos = self.velocity * self.simulator.dt 
            self.position = self.velocity * self.simulator.dt 
            self.altitude += self.vz * self.simulator.dt 
            
            # update camera origin (world frame)
            self.origin += self.delta_pos
        else:
            # new stuffs
            self.prev_delta_pos = deepcopy(self.delta_pos)
            self.prev_origin = deepcopy(self.origin)

            # compute required force
            F = (self.g - self.az) * (self.INERTIA.m / (cos(self.phi)*cos(self.theta)))

            # compute commanded attitude using commanded accelerations
            phi_c = atan(self.acceleration[1] * cos(self.theta)/(self.g - self.az))
            theta_c = atan(self.acceleration[0] / (self.az - self.g))
            psi_c = 0

            # compute required torque
            tau_phi = self.GAINS.KP_phi*self.norm_ang(phi_c - self.phi) + self.GAINS.KD_phi*(0-self.p)
            tau_theta = self.GAINS.KP_theta*self.norm_ang(theta_c - self.theta) + self.GAINS.KD_theta*(0-self.q)
            tau_psi = self.GAINS.KP_psi*self.norm_ang(psi_c - self.psi) + self.GAINS.KD_psi*(0-self.r)

            # update state
            num_inner_loop = 10
            outer_loop_rate = 1/DELTA_TIME
            inner_loop_rate = outer_loop_rate * num_inner_loop

            for i in range(num_inner_loop):
                # collect state
                state = self.get_quadrotor_state()

                # compute total thrust 
                F_app = F

                # evaluate dynamics and compute state dot
                state_dot = self.get_quadrotor_state_dot(state, F_app, tau_phi, tau_theta, tau_psi)

                # euler integration
                state += state_dot/inner_loop_rate

                # # runge-kutta
                # k1 = self.get_quadrotor_state_dot(state, F_app, tau_phi, tau_theta, tau_psi)
                # k2 = self.get_quadrotor_state_dot(state + k1*(1/(2*inner_loop_rate)), F_app, tau_phi, tau_theta, tau_psi)
                # k3 = self.get_quadrotor_state_dot(state + k2*(1/(2*inner_loop_rate)), F_app, tau_phi, tau_theta, tau_psi)
                # k4 = self.get_quadrotor_state_dot(state + k3*(1/(inner_loop_rate)), F_app, tau_phi, tau_theta, tau_psi)

                # state += (k1 + 2*k2 + 2*k3 + k4) * (1/(6*inner_loop_rate))

                # update state
                self.update_quadrotor_state(state)

            self.print_states()
            print(f'            {g("comm_sig: F=")}{gb(f"{F:.4f}")}{g(", τφ=")}{gb(f"{tau_phi:.4f}")}{g(", τθ=")}{gb(f"{tau_theta:.4f}")}{g(", τψ=")}{gb(f"{tau_psi:.4f}")}')

            # # construct Rotation matrix from Drone local NED to Drone body attached frame
            # N_R_A = np.array([[cos(self.theta)*cos(self.psi),                                           cos(self.theta)*sin(self.psi),                                           -sin(self.theta)],
            #                   [sin(self.phi)*sin(self.theta)*cos(self.psi)-cos(self.phi)*sin(self.psi), sin(self.phi)*sin(self.theta)*sin(self.psi)+cos(self.phi)*cos(self.psi), sin(self.phi)*cos(self.theta)],
            #                   [cos(self.phi)*sin(self.theta)*cos(self.psi)+sin(self.phi)*sin(self.psi), cos(self.phi)*sin(self.theta)*sin(self.psi)-sin(self.phi)*cos(self.psi), cos(self.phi)*cos(self.theta)]])

            # # compute transformed velocity in inertial world frame
            # vel_inertial = N_R_A.T @ np.array([[self.velocity[0]], 
            #                                     [self.velocity[1]],
            #                                     [self.vz]])

            # # update velocity
            # self.velocity = pygame.Vector2(vel_inertial.flatten()[0], vel_inertial.flatten()[1])
            # self.vz = vel_inertial.flatten()[2]

            # save camera origin
            self.origin += self.position
            self.origin_z = self.altitude


    def get_quadrotor_state_dot(self, state, F, tau_phi, tau_theta, tau_psi):
        """Given the state and actuation wrench, computes and returns state_dot

        Args:
            state (tuple): Tuple with 12 components of quadrotor state
            F (float): Thrust force along k_A in body frame A
            tau_phi (float): Rolling torque
            tau_theta (float): Pitching torque
            tau_psi (float): Yawing torque

        Returns:
            tuple: Tuple of with 12 components of quadrotor state dynamics
        """
        # evaluate dynamics
        dpN,  dpE,    dpH   = self.evaluate_position_dynamics(phi=state[6], theta=state[7], psi=state[8], u=state[3], v=state[4], w=state[5])
        du,   dv,     dw    = self.evaluate_velocity_dynamics(u=state[3], v=state[4], w=state[5], p=state[9], q=state[10], r=state[11], phi=state[6], theta=state[7], F=F)
        dphi, dtheta, dpsi  = self.evaluate_attitude_dynamics(p=state[9], q=state[10], r=state[11], phi=state[6], theta=state[7])
        dp,   dq,     dr    = self.evaluate_angular_rate_dynamics(tau_phi, tau_theta, tau_psi, p=state[9], q=state[10], r=state[11])

        # collect state dot
        state_dot = np.array([dpN,
                                dpE,
                                dpH,
                                du,
                                dv,
                                dw,
                                dphi,
                                dtheta,
                                dpsi,
                                dp,
                                dq,
                                dr])

        return state_dot


    def update_quadrotor_state(self, state):
        self.position = pygame.Vector2(state[0], state[1])
        self.altitude = state[2]
        self.velocity = pygame.Vector2(state[3], state[4])
        self.vz = state[5]
        self.phi = state[6]
        self.theta = state[7]
        self.psi = state[8]
        self.p = state[9]
        self.q = state[10]
        self.r = state[11]


    def evaluate_position_dynamics(self, phi, theta, psi, u, v, w):
        """Evaluates dynamics of quadrotor inertial (local NED) position components in inertial world frame

        Args:
            φ (float): Roll angle with respect to frame N2 (local NEU) about i_N2 axis (N2 → A)
            θ (float): Pitch angle with respect to frame N1 (local NEU) about j_N1 axis (N1 → N2 )
            ψ (float): Yaw angle with respect to frame N (local NEU) about k_N axis (N → N1)
            u (float): velocity along i_A in body frame A
            v (float): velocity along j_A in body frame A
            w (float): velocity along k_A in body frame A

        Returns:
            (1D np.ndarray): pN_dot, pE_dot, h_dot
        """
        # construct rotation matrix transforming body frame (A) to local NED (N)
        A_R_N = np.array([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],
                          [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)],
                          [-sin(theta),         sin(phi)*cos(theta),                            cos(phi)*cos(theta)]])
        
        # transform and compute velocities of inertial frame quantities pN, pE and h
        vel_NEU = A_R_N @ np.array([[u],
                                    [v],
                                    [w]])

        # return pN_dot, pE_dot, h_dot
        return vel_NEU.flatten()


    def evaluate_velocity_dynamics(self, u, v, w, p, q, r, phi, theta, F):
        """Evaluate dynamics of quadrotor body frame velocities u, v, w (body frame A)

        Args:
            u (float): velocity along i_A in body frame A
            v (float): velocity along j_A in body frame A
            w (float): velocity along k_A in body frame A
            p (float): Roll rate measured along i_A in body frame A
            q (float): Pitch rate measured along j_A in body frame A
            r (float): Yaw rate measured along k_A in body frame A
            φ (float): Roll angle with respect to frame N2 (local NEU) about i_N2 axis (N2 → A)
            θ (float): Pitch angle with respect to frame N1 (local NEU) about j_N1 axis (N1 → N2 )
            F (float): Total thrust produced by four motors 

        Returns:
            (1D np.ndarray): u_dot, v_dot, w_dot
        """
        # computer velocity rates
        vel_dot = np.array([[r*v - q*w],
                            [p*w - r*u],
                            [q*u - p*v]]) \
                + np.array([[-self.g*sin(theta)],
                            [self.g*cos(theta)*sin(phi)],
                            [self.g*cos(theta)*cos(phi)]]) \
                + np.array([[0],
                            [0],
                            [-F/self.INERTIA.m]])
        
        # return u_dot, v_dot, w_dot
        return vel_dot.flatten()
        
    def evaluate_attitude_dynamics(self, p, q, r, phi, theta):
        """Evaluate dynamics of quadrotor attitude 

        Args:
            p (float): Roll rate measured along i_A in body frame A
            q (float): Pitch rate measured along j_A in body frame A
            r (float): Yaw rate measured along k_A in body frame A
            φ (float): Roll angle with respect to frame N2 (local NEU) about i_N2 axis (N2 → A)
            θ (float): Pitch angle with respect to frame N1 (local NEU) about j_N1 axis (N1 → N2 )

        Returns:
            (1D np.ndarray): phi_dot, theta_dot, psi_dot
        """
        # compute the transform matrix
        transform_mat = np.array([[1, sin(phi)*tan(theta), cos(phi)*tan(theta)],
                                  [0, cos(phi),            -sin(phi)],
                                  [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]])

        # compute attitude rate dynamics
        attitude_dot = transform_mat @ np.array([[p],
                                                 [q],
                                                 [r]])
        # return phi_dot, theta_dot, psi_dot
        return attitude_dot.flatten()

    def evaluate_angular_rate_dynamics(self, tau_phi, tau_theta, tau_psi, p, q, r):
        """Evaluate dynamics of quadrotor angular rates (body frame A)

        Args:
            tau_phi (float): Rolling torque
            tau_theta (float): Pitching torque
            tau_psi (float): Yawing torque
            p (float): Roll rate measured along i_A in body frame A
            q (float): Pitch rate measured along j_A in body frame A
            r (float): Yaw rate measured along k_A in body frame A

        Returns:
            (1D np.ndarray): p_dot, q_dot, r_dot
        """
        # compute coriolis term
        coriolis = np.array([[((self.INERTIA.Iyy - self.INERTIA.Izz)/self.INERTIA.Ixx)*q*r],
                             [((self.INERTIA.Izz - self.INERTIA.Ixx)/self.INERTIA.Iyy)*p*r],
                             [((self.INERTIA.Ixx - self.INERTIA.Iyy)/self.INERTIA.Izz)*p*q]])

        # compute F/m analog in Euler equations - τ/I
        euler_eom = np.array([[(1/self.INERTIA.Ixx)*tau_phi],
                                  [(1/self.INERTIA.Iyy)*tau_theta],
                                  [(1/self.INERTIA.Izz)*tau_psi]])

        # compute angular velocity rate dynamics
        ang_vel_dot = euler_eom + coriolis

        # return p_dot, q_dot, r_dot
        return ang_vel_dot.flatten()


    def print_states(self):
        """helper function to print states"""
        print(f'{cb(f"{self.simulator.time:.2f}")}{c(" secs: ")}' + 
              f'{y("cam_origin=")}' + 
              f'{yb(f"({self.origin[0]:.1f}, {self.origin[1]:.1f})  ")}' + 
              f'{y("pos=")}' + 
              f'{yb(f"({self.position[0]:.1f}, {self.position[1]:.1f}, {self.altitude:.1f})  ")}' +  
              f'{y("vel=")}' + 
              f'{yb(f"({self.velocity[0]:.1f}, {self.velocity[1]:.1f}, {self.vz:.1f})  ")}' +  
              f'{y("ang_pos=")}' + 
              f'{yb(f"({self.phi:.1f}, {self.theta:.1f}, {self.psi:.1f})  ")}' + 
              f'{y("ang_vel=")}' + 
              f'{yb(f"({self.p:.1f}, {self.q:.1f}, {self.r:.1f})  ")}' + 
              f'{y("acc_command=")}' + 
              f'{yb(f"({self.acceleration[0]:.2f}, {self.acceleration[1]:.2f}, {self.az:.2f})")}'
            )



    def update(self):
        """helper function update kinematics
        """
        self.update_kinematics()
        # self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER

    def update_rect(self):
        """update drone sprite's rect.
        """
        # position is in meters, convert it to pixels
        # and flip y axis
        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac

        # set rect, convert to integer, 
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT

        # translate center to screen center
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)


    def compensate_camera_motion(self, sprite_obj):
        """Compensates camera motion by updating position of sprite object.

        Args:
            sprite_obj (pygame.sprite.Sprite): Sprite object whose motion needs compensation.
        """
        sprite_obj.position -= self.position
        sprite_obj.update_rect()


    def apply_accleration_command(self, ax, ay, az=0):
        self.acceleration = pygame.Vector2((ax, ay))
        # self.az = self.GAINS.KP_z* (100 - self.altitude) + self.GAINS.KD_z * (0 - self.vz)
        self.az = az


    def convert_px_to_m(self, p):
        """Convert pixels to meters

        Args:
            p (float): Value in pixel units

        Returns:
            float: Value in SI units
        """
        return p * ((self.altitude * PIXEL_SIZE) / FOCAL_LENGTH)

    def convert_m_to_px(self, x):
        """Convert meters to pixel units

        Args:
            x (float): Value in SI units

        Returns:
            float: Value in pixels
        """
        return x / ((self.altitude * PIXEL_SIZE) / FOCAL_LENGTH)

    def fly_higher(self):
        """Helper function to implement drone raise altitude. Updates altitude and its change factor.
        """
        self.simulator.alt_change_fac = 1.0 + self.alt_change / self.altitude
        self.altitude += self.alt_change

    def fly_lower(self):
        """Helper function to implement drone lower altitude. Updates altitude and its change factor.
        """
        self.simulator.alt_change_fac = 1.0 - self.alt_change / self.altitude
        self.altitude -= self.alt_change
