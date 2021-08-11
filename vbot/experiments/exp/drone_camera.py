from copy import deepcopy
from math import cos, sin, atan
import pygame
import numpy as np

from .settings import *

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
        self.m = DRONE_MASS
        self.I_XX = DRONE_I_XX
        self.I_YY = DRONE_I_YY
        self.I_ZZ = DRONE_I_ZZ

        # set acceleration due to gravity
        self.g = ACC_GRAVITY

        # set gains
        self.gains = {
            'K_P_THETA': K_P_THETA,
            'K_D_THETA': K_D_THETA,
            'K_P_PHI': K_P_PHI,
            'K_D_PHI': K_D_PHI,
            'K_P_PSI': K_P_PSI,
            'K_D_PSI': K_D_PSI
            }

        # reset kinematics and dynamics
        self.reset_kinematics()

        # update sprite rect 
        self.update_rect()

        # set accelerations and velocity clips
        self.vel_limit = DRONE_VELOCITY_LIMIT
        self.acc_limit = DRONE_ACCELERATION_LIMIT

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
        self.P = 0.0
        self.Q = 0.0
        self.R = 0.0

        # set initial acceleration
        self.acceleration = pygame.Vector2(0, 0)
        self.az = 0.0

        # set camera origin 
        self.origin = self.position
        self.origin_z = ALTITUDE
        self.prev_origin = self.position
        self.prev_origin_z = ALTITUDE

        self.delta_pos = pygame.Vector2(0.0,0.0)
        self.delta_pos_z = 0.0


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
            if abs(self.velocity.length()) > self.vel_limit:
                self.velocity -= self.acceleration * self.simulator.dt

            # update position
            self.delta_pos = self.velocity * self.simulator.dt 
            self.position = self.velocity * self.simulator.dt 
            
            # update camera origin (world frame)
            self.origin += self.delta_pos
        else:
            # new stuffs
            # self.acceleration = pygame.Vector2(0.02, 0.0)
            # self.az = 0.0
            self.prev_delta_pos = deepcopy(self.delta_pos)
            self.prev_origin = deepcopy(self.origin)

            # compute force
            F = self.m * (self.g - self.az) / (cos(self.phi)*cos(self.theta))

            # compute desired attitude
            phi_des = atan(self.acceleration[1] * cos(self.theta)/(self.g - self.az))
            theta_des = atan(self.acceleration[0] / (self.az - self.g))
            psi_des = 0

            # compute required
            tau_theta = self.gains['K_P_THETA']*(theta_des - self.theta) + self.gains['K_D_THETA']*(0-self.Q)
            tau_phi = self.gains['K_P_PHI']*(phi_des - self.phi) + self.gains['K_D_PHI']*(0-self.P)
            tau_psi = self.gains['K_P_PSI']*(psi_des - self.psi) + self.gains['K_D_PSI']*(0-self.R)

            # update state
            num_inner_loop = 10
            outer_loop_rate = 1/DELTA_TIME
            inner_loop_rate = outer_loop_rate * num_inner_loop

            for i in range(num_inner_loop):
                pN = self.position[0]
                pE = self.position[1]
                pH = self.altitude
                U = self.velocity[0]
                V = self.velocity[1]
                W = self.vz
                phi = self.phi
                theta = self.theta
                psi = self.psi
                P = self.P
                Q = self.Q
                R = self.R

                F_app = F * ((i+1) / num_inner_loop)**-2

                dpN = U*cos(theta)*cos(psi) + V*(-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi)) + W*(sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi))
                dpE = U*cos(theta)*sin(psi) + V*(cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi)) + W*(-sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi))
                dpH = -(U*sin(theta) - V*sin(phi)*cos(theta) - W*cos(phi)*cos(theta))
                dU = R*V - Q*W - self.g*sin(theta)
                dV = -R*U + P*W + self.g*sin(phi)*cos(theta)
                dW = (Q*U - P*V + self.g*cos(phi)*cos(theta) - F_app/self.m)
                dphi = P + tan(theta)*(Q*sin(phi)+R*cos(phi))
                dtheta = Q*cos(phi) - R*sin(phi)
                dpsi = (Q*sin(phi) + R*cos(phi))/cos(theta)
                dP = (self.I_YY - self.I_ZZ) / self.I_XX *Q*R + tau_phi/self.I_XX
                dQ = (self.I_ZZ - self.I_XX) / self.I_YY *P*R + tau_theta/self.I_YY
                dR = (self.I_XX - self.I_YY) / self.I_ZZ *P*Q + tau_psi/self.I_ZZ

                pN += (dpN) / inner_loop_rate
                pE += (dpE) / inner_loop_rate
                pH += (dpH) / inner_loop_rate
                U += (dU) / inner_loop_rate
                V += (dV) / inner_loop_rate
                W += (dW) / inner_loop_rate
                phi += (dphi) / inner_loop_rate
                theta += (dtheta) / inner_loop_rate
                psi += (dpsi) / inner_loop_rate
                P += (dP) / inner_loop_rate
                Q += (dQ) / inner_loop_rate
                R += (dR) / inner_loop_rate

                self.position = pygame.Vector2(pN, pE)
                self.altitude = pH
                self.velocity = pygame.Vector2(U, V)
                self.vz = W
                self.phi = phi
                self.theta = theta
                self.psi = psi
                self.P = P
                self.Q = Q
                self.R = R

            print(f'{self.acceleration[0]:.2f}, {self.acceleration[1]:.2f}, {self.az:.2f}')
            self.print_states()

            # construct Rotation matrix for Drone attached frame
            W_R_A = np.array([[cos(self.theta)*cos(self.psi), cos(self.theta)*sin(self.psi),  -sin(self.theta)],
                               [sin(self.phi)*sin(self.theta)*cos(self.psi)-cos(self.phi)*sin(self.psi), sin(self.phi)*sin(self.theta)*sin(self.psi)+cos(self.phi)*cos(self.psi), sin(self.phi)*cos(self.theta)],
                               [cos(self.phi)*sin(self.theta)*cos(self.psi)+sin(self.phi)*sin(self.psi), cos(self.phi)*sin(self.theta)*sin(self.psi)-sin(self.phi)*cos(self.psi), cos(self.phi)*cos(self.theta)]])

            # compute transformed position and velocity in inertial world frame
            pos_inertial = W_R_A.T @ np.array([[self.position[0]],
                                               [self.position[1]],
                                               [self.altitude]])

            vel_inertial = W_R_A.T @ np.array([[self.velocity[0]], 
                                                [self.velocity[1]],
                                                [self.vz]])

            # update position and velocity
            # self.position = pygame.Vector2(pos_inertial.flatten()[0], pos_inertial.flatten()[1])
            # self.altitude = pos_inertial.flatten()[2]
            self.velocity = pygame.Vector2(vel_inertial.flatten()[0], vel_inertial.flatten()[1])
            self.vz = vel_inertial.flatten()[2]

            self.origin += self.position
            self.origin_z = self.altitude



    def print_states(self):
        print(f'{self.origin[0]:.1f}, ' + 
        f'{self.origin[1]:.1f} | , ' +
        f'{self.position[0]:.1f}, ' +
        f'{self.position[1]:.1f}, ' +
        f'{self.altitude:.1f}, ' +
        f'{self.velocity[0]:.1f}, ' +
        f'{self.velocity[1]:.1f}, ' +
        f'{self.vz:.1f}, ' +
        f'{self.phi:.1f}, ' +
        f'{self.theta:.1f}, ' +
        f'{self.psi:.1f}, ' +
        f'{self.P:.1f}, ' +
        f'{self.Q:.1f}, ' +
        f'{self.R:.1f}'
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
