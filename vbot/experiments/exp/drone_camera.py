from copy import deepcopy
from math import cos, sin, atan
from collections import namedtuple
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
        Gains = namedtuple('Gains', 'KP_phi KD_phi KP_theta KD_theta KP_psi KD_psi KP_zdot KD_zdot')
        self.GAINS = Gains(K_P_PHI, K_D_PHI, K_P_THETA, K_D_THETA, K_P_PSI, K_D_PSI, K_P_ZDOT, K_D_ZDOT)

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
            F = self.INERTIA.m * (self.g - self.az) / (cos(self.phi)*cos(self.theta))

            # compute desired attitude
            phi_c = atan(self.acceleration[1] * cos(self.theta)/(self.g - self.az))
            theta_c = atan(self.acceleration[0] / (self.az - self.g))
            psi_c = 0

            # compute required
            tau_theta = self.GAINS.KP_theta*(theta_c - self.theta) + self.GAINS.KD_theta*(0-self.q)
            tau_phi = self.GAINS.KP_phi*(phi_c - self.phi) + self.GAINS.KD_phi*(0-self.p)
            tau_psi = self.GAINS.KP_psi*(psi_c - self.psi) + self.GAINS.KD_psi*(0-self.r)

            # update state
            num_inner_loop = 10
            outer_loop_rate = 1/DELTA_TIME
            inner_loop_rate = outer_loop_rate * num_inner_loop

            for i in range(num_inner_loop):
                pN = self.position[0]
                pE = self.position[1]
                pH = self.altitude
                u = self.velocity[0]
                v = self.velocity[1]
                w = self.vz
                phi = self.phi
                theta = self.theta
                psi = self.psi
                p = self.p
                q = self.q
                r = self.r

                F_app = F * ((i+1) / num_inner_loop)**-2

                dpN = u*cos(theta)*cos(psi) + v*(-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi)) + w*(sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi))
                dpE = u*cos(theta)*sin(psi) + v*(cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi)) + w*(-sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi))
                dpH = -(u*sin(theta) - v*sin(phi)*cos(theta) - w*cos(phi)*cos(theta))
                dU = r*v - q*w - self.g*sin(theta)
                dV = -r*u + p*w + self.g*sin(phi)*cos(theta)
                dW = (q*u - p*v + self.g*cos(phi)*cos(theta) - F_app/self.INERTIA.m)
                dphi = p + tan(theta)*(q*sin(phi)+r*cos(phi))
                dtheta = q*cos(phi) - r*sin(phi)
                dpsi = (q*sin(phi) + r*cos(phi))/cos(theta)
                dP = (self.INERTIA.Iyy - self.INERTIA.Izz) / self.INERTIA.Ixx *q*r + tau_phi/self.INERTIA.Ixx
                dQ = (self.INERTIA.Izz - self.INERTIA.Ixx) / self.INERTIA.Iyy *p*r + tau_theta/self.INERTIA.Iyy
                dR = (self.INERTIA.Ixx - self.INERTIA.Iyy) / self.INERTIA.Izz *p*q + tau_psi/self.INERTIA.Izz

                pN += (dpN) / inner_loop_rate
                pE += (dpE) / inner_loop_rate
                pH += (dpH) / inner_loop_rate
                u += (dU) / inner_loop_rate
                v += (dV) / inner_loop_rate
                w += (dW) / inner_loop_rate
                phi += (dphi) / inner_loop_rate
                theta += (dtheta) / inner_loop_rate
                psi += (dpsi) / inner_loop_rate
                p += (dP) / inner_loop_rate
                q += (dQ) / inner_loop_rate
                r += (dR) / inner_loop_rate

                self.position = pygame.Vector2(pN, pE)
                self.altitude = pH
                self.velocity = pygame.Vector2(u, v)
                self.vz = w
                self.phi = phi
                self.theta = theta
                self.psi = psi
                self.p = p
                self.q = q
                self.r = r

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
        """helper function to print states"""
        print(f'{self.simulator.time:.2f}secs: cam_origin=(' + 
            f'{self.origin[0]:.1f}, ' + 
        f'{self.origin[1]:.1f})  pos=(' +
        f'{self.position[0]:.1f}, ' +
        f'{self.position[1]:.1f}, ' +
        f'{self.altitude:.1f})  vel=(' +
        f'{self.velocity[0]:.1f}, ' +
        f'{self.velocity[1]:.1f}, ' +
        f'{self.vz:.1f})  ang_pos=(' +
        f'{self.phi:.1f}, ' +
        f'{self.theta:.1f}, ' +
        f'{self.psi:.1f})  ang_vel=(' +
        f'{self.p:.1f}, ' +
        f'{self.q:.1f}, ' +
        f'{self.r:.1f})' + 
        f'  acc_command=({self.acceleration[0]:.2f}, {self.acceleration[1]:.2f}, {self.az:.2f})'
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
        self.az = self.GAINS.KP_zdot* (100 - self.altitude) + self.GAINS.KD_zdot * (0 - self.vz)


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
