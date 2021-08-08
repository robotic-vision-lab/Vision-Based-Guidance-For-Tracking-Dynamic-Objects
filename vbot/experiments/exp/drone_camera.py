from copy import deepcopy
import pygame

from .settings import *

class DroneCamera(pygame.sprite.Sprite):
    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.drone_sprites]

        # call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        self.image, self.rect = simulator.drone_img_rect
        self.image.fill((255, 255, 255, DRONE_IMG_ALPHA), None, pygame.BLEND_RGBA_MULT)
        self.reset_kinematics()
        self.origin = self.position
        self.prev_origin = self.position
        self.delta_pos = pygame.Vector2(0.0,0.0)
        self.altitude = ALTITUDE    # same as z (subject to accleratiosns)
        self.vz = 0.0
        self.az = 0.0

        # self.rect.center = self.position + SCREEN_CENTER
        self.simulator = simulator
        self.update_rect()

        self.vel_limit = DRONE_VELOCITY_LIMIT
        self.acc_limit = DRONE_ACCELERATION_LIMIT

    def reset_kinematics(self):
        """helper function to reset kinematics
        """
        self.position = pygame.Vector2(DRONE_POSITION)
        self.velocity = pygame.Vector2(DRONE_INITIAL_VELOCITY)
        self.acceleration = pygame.Vector2(0, 0)

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

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position (in the XY plane)
        self.velocity += self.acceleration * self.simulator.dt
        if abs(self.velocity.length()) > self.vel_limit:
            self.velocity -= self.acceleration * self.simulator.dt

        self.prev_delta_pos = deepcopy(self.delta_pos)
        self.delta_pos = self.velocity * self.simulator.dt #+ 0.5 * self.acceleration * \
            # self.simulator.dt**2      # i know how this looks like but,   pylint: disable=line-too-long
        self.position = self.velocity * self.simulator.dt #+ 0.5 * self.acceleration * \
            # self.simulator.dt**2  # donot touch ☠                    pylint: disable=line-too-long
        self.prev_origin = deepcopy(self.origin)
        self.origin += self.delta_pos

        # update kinematics along Z

    def apply_accleration_command(self, ax, ay, az=0):
        self.acceleration = pygame.Vector2((ax, ay))


    def compensate_camera_motion(self, sprite_obj):
        """Compensates camera motion by updating position of sprite object.

        Args:
            sprite_obj (pygame.sprite.Sprite): Sprite object whose motion needs compensation.
        """
        sprite_obj.position -= self.position
        sprite_obj.update_rect()

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
