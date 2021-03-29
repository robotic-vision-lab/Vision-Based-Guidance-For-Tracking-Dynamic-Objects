from copy import deepcopy
import pygame

from settings import *

class DroneCamera(pygame.sprite.Sprite):
    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.drone_sprite]

        # call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        self.image, self.rect = simulator.drone_img
        self.image.fill((255, 255, 255, DRONE_IMG_ALPHA), None, pygame.BLEND_RGBA_MULT)
        self.reset_kinematics()
        self.origin = self.position
        self.prev_origin = self.position
        self.delta_pos = pygame.Vector2(0.0,0.0)
        self.altitude = ALTITUDE
        self.alt_change = 1.0

        # self.rect.center = self.position + SCREEN_CENTER
        self.simulator = simulator
        self.update_rect()

        self.vel_limit = DRONE_VELOCITY_LIMIT
        self.acc_limit = DRONE_ACCELERATION_LIMIT

    def update(self):
        """helper function update kinematics
        """
        self.update_kinematics()
        # self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER

    def update_rect(self):
        """update drone sprite's rect.
        """
        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)

    def reset_kinematics(self):
        """helper function to reset kinematics
        """
        self.position = pygame.Vector2(DRONE_POSITION)
        self.velocity = pygame.Vector2(DRONE_INITIAL_VELOCITY)
        self.acceleration = pygame.Vector2(0, 0)

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # set a drag coefficient
        # COEFF = 0.1
        # print(f'a {self.acceleration}, v {self.velocity}, dt {self.game.dt}')
        # self.acceleration -= self.acceleration * COEFF
        # print(f'a {self.acceleration}, v {self.velocity}')

        # update velocity and position
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

    def compensate_camera_motion(self, sprite_obj):
        """Compensates camera motion by updating position of sprite object.

        Args:
            sprite_obj (pygame.sprite.Sprite): Sprite object whose motion needs compensation.
        """
        sprite_obj.position -= self.position
        # sprite_obj.velocity -= self.velocity    # consider investigating for correctness
        sprite_obj.update_rect()

    def change_acceleration(self, command_vec):
        """Changes the drone acceleration appropriately in reponse to given command vector.

        Args:
            command_vec (tuple(float, float)): Command vector tuple. Indicates acceleration change to be made.
        """
        # update acceleration
        COMMAND_SENSITIVITY = 0.1
        command_vec *= COMMAND_SENSITIVITY
        self.acceleration += command_vec

        # counter floating point arithmetic noise
        if abs(self.acceleration[0]) < COMMAND_SENSITIVITY:
            self.acceleration[0] = 0.0
        if abs(self.acceleration[1]) < COMMAND_SENSITIVITY:
            self.acceleration[1] = 0.0

        # make sure acceleration magnitude stays within a set limit
        if abs(self.acceleration.length()) > self.acc_limit:
            self.acceleration -= command_vec

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
        """Helper function to implement drone raise altitude
        """
        self.simulator.alt_change_fac = 1.0 + self.alt_change / self.altitude
        self.altitude += self.alt_change

    def fly_lower(self):
        """Helper function to implement drone lower altitude
        """
        self.simulator.alt_change_fac = 1.0 - self.alt_change / self.altitude
        self.altitude -= self.alt_change
