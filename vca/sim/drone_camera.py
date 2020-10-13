import pygame

from pygame.locals import *
from settings import *
from math import copysign


class DroneCamera(pygame.sprite.Sprite):
    def __init__(self, game):
        self.groups = [game.all_sprites, game.drone_sprite]

        # call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        # self.drone = pygame.Rect(0, 0, WIDTH, HEIGHT)
        # self.image = pygame.Surface((20, 20))
        # self.image.fill(BLUE)
        # self.rect = self.image.get_rect()
        self.image, self.rect = game.drone_img
        self.image.fill((255, 255, 255, 204), None, pygame.BLEND_RGBA_MULT)
        self.reset_kinematics()
        
        self.rect.center = self.position + SCREEN_CENTER
        self.game = game
        
        self.vel_limit = DRONE_VELOCITY_LIMIT
        self.acc_limit = DRONE_ACCELERATION_LIMIT

    def update(self):
        """[summary]
        """
        self.update_kinematics()
        self.rect.center = self.position + SCREEN_CENTER

    def reset_kinematics(self):
        """[summary]
        """
        self.position = pygame.Vector2(0, 0)
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
        self.velocity += self.acceleration * self.game.dt
        if abs(self.velocity.length()) > self.vel_limit:
            self.velocity -= self.acceleration * self.game.dt

        self.position += self.velocity * self.game.dt + 0.5 * self.acceleration * self.game.dt**2

    def compensate_camera_motion(self, sprite_obj):
        """[summary]

        Args:
            sprite_obj ([type]): [description]
        """
        sprite_obj.position -= self.position #self.velocity * self.game.dt + 0.5 * self.acceleration * self.game.dt**2

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
        