import pygame

from pygame.locals import *
from settings import *



class DroneCamera:
    def __init__(self, game):
        self.drone = pygame.Rect(0, 0, WIDTH, HEIGHT)
        self.position = pygame.Vector2(0, 0)
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.game = game
        self.vel_limit = 500.0
        self.acc_limit = 100.0

    def update_kinematics(self):
        """helper function to update kinematics of object
        """

        # set a drag coefficient
        # COEFF = 0.1
        # self.acceleration -= self.velocity * COEFF 
        # update velocity and position
        self.velocity += self.acceleration * self.game.dt
        if abs(self.velocity.length()) > self.vel_limit:
            # self.acceleration = pygame.Vector2(0,0)
            self.velocity -= self.acceleration * self.game.dt
            self.position += self.velocity * self.game.dt
        else:
            self.position += self.velocity * self.game.dt + 0.5 * self.acceleration * self.game.dt**2


    def compensate_camera_motion(self, sprite_obj):
        sprite_obj.position -= self.velocity * self.game.dt + 0.5 * self.acceleration * self.game.dt**2

    def move(self, command_vec):
        if not (command_vec.x == 0 or command_vec.y == 0):
            command_vec *= 0.7071

        COMMAND_SENSITIVITY = 0.8
        command_vec *= COMMAND_SENSITIVITY
        self.acceleration += command_vec
        if abs(self.acceleration.length()) > self.acc_limit:
            self.acceleration -= command_vec
        self.update_kinematics()
        # print(f'Kinematics {self.position} | {self.velocity} | {self.acceleration}')
        x = int(self.position.x)
        y = int(self.position.y)
        # print(x,y)
        self.drone = pygame.Rect(x, y, WIDTH, HEIGHT)