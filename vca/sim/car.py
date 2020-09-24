import os
import sys
import random
import pygame

from pygame.locals import *

from settings import CAR_IMG

class Car(pygame.sprite.Sprite):
    """Defines a car sprite.
    """
    def __init__(self, game, x, y, vx=0, vy=0, ax=0, ay=0):
        # assign itself to the all_sprites group 
        self.groups = [game.all_sprites, game.car_block_sprites]

        # call Sprite initializer with group info
        pygame.sprite.Sprite.__init__(self, self.groups) 
        
        # assign Sprite.image and Sprite.rect attributes for this Sprite
        self.image, self.rect = game.car_img

        # set kinematics
        # note the velocity and acceleration we assign below 
        # will be interpreted as pixels/sec
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(vx, vy)
        self.acceleration = pygame.Vector2(ax, ay)

        # set initial rect location to position
        self.rect.center = self.position

        # hold onto the game reference
        self.game = game


    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position
        self.velocity += self.acceleration * self.game.dt
        self.position += self.velocity * self.game.dt + 0.5 * self.acceleration * self.game.dt**2


    def update(self):
        """ update sprite attributes. 
            This will get called in game loop for every frame
        """
        self.update_kinematics()
        self.rect.center = self.position
    

