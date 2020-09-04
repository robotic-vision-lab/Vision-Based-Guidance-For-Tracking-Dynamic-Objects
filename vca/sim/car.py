import os
import sys
import random
import pygame

from pygame.locals import *

from settings import CAR_IMG

class Car(pygame.sprite.Sprite):
    """Defines a car sprite.
    """
    def __init__(self, game, x, y):
        pygame.sprite.Sprite.__init__(self) # call Sprite initializer
        
        # assign Sprite.image and Sprite.rect attributes for this Sprite
        self.image = game.car_img
        self.rect = self.image.rect()

        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2()
        self.acceleration = pygame.Vector2()

        # set initial rect position
        self.rect.center = self.position

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
    
         


        
