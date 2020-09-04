import os
import sys
import random
import pygame

from pygame.locals import *

from settings import CAR_IMG
from game_utils import load_image

class Car(pygame.sprite.Sprite):
    """Defines a car sprite.
    """
    def __init__(self, game, x, y):
        pygame.sprite.Sprite.__init__(self) # call Sprite initializer
        
        # assign Sprite.image and Sprite.rect attributes for this Sprite
        self.image, self.rect = load_image(img_name=CAR_IMG, colorkey=BLACK)

        # set initial rect position
        # self.rect.center = 
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2()
        self.acceleration = pygame.Vector2()

        self.game = game

    def update(self):
        """ update sprite attributes
        """
        # update velocity and position
        self.velocity = self.velocity + self.acceleration * self.game.dt
         


        
