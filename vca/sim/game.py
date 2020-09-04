import os
import sys
import pygame

from pygame.locals import *
from settings import *
from car import Car

class Game:
    """Simulation Game
    """
    def __init__(self):
        # initialize screen
        pygame.init()
        self.screen_surface = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption(SCREEN_DISPLAY_TITLE)


