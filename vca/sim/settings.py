import os
import pygame

# define some colors to be used with pygame 
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (31, 31, 31)
LIGHT_GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 255)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


#   S E T T I N G S  #

# screen settings
WIDTH, HEIGHT = 512, 512
TITLE = "Car Simulation"
BG_COLOR = DARK_GRAY

# frame settings
FPS = 30

# resources settings
ASSET_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets')

# car settings
CAR_IMG = 'car.png'
