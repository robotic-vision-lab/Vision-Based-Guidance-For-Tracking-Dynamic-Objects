import os
import pygame
from math import tan, radians

# define some colors to be used with pygame 
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (31, 31, 31)
DARK_GRAY_2 = (40, 40, 40)
LIGHT_GRAY = (128, 128, 128)
LIGHT_GRAY_2 = (153, 153, 153)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (102, 102, 255)
YELLOW = (255, 255, 0)

# define colors to be used with opencv (BGR)
RED_CV = (102, 102, 255)
GREEN_CV = (102, 255, 102)
BLUE_CV = (255, 102, 102)
YELLOW_CV = (0, 255, 255)



#   S E T T I N G S  #

# resources settings
ASSET_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets')
TEMP_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tmp')

# frame settings
FPS = 30

# screen settings
SCREEN_SIZE = WIDTH, HEIGHT = 800, 600
SCREEN_CENTER = (WIDTH//2, HEIGHT//2)
SCREEN_DISPLAY_TITLE = "Car Simulation"
SCREEN_BG_COLOR = DARK_GRAY

# camera image formation settings
FOV = 47.0                                      # degrees
PIXEL_SIZE = 6.25 * 10**-6                      # meters
ALTITUDE = 200                                  # meters
SENSOR_WIDTH = PIXEL_SIZE * WIDTH
FOCAL_LENGTH = (SENSOR_WIDTH / 2) / tan(radians(FOV/2))
HORIZONTAL_SPAN = (ALTITUDE * SENSOR_WIDTH) / FOCAL_LENGTH
PIXEL_TO_METERS_FACTOR = HORIZONTAL_SPAN / WIDTH

# car settings
CAR_IMG = 'car.png'
CAR_LENGTH = 6
CAR_LENGTH_PX = 128
CAR_SCALE = CAR_LENGTH / (CAR_LENGTH_PX * PIXEL_TO_METERS_FACTOR)
# note (0,0) at image center, axes: x points right [>], y points down [v]
CAR_INITIAL_POSITION = (-300, -200)#(50, HEIGHT//2)
CAR_INITIAL_VELOCITY = (45, 0)
CAR_ACCELERATION = (0, 0)
CAR_RADIUS = 1

# block settings
BLOCK_COLOR = DARK_GRAY_2
BLOCK_SIZE = BLOCK_WIDTH, BLOCK_HEIGHT = 12, 8
NUM_BLOCKS = 50

# drone camera settings
DRONE_IMG = 'cross_hair2.png'
DRONE_SCALE = 0.2
# note (0,0) at image center, axes: x points right [>], y points down [v]
DRONE_POSITION = (0, 0)
DRONE_INITIAL_VELOCITY = (30, 0)
DRONE_VELOCITY_LIMIT = 500      # +/-
DRONE_ACCELERATION_LIMIT = 20   # +/-

# time settings
TIME_FONT = 'consolas'
TIME_FONT_SIZE = 16
TIME_COLOR = LIGHT_GRAY_2

# Bounding box settings
BB_COLOR = BLUE