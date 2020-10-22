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
LIGHT_GRAY_3 = (250, 250, 250)
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
ALTITUDE = 500.0                                  # meters
SENSOR_WIDTH = PIXEL_SIZE * WIDTH
FOCAL_LENGTH = (SENSOR_WIDTH / 2) / tan(radians(FOV/2))
HORIZONTAL_SPAN = (ALTITUDE * SENSOR_WIDTH) / FOCAL_LENGTH
PIXEL_TO_METERS_FACTOR = HORIZONTAL_SPAN / WIDTH

# car settings
CAR_IMG = 'car.png'
CAR_LENGTH = 6          # meters
CAR_LENGTH_PX = 128
CAR_SCALE = CAR_LENGTH / (CAR_LENGTH_PX * PIXEL_TO_METERS_FACTOR)
# note for exp 3 (0,0) at image center, axes: x points right [>], y points down [v]
# for exp4 world coords in SI units, x ->, y ^.
# CAR_INITIAL_POSITION = (-70.0, -70.0)#(966.94, -150.00)#(-200.0, -150.0)#(-200.0, 200.0)#(-30.0, 30.0)#(50, HEIGHT//2)
# CAR_INITIAL_VELOCITY = (22.22, 0.0)#(30.0, 0.0)#(30.0, 0.0)#
CAR_ACCELERATION = (0.0, 0.0)
CAR_RADIUS = 10.0
TRACK_COLOR = (102, 255, 102)

# block settings
BLOCK_COLOR = DARK_GRAY_2
BLOCK_SIZE = BLOCK_WIDTH, BLOCK_HEIGHT = 5.0, 3.0
NUM_BLOCKS = 50

# drone camera settings
DRONE_IMG = 'cross_hair2.png'
DRONE_SCALE = 0.2
# note (0,0) at image center, axes: x points right [>], y points down [v]
# DRONE_POSITION = (0.0, 0.0)#(943.15, -203.48)#(0.0, 50.0)
# DRONE_INITIAL_VELOCITY = (31.11, 0.0)#(-11.11, 0.0)#(20.0, 0.0)
DRONE_VELOCITY_LIMIT = 500      # +/-
DRONE_ACCELERATION_LIMIT = 20   # +/-

# time settings
TIME_FONT = 'consolas'
TIME_FONT_SIZE = 16
TIME_COLOR = LIGHT_GRAY_2 # used for all simulator texts

# metrics
METRICS_COLOR = LIGHT_GRAY_2

# Bounding box settings
BB_COLOR = BLUE     # pygame color

# theme
DARK_ON = 0
if DARK_ON:
    BLOCK_COLOR = DARK_GRAY_2
    SCREEN_BG_COLOR = DARK_GRAY
    TRACK_COLOR = (102, 255, 102)
    TIME_COLOR = LIGHT_GRAY_2
    METRICS_COLOR = LIGHT_GRAY_2
    DOT_COLOR = WHITE
    DRONE_IMG_ALPHA = 250
    BB_COLOR = BLUE
else:
    BLOCK_COLOR = (230, 220, 230)
    SCREEN_BG_COLOR = (250, 250, 250)
    TRACK_COLOR = (8, 255, 8)
    TIME_COLOR = LIGHT_GRAY
    METRICS_COLOR = LIGHT_GRAY
    DOT_COLOR = DARK_GRAY
    DRONE_IMG_ALPHA = 102
    BB_COLOR = (51, 51, 255)


# tracker settings
USE_WORLD_FRAME = 1

# filter choice
USE_KALMAN = 0  # else Moving average


#--------------------------------------------------------------------------------
CAR_RADIUS = 1
# initial conditions

# # 1 OPEN
# CAR_INITIAL_POSITION    = (-70.0, -70.0)
# CAR_INITIAL_VELOCITY    = (31.11, 0.0)
# DRONE_POSITION          = (0.0, 0.0)
# DRONE_INITIAL_VELOCITY  = (22.22, 0.0)
# K_1                     = 0.15
# K_2                     = 0.02
# w_                      = -0.1

# # 2 CLOSED [cam frame, truekin, c1 with den .05, bound=3]
# CAR_INITIAL_POSITION    = (-70.0, -10.0)    # DO NOT TOUCH
# CAR_INITIAL_VELOCITY    = (22.22, 0.0)      # DO NOT TOUCH
# DRONE_POSITION          = (0.0, 0.0)        # DO NOT TOUCH
# DRONE_INITIAL_VELOCITY  = (11.11, 0.0)      # DO NOT TOUCH
# K_1                     = 0.3               # DO NOT TOUCH
# K_2                     = 0.05              # DO NOT TOUCH
# w_                      = -0.1              # DO NOT TOUCH

# # 3 CLOSED [world frame, truekin, c2 with den .01, bound=10]
# CAR_INITIAL_POSITION    = (0.0, 0.0)        #DO NOT TOUCH
# CAR_INITIAL_VELOCITY    = (22.2222, 0.0)    #DO NOT TOUCH
# DRONE_POSITION          = (0.0, 50.0)       #DO NOT TOUCH
# DRONE_INITIAL_VELOCITY  = (31.1111, 0.0)    #DO NOT TOUCH
# K_1                     = 0.1               #DO NOT TOUCH
# K_2                     = 0.05              #DO NOT TOUCH
# w_                      = -0.1              #DO NOT TOUCH

# #4 CLOSED [world frame, truekin, c2 with den .01, bound=10, R=10]
CAR_INITIAL_POSITION    = (200.0, 100.0)    # DO NOT TOUCH
CAR_INITIAL_VELOCITY    = (22.22, 0.0)      # DO NOT TOUCH
DRONE_POSITION          = (0.0, 0.0)        # DO NOT TOUCH
DRONE_INITIAL_VELOCITY  = (31.11, 0.0)      # DO NOT TOUCH
K_1                     = 0.1               # DO NOT TOUCH
K_2                     = 0.05              # DO NOT TOUCH
w_                      = -0.1              # DO NOT TOUCH

# 5 open
# CAR_INITIAL_POSITION    = (100.0, 100.0)
# CAR_INITIAL_VELOCITY    = (22.22, 0.0)
# DRONE_POSITION          = (0.0, 0.0)
# DRONE_INITIAL_VELOCITY  = (31.11, 0.0)
# K_1                     = 0.1
# K_2                     = 0.05
# w_                      = -0.1

# # 6 open
# CAR_INITIAL_POSITION    = (-70.0, -70.0)
# CAR_INITIAL_VELOCITY    = (22.22, 0.0)
# DRONE_POSITION          = (0.0, 0.0)
# DRONE_INITIAL_VELOCITY  = (31.11, 0.0)
# K_1                     = 0.1
# K_2                     = 0.05
# w_                      = -0.1

# # 7 open
# CAR_INITIAL_POSITION    = (-70.0, -70.0)
# CAR_INITIAL_VELOCITY    = (22.22, 0.0)
# DRONE_POSITION          = (0.0, 0.0)
# DRONE_INITIAL_VELOCITY  = (31.11, 0.0)
# K_1                     = 0.1
# K_2                     = 0.05
# w_                      = -0.1
