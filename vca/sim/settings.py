import os
import pygame
from math import tan, radians
from colors import *

# special constants
POS_INF = float('inf')
NEG_INF = float('-inf')
NAN = float('nan')

#   S E T T I N G S  #

# resources settings
ASSET_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets')
TEMP_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tmp')
TRACKER_TEMP_FOLDER = os.path.join(os.path.abspath(TEMP_FOLDER), 'track_tmp')
SIMULATOR_TEMP_FOLDER = os.path.join(os.path.abspath(TEMP_FOLDER), 'sim_tmp')

# frame settings
FPS = 30

# screen settings
SCREEN_SIZE = WIDTH, HEIGHT = 800, 600  # 640, 480 # 800, 600
SCREEN_CENTER = (WIDTH//2, HEIGHT//2)
SCREEN_DISPLAY_TITLE = "Car Simulation"
SCREEN_BG_COLOR = DARK_GRAY

# camera image formation settings
FOV = 47.0                                      # degrees
PIXEL_SIZE = 6.25 * 10**-6                      # meters
ALTITUDE = 250.0                                  # meters
SENSOR_WIDTH = PIXEL_SIZE * WIDTH
FOCAL_LENGTH = (SENSOR_WIDTH / 2) / tan(radians(FOV/2))
HORIZONTAL_SPAN = (ALTITUDE * SENSOR_WIDTH) / FOCAL_LENGTH
PIXEL_TO_METERS_FACTOR = HORIZONTAL_SPAN / WIDTH

# car settings
CAR_IMG = 'car3.png'
CAR_LENGTH = 6          # meters
CAR_LENGTH_PX = 128
CAR_SCALE = CAR_LENGTH / (CAR_LENGTH_PX * PIXEL_TO_METERS_FACTOR)
# note for exp 3 (0,0) at image center, axes: x points right [>], y points down [v]
# for exp4 world coords in SI units, x ->, y ^.
# CAR_INITIAL_POSITION = (-70.0, -70.0)#(966.94, -150.00)#(-200.0, -150.0)#(-200.0, 200.0)#(-30.0, 30.0)#(50, HEIGHT//2)
# CAR_INITIAL_VELOCITY = (22.22, 0.0)#(30.0, 0.0)#(30.0, 0.0)#
CAR_INITIAL_POSITION_2 = (50.0, 20.0)#(966.94, -150.00)#(-200.0, -150.0)#(-200.0, 200.0)#(-30.0, 30.0)#(50, HEIGHT//2)
CAR_INITIAL_VELOCITY_2 = (23.22, 0.0)#(30.0, 0.0)#(30.0, 0.0)#
CAR_ACCELERATION = (0.0, 0.0)
CAR_RADIUS = 10.0
TRACK_COLOR = (102, 255, 102)

DEFAULT_TRAJECTORY = 0
ONE_HOLE_TRAJECTORY = 1
TWO_HOLE_TRAJECTORY = 2
LANE_CHANGE_TRAJECTORY = 3
SQUIRCLE_TRAJECTORY = 4

USE_TRAJECTORY = SQUIRCLE_TRAJECTORY
TWO_HOLE_PERIOD = 120
TWO_HOLE_SIZE = 30
ONE_HOLE_PERIOD = 60
ONE_HOLE_SIZE = 30
SQUIRCLE_PARAM_S = 0.95
SQUIRCLE_PARAM_R = 90
SQUIRCLE_PERIOD = 180


# block settings
BLOCK_COLOR = DARK_GRAY_2
BLOCK_SIZE = BLOCK_WIDTH, BLOCK_HEIGHT = 15.0, 0.5 #18.0, 1.0
NUM_BLOCKS = 10

# bar settings
BAR_COLOR = DARK_GRAY_3
BAR_COLOR_DELTA = (8, 8, 8)
BAR_SIZE = BAR_WIDTH, BAR_HEIGHT = 4.0, (HEIGHT-1) * PIXEL_TO_METERS_FACTOR
NUM_BARS = 4

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
TIME_COLOR = LIGHT_GRAY_2   # used for all simulator texts
DELTA_TIME = 0.1           # used in full blocking mode

# Bounding box settings
BB_COLOR = BLUE     # pygame color

# tracker settings
METRICS_COLOR = LIGHT_GRAY_2
TRACK_COLOR = (102, 255, 102)
ARROW_SCALE = 15.0
TRACKER_BLANK = 31
ADD_METRICS = 1
ADD_ALTITUDE_INFO = 1
SHOW_EXTRA = 0
DRAW_KEYPOINT_TRACKS = 0

# theme
DARK_ON = 0
if DARK_ON:
    BLOCK_COLOR = (40, 40, 40)
    BLOCK_COLOR_DELTA = 24
    SCREEN_BG_COLOR = (31, 31, 31)
    TRACK_COLOR = CAROLINA_BLUE_BGR
    TRACKER_BLANK = 31
    TIME_COLOR = (153, 153, 153)
    METRICS_COLOR = (153, 153, 153)
    DOT_COLOR = WHITE
    DRONE_IMG_ALPHA = 250
    BB_COLOR = (102, 102, 255)
else:
    BLOCK_COLOR = (230, 230, 230)
    BLOCK_COLOR_DELTA = 18
    SCREEN_BG_COLOR = (250, 250, 250)
    TRACK_COLOR = TURQUOISE_GREEN_BGR
    TRACKER_BLANK = 250
    TIME_COLOR = (128, 128, 128)
    METRICS_COLOR = (128, 128, 128)
    DOT_COLOR = (31, 31, 31)
    DRONE_IMG_ALPHA = 102
    BB_COLOR = (102, 102, 255)


# tracker settings
USE_WORLD_FRAME = 0

# simulator settings
CLEAR_TOP = 1

# console settings
CLEAN_CONSOLE = 0

# resolution settings
SCALE_1 = 1.0
SCALE_2 = 1.0

OPTION = 0
if OPTION == 0:
    SCALE_1 = 1.0
    SCALE_2 = 1.0
if OPTION == 1:
    SCALE_1 = 0.5
    SCALE_2 = 2.0
if OPTION == 2:
    SCALE_1 = 0.25
    SCALE_2 = 4.0

# salt pepper SNR settings
SNR = 1.0   #0.99

# filter choice
# use filter will make tracker use filter, it may use kalman or moving average
USE_TRACKER_FILTER = 0  
USE_KALMAN = 0
USE_MA = 0
# EKF used by controller
USE_EXTENDED_KALMAN = 1
USE_NEW_EKF = 1

# plot settings
LINE_WIDTH_1 = 1.0
LINE_WIDTH_2 = 1.5
TITLE_FONT_SIZE = 14
SUB_TITLE_FONT_SIZE = 11
SUPTITLE_ON = 1

# --------------------------------------------------------------------------------
CAR_RADIUS = 0.1
# initial conditions

# # 1 OPEN
# CAR_INITIAL_POSITION    = (-70.0, -70.0)
# CAR_INITIAL_VELOCITY    = (31.11, 0.0)
# DRONE_POSITION          = (0.0, 0.0)
# DRONE_INITIAL_VELOCITY  = (22.22, 0.0)
# K_1                     = 0.15
# K_2                     = 0.02
# K_W                      = -0.1

# # 2 CLOSED [cam frame, truekin, c1 with den .05, bound=3]
# CAR_INITIAL_POSITION    = (-70.0, -10.0)    # DO NOT TOUCH
# CAR_INITIAL_VELOCITY    = (22.22, 0.0)      # DO NOT TOUCH
# DRONE_POSITION          = (0.0, 0.0)        # DO NOT TOUCH
# DRONE_INITIAL_VELOCITY  = (11.11, 0.0)      # DO NOT TOUCH
# K_1                     = 0.3               # DO NOT TOUCH
# K_2                     = 0.05              # DO NOT TOUCH
# K_W                      = -0.1              # DO NOT TOUCH

# # 3 CLOSED [world frame, truekin, c2 with den .01, bound=10]
# CAR_INITIAL_POSITION    = (0.0, 0.0)        #DO NOT TOUCH
# CAR_INITIAL_VELOCITY    = (22.2222, 0.0)    #DO NOT TOUCH
# DRONE_POSITION          = (0.0, 50.0)       #DO NOT TOUCH
# DRONE_INITIAL_VELOCITY  = (31.1111, 0.0)    #DO NOT TOUCH
# K_1                     = 0.1               #DO NOT TOUCH
# K_2                     = 0.05              #DO NOT TOUCH
# K_W                      = -0.1              #DO NOT TOUCH

# #4 CLOSED [world frame, truekin, c2 with den .01, bound=10, R=10]
# CAR_INITIAL_POSITION    = (200.0, 100.0)    # DO NOT TOUCH
# CAR_INITIAL_VELOCITY    = (22.22, 0.0)      # DO NOT TOUCH
# DRONE_POSITION          = (0.0, 0.0)        # DO NOT TOUCH
# DRONE_INITIAL_VELOCITY  = (31.11, 0.0)      # DO NOT TOUCH
# K_1                     = 0.1               # DO NOT TOUCH
# K_2                     = 0.05              # DO NOT TOUCH
# K_W                      = -0.1              # DO NOT TOUCH

# 5 open
CAR_INITIAL_POSITION    = (0.0, -20.0)
CAR_INITIAL_VELOCITY    = (22.22, 0.0)
DRONE_POSITION          = (0.0, 0.0)
DRONE_INITIAL_VELOCITY  = (0.0, 0.0)
K_1                     = 0.1
K_2                     = 0.05
K_W                     = -0.1

# # 6 open
# CAR_INITIAL_POSITION    = (-200.0, -100.0)
# CAR_INITIAL_VELOCITY    = (31.11, 0.0)
# DRONE_POSITION          = (0.0, 0.0)
# DRONE_INITIAL_VELOCITY  = (22.22, 0.0)
# K_1                     = 0.1
# K_2                     = 0.05
# K_W                      = -0.1

# 7 open
# CAR_INITIAL_POSITION    = (-50.0, 10.0)
# CAR_INITIAL_VELOCITY    = (22.22, 0.0)
# DRONE_POSITION          = (0.0, 0.0)
# DRONE_INITIAL_VELOCITY  = (0.11, 0.0)
# K_1                     = 0.1
# K_2                     = 0.05
# K_W                      = -0.1
