import os
import pygame
from math import tan, radians
from .colors import *

# special constants
POS_INF = float('inf')
NEG_INF = float('-inf')
NAN = float('nan')

#   S E T T I N G S  #

# resources settings
ASSET_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../assets')
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
ALTITUDE = 100.0                                  # meters
SENSOR_WIDTH = PIXEL_SIZE * WIDTH
FOCAL_LENGTH = (SENSOR_WIDTH / 2) / tan(radians(FOV/2))
HORIZONTAL_SPAN = (ALTITUDE * SENSOR_WIDTH) / FOCAL_LENGTH
PIXEL_TO_METERS_FACTOR = HORIZONTAL_SPAN / WIDTH

# car settings
CAR_IMG = 'cars_7.png'
CAR_IMG_2 = 'cars_11.png'
CAR_IMG_3 = 'cars_4.png'
CAR_LENGTH = 6          # meters
CAR_LENGTH_PX = 128
CAR_SCALE = CAR_LENGTH / (CAR_LENGTH_PX * PIXEL_TO_METERS_FACTOR)
# CAR_INITIAL_POSITION_2 = (10.0, 10.0)
# CAR_INITIAL_VELOCITY_2 = (5.22, 0.0)
CAR_ACCELERATION = (0.0, 0.0)
CAR_RADIUS = 10.0
TRACK_COLOR = (102, 255, 102)

DEFAULT_TRAJECTORY = 0
ONE_HOLE_TRAJECTORY = 1
TWO_HOLE_TRAJECTORY = 2
LANE_CHANGE_TRAJECTORY = 3
SQUIRCLE_TRAJECTORY = 4

USE_TRAJECTORY = LANE_CHANGE_TRAJECTORY
TWO_HOLE_PERIOD = 120
TWO_HOLE_SIZE = 30
ONE_HOLE_PERIOD = 60
ONE_HOLE_SIZE = 30
SQUIRCLE_PARAM_S = 0.9
SQUIRCLE_PARAM_R = 1000
SQUIRCLE_PERIOD = 360

# occlusion case sentinels
NO_OCC = 0
PARTIAL_OCC = 1
TOTAL_OCC = 2

# none kinematics sentinel
NONE_KINEMATICS = [[None, None], [None, None]]

# block settings
BLOCK_COLOR = DARK_GRAY_2
BLOCK_SIZE = BLOCK_WIDTH, BLOCK_HEIGHT = 15.0, 0.5 #18.0, 1.0
NUM_BLOCKS = 0

# bar settings
BAR_COLOR = DARK_GRAY_3
BAR_COLOR_DELTA = (8, 8, 8)
BAR_SIZE = BAR_WIDTH, BAR_HEIGHT = 25.0, (HEIGHT-1) * PIXEL_TO_METERS_FACTOR
NUM_BARS = 3

# drone camera settings
DRONE_IMG = 'cross_hair.png'
DRONE_SCALE = 1.0
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
BB_COLOR = SAFETY_YELLOW_RGB#BLUE     # pygame color

# Ellipse settings
ELLIPSE_TOLERANCE = 0.1
ELLIPSE_COLOR = (232,232,232)#(66, 61, 78)
ELLIPSE_OPACITY = 0.4
ELLIPSE_FP_RADIUS = 4

# reference frame
IMAGE_REF_FRAME = 0
WORLD_INERTIAL_REF_FRAME = 1
WORLD_REF_FRAME = 2


# tracker settings
METRICS_COLOR = LIGHT_GRAY_2
TRACK_COLOR = (102, 255, 102)
ARROW_SCALE = 15.0
TRACKER_BLANK = 31
ADD_METRICS = 0
ADD_ALTITUDE_INFO = 1
SHOW_EXTRA = 0
DRAW_KEYPOINT_TRACKS = 1
DRAW_KEYPOINTS_ONLY_WITHOUT_TRACKS = 1
TRACK_SCALE = 2.5

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
	BLOCK_COLOR = (248, 248, 248)
	BLOCK_COLOR_DELTA = 3
	SCREEN_BG_COLOR = (250, 250, 250)
	TRACK_COLOR = TURQUOISE_GREEN_BGR
	TRACKER_BLANK = 250
	TIME_COLOR = (128, 128, 128)
	METRICS_COLOR = (128, 128, 128)
	DOT_COLOR = (31, 31, 31)
	DRONE_IMG_ALPHA = 153
	BB_COLOR = MIDDLE_YELLOW_RGB#(102, 102, 255)


# tracker settings
USE_WORLD_FRAME = 0

# simulator settings
CLEAR_TOP = 1

# console settings
CLEAN_CONSOLE = 1

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

# open
CAR_INITIAL_POSITION    = (10.0, -10.0)
CAR_INITIAL_VELOCITY    = (5.22, 0.0)
CAR_INITIAL_POSITION_2  = (10.0, 10.0)
CAR_INITIAL_VELOCITY_2  = (5.22, 0.0)
CAR_INITIAL_POSITION_3  = (-15.0, 0.0)
CAR_INITIAL_VELOCITY_3  = (5.22 , 0.0)
DRONE_POSITION          = (0.0, 0.0)
DRONE_INITIAL_VELOCITY  = (5.31, 0.0)
K_1                     = 0.5
K_2                     = 0.2
K_W                     = -0.1
