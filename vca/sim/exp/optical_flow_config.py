import os
import cv2 as cv

#   F A R N E B A C K   P A R A M S 
# set farneback parameters
PYRAMID_SCALE = 0.5         
NUM_LEVELS = 3
WINDOW_SIZE = 15
NUM_ITER = 3
POLY_NEIGHBORHOOD = 5
POLY_SIGMA = 1.2
OPERATION_FLAGS = 0

# create dictionary of farneback params
FARNEBACK_PARAMS = dict( pyr_scale=PYRAMID_SCALE,
                         levels=NUM_LEVELS,
                         winsize=WINDOW_SIZE,
                         iterations=NUM_ITER,
                         poly_n=POLY_NEIGHBORHOOD,
                         poly_sigma=POLY_SIGMA,
                         flags=OPERATION_FLAGS )


#   S H I - T O M A S I   F E A T U R E   P A R A M S
# set params for ShiTomasi corner detection
MAX_NUM_CORNERS = 4
WORST_TO_BEST_CORNER_QUALITY_RATIO = 0.3
MIN_DISTANCE_BETWEEN_CORNERS = 10
DERIVATIVE_COVARIATION_BLOCK_SIZE = 5

# create a dictionary for goodFeaturesToTrack parameters
FEATURE_PARAMS = dict( maxCorners = MAX_NUM_CORNERS,
                       qualityLevel = WORST_TO_BEST_CORNER_QUALITY_RATIO,
                       minDistance = MIN_DISTANCE_BETWEEN_CORNERS,
                       blockSize = DERIVATIVE_COVARIATION_BLOCK_SIZE )


#   L U C A S - K A N A D E   P A R A M S 
# set lucas-kanade parameters
SEARCH_WINDOW_SIZE_AT_EACH_PYR_LEVEL = (15,15)
MAX_NUM_PYR_LEVELS = 3
TERM_CRITERIA_TYPE = cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT
TERM_CRITERIA_MAX_COUNT = 10
TERM_CRITERIA_EPSILON = 0.01
SEARCH_TERMINATION_CRITERIA = (TERM_CRITERIA_TYPE, TERM_CRITERIA_MAX_COUNT, TERM_CRITERIA_EPSILON)

# create a dictionary for LK params
LK_PARAMS = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = SEARCH_TERMINATION_CRITERIA )



# FOLDERS
FARN_TEMP_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tmp_farn')
LK_TEMP_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tmp_lk')
