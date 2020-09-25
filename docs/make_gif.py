import os
import sys
import cv2 as cv
import glob
import imageio
import pygifsicle 


# add vca/ to sys path
vca_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../vca'))

if not vca_path in sys.path:
    sys.path.append(vca_path)

from utils import *

# list vid folders
EXPERIMENT_VIDS_FOLDER = os.path.realpath(os.path.join(os.path.dirname(__file__), './experiment_vids'))
SIMPLE_CAR_FOLDER = os.path.realpath(os.path.join(EXPERIMENT_VIDS_FOLDER, './simple_car'))
CAR_BLOCKS_FOLDER = os.path.realpath(os.path.join(EXPERIMENT_VIDS_FOLDER, './car_blocks'))
SIM_TRACK_FOLDER = os.path.realpath(os.path.join(EXPERIMENT_VIDS_FOLDER, './simulation_tracker'))


# create a list of paths of video files in each folder
SIMPLE_CAR_PATHS = [ os.path.realpath(path) for path in glob.glob(SIMPLE_CAR_FOLDER + '/*.avi')]
CAR_BLOCKS_PATHS = [ os.path.realpath(path) for path in glob.glob(CAR_BLOCKS_FOLDER + '/*.avi')]
SIM_TRACK_PATHS = [ os.path.realpath(path) for path in glob.glob(SIM_TRACK_FOLDER + '/*.avi')]

# list 'em up
VID_FILE_PATHS_LIST = [ SIMPLE_CAR_PATHS,
                        CAR_BLOCKS_PATHS,
                        SIM_TRACK_PATHS ]


## What are we going to do in the module?

# make gifs
# we are in vca/../docs
# the vid folders are in docs/experiment_vids
# namely, simple_car, car_blocks, etc
# we need to
# 1. make gifs of individual videos
# 2. make gifs of all videos in a folder stacked together
# currently working on 
# * 1. make gifs of individual vids in each folder 
#       fetch the vid file handle, capture all frames and write it into a gif file
#       how? 
#       * imageio library can be used to create animated images


def vid_to_gif(vid_path, gif_name=None, fps=30.0):
    """Extract the video directory name from vid_path uri
        Make gif from the given video file and save it into a gif folder with given name 

    Args:
        vid_path (str): Full path of hte video file uri. Eg., 'simple_car/simple_car.avi'
        gif_name (str): Name of the gif file to be created.
    """
    # extract folder information and form the gif_folder string
    gif_folder = os.path.join(os.path.dirname(vid_path), 'gifs')

    # check if gif folder exists, if it does not then create it
    if not os.path.isdir(gif_folder):
        os.mkdir(gif_folder)

    # extract vid name from vid_path, form a gif_name and gif_path
    if not gif_name:
        gif_name = os.path.split(vid_path)[1].split('.')[0] + '.gif'
    gif_path = os.path.join(gif_folder, gif_name)

    # read the video file using the 
    video_reader = imageio.get_reader(vid_path) # mimread(vid_path)

    # write into a gif
    # imageio.mimwrite(gif_path, video_read, fps=fps)
    writer = imageio.get_writer(gif_path, fps=fps)
    for f in video_reader:
        writer.append_data(f)
    writer.close()

    # optimize gif
    # pygifsicle.optimize(gif_path) # TODO fix file not found error with gifsicle


if __name__ == "__main__":
    # for each vid folder make directory to store gifs
    for v_paths in VID_FILE_PATHS_LIST:
        for v_path in v_paths:
            print(f'Process make_gif running on \n{v_path}')
            vid_to_gif(v_path)
    