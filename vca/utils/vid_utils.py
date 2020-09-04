import cv2 as cv
import numpy as np

if __name__ == "__main__":
    from img_utils import get_image_paths
else:
    from .img_utils import get_image_paths


def create_video_from_images(path, image_type, video_name, fps, fourcc_codec='DIVX'):
    """Creates a video file from frames in a given folder

    Args:
        path (str): folder path where the frames are 
        image_type (str): type of images in the folder eg: jpg, png, etc
        video_name (str): name for the created video file. For example: 'vid_1.avi'
        fps (double): frame for the created video stream
        fourcc_codec (str): 4-character code of codec used to compress the frames. Default 'DIVX'
                            For example, 'PIM1' is a MPEG-1 codec, 'MJPG' is a motion-jpeg codec etc. 
                            List of codes can be obtained here http://www.fourcc.org/codecs.php

    Usage:
        create_video_from_images('./', 'jpg', 'vid_1.avi', 30)
    """
    # collect image paths
    img_paths = get_image_paths(path=path, image_type=image_type)

    # extract size information from 1st image, to initialize a video writer
    im = cv.imread(img_paths[0])
    im_size = (im.shape[1], im.shape[0])

    # initialize a video writer
    vid_out = cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*fourcc_codec), fps, im_size)

    # read each frame and write into video
    for img_path in img_paths:
        img = cv.imread(img_path)
        vid_out.write(img)

    # release the video writer
    vid_out.release()
