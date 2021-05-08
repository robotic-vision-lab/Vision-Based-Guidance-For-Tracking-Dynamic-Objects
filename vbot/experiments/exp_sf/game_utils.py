import os
import shutil
import pygame
import cv2 as cv
from pygame.locals import *
from .settings import *

class ImageDumper:
    def __init__(self, path):
        self.path = os.path.realpath(path)
        
        self._folder_prep_success = _prep_temp_folder(self.path)
        self._frame_num = 0

    def _get_next_path(self):
        self._frame_num += 1
        img_name = f'frame_{str(self._frame_num).zfill(6)}.png'
        file_path = os.path.join(self.path, img_name)

        return file_path 

    def dump(self, img):
        file_name = self._get_next_path()
        cv.imwrite(file_name, img)


# helper function to load images
def load_image(img_name, colorkey=None, alpha=True, scale=1.0):
    """loads pygame image and its rect and returns them as a tuple.
        It will look into the ASSET_FOLDER for the img_name.

    Args:
        img_name (str): name of the image file
        colorkey (tuple, optional): colorkey to set as transparent. For example, (0, 0, 0). Defaults to None.
    """
    # construct full path for the image
    full_name = os.path.join(ASSET_FOLDER, img_name)

    # try to load the image
    try:
        image = pygame.image.load(full_name)
    except pygame.error:
        print('Cannot load image: ', full_name)
        raise SystemExit(str(pygame.get_error()))

    # convert image for pygame compatibility
    image = image.convert_alpha() if alpha else image.convert()

    # scale image
    image = scale_img(image, scale)

    # adjust according to colorkey
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    return image, image.get_rect()


def scale_img(img, scale):
    """Scales a pygame image by given scale

    Args:
        img (pygame.surface.Surface): Image to be scaled
        scale (float): Scale of output image
    """
    if not scale == 1.0:
        new_width = int(img.get_width() * scale)
        new_height = int(img.get_height() * scale)
        img = pygame.transform.scale(img, (new_width, new_height))

    return img


def screen_saver(screen, path):
    """ generator function saves current frame on given screen as .jpg image file """
    # prep folder path 
    _prep_temp_folder(path)
    
    frame_num = 0
    while True:
        frame_num += 1
        image_name = f'frame_{str(frame_num).zfill(4)}.png'
        file_path = os.path.join(path, image_name)
        pygame.image.save(screen, file_path)
        yield


def _prep_temp_folder(folder_path):
    """Prep the temp folder for next screen dump.
        If folder didn't exist before, make it.
        If it exists then remove it and make it.
    """
    # make dir if it doesnot exist
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

    return os.path.isdir(folder_path)


def vec_str(vec):
    """Helper function to get pygame.Vector2 vectors in 0.2f format string.

    Args:
        vec (pygame.Vector2): pygame.Vector2 to be formatted

    Returns:
        (str): formatted string representation
    """

    return f'[{vec[0]:0.2f}, {vec[1]:0.2f}]'