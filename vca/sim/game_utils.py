import os
import pygame

from settings import *

# helper function to load images
def load_image(img_name, colorkey=None, alpha=True):
    """loads pygame image and its rect and returns them as a tuple

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
        raise SystemExit(str(geterror()))

    # convert image for pygame compatibility
    image = image.convert_alpha() if alpha else image.convert()

    # adjust according to colorkey
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    return image, image.get_rect()