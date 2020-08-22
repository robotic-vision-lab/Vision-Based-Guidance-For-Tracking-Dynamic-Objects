import cv2 as cv
from math import sin, cos
import numpy as np
import os


def get_blank_image(size, value=0, dtype=np.uint8):
    """ returns a blank image with value at every pixel """
    return np.ones(size, dtype=dtype) * value


def dir_exist(dir_name, path):
    """ returns bool indicating if directory exists in path """
    return dir_name in [item for item in os.listdir(path) if os.path.isdir(os.path.join(os.path.relpath(path), item))]


def create_dir(folder_name, path):
    """ creates folder in the given path if it didn't already exist """
    if not dir_exist(folder_name, path):
        os.mkdir(os.path.join(os.path.relpath(path), folder_name))


def generate_synth_data(img_size=(20,20), 
                        path='./', 
                        num_images=4, 
                        folder_name='synth_data'):
    """ takes in size and path, creates synthetic data of given size in the given path """

    # create folder if the folder did not already exist in the path
    create_dir(folder_name, path)

    # synthesise images
    # set parameters for circle
    h, w = img_size
    center = (h//2, w//2)
    radius = 5
    num_dots = 8

    # create images with circles
    for i in range(num_images):
        # create image 
        blank_img = get_blank_image(img_size, value=255)
        # draw the circle
        circle_img = draw_radial_dots(blank_img, radius, center, 0, num_dots)
        
        
        # save the image
        image_name = f'frame_{str(i+1).zfill(2)}.jpg'
        cv.imwrite(os.path.join(os.path.relpath(path), folder_name, image_name), circle_img)    # TODO make sure to write image appropriately
                                                                                                # currently it will overwrite if the images were already there
        # increment radius for next image
        radius += 1

    return os.path.join(os.path.relpath(path), folder_name)


def draw_radial_dots(img, 
                    radius, 
                    center, 
                    pixel_value, 
                    num_dots=8):
    """ draws radial dots in a the given image and return the modified image """
    thetas = [t*np.pi*(2.0/num_dots) for t in range(num_dots)]
    for t in thetas:
        x = int(radius * cos(t)) + center[0]
        y = int(radius * sin(t)) + center[1]
        img[x][y] = pixel_value

    return img


if __name__ == "__main__":
    generate_synth_data()
