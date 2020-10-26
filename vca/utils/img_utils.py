import cv2 as cv
import os
import numpy as np
import logging

from PIL import Image


DEBUG_LOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../logs/debug.log')

# set the debug file name 
logging.basicConfig(filename=DEBUG_LOG_PATH, filemode='w', level=logging.DEBUG)

def l_print(line):
    """ custom log printer for debugging """
    logging.debug(line)

# initialise image path variable 
DATASET_PATH = "./datasets/RubberWhale/"


def get_image_shape(img):
    """ returns the shape of the image """
    return img.shape


def get_image_size(img):
    """ returns the size of the image """
    return img.size


def get_image_paths(path=DATASET_PATH, image_type='png'):
    """ returns list of image filenames of type image_type in the given location path """
    return [os.path.join(path, item) for item in os.listdir(path) if item.split('.')[-1] == image_type]


def convert_to_grayscale(img):
    """ returns grayscale of the input image """
    # check if image shape has 3 dimensions before converting
    if len(img.shape) == 3:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img_gray = img

    return img_gray


def convert_grayscale_to_BGR(img):
    """ returns BGR image given a grayscale image """
    # check if image shape has 2 dimensions before converting
    if len(img.shape) == 2:
        img_BGR = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        img_BGR = img

    return img_BGR


def convert_BGR_to_RGB(img):
    """ converts the image from BGR (OpenCV) to RGB """
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def normalize_to_unit_range(img):
    """ takes in an image and normalizes to range 0.0 to 1.0  """
    return cv.normalize(img.astype('float32'), None, 0.0, 1.0, norm_type=cv.NORM_MINMAX)


def normalize_to_255_range(img):
    """ takes in an image and normalizes to range 0 to 255  """
    return cv.normalize(img, None, 0, 255, norm_type=cv.NORM_MINMAX).astype('uint8') 


def get_images(path=DATASET_PATH, image_type='png'):
    """ returns list of PIL Image objects using files in the given location path """
    
    # save the current directory and change directory to path
    current_directory = os.getcwd()
    os.chdir(path)

    # list items in directory
    folder_items = os.listdir()

    # build the list of Image objects
    img_list = []
    if len(folder_items) > 0:
        img_list = [Image.open(item) for item in folder_items if item.split('.')[-1] == image_type]
    
    # change back to the previously saved current directory
    os.chdir(current_directory)

    return img_list


def get_laplacian_kernel():
    """ returns a laplacian kernel """
    return np.array([[1/12, 1/6, 1/12], 
                     [1/6, -1, 1/6], 
                     [1/12, 1/6, 1/12]]).astype('float32')


def get_average_kernel():
    """ returns a average kernel """
    return np.array([[1/12, 1/6, 1/12], 
                     [1/6, 0, 1/6], 
                     [1/12, 1/6, 1/12]]).astype('float32')


def preprocess_image(img, blur=True):
    """ returns grayscaled, gaussian blurred and normalised image """
    img = convert_to_grayscale(img)
    if blur:
        img = cv.GaussianBlur(img, (3,3), 0)
    img = normalize_to_unit_range(img)

    return img


def add_salt_pepper(img, SNR):
    img_ = img.copy()
    img_ = img_.transpose(2,1,0)
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0) # Copy by channel to have the same shape as img
    img_[mask == 1] = 255 # salt noise
    img_[mask == 2] = 0 # 
    img_ = img_.transpose(2,1,0)
    return img_

def scale_image(img, scale_factor=1.0):
    """ takes in img and scale_factor and returns scaled image with aspect ratio preserved """
    # compute dsize
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    
    # choose appropriate interpolation model
    if scale_factor < 1.0:
        # image is to be shrunk
        interpolation = cv.INTER_AREA
    else:
        # image is to be enlarged
        interpolation = cv.INTER_CUBIC      # slow but looks good
        # interpolation = cv.INTER_LINEAR   # faster and looks OK
        
    scaled_image = cv.resize( src=img,
                              dst=None, 
                              dsize=(width, height), 
                              interpolation=interpolation )

    return scaled_image


def add_border(img, border_color=[255,255,255], thickness=None, min_border=1):
    """ inputs image and border parameters and returns image with added border.
        Added border is of uniform thickness (≥ min_border)
    """
    # compute border thickness
    border_thickness_ratio = 0.01
    if thickness is None:
        border_thickness = max(min_border, int(max(img.shape[0], img.shape[1])*border_thickness_ratio))
    else:
        border_thickness = thickness

    # set top, bottom, left, right border thickness
    t = b = l = r = border_thickness

    bordered_image = cv.copyMakeBorder( src=img,
                                        top=t,
                                        bottom=b,
                                        left=l,
                                        right=r,
                                        borderType=cv.BORDER_CONSTANT,
                                        dst=None,
                                        value=border_color )
    
    return bordered_image
    

def images_assemble(images, 
                    grid_shape, 
                    scale_factor=1.0, 
                    bg_color=[255, 255, 255], 
                    border=True, 
                    scale_to_fit=False):
    """ Assembles an array of images into a single image grid.
        Also, scale all images by the scale_factor.
        Images (1D list) of different sizes may be scaled to fit the row.

    """
    # validate grid_shape and num of elements in images
    if not len(grid_shape) == 2:
        print(f'grid_shape needs to be a tuple with 2 elements (height, width).')
        return
    elif not len(images) == grid_shape[0]*grid_shape[1]:
        print(f'length of the list images (1D list) needs to be same as area of grid_shape (height*width). ')
    
    # extract height and width
    height = grid_shape[0]
    width = grid_shape[1]

    # add border
    if border:
        images = [add_border(img, thickness=5) for img in images]

    # compute max heights and widths
    max_heights = np.zeros((height,))
    max_widths = np.zeros((width,))
    for row in range(height):
        for col in range(width):
            # select appropriate image
            img = images[width * row + col]
            
            # update max_height
            max_heights[row] = max(max_heights[row], img.shape[0])

    for col in range(width):
        for row in range(height):
            # select appropriate image
            img = images[width * row + col]
            
            # update max_width
            max_widths[col] = max(max_widths[col], img.shape[1])

    # initialise the aggregate image with zeros
    img_assembled = np.zeros((int(sum(max_heights)), int(sum(max_widths)), 3), dtype='uint8') 
    
    # set background color
    img_assembled[..., :] = bg_color

    # assemble images into one by placing them at appropriate offsets
    for r in range(height):
        for c in range(width):
            # select appropriate image
            img = images[width * r + c]

            # make sure that image has 3 channels
            if len(img.shape) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

            # compute offset (img_x, img_y) for image placement
            img_h = img.shape[0]
            img_w = img.shape[1]
            cell_h = max_heights[r]
            cell_w = max_widths[c]

            y = sum(max_heights[:r]) + max_heights[r] // 2
            x = sum(max_widths[:c]) + max_widths[c] // 2

            img_y = int(y - img_h // 2)
            img_x = int(x - img_w // 2)

            # place the image
            img_assembled[img_y:img_y+img_h, img_x:img_x+img_w, ...] = img

    return scale_image(img_assembled, scale_factor)


def put_text(img, 
             text, 
             coords, 
             font=cv.FONT_HERSHEY_SIMPLEX, 
             font_scale=1, 
             color=(51, 255, 51),
             thickness=2 ):
    """
    helper method lets put text on images
    """
    img = cv.putText(img, text, coords, font, font_scale, color, thickness, cv.LINE_AA)

    return img


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # test images_assemble() usage
    # ---------------------------------------------------------------
    # make a list of images
    img = cv.imread(get_image_paths('./datasets/Dimetrodon')[0])   
    img_b = np.hstack((img, img))
    imgs = [img, img_b, img_b, img_b, img, img_b]

    # assemble images
    a = images_assemble(imgs, (2,3), scale_factor=0.4)

    # display
    cv.imshow('assembled', a)
    cv.waitKey(0)
    cv.destroyAllWindows() 
    # ---------------------------------------------------------------