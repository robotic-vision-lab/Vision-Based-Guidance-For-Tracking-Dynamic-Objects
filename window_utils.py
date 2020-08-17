import cv2 as cv
import os

class ImageWindow:
    """ Defines a window object to handle image displaying tasks using Opencv.
        OpenCV image window displaying is not intuitive to work with.
        Purpose of  ImageWindow is to make it easier to handle images in a window in a rather object-oriented fashion.
        Usage:
            (1)
            # create a window
            my_window = ImageWindow(window_name='My Window')

            # display image 
            my_window.show(image_1)         

            # save the image in window
            my_window.save()                # >>> Saving ./image_1.jpg.

            # close the window
            my_window.close()
    """

    def __init__(self, window_name, callback=None, window_flags=None):
        self.window_name = window_name
        self.callback = callback
        self.window_flags = window_flags
        
        self.create_window(self.window_flags)
        
    def create_window(self, window_flags=None):
        """ Creates a single window that can be a place holder for images and trackbars.
            Created window will be referred to by the window name.
            If a window already exists by the name, it would do nothing.
        
            note: 
            Windows need to be created before waitKey() can be used to process events.
            For window flags check 
            https://docs.opencv.org/3.4/d7/dfc/group__highgui.html#gabf7d2c5625bc59ac130287f925557ac3
        """
        cv.namedWindow(self.window_name, window_flags)
        if self.window_exists(self.window_name):
            print(f'Window {self.window_name} already exists.')

    def window_exists(self, window_name):
        """ Checks if a window by the given name exists and is visible """
        return True if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE)==1.0 else False

    def show(self, img):
        """ displays the image in the window """
        self.img = img
        cv.imshow(self.window_name, img)

    def save(self, path='./', image_type='jpg'):
        """ saves the displayed image in an image file"""
        filename = os.path.join(path, f'{self.window_name}.{image_type}')
        cv.imwrite(filename, self.img)
        print(f'Saving {filename}.')


    def close(self):
        """ destroys window, de-allocates memory usage """
        cv.destroyWindow(self.window_name)

    def handle_events(self):
        """ calls the waitKey() to process events and calls callback if it exists """
        # get keycode 
        keycode = cv2.waitKey(1)

        # if callback function exists and some keypress happened, call it
        if self.callback is not None and keycode != -1:
            self.callback(keycode)
