import cv2 as cv

class ImageWindow:
    """ Defines a window object to handle image displaying tasks using opencv.
        OpenCV image window displaying is not intuitive to work with.
        Using ImageWindow it should become easier to handle images in a window in an object-oriented fashion.
    """

    def __init__(self, window_name, callback=None):
        self.window_name = window_name
        
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
        cv.imshow(self.window_name, img)

    def destroy(self):
        """ destroys window, de-allocates memory usage """
        cv.destroyWindow(self.window_name)

    def handle_events(self):
        """ calls the waitKey() to process events and calls callback if it exists """
        # get keycode 
        keycode = cv2.waitKey(1)

        # if callback function exists and some keypress happened, call it
        if self.callback is not None and keycode != -1:
            self.callback(keycode)
