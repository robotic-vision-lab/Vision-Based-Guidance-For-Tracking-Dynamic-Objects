import cv2 as cv

class Sift:
    def __init__(self):
        self.detector = cv.SIFT_create()

    @staticmethod
    def draw_keypoints(img, keypoints):
        """Takes as input an image and SIFT keypoints and outputs the image with keypoints drawn over it

        Args:
            img (np.ndarray): Image over which keypoints are to be drawn
            keypoints (np.ndarray): Keypoints of the given image

        Returns:
            np.ndarray: Image with keypoints drawn over it
        """
        return cv.drawKeypoints(img, keypoints, None)

    @staticmethod
    def draw_rich_keypoints(img, keypoints):
        """Takes as input an image and SIFT keypoints and outputs the image with rich representation of keypoints drawn over it

        Args:
            img (np.ndarray): Image over which keypoints will be richly drawn
            keypoints (np.ndarray): Keypoints of the given image

        Returns:
            np.ndarray: Image with richly drawn keypoints
        """
        return cv.drawKeypoints(img, 
                                keypoints,
                                None,
                                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def get_keypoints(self, img, mask=None):
        """Computes SIFT keypoints in image. Takes in the image and optional mask.
        Returns computed SIFT keypoints 

        Args:
            img (np.ndarray): Image in which keypoints are to be computed
            mask (np.ndarray, optional): Mask specifies where to look for keypoints in the image. Defaults to None.

        Returns:
            np.ndarray: Computed keypoints
        """
        return self.detector.detect(img, mask)

    def detect_and_show(self, img, mask=None):
        """Takes in an image, detects SIFT features and displays original image and image with rich keypoints drawn over it.

        Args:
            img (np.ndarray): Image in which SIFT features are to be detected
            mask (np.ndarray, optional): Specify using a mask where to look in the image for features. Defaults to None.
        """
        kp = self.detector.detect(img, mask)
        img_SFD = self.draw_rich_keypoints(img, kp)
        cv.imshow('Original', img)
        cv.imshow('SIFT Features Detected', img_SFD)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def get_descriptors_at_keypoints(self, img, keypoints):
        keypoints, descriptors = self.detector.compute(img, keypoints)

        return keypoints, descriptors

    def get_descriptors(self, img, mask=None):
        _, descriptors = self.detector.detectAndCompute(img, mask)

        return descriptors

    def get_keypoints_and_descriptors(self, img, mask=None):
        keypoints, descriptors = self.detector.detectAndCompute(img, mask)

        return keypoints, descriptors