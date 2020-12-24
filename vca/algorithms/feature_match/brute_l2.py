import cv2 as cv
import numpy as np

class BruteL2:
    def __init__(self):
        self.matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)

    def compute_matches(self, descriptors_1, descriptors_2):
        return self.matcher.match(descriptors_1, descriptors_2)

    @staticmethod
    def draw_matches(self, img_1, kp_1, img_2, kp_2, matches):
        img = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1]+img_2.shape[1], 3), 
                        dtype=np.uint8)

        distances = [m.distance for m in matches]
        mxd = max(distances)
        mnd = min(distances)
        matches_ = [m for m in matches if m.distance/mxd < mnd/mxd + 0.15]

        img = cv.drawMatches(img_1, kp_1, img_2, kp_2, matches_, img)
        cv.imshow('Good Matches', img)
        cv.waitKey(0)
        cv.destroyAllWindows()