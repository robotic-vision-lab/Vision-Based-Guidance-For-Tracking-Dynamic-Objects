import cv2 as cv
import numpy as np

class BruteL2:
    def __init__(self):
        self.matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)

    def compute_matches(self, descriptors_1, descriptors_2, threshold=-1, distance_threshold=float('inf')):
        matches = self.matcher.match(descriptors_1, descriptors_2)

        if threshold == -1:
            return matches

        # use of the other threshold is deprecated 
        distances = [m.distance for m in matches]
        mnd = min(distances)
        mxd = max(distances)
        if mxd == 0.0:
            return matches

        matches_ = [m for m in matches if m.distance/mxd < mnd/mxd + threshold]

        return matches_

    @staticmethod
    def draw_matches(img_1, kp_1, img_2, kp_2, matches):
        img = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1]+img_2.shape[1], 3), 
                        dtype=np.uint8)

        img = cv.drawMatches(img_1, kp_1, img_2, kp_2, matches, img)
        cv.imshow('Matches', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def compute_descriptor_match_distance(self, des_1, des_2):
        des_1 = np.array([des_1])
        des_2 = np.array([des_2])

        m = self.matcher.match(des_1, des_2)

        return m[0].distance