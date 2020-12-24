import cv2 as cv
import numpy as np

class CorrelationCoeffNormed:
    def __init__(self):
        self.template_match_method = cv.TM_CCOEFF_NORMED

    def compute_match(self, img, template):
        result = cv.matchTemplate(img, template, self.template_match_method)

        return result

    
    