import cv2 as cv
import numpy as np

class CorrelationCoeffNormed:
    def __init__(self):
        self.template_match_method = cv.TM_CCOEFF_NORMED

    def compute_match(self, img, template):
        result = cv.matchTemplate(img, template, self.template_match_method)

        return result

class TemplateMatcher:
    def __init__(self, template, template_matcher):
        self.template = template
        self.matcher = template_matcher
        self.match_result = None

    def match_in_image(self, img):
        self.match_result = self.matcher.compute_match(img, self.template)
        return self.match_result

    def find_template_in_image(self, img):
        self.match_in_image(img)
        self.min_val, self.max_val, self.min_loc, self.max_loc = cv.minMaxLoc(self.match_result)
        w = self.template.shape[1]
        h = self.template.shape[0]
        self.top_left = self.max_loc
        self.bottom_right = (self.top_left[0]+w, self.top_left[1]+h)

        return self.top_left, self.bottom_right

    def find_template_in_image_bb(self, img, bb):
        x,y,w,h = bb
        img_patch = img[y:y+h, x:x+w]
        tl, br =  self.find_template_in_image(img_patch)
        
        self.top_left = (tl[0]+x, tl[1]+y)
        self.bottom_right = (br[0]+x, br[1]+y)
        
        return self.top_left, self.bottom_right

    def find_template_center_in_image_bb(self, img, bb):
        tl, br = self.find_template_in_image_bb(img, bb)
        cx = (tl[0] + br[0]) // 2
        cy = (tl[1] + br[1]) // 2
        return cx, cy

    
    