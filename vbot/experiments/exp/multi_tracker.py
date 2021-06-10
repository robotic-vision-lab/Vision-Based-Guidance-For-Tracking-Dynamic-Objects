from datetime import timedelta
from math import isnan

import cv2 as cv
import numpy as np
from .settings import *


from .my_imports import (Sift,
                        BruteL2,
                        CorrelationCoeffNormed,
                        ImageDumper,
                        convert_to_grayscale,
                        FEATURE_PARAMS,
                        LK_PARAMS,
                        MAX_NUM_CORNERS,
                        TemplateMatcher,
                        compute_optical_flow_LK,
                        draw_tracks,
                        draw_point,
                        draw_sparse_optical_flow_arrows,
                        put_text,
                        images_assemble,)

class MultiTracker:
    """MultiTracker object is designed to work with and ExperimentManager object.
    It can be used to process screen captures and produce tracking information for feature points.
    Computer Vision techniques employed here.
    """

    def __init__(self, manager):

        self.manager = manager

        self.frame_old_gray = None
        self.frame_old_color = None
        self.frame_new_gray = None
        self.frame_new_color = None

        self.targets = None
        # self.frame_new_color_edited = None
        # self.img_tracker_display = None

        self.patch_size = round(1.5/PIXEL_TO_METERS_FACTOR)   # meters

        self.detector = Sift()
        self.descriptor_matcher = BruteL2()
        self.template_matcher = CorrelationCoeffNormed()

        self.cur_img = None

        self._can_begin_control_flag = False    # will be modified in process_image
        self.window_size = 5

        self.display_arrow_color = {NO_OCC:GREEN_CV, PARTIAL_OCC:ORANGE_PEEL_BGR, TOTAL_OCC:TOMATO_BGR}

        self._frame_num = 0
        # self.track_length = 10 # for lifetime management
        self.tracker_info_mask = None   # mask over which tracker information is drawn persistently
        self.win_name = 'Tracking in progress'
        self.img_dumper = ImageDumper(TRACKER_TEMP_FOLDER)
        self.DES_MATCH_DISTANCE_THRESH = 250 #450
        self.DES_MATCH_DEV_THRESH = 0.50 # float('inf') to get every match
        self.TEMP_MATCH_THRESH = 0.9849

        self._FAILURE = False
        self._SUCCESS = True
        self.MAX_ERR = 15

    def set_targets(self, targets):
        self.targets = targets

    def is_first_time(self):
        """Indicates if tracker never received a frame for the first time

        Returns:
            bool: Boolean indicating first time
        """
        # this function is called in process_image in the beginning. 
        # it indicates if this the first time process_image received a frame
        return self.frame_old_gray is None

    def is_total_occlusion(self, target):
        return target.occlusion_case_new == TOTAL_OCC

    def can_begin_control(self):
        """Returns boolean check indicating if controller can be used post tracking.

        Returns:
            bool: Indicator for controller begin doing it's thing
        """
        return self._can_begin_control_flag  # and self.prev_car_pos is not None

    def save_initial_target_descriptors(self, target):
        """Helper function used after feature keypoints and centroid computation. Saves initial target descriptors.
        """
        # use keypoints from new frame, 
        # save descriptors of new keypoints(good)
        keyPoints = [cv.KeyPoint(*kp.ravel(), 15) for kp in target.initial_keypoints]
        target.initial_kps, target.initial_target_descriptors = self.detector.get_descriptors_at_keypoints(
                                                                self.frame_new_gray, 
                                                                keyPoints,
                                                                target.bounding_box)

    def save_initial_target_template(self, target):
        """Helper function used after feature keypoints and centroid computation. Saves initial target template.
        """
        # use the bounding box location to save the target template
        # x, y, w, h = bb = self.manager.get_target_bounding_box()
        # center = tuple(map(int, (x+w/2, y+h/2)))
        target.initial_target_template_color = self.get_bb_patch_from_image(self.frame_new_color, target.bounding_box)
        target.initial_target_template_gray = self.get_bb_patch_from_image(self.frame_new_gray, target.bounding_box)

    def save_initial_patches(self, target):
        """Helper function used after feature keypoints and centroid computation. Saves initial patches around keypoints.
        Also, initializes dedicated template matchers
        """
        target.initial_patches_color = [self.get_neighborhood_patch(self.frame_new_color, tuple(map(int,kp.flatten())), self.patch_size) for kp in target.initial_keypoints]
        target.initial_patches_gray = [self.get_neighborhood_patch(self.frame_new_gray, tuple(map(int,kp.flatten())), self.patch_size) for kp in target.initial_keypoints]
        
        # initialize template matcher object for each patch
        target.template_matchers = [TemplateMatcher(patch, self.template_matcher) for patch in target.initial_patches_gray]

    def update_patches(self, target):
        # pass
        self.patch_size = round(1.5 / self.manager.simulator.pxm_fac)
        target.patches_gray = [self.get_neighborhood_patch(self.frame_new_gray, tuple(map(int,kp.flatten())), self.patch_size) for kp in target.keypoints_new_good]
        target.template_matchers = [TemplateMatcher(patch, self.template_matcher) for patch in target.patches_gray]

    def update_template(self):
        for target in self.targets:
            target.template_gray = self.get_bb_patch_from_image(self.frame_new_gray, target.bounding_box)

    def augment_old_frame(self):
        # keypoints that were not found in new frame would get discounted by the next iteration
        # these bad points from old frame can be reconstructed in new frame
        # and then corresponding patches can be drawn in new frame after the flow computation
        # then save to old
        pass

    def find_saved_patches_in_img_bb(self, img, target):
        """Uses patch template matchers to locate patches in given image inside given bounding box.

        Args:
            img (numpy.ndarray): Image in which patches are to be found.
            target (Target): Target for which patches need to be found 

        Returns:
            tuple: Best matched template locations, best match values
        """
        bb = target.get_updated_bounding_box()
        target.template_points = np.array([
            temp_matcher.find_template_center_in_image_bb(img, bb)
            for temp_matcher in target.template_matchers
            ]).reshape(-1, 1, 2)

        target.template_scores = np.array([
            temp_matcher.get_best_match_score()
            for temp_matcher in target.template_matchers
            ]).reshape(-1, 1)

        return target.template_points, target.template_scores

    def get_relative_associated_patch(self, keypoint, centroid, target):
        # make sure shape is consistent
        keypoint = keypoint.reshape(-1, 1, 2)
        centroid = centroid.reshape(-1, 1, 2)
        rel = keypoint - centroid

        # find index of closest relative keypoints
        index = ((target.rel_keypoints - rel)**2).sum(axis=2).argmin()

        # return corresponding patch
        return target.initial_patches_gray[index]

    def put_patch_at_point(self, img, patch, point):
        """Stick a given patch onto given image at given point

        Args:
            img (numpy.ndarray): Image onto which we want to put a patch
            patch (numpy.ndarray): Patch that we want to stick on the image 
            point (tuple): Point at which patch center would go

        Returns:
            numpy.ndarray: Image after patch is put with it's center aligned with the given point
        """
        # assumption: patch size is fixed by tracker
        x_1 = point[0] - self.patch_size//2
        y_1 = point[1] - self.patch_size//2
        x_2 = x_1 + self.patch_size
        y_2 = y_1 + self.patch_size
        img[y_1:y_2, x_1:x_2] = patch

        return img

    @staticmethod
    def get_bb_patch_from_image(img, bounding_box):
        """Returns a patch from image using bounding box

        Args:
            img (numpy.ndarray): Image from which patch is a to be drawn
            bounding_box (tuple): Bounding box surrounding the image patch of interest

        Returns:
            numpy.ndarray: The image patch 
        """
        x, y, w, h = bounding_box
        return img[y:y+h, x:x+w] # same for color or gray

    @staticmethod
    def get_neighborhood_patch(img, center, size):
        """Returns a patch from image using a center point and size

        Args:
            img (numpy.ndarray): Image from which a patch is to be drawn
            center (tuple): Point in image at which patch center would align
            size (int): Size of patch 

        Returns:
            numpy.ndaray: Patch
        """
        size = (size, size) if isinstance(size, int) else size
        x = center[0] - size[0]//2
        y = center[1] - size[1]//2
        w, h = size

        return img[y:y+h, x:x+w] # same for color or gray

    @staticmethod
    def get_patch_mask(img, patch_center, patch_size):
        """Returns a mask, given a patch center and size

        Args:
            img (numpy.ndarray): Image using which mask is to be created
            patch_center (tuple): Center location of patch
            patch_size (tuple): Size of patch

        Returns:
            numpy.ndarray: Mask
        """
        x = patch_center[0] - patch_size //2
        y = patch_center[1] - patch_size //2
        mask = np.zeros_like(img)
        mask[y:y+patch_size[1], x:x+patch_size[0]] = 255
        return mask

    @staticmethod
    def get_bounding_box_mask(img, x, y, width, height):
        """Returns mask, using bounding box

        Args:
            img (numpy.ndarray): Image using which mask is to be created
            x (int): x coord of top left of bounding box
            y (int): y coord of top left of bounding box
            width (int): width of bounding box
            height (int): height of bounding box

        Returns:
            numpt.ndarray: mask
        """
        # assume image is grayscale
        mask = np.zeros_like(img)
        mask[y:y+height, x:x+width] = 255
        return mask

    @staticmethod
    def get_centroid(points):
        """Returns centroid of given list of points

        Args:
            points (np.ndarray): Centroid point. [shape: (1,2)]
        """
        points_ = np.array(points).reshape(-1, 1, 2)
        return np.mean(points_, axis=0)

    def get_feature_keypoints_from_mask(self, img, mask, bb=None):
        """Returns feature keypoints compute in given image using given mask

        Args:
            img (numpy.ndarray): Image in which feature keypoints are to be computed
            mask (numpy.ndarray): Mask indicating selected region

        Returns:
            numpy.ndarray: Feature keypoints
        """
        shi_tomasi_kpts = cv.goodFeaturesToTrack(img, mask=mask, **FEATURE_PARAMS)
        detector_kpts = self.detector.get_keypoints(img, mask, bb)
        detector_kpts = np.array([pt.pt for pt in detector_kpts]).astype(np.float32).reshape(-1, 1, 2)
        if shi_tomasi_kpts is None and (detector_kpts is None or len(detector_kpts) == 0):
            return None

        if shi_tomasi_kpts is None and not (detector_kpts is None or len(detector_kpts) == 0):
            return detector_kpts
        
        if shi_tomasi_kpts is not None and (detector_kpts is None or len(detector_kpts) == 0):
            return shi_tomasi_kpts
        
        comb_kpts = np.concatenate((shi_tomasi_kpts, detector_kpts), axis=0)
        return comb_kpts

    def get_descriptors_at_keypoints(self, img, keypoints, bb=None):
        """Returns computed descriptors at keypoints

        Args:
            img (numpy.ndarray): Image
            keypoints (numpy.ndarray): Keypoints of interest

        Returns:
            list: List of descriptors corresponding to given keypoints.
        """
        kps = [cv.KeyPoint(*kp.ravel(), 15) for kp in keypoints]
        kps, descriptors = self.detector.get_descriptors_at_keypoints(self.frame_new_gray, kps, bb)
        return kps, descriptors

    def get_true_bb_from_oracle(self, target):
        """Helper function to get the true target bounding box.

        Returns:
            tuple: True target bounding box
        """
        # return self.manager.get_target_bounding_box_from_offset()
        return target.get_updated_bounding_box()

    def _get_kin_from_manager(self):
        """Helper function to fetch appropriate kinematics from manager

        Returns:
            tuple: Kinematics tuple
        """
        #TODO switch based true or est 
        return self.manager.get_true_kinematics()

    def _get_target_image_location(self):
        """Helper function returns true target location measured in pixels

        Returns:
            [type]: [description]
        """
        kin = self._get_kin_from_manager()
        x,y = kin[2].elementwise()* (1,-1) / self.manager.simulator.pxm_fac
        target_location = (int(x), int(y) + HEIGHT)
        target_location += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)
        return target_location

    def process_image_complete(self, new_frame):
        """Processes new frame and performs and delegated various tracking based tasks.
            1. Extracts target attributes and stores them
            2. Processes each next frame and tracks target
            3. Delegates kinematics computation
            4. Handles occlusions

        Args:
            new_frame (numpy.ndarray): Next frame to be processed

        Returns:
            tuple: Sentinel tuple consisting indicator of process success/failure and kinematics if computed successfully.
        """
        # save new frame, compute grayscale
        self.frame_new_color = new_frame
        self.frame_new_gray = convert_to_grayscale(self.frame_new_color)
        # self.frame_new_gray = cv.GaussianBlur(self.frame_new_gray, (3,3), 0)
        # self.true_new_pt = self._get_target_image_location()
        # cv.imshow('nxt_frame', self.frame_new_gray); cv.waitKey(1)
        if self.is_first_time():
            for target in self.targets:
                # compute bb
                target.bounding_box_mask = self.get_bounding_box_mask(self.frame_new_gray, *target.bounding_box)

                # compute initial feature keypoints and centroid
                target.initial_keypoints = cv.goodFeaturesToTrack(self.frame_new_gray, mask=target.bounding_box_mask, **FEATURE_PARAMS)
                target.initial_centroid = self.get_centroid(target.initial_keypoints)
                target.rel_keypoints = target.initial_keypoints - target.initial_centroid

                # compute and save descriptors at keypoints, save target template
                self.save_initial_target_descriptors(target)
                self.save_initial_target_template(target)
                self.save_initial_patches(target)

                # posterity - keypoints, centroid, occ_case, centroid location relative to rect center
                target.keypoints_old = target.keypoints_new = target.initial_keypoints
                # target.keypoints_old_good = target.keypoints_new_good # not needed, since next iter will have from_no_occ
                target.centroid_old = target.centroid_new = target.initial_centroid
                target.occlusion_case_old = target.occlusion_case_new
                self.manager.set_target_centroid_offset(self.targets[0])
                target.track_status = self._FAILURE
            
            # posterity - save frames
            self.frame_old_gray = self.frame_new_gray
            self.frame_old_color = self.frame_new_color
            # return self._FAILURE

        # cv.imshow('cur_frame', self.frame_old_gray); cv.waitKey(1)
        self._can_begin_control_flag = True


        for target in self.targets:
            # ################################################################################
            # CASE |NO_OCC, _>
            if target.occlusion_case_old == NO_OCC:
                # (a priori) older could have been start or no_occ, or partial_occ or total_occ
                # we should have all keypoints as good ones, and old centroid exists, if old now has no_occ

                # try to target compute flow at keypoints and infer next occlusion case
                self.compute_flow(target)

                # amplify bad errors
                target.cross_feature_errors[(target.cross_feature_errors > 0.75 * self.MAX_ERR) & (target.cross_feature_errors > 100*target.cross_feature_errors.min())] *= 10

                # ---------------------------------------------------------------------
                # |NO_OCC, NO_OCC>
                if (target.feature_found_statuses.all() and target.feature_found_statuses.shape[0] == MAX_NUM_CORNERS and 
                        target.cross_feature_errors.max() < self.MAX_ERR):
                    target.occlusion_case_new = NO_OCC

                    # set good points (since no keypoints were occluded all are good, no need to compute)
                    target.keypoints_new_good = target.keypoints_new
                    target.keypoints_old_good = target.keypoints_old
                    self.update_patches(target)

                    # compute target centroid
                    target.centroid_new = self.get_centroid(target.keypoints_new_good)

                    # compute target kinematics measurements using centroid
                    target.kinematics = self.compute_kinematics_by_centroid(target.centroid_old, target.centroid_new)

                    # posterity
                    target.keypoints_old = target.keypoints_new
                    target.keypoints_old_good = target.keypoints_new_good
                    target.centroid_adjustment = None
                    target.centroid_old = target.centroid_new
                    target.occlusion_case_old = target.occlusion_case_new
                    target.track_status = self._SUCCESS

                # ---------------------------------------------------------------------
                # |NO_OCC, TOTAL_OCC>
                elif not target.feature_found_statuses.all() or target.cross_feature_errors.min() >= self.MAX_ERR:
                    target.occlusion_case_new = TOTAL_OCC

                    # cannot compute target kinematics
                    target.kinematics = None

                    # posterity
                    target.centroid_adjustment = None
                    target.centroid_old_true = self.manager.get_target_centroid(target)
                    target.occlusion_case_old = target.occlusion_case_new
                    target.track_status = self._FAILURE

                # ---------------------------------------------------------------------
                # |NO_OCC, PARTIAL_OCC>
                else:
                    target.occlusion_case_new = PARTIAL_OCC

                    # in this case of from no_occ to partial_occ, no more keypoints are needed to be found
                    # good keypoints need to be computed for kinematics computation as well as posterity
                    target.keypoints_new_good = target.keypoints_new[(target.feature_found_statuses==1) & (target.cross_feature_errors < self.MAX_ERR)].reshape(-1, 1, 2)
                    target.keypoints_old_good = target.keypoints_old[(target.feature_found_statuses==1) & (target.cross_feature_errors < self.MAX_ERR)].reshape(-1, 1, 2)

                    # compute adjusted target centroid
                    centroid_old_good = self.get_centroid(target.keypoints_old_good)
                    centroid_new_good = self.get_centroid(target.keypoints_new_good)
                    target.centroid_adjustment = target.centroid_old - centroid_old_good
                    target.centroid_new = centroid_new_good + target.centroid_adjustment

                    # compute target kinematics measurements
                    target.kinematics = self.compute_kinematics_by_centroid(target.centroid_old, target.centroid_new)

                    # adjust missing old keypoints (no need to check recovery)
                    keypoints_missing = target.keypoints_old[(target.feature_found_statuses==0) | (target.cross_feature_errors >= self.MAX_ERR)]
                    target.keypoints_new_bad = keypoints_missing - target.centroid_old + target.centroid_new

                    # put patches over bad points in new frame
                    for kp in target.keypoints_new_bad:
                        # fetch appropriate patch
                        patch = self.get_relative_associated_patch(kp, target.centroid_new, target)
                        # paste patch at appropriate location
                        self.put_patch_at_point(self.frame_new_gray, patch, tuple(map(int,kp.flatten())))

                    # add revived bad points to good points
                    target.keypoints_new_good = np.concatenate((target.keypoints_new_good, target.keypoints_new_bad.reshape(-1, 1, 2)), axis=0)

                    self.update_patches(target)

                    # posterity
                    target.keypoints_old = target.keypoints_new
                    target.keypoints_old_good = target.keypoints_new_good.reshape(-1, 1, 2)
                    target.keypoints_old_bad = target.keypoints_new_bad.reshape(-1, 1, 2)
                    target.centroid_old = target.centroid_new
                    target.centroid_old_true = self.manager.get_target_centroid(target)
                    target.occlusion_case_old = target.occlusion_case_new
                    target.track_status = self._SUCCESS

            # ################################################################################
            # CASE |PARTIAL_OCC, _>
            elif target.occlusion_case_old == PARTIAL_OCC:
                # (a priori) older could have been no_occ, partial_occ or total_occ
                # we should have some keypoints that are good, if old now has partial_occ
                
                # use good keypoints, to compute flow (keypoints_old_good used as input, and outputs into keypoints_new)
                self.compute_flow(target, use_good=True)

                # amplify bad errors 
                if target.cross_feature_errors.shape[0] > 0.5 * MAX_NUM_CORNERS:
                    target.cross_feature_errors[(target.cross_feature_errors > 0.75 * self.MAX_ERR) & (target.cross_feature_errors > 100*target.cross_feature_errors.min())] *= 10
                
                # update good old and new
                # good keypoints are used for kinematics computation as well as posterity
                target.keypoints_new_good = target.keypoints_new[(target.feature_found_statuses==1) & (target.cross_feature_errors < self.MAX_ERR)].reshape(-1, 1, 2)
                target.keypoints_old_good = target.keypoints_old[(target.feature_found_statuses==1) & (target.cross_feature_errors < self.MAX_ERR)].reshape(-1, 1, 2)

                # these good keypoints may not be sufficient for precise partial occlusion detection
                # for precision, we will need to check if any other keypoints can be recovered or reconstructed
                # we should still have old centroid
                target.bounding_box = self.get_true_bb_from_oracle(target)
                target.bounding_box_mask = self.get_bounding_box_mask(self.frame_new_gray, *target.bounding_box)

                # perform part template matching and update template points and scores
                self.find_saved_patches_in_img_bb(self.frame_new_gray, target)

                # compute good feature keypoints in the new frame (shi-tomasi + SIFT)
                good_keypoints_new = self.get_feature_keypoints_from_mask(self.frame_new_gray, mask=target.bounding_box_mask, bb=target.bounding_box)

                if good_keypoints_new is None or good_keypoints_new.shape[0] == 0:
                    good_distances = []
                else:
                    # compute descriptors at the new keypoints
                    kps, descriptors = self.get_descriptors_at_keypoints(self.frame_new_gray, good_keypoints_new, bb=target.bounding_box)

                    # match descriptors 
                    matches = self.descriptor_matcher.compute_matches(target.initial_target_descriptors, 
                                                                    descriptors, 
                                                                    threshold=-1)

                    distances = np.array([m.distance for m in matches]).reshape(-1, 1)
                    good_distances = distances[distances < self.DES_MATCH_DISTANCE_THRESH]

                # ---------------------------------------------------------------------
                # |PARTIAL_OCC, TOTAL_OCC>
                if ((not target.feature_found_statuses.all() or 
                        target.cross_feature_errors.min() >= self.MAX_ERR) and 
                        len(good_distances) == 0 and 
                        (target.template_scores > self.TEMP_MATCH_THRESH).sum() == 0):
                    target.occlusion_case_new = TOTAL_OCC

                    # cannot compute kinematics
                    target.kinematics = None

                    # posterity
                    target.centroid_old_true = self.manager.get_target_centroid(target)
                    target.occlusion_case_old = target.occlusion_case_new
                    target.track_status = self._FAILURE

                # ---------------------------------------------------------------------
                # |PARTIAL_OCC, NO_OCC>
                elif ((target.keypoints_new_good.shape[0] > 0) and
                        len(good_distances) == MAX_NUM_CORNERS and
                        (target.template_scores > self.TEMP_MATCH_THRESH).sum() == MAX_NUM_CORNERS):
                    target.occlusion_case_new = NO_OCC

                    # compute centroid 
                    target.centroid_new = self.get_centroid(target.keypoints_new_good)

                    # update keypoints
                    if len(good_distances) == MAX_NUM_CORNERS:
                        good_matches = np.array(matches).reshape(-1, 1)[distances < self.DES_MATCH_DISTANCE_THRESH]
                        target.keypoints_new_good = np.array([list(good_keypoints_new[gm.trainIdx]) for gm in good_matches.flatten()]).reshape(-1,1,2)
                        target.keypoints_new = target.keypoints_new_good
                        target.centroid_new = self.get_centroid(target.keypoints_new_good)
                        target.rel_keypoints = target.keypoints_new - target.centroid_new
                        
                    if (target.template_scores > self.TEMP_MATCH_THRESH).sum()==MAX_NUM_CORNERS:
                        target.keypoints_new_good = target.template_points
                        target.keypoints_new = target.keypoints_new_good
                        target.centroid_new = self.get_centroid(target.keypoints_new_good)
                        target.rel_keypoints = target.keypoints_new - target.centroid_new


                    # compute kinematics
                    target.kinematics = self.compute_kinematics_by_centroid(target.centroid_old, target.centroid_new)

                    # adjust centroid
                    if len(good_distances) == MAX_NUM_CORNERS:
                        target.centroid_adjustment = None
                    else: 
                        centroid_new_good = self.get_centroid(good_keypoints_new)
                        target.centroid_adjustment = centroid_new_good - target.centroid_new
                        target.centroid_new = centroid_new_good

                        # update keypoints
                        self.keypoints_new_good = self.keypoints_new = self.centroid_new + self.rel_keypoints         
                    
                    self.update_patches(target)

                    # posterity
                    target.keypoints_old = target.keypoints_new_good
                    target.keypoints_old_good = target.keypoints_new_good
                    target.centroid_adjustment = None
                    target.centroid_old = target.centroid_new
                    target.occlusion_case_old = target.occlusion_case_new
                    target.track_status = self._SUCCESS

                # ---------------------------------------------------------------------
                # |PARTIAL_OCC, PARTIAL_OCC>
                else:
                    # if we come to this, it means at least something can be salvaged
                    target.occlusion_case_new = PARTIAL_OCC

                    if target.keypoints_new_good.shape[0] == 0:
                        # flow failed, matching succeeded (feature or template)
                        # compute new good keypoints using matching
                        if len(good_distances) > 0: 
                            good_matches = np.array(matches).reshape(-1, 1)[distances < self.DES_MATCH_DISTANCE_THRESH]
                            target.keypoints_new_good = np.array([list(good_keypoints_new[gm.trainIdx]) for gm in good_matches.flatten()]).reshape(-1,1,2)
                            target.keypoints_new = target.keypoints_new_good
                            target.centroid_old = target.centroid_old_true
                            target.centroid_new = self.manager.get_target_centroid(target)
                        if (target.template_scores > self.TEMP_MATCH_THRESH).sum() > 0:
                            target.keypoints_new_good = target.template_points[target.template_scores > self.TEMP_MATCH_THRESH].reshape(-1, 1, 2)
                            target.keypoints_new = target.keypoints_new_good
                            target.centroid_old = target.centroid_old_true
                            target.centroid_new = self.manager.get_target_centroid(target)
                    
                    elif target.keypoints_new_good.shape[0] > 0:
                        # flow succeeded, (at least one new keypoint was found)
                        # compute adjusted centroid and compute kinematics
                        centroid_old_good = self.get_centroid(target.keypoints_old_good)
                        centroid_new_good = self.get_centroid(target.keypoints_new_good)
                        target.centroid_adjustment = target.centroid_old - centroid_old_good
                        target.centroid_new = centroid_new_good + target.centroid_adjustment

                    target.kinematics = self.compute_kinematics_by_centroid(target.centroid_old, target.centroid_new)

                    # treat keypoints that were lost during flow
                    if (target.keypoints_new_good.shape[0] > 0 and 
                            # not (len(good_distances) > 0 or (self.template_scores > self.TEMP_MATCH_THRESH).sum() > 0) and
                            ((target.feature_found_statuses==0) | (target.cross_feature_errors >= self.MAX_ERR)).sum() > 0):
                        # adjust missing old keypoints (need to check recovery)
                        keypoints_missing = target.keypoints_old[(target.feature_found_statuses==0) | (target.cross_feature_errors >= self.MAX_ERR)]
                        target.keypoints_new_bad = keypoints_missing - target.centroid_old + target.centroid_new

                        # put patches over bad points in new frame
                        for kp in target.keypoints_new_bad:
                            # fetch appropriate patch
                            patch = self.get_relative_associated_patch(kp, target.centroid_new, target)
                            # paste patch at appropriate location
                            self.put_patch_at_point(self.frame_new_gray, patch, tuple(map(int,kp.flatten())))
                    else:
                        target.keypoints_new_bad = None

                    # add bad points to good 
                    if target.keypoints_new_bad is not None:
                        target.keypoints_new_good = np.concatenate((target.keypoints_new_good, target.keypoints_new_bad.reshape(-1, 1, 2)), axis=0)

                    if (len(good_distances) > target.keypoints_new_good.shape[0] and
                            (target.template_scores > self.TEMP_MATCH_THRESH).sum() > target.keypoints_new_good.shape[0]):
                        self.keypoints_new_good = self.template_points[target.template_scores > self.TEMP_MATCH_THRESH].reshape(-1, 1, 2)
                        target.keypoints_new = target.keypoints_new_good
                        target.centroid_new = self.manager.get_target_centroid(target)

                    self.update_patches(target)

                    # posterity
                    target.keypoints_old = target.keypoints_new
                    target.keypoints_old_good = target.keypoints_new_good.reshape(-1, 1, 2)
                    target.centroid_old = target.centroid_new
                    target.centroid_old_true = self.manager.get_target_centroid(target)
                    target.occlusion_case_old = target.occlusion_case_new
                    target.track_status = self._SUCCESS

            # ################################################################################
            # CASE FROM_TOTAL_OCC
            elif target.occlusion_case_old == TOTAL_OCC:
                # (a priori) older could have been no_occ, partial_occ or total_occ
                # we should have no good keypoints, if old now has total_occ
                # here we are looking for the target again, see if we can spot it again
                # purpose being redetecting target to recover from occlusion
                
                # where do we start, nothing in the old frame to work off of
                # no flow computations, so ask help from oracle or estimator (KF or EKF)
                target.target_bounding_box = self.get_true_bb_from_oracle(target)
                target.target_bounding_box_mask = self.get_bounding_box_mask(self.frame_new_gray, *target.bounding_box)

                # perform template matching for patches to update template points and scores
                self.find_saved_patches_in_img_bb(self.frame_new_gray, target)

                # compute good feature keypoints in the new frame
                # good_keypoints_new = cv.goodFeaturesToTrack(self.frame_new_gray, mask=self.target_bounding_box_mask, **FEATURE_PARAMS)
                good_keypoints_new = self.get_feature_keypoints_from_mask(self.frame_new_gray, mask=target.bounding_box_mask, bb=target.bounding_box)

                if good_keypoints_new is None or good_keypoints_new.shape[0] == 0:
                    good_distances = []
                else:
                    # compute descriptors at the new keypoints
                    kps, descriptors = self.get_descriptors_at_keypoints(self.frame_new_gray, good_keypoints_new, bb=target.bounding_box)

                    # match descriptors 
                    # note, matching only finds best matching/pairing, 
                    # no guarantees of quality of match
                    matches = self.descriptor_matcher.compute_matches(target.initial_target_descriptors, 
                                                                    descriptors, 
                                                                    threshold=-1)

                    distances = np.array([m.distance for m in matches]).reshape(-1, 1)  # redundant TODO clean it

                    # good distances indicate good matches
                    good_distances = distances[distances < self.DES_MATCH_DISTANCE_THRESH]
                    # if (distances < self.DES_MATCH_DISTANCE_THRESH).sum()


                # ---------------------------------------------------------------------
                # |TOTAL_OCC, NO_OCC>
                if (len(good_distances) == MAX_NUM_CORNERS and 
                        (target.template_scores > self.TEMP_MATCH_THRESH).sum()==MAX_NUM_CORNERS):
                    target.occlusion_case_new = NO_OCC

                    good_matches = np.array(matches).reshape(-1, 1)[distances < self.DES_MATCH_DISTANCE_THRESH]
                    target.keypoints_new = np.array([list(good_keypoints_new[gm.trainIdx]) for gm in good_matches.flatten()]).reshape(-1,1,2)
                    target.keypoints_new_good = target.keypoints_new
                    target.centroid_new = self.get_centroid(target.keypoints_new)
                    target.rel_keypoints = target.keypoints_new - target.centroid_new

                    self.update_patches(target)

                    # posterity
                    target.keypoints_old = target.keypoints_new
                    target.initial_keypoints = target.keypoints_old = target.keypoints_new
                    target.initial_centroid = target.centroid_old = target.centroid_new
                    target.centroid_adjustment = None
                    target.centroid_old = target.centroid_new
                    target.centroid_old_true = self.manager.get_target_centroid(target)
                    target.occlusion_case_old = target.occlusion_case_new
                    self.manager.set_target_centroid_offset(target)
                    target.track_status = self._FAILURE

                # ---------------------------------------------------------------------
                # |TOTAL_OCC, TOTAL_OCC>
                elif (len(good_distances) == 0 and 
                        (target.template_scores > self.TEMP_MATCH_THRESH).sum()==0):
                    target.occlusion_case_new = TOTAL_OCC

                    target.centroid_old = target.centroid_old_true
                    target.centroid_new = self.manager.get_target_centroid(target)

                    # cannot compute kinematics
                    target.kinematics = None

                    # posterity
                    target.centroid_old_true = self.manager.get_target_centroid(target)
                    target.occlusion_case_old = target.occlusion_case_new
                    target.track_status = self._FAILURE

                # ---------------------------------------------------------------------
                # |TOTAL_OCC, PARTIAL_OCC>
                else: 
                    target.occlusion_case_new = PARTIAL_OCC

                    target.centroid_old = target.centroid_old_true
                    target.centroid_new = self.manager.get_target_centroid(target)

                    if len(good_distances) > 0:
                        # compute good matches
                        good_matches = np.array(matches).reshape(-1, 1)[distances < self.DES_MATCH_DISTANCE_THRESH]
                        
                        # compute good points, centroid adjustments
                        target.keypoints_new_good = np.array([list(good_keypoints_new[gm.trainIdx]) for gm in good_matches.flatten()]).reshape(-1,1,2)    #NOTE changed queryIdx to trainIdx .. double check later
                    
                    if (target.template_scores > self.TEMP_MATCH_THRESH).sum() > 0:
                        target.keypoints_new_good = target.template_points[target.template_scores > self.TEMP_MATCH_THRESH].reshape(-1, 1, 2)

                    self.update_patches(target)

                    # posterity
                    target.keypoints_old = target.keypoints_new  = target.keypoints_new_good
                    target.keypoints_old_good = target.keypoints_new_good
                    target.centroid_old = target.centroid_new
                    target.centroid_old_true = self.manager.get_target_centroid(target)
                    target.occlusion_case_old = target.occlusion_case_new
                    self.manager.set_target_centroid_offset(target)
                    target.track_status = self._FAILURE

        
        self.display()

        self.frame_old_gray = self.frame_new_gray
        self.frame_old_color = self.frame_new_color


    def compute_flow(self, target, use_good=False):
        # it's main purpose is to compute new points
        # looks at 2 frames, uses flow, tells where old points went
        # make clever use of this function, we want to use good 
        # for from_partial_occ case

        if use_good:
            flow_output = compute_optical_flow_LK(self.frame_old_gray,
                                                self.frame_new_gray,
                                                target.keypoints_old_good.astype('float32'), # good from previous frame
                                                LK_PARAMS)
            # self.keypoints_old_good = flow_output[0]
        else:
            flow_output = compute_optical_flow_LK(self.frame_old_gray,
                                                self.frame_new_gray,
                                                target.keypoints_old.astype('float32'), # good from previous frame
                                                LK_PARAMS)

        # note that new keypoints are going to be of cardinality at most that of old keypoints
        target.keypoints_old = flow_output[0]
        target.keypoints_new = flow_output[1]
        target.feature_found_statuses = flow_output[2]
        target.cross_feature_errors  = flow_output[3]

    def compute_kinematics_by_centroid(self, old_centroid, new_centroid):

        # assumptions:
        # - centroids are computed using get_centroid, therefore, centroid shape (1,2)
        # - centroids represent the target location in old and new frames

        # form pygame.Vector2 objects representing measured car_position and car_velocity 
        # in corner image coord frame in spatial units of *pixels* 
        measured_car_pos = pygame.Vector2(list(new_centroid.flatten()))
        dt = self.manager.get_sim_dt()
        measured_car_vel = pygame.Vector2(list( ((new_centroid-old_centroid)/dt).flatten() ))

        # collect fov and true drone position and velocity from simulator
        fov = self.manager.get_drone_cam_field_of_view()
        true_drone_pos = self.manager.get_true_drone_position()
        true_drone_vel = self.manager.get_true_drone_velocity()

        # transform measured car kinematics from topleft img coord frame to centered world coord frame
        # also, convert spatial units from image pixels to meters
        measured_car_pos_cam_frame_meters = self.manager.transform_pos_corner_img_pixels_to_center_cam_meters(measured_car_pos)
        measured_car_vel_cam_frame_meters = self.manager.transform_vel_img_pixels_to_cam_meters(measured_car_vel)

        #TODO consider handling the filtering part in a separate function
        # filter tracked measurements
        if USE_TRACKER_FILTER:
            if USE_MA:
                if not self.manager.MAF.ready:
                    self.manager.MAF.init_filter(measured_car_pos_cam_frame_meters, measured_car_vel_cam_frame_meters)    
                else:
                    self.manager.MAF.add_pos(measured_car_pos_cam_frame_meters)
                    maf_est_car_pos = self.manager.MAF.get_pos()
                    if dt == 0:
                        maf_est_car_vel = self.manager.MAF.get_vel()
                    else:
                        maf_est_car_vel = (self.manager.MAF.new_pos - self.manager.MAF.old_pos) / dt

                    self.manager.MAF.add_vel(measured_car_vel_cam_frame_meters)
            else:
                maf_est_car_pos = NAN
                maf_est_car_vel = NAN


            if USE_KALMAN:
                if not self.manager.KF.ready:
                    self.manager.KF.init_filter(measured_car_pos_cam_frame_meters, measured_car_vel_cam_frame_meters)
                else:
                    self.manager.KF.add(measured_car_pos_cam_frame_meters, measured_car_vel_cam_frame_meters)
                    kf_est_car_pos = self.manager.KF.get_pos()
                    kf_est_car_vel = self.manager.KF.get_vel()
            else:
                kf_est_car_pos = NAN
                kf_est_car_vel = NAN
        else:
            maf_est_car_pos = NAN
            maf_est_car_vel = NAN
            kf_est_car_pos = NAN
            kf_est_car_vel = NAN


        # return kinematics in camera frame in spatial units of meters
        ret_maf_est_car_vel = NAN if isnan(maf_est_car_vel) else maf_est_car_vel + true_drone_vel
        ret_kf_est_car_vel = NAN if isnan(kf_est_car_vel) else kf_est_car_vel + true_drone_vel

        return (
            true_drone_pos,
            true_drone_vel,
            maf_est_car_pos,
            ret_maf_est_car_vel,
            kf_est_car_pos,
            ret_kf_est_car_vel,
            measured_car_pos_cam_frame_meters,
            measured_car_vel_cam_frame_meters
        )

    def display(self):
        if self.manager.tracker_display_on:
            # add cosmetics to frame_2 for display purpose
            self.frame_color_edited, self.tracker_info_mask = self.add_cosmetics(self.frame_new_color.copy(), 
                                                                                 self.tracker_info_mask)

            # set cur_img; to be used for saving # TODO investigated it's need, used in Simulator to save screen, fix it
            self.cur_img = self.frame_color_edited

            # show resultant img
            cv.imshow(self.win_name, self.frame_color_edited)
        
        # if SHOW_EXTRA:
        #     patches = self.initial_patches_gray if self.patches_gray is None else self.patches_gray
        #     p = images_assemble(patches, grid_shape=(1,len(patches)))
        #     of = self.frame_old_gray.copy()
        #     of[0:p.shape[0],0:p.shape[1]] = convert_to_grayscale(p)
        #     cv.imshow("cur_frame", of)
        #     self.show_me_something()
        #     assembled_img = images_assemble([self.frame_old_gray.copy(), self.nf6.copy(), self.frame_color_edited.copy()], (1,3))
        # else:
        #     # dump frames for analysis
        #     assembled_img = images_assemble([self.frame_old_gray.copy(), self.frame_new_gray.copy(), self.frame_color_edited.copy()], (1,3))
        # self.img_dumper.dump(assembled_img)

        cv.waitKey(1)

    def add_cosmetics(self, frame, mask):
        img = frame
        
        for target in self.targets:
            # est_cents = self.manager.get_estimated_centroids(target)
            # old_pt = (est_cents[0],est_cents[1])
            # new_pt = (est_cents[2],est_cents[3])
            _ARROW_COLOR = self.display_arrow_color[target.occlusion_case_new]
                
            # draw circle and tracks between old and new centroids
            img, mask = draw_tracks(frame, target.centroid_old, target.centroid_new, [TRACK_COLOR], mask, track_thickness=int(2*TRACK_SCALE), radius=int(7*TRACK_SCALE), circle_thickness=int(2*TRACK_SCALE))
            
            if DRAW_KEYPOINT_TRACKS:
                if not DRAW_KEYPOINTS_ONLY_WITHOUT_TRACKS:
                    # draw tracks between old and new keypoints
                    for cur, nxt in zip(target.keypoints_old_good, target.keypoints_new_good):
                        img, mask = draw_tracks(frame, [cur], [nxt], [TURQUOISE_GREEN_BGR], mask, track_thickness=int(1*TRACK_SCALE), radius=int(7*TRACK_SCALE), circle_thickness=int(1*TRACK_SCALE))
                # draw circle for new keypoints
                for nxt in target.keypoints_new_good:
                    img, mask = draw_tracks(frame, None, [nxt], [TURQUOISE_GREEN_BGR], mask, track_thickness=int(1*TRACK_SCALE), radius=int(7*TRACK_SCALE), circle_thickness=int(1*TRACK_SCALE))
                

            # add optical flow arrows
            img = draw_sparse_optical_flow_arrows(img,
                                                  target.centroid_old, # self.get_centroid(good_cur),
                                                  target.centroid_new, # self.get_centroid(good_nxt),
                                                  thickness=int(2*TRACK_SCALE),
                                                  arrow_scale=int(ARROW_SCALE*TRACK_SCALE),
                                                  color=_ARROW_COLOR)

        # add axes
        img = self.add_axes_at_point(img, SCREEN_CENTER)

        # add a drone center
        img = cv.circle(img, SCREEN_CENTER, radius=1, color=DOT_COLOR, thickness=2)

        # put metrics text
        # img = self.put_metrics(img, kin)

        return img, mask

    def add_axes_at_point(self, img, point, size=25, thickness=2):
        x, y = point
        img = cv.arrowedLine(img, (x+1, y), (x+1+size, y), (51, 51, 255), thickness)
        img = cv.arrowedLine(img, (x, y-1 ), (x, y-1-size), (51, 255, 51), thickness)
        return img

    def put_metrics(self, img, k):
        """Helper function, put metrics and stuffs on opencv image.

        Args:
            k (tuple): drone_position, drone_velocity, car_position, car_velocity

        Returns:
            [np.ndarray]: Image after putting all kinds of crap
        """
        if ADD_ALTITUDE_INFO:
            img = put_text(
                img,
                f'Altitude = {self.manager.simulator.camera.altitude:0.2f} m',
                (WIDTH - 175,
                 HEIGHT - 15),
                font_scale=0.5,
                color=METRICS_COLOR,
                thickness=1)
            img = put_text(
                img,
                f'1 pixel = {self.manager.simulator.pxm_fac:0.4f} m',
                (WIDTH - 175,
                 HEIGHT - 40),
                font_scale=0.5,
                color=METRICS_COLOR,
                thickness=1)

        if ADD_METRICS:
            if k is None:
                dpos = self.manager.simulator.camera.position
                dvel = self.manager.simulator.camera.velocity
            kin_str_1 = f'car_pos (m) : '      .rjust(20)
            kin_str_2 = '--' if k is None else f'<{k[6][0]:6.2f}, {k[6][1]:6.2f}>'
            kin_str_3 = f'car_vel (m/s) : '    .rjust(20)
            kin_str_4 = '--' if k is None else f'<{k[7][0]:6.2f}, {k[7][1]:6.2f}>'
            kin_str_5 = f'drone_pos (m) : '    .rjust(20)
            kin_str_6 = f'<{dpos[0]:6.2f}, {dpos[1]:6.2f}>' if k is None else f'<{k[0][0]:6.2f}, {k[0][1]:6.2f}>'
            kin_str_7 = f'drone_vel (m/s) : '  .rjust(20)
            kin_str_8 = f'<{dvel[0]:6.2f}, {dvel[1]:6.2f}>' if k is None else f'<{k[1][0]:6.2f}, {k[1][1]*-1:6.2f}>'
            kin_str_9 = f'drone_acc (m/s^2) : '.rjust(20)
            kin_str_0 = f'<{self.manager.simulator.camera.acceleration[0]:6.2f}, {self.manager.simulator.camera.acceleration[1]:6.2f}>'
            kin_str_11 = f'r (m) : '       .rjust(20)
            kin_str_12 = f'{self.manager.simulator.camera.position.distance_to(self.manager.simulator.car.position):0.4f}'
            kin_str_13 = f'theta (degrees) : '  .rjust(20)
            kin_str_14 = f'{(self.manager.simulator.car.position - self.manager.simulator.camera.position).as_polar()[1]:0.4f}'
            kin_str_15 = f'cam origin : <{self.manager.simulator.camera.origin[0]:6.2f}, {self.manager.simulator.camera.origin[1]:6.2f}>'

            img = put_text(img, kin_str_1, (WIDTH - (330 + 25), 25),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_2, (WIDTH - (155 + 25), 25),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_3, (WIDTH - (328 + 25), 50),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_4, (WIDTH - (155 + 25), 50),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_5, (WIDTH - (332 + 25), 75),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_6, (WIDTH - (155 + 25), 75),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_7, (WIDTH - (330 + 25), 100),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_8, (WIDTH - (155 + 25), 100),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_9, (WIDTH - (340 + 25), 125),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_0, (WIDTH - (155 + 25), 125),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_11, (WIDTH - (323 + 25), 150),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_12, (WIDTH - (155 + 25), 150),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_13, (WIDTH - (323 + 25), 175),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_14, (WIDTH - (155 + 25), 175),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)
            img = put_text(img, kin_str_15, (15, HEIGHT - 15),
                           font_scale=0.45, color=METRICS_COLOR, thickness=1)

        occ_str_dict = {self._NO_OCC:'NO OCCLUSION', self._PARTIAL_OCC:'PARTIAL OCCLUSION', self._TOTAL_OCC:'TOTAL OCCLUSION'}
        occ_color_dict = {self._NO_OCC:EMERALD_BGR, self._PARTIAL_OCC:MARIGOLD_BGR, self._TOTAL_OCC:VERMILION_BGR}
        occ_str = occ_str_dict[self.target_occlusion_case_new]
        occ_str_old = occ_str_dict[self.target_occlusion_case_old]
        occ_color = occ_color_dict[self.target_occlusion_case_new]
        occ_color_old = occ_color_dict[self.target_occlusion_case_old]
        img = put_text(img, occ_str, (WIDTH//2 - 50, HEIGHT - 40),
                           font_scale=0.55, color=occ_color, thickness=1)
        img = put_text(img, occ_str_old, (WIDTH//2 - 30, HEIGHT - 15),
                           font_scale=0.35, color=occ_color_old, thickness=1)
        # if self.target_occlusion_case_new==self._TOTAL_OCC:
            # print(f"Total at {self.manager.simulator.time}")
        return img

    def print_to_console(self):
        if not CLEAN_CONSOLE:
            # NOTE: 
            # drone kinematics are assumed to be known (IMU and/or FPGA optical flow)
            # here, the drone position and velocity is known from Simulator
            # only the car kinematics are tracked/measured by tracker
            # drone_position, drone_velocity, car_position, car_velocity, cp_, cv_ = self.kin
            if self.kin is not None:
                drone_position, drone_velocity, _, _, _, _, car_position, car_velocity= self.kin
                print(f'TTTT >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:{vec_str(drone_position)} | v:{vec_str(drone_velocity)} | CAR - x:{vec_str(car_position)} | v:{vec_str(car_velocity)}')

    def show_me_something(self):
        """Worker function to aid debugging
        """
        # 1 for shi-tomasi, 2 for SIFT, 3 for combination, 4 for template match, 5 for new good flow
        self.nf1 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf2 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf3 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf4 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf5 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)

        # set some colors
        colors = [(16,16,16), (16,16,255), (16,255,16), (255, 16, 16)]

        # compute bounding box
        self.target_bounding_box = self.get_true_bb_from_oracle()
        self.target_bounding_box_mask = self.get_bounding_box_mask(self.frame_new_gray, *self.target_bounding_box)

        # compute shi-tomasi good feature points 
        good_keypoints_new = cv.goodFeaturesToTrack(self.frame_new_gray, 
                                                        mask=self.target_bounding_box_mask, 
                                                        **FEATURE_PARAMS)
        # draw shi-tomasi good feature points
        if good_keypoints_new is not None:
            for i, pt in enumerate(good_keypoints_new): self.nf1 = draw_point(self.nf1, tuple(pt.flatten()), colors[i])
            

        # compute SIFT feature keypoints and draw points and matches
        gkn = self.detector.get_keypoints(self.frame_new_gray, mask=self.target_bounding_box_mask)
        if gkn is not None and len(gkn) > 0:
            self.gpn = np.array([list(gp.pt) for gp in gkn]).reshape(-1,1,2)
            for i, pt in enumerate(self.gpn): self.nf2 = draw_point(self.nf2, tuple(map(int,pt.flatten())))
            self.k,self.d = self.get_descriptors_at_keypoints(self.frame_new_gray, self.gpn, bb=self.target_bounding_box)
            self.mat = self.descriptor_matcher.compute_matches(self.initial_target_descriptors, 
                                                                    self.d, 
                                                                    threshold=-1)
            self.dist = np.array([m.distance for m in self.mat]).reshape(-1, 1)
            for i,m in enumerate(self.mat):
                pt = tuple(map(int,self.gpn[m.trainIdx].flatten()))
                self.nf2 = cv.circle(self.nf2, pt, 5, colors[i], 2)

        # use combination of both shi-tomasi and SIFT keypoints, and draw points and matches
        self.cmb_pts = self.get_feature_keypoints_from_mask(self.frame_new_gray, mask=self.target_bounding_box_mask, bb=self.target_bounding_box)
        if self.cmb_pts is not None and len(self.cmb_pts) > 0:
            for i, pt in enumerate(self.cmb_pts): self.nf3 = draw_point(self.nf3, tuple(map(int,pt.flatten())))
            self.kc,self.dc = self.get_descriptors_at_keypoints(self.frame_new_gray, self.cmb_pts, bb=self.target_bounding_box)
            self.matc = self.descriptor_matcher.compute_matches(self.initial_target_descriptors, 
                                                                    self.dc, 
                                                                    threshold=-1)
            self.distc = np.array([m.distance for m in self.matc]).reshape(-1, 1)
            for i,m in enumerate(self.matc):
                pt = tuple(map(int,self.cmb_pts[m.trainIdx].flatten()))
                self.nf3 = cv.circle(self.nf3, pt, 5, colors[i], 2)
                if m.distance < self.DES_MATCH_DISTANCE_THRESH:
                    self.nf3 = cv.circle(self.nf3, pt, 9, colors[i], 1)
        
        # find patch templates, and draw location points and matches
        self.find_saved_patches_in_img_bb(convert_to_grayscale(self.nf4), self.target_bounding_box)
        for i, t_pt in enumerate(self.template_points):
            pt = tuple(t_pt.flatten())
            self.nf4 = draw_point(self.nf4, pt, colors[i])
            if self.template_scores.flatten()[i] > self.TEMP_MATCH_THRESH:
                self.nf4 = cv.circle(self.nf4, pt, 7, colors[i], 2)

        # draw new good flow points
        if self.keypoints_new is not None:
            for i, pt in enumerate(self.keypoints_new):
                pt = tuple(map(int, pt.flatten()))
                self.nf5 = draw_point(self.nf5, pt, colors[i])
        if self.keypoints_new_good is not None:
            for i, pt in enumerate(self.keypoints_new_good):
                pt = tuple(map(int, pt.flatten()))
                self.nf5 = cv.circle(self.nf5, pt, 7, colors[i], 2)

        p1 = self.get_bb_patch_from_image(self.nf1, self.target_bounding_box)
        p2 = self.get_bb_patch_from_image(self.nf2, self.target_bounding_box)
        p3 = self.get_bb_patch_from_image(self.nf3, self.target_bounding_box)
        p4 = self.get_bb_patch_from_image(self.nf4, self.target_bounding_box)
        p5 = self.get_bb_patch_from_image(self.nf5, self.target_bounding_box)
        self.patches_assembled = images_assemble([p1, p2, p3, p4, p5], grid_shape=(5,1))
        self.nf1[0:self.patches_assembled.shape[0], 0:self.patches_assembled.shape[1]] = self.patches_assembled
        self.nf2[0:self.patches_assembled.shape[0], 0:self.patches_assembled.shape[1]] = self.patches_assembled
        self.nf3[0:self.patches_assembled.shape[0], 0:self.patches_assembled.shape[1]] = self.patches_assembled
        self.nf4[0:self.patches_assembled.shape[0], 0:self.patches_assembled.shape[1]] = self.patches_assembled
        self.nf5[0:self.patches_assembled.shape[0], 0:self.patches_assembled.shape[1]] = self.patches_assembled
        self.nf6 = cv.cvtColor(self.frame_new_gray.copy(), cv.COLOR_GRAY2BGR)
        self.nf6[0:self.patches_assembled.shape[0], 0:self.patches_assembled.shape[1]] = self.patches_assembled

        cv.imshow('nxt_frame', self.nf6); cv.waitKey(1)

