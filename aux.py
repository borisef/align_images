import cv2 as cv
import numpy as np
import copy


orb = cv.ORB_create(
    nfeatures=10000,
    scaleFactor=1.2,
    scoreType=cv.ORB_HARRIS_SCORE)


def add_noise(img,w):
    row = img.shape[0]
    col = img.shape[1]
    # noise
    gaussian = np.random.random((row, col)).astype(np.float32)
    gaussian = gaussian * 255
    gaussian_img = img * (1 - w) + w * gaussian
    img = gaussian_img
    img = img.astype('uint8')
    return img

class FeatureExtraction:
    def __init__(self, img, roi = None):
        self.img = copy.copy(img)
        if(len(img.shape)==3):
            self.gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            self.gray_img = img
        if(roi is None):
            mask = None
        else:
            mask = np.zeros_like(img)
            x1,y1,x2,y2 = roi
            mask[y1:y2, x1:x2] = 1

        self.kps, self.des = orb.detectAndCompute(self.gray_img, mask = mask)

        if(roi is not None): #TODO filter ot features not in ROI see kps pt
            self.img_kps = cv.drawKeypoints( \
            self.img, self.kps, 0, \
            flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
            cv.imwrite("data/outputs/aaa.png",self.img)

        self.matched_pts = []


LOWES_RATIO = 0.8 # 0.7
MIN_MATCHES = 10 #50
index_params = dict(
    algorithm = 6, # FLANN_INDEX_LSH
    table_number = 6,
    key_size = 10,
    multi_probe_level = 2)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(
    index_params,
    search_params)

def feature_matching(features0, features1, new_lowes_ratio = None, new_min_matches = None):
    if(new_lowes_ratio is None):
        new_lowes_ratio = LOWES_RATIO
    if(new_min_matches is None):
        new_min_matches = MIN_MATCHES
    matches = [] # good matches as per Lowe's ratio test
    if(features0.des is not None and len(features0.des) > 2):
        all_matches = flann.knnMatch( \
            features0.des, features1.des, k=2)
        try:
            for m,n in all_matches:
                if m.distance < new_lowes_ratio * n.distance:
                    matches.append(m)
        except ValueError:
            pass
        if(len(matches) > new_min_matches):
            features0.matched_pts = np.float32( \
                [ features0.kps[m.queryIdx].pt for m in matches ] \
                ).reshape(-1,1,2)
            features1.matched_pts = np.float32( \
                [ features1.kps[m.trainIdx].pt for m in matches ] \
                ).reshape(-1,1,2)
    return matches