#import boris_feature_align as bfa
import cv2 as cv
import numpy as np
import aux
from aux import FeatureExtraction, feature_matching
from align_images_new import featureAlign_boris, eccAlign_boris

def get_warped(img0,img1,H):
    h, w = img1.shape
    warped = cv.warpPerspective(img0, H, (w, h), \
                                borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped

def feature_align_1(img0,img1,params):
    features0 = FeatureExtraction(img0)
    features1 = FeatureExtraction(img1)

    matches = feature_matching(features0, features1,new_lowes_ratio=params['lowes_ratio'], new_min_matches=params['min_matches'])
    matched_image = cv.drawMatches(img0, features0.kps, \
                                   img1, features1.kps, matches, None, flags=2)

    if(params['save_matched']==True):
        cv.imwrite("matched_image.jpg", matched_image)

    H, _ = cv.findHomography(features0.matched_pts, \
                             features1.matched_pts, cv.RANSAC, 5.0)

    warped = get_warped(img0,img1,H)
    return warped

def feature_align_2(img0,img1,params):
    im1Reg, H = featureAlign_boris(img0,img1,max_features = params['max_features'],feature_retention=params['feature_retention'],save_matches=params['save_matches'])

    warped = get_warped(img0, img1, H)
    return warped

def ecc_align(img0,img1,params):
    im2_aligned, warp_matrix = eccAlign_boris(img0,img1,number_of_iterations=params['number_of_iterations'],termination_eps=params['termination_eps'],
                                              warp_mode=cv.MOTION_EUCLIDEAN)

    return im2_aligned


def align_two_images(img1,img2,method="feature_align_1",params=None):
    #method: feature_align_1,feature_align_2,ECC,FFT_phase_corr, brute_force
    #return img2 aligned to img1, H

    im1 = cv.imread(img1, cv.COLOR_BGR2RGBA)
    im2 = cv.imread(img2, cv.COLOR_BGR2RGBA)

    if(method == "feature_align_1"):
        img_reg2 = feature_align_1(im1,im2,params)

    if (method == "feature_align_2"):
        img_reg2 = feature_align_2(im1, im2, params)

    if (method == "ECC"):
        img_reg2 = ecc_align(im2, im1, params)

    return img_reg2



if __name__ == "__main__":
    img1 = "/home/borisef/projects/align_images/nir_image.jpg"
    img2 = "/home/borisef/projects/align_images/rgb_image.jpg"

    params_fa_1 ={'save_matched': True,
                  'lowes_ratio': 0.8, #0.7
                  'min_matches': 15,#50
                  }
    params_fa_2 = {'max_features' : 1000,'feature_retention': 0.1,'save_matches':True}

    params_ecc = {'number_of_iterations': 1000,'termination_eps': 1e-8, 'warp_mode':cv.MOTION_EUCLIDEAN}

    #img21 = align_two_images(img1,img2,method="feature_align_1",params=params_fa_1)
    #img21 = align_two_images(img1, img2, method="feature_align_2", params=params_fa_2)
    img21 = align_two_images(img1, img2, method="ECC", params=params_ecc)
    cv.imwrite("output_image_21ecc.jpg", img21)



