#import boris_feature_align as bfa
import cv2 as cv
import numpy as np
import aux
from aux import FeatureExtraction, feature_matching
from align_images_new import featureAlign_boris, eccAlign_boris, translation_boris

def get_warped(img0,img1,H):
    h, w = img1.shape
    if(H.shape == (2,3)):
        warped = cv.warpAffine(img0, H, (w, h), \
                                    borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    else:
        warped = cv.warpPerspective(img0, H, (w, h), \
                                    borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped

def feature_align_1(img0,img1,params):
    features0 = FeatureExtraction(img0)
    features1 = FeatureExtraction(img1)

    matches = feature_matching(features0, features1,new_lowes_ratio=params['lowes_ratio'], new_min_matches=params['min_matches'])
    matched_image = cv.drawMatches(img0, features0.kps, \
                                   img1, features1.kps, matches, None, flags=2) # visalization

    if(params['save_matched']==True):
        cv.imwrite("matched_image.jpg", matched_image)

    if(params['transformation'] == 'Homography'):
        T, _ = cv.findHomography(features0.matched_pts, \
                                 features1.matched_pts, cv.RANSAC, 4.0)
    else: #affine
        T, _ = cv.estimateAffine2D(features0.matched_pts,features1.matched_pts, method = cv.RANSAC, ransacReprojThreshold = 4)

    warped = get_warped(img0,img1,T)
    return warped

def feature_align_2(img0,img1,params):
    im1Reg, H = featureAlign_boris(img0,img1,max_features = params['max_features'],feature_retention=params['feature_retention'],
                                   save_matches=params['save_matches'],transform=params['transformation'])

    warped = get_warped(img0, img1, H)
    return warped

def ecc_align(img0,img1,params):
    im1_aligned, warp_matrix = eccAlign_boris(img0,img1,number_of_iterations=params['number_of_iterations'],termination_eps=params['termination_eps'],
                                              warp_mode=params['warp_mode'])

    return im1_aligned

def translation_align(im1, im2, params):
    H= translation_boris(im1, im2)
    warped = get_warped(im1, im2, H)
    return warped

def align_two_images(img1,img2,method="feature_align_1",params=None):
    #method: feature_align_1,feature_align_2,ECC,FFT_phase_corr, brute_force
    #return img2 aligned to img1, H

    im1 = cv.imread(img1, cv.COLOR_BGR2RGBA)
    im2 = cv.imread(img2, cv.COLOR_BGR2RGBA)

    if(method == "feature_align_1"):
        img_reg1 = feature_align_1(im1,im2,params)

    if (method == "feature_align_2"):
        img_reg1 = feature_align_2(im1, im2, params)

    if (method == "ECC"): #NOT good
        img_reg1 = ecc_align(im1, im2, params)

    if (method == "translation"):
        img_reg1 = translation_align(im1, im2, params)

    return img_reg1



if __name__ == "__main__":
    img1 = "/home/borisef/projects/align_images/im1gray.jpg" #source will be transformed
    img2 = "/home/borisef/projects/align_images/im2gray.jpg" #target

    params_fa_1 ={'save_matched': True,
                  'lowes_ratio': 0.8, #0.7
                  'min_matches': 15,#50
                  'transformation': 'Homography', #or Affine or Homography
                  'roi_4_features': None # only select features n ROI #TODO
                  }
    params_fa_2 = {'max_features' : 1000,'feature_retention': 0.1,'save_matches':True,
                   'transformation': 'Affine', #or Affine or Homography
                   }

    params_ecc = {'number_of_iterations': 100,'termination_eps': 1e-7, 'warp_mode':cv.MOTION_AFFINE}

    params_translation = None

    if (1):
        img21 = align_two_images(img1,img2,method="feature_align_1",params=params_fa_1)
        cv.imwrite("output_image_fa1_homo.jpg", img21)
    if (0):
        img21 = align_two_images(img1, img2, method="feature_align_2", params=params_fa_2)
        cv.imwrite("output_image_fa2_affine.jpg", img21)
    if (0):
        img21 = align_two_images(img1, img2, method="ECC", params=params_ecc)
        cv.imwrite("output_image_ecc.jpg", img21)
    if (0):
        # img1 = "/home/borisef/projects/align_images/im1gray.jpg"  # source will be transformed
        # img2 = "/home/borisef/projects/align_images/im1gray.jpg"  # target
        img21 = align_two_images(img1, img2, method="translation", params=params_translation)
        cv.imwrite("output_image_trans.jpg", img21)



