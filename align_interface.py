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
    features0 = FeatureExtraction(img0,roi = params['bbox1'])
    features1 = FeatureExtraction(img1, roi = params['bbox2'])


    matches = feature_matching(features0, features1,new_lowes_ratio=params['lowes_ratio'], new_min_matches=params['min_matches'])
    matched_image = cv.drawMatches(img0, features0.kps, \
                                   img1, features1.kps, matches, None, flags=2) # visalization

    if(params['save_matched']==True):
        cv.imwrite("data/outputs/matched_image.jpg", matched_image)

    if(params['transformation'] == 'Homography'):
        T, _ = cv.findHomography(features0.matched_pts, \
                                 features1.matched_pts, cv.RANSAC, 4.0)
    else: #affine
        T, _ = cv.estimateAffine2D(features0.matched_pts,features1.matched_pts, method = cv.RANSAC, ransacReprojThreshold = 4)

    warped = get_warped(img0,img1,T)
    return warped

def feature_align_2(img0,img1,params):
    im1Reg, H = featureAlign_boris(img0,img1,
                                   max_features = params['max_features'],
                                   save_matches=params['save_matches'],transform=params['transformation'],
                                   roi1=params['bbox1'], roi2=params['bbox2'],
                                   min_matches=params['min_matches'],
                                   max_matches=params['max_matches'])

    warped = get_warped(img0, img1, H)
    return warped

def ecc_align(img0,img1,params):
    im1_aligned, warp_matrix = eccAlign_boris(img0,img1,number_of_iterations=params['number_of_iterations'],termination_eps=params['termination_eps'],
                                              warp_mode=params['warp_mode'], bbox1=params['bbox1'],bbox2=params['bbox2'])

    return im1_aligned

def translation_align(im1, im2, params):
    H= translation_boris(im1, im2, params['bbox1'], params['bbox2'])
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
    img1 = "/home/borisef/projects/align_images/data/im1gray.jpg" #source will be transformed
    img2 = "/home/borisef/projects/align_images/data/im2gray.jpg" #target

    bbox1 = [350,160,650,600] # (x1,y1,x2,y2)
    bbox2 = [275, 150, 600, 625]  # (x1,y1,x2,y2)


    img1 = "/home/borisef/projects/align_images/data/767.jpg"
    img2 = "/home/borisef/projects/align_images/data/768.jpg"

    bbox1 = [200, 200, 300, 300]  # (x1,y1,x2,y2)
    bbox2 = [200, 200, 350, 350]  # (x1,y1,x2,y2)

    params_fa_1 ={'save_matched': True,
                  'lowes_ratio': 1.1, #0.7 # larger number => more points
                  'min_matches': 5,#50
                  'transformation': 'Affine', #or Affine or Homography
                  'bbox1': bbox1, #roi1 (x1,y1,x2,y2)
                  'bbox2': bbox2 #roi2  (x1,y1,x2,y2)
                  }
    params_fa_2 = {'max_features' : 1000,
                   #'feature_retention': 0.9, # larger number more matches
                   'min_matches': 5,  # 50
                   'max_matches': 100,  #50
                   'save_matches':True,
                   'transformation': 'Homography',  #or Affine or Homography
                   'bbox1': bbox1,  # roi1 (x1,y1,x2,y2)
                   'bbox2': bbox2  # roi2  (x1,y1,x2,y2)
                   }

    params_ecc = {'number_of_iterations': 1000,
                  'termination_eps': 1e-8,
                  'warp_mode':cv.MOTION_AFFINE,
                  'bbox1': bbox1,  # roi1 (x1,y1,x2,y2)
                  'bbox2': bbox2  # roi2  (x1,y1,x2,y2)
                  }

    params_translation = {
                  'bbox1': bbox1,  # roi1 (x1,y1,x2,y2)
                  'bbox2': bbox2  # roi2  (x1,y1,x2,y2)
                  }

    if (0): #OK
        img21 = align_two_images(img1,img2,method="feature_align_1",params=params_fa_1)
        outname = "data/outputs/output_image_fa1_" + params_fa_1["transformation"] + ".png"
        cv.imwrite(outname, img21)
    if (1): #OK
        img21 = align_two_images(img1, img2, method="feature_align_2", params=params_fa_2)
        outname = "data/outputs/output_image_fa2_" + params_fa_2["transformation"] + ".png"
        cv.imwrite(outname, img21)
    if (0): #bad results
        img21 = align_two_images(img1, img2, method="ECC", params=params_ecc)
        cv.imwrite("data/outputs/output_image_ecc.jpg", img21)
    if (0):
        img21 = align_two_images(img1, img2, method="translation", params=params_translation)
        cv.imwrite("data/outputs/output_image_trans.jpg", img21)



