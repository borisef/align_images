#!/usr/bin/env python

# Import necessary libraries.
import os, argparse
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def legal_bbox(bbox, im):
    x1, y1, x2, y2 = bbox
    H = im.shape[0]
    W = im.shape[1]
    if(x1<0 or y1 <0 or x2 >=W or y2 >=H):
        return False
    else:
        return True


def same_size_bboxes(bbox0,bbox1, im0,im1):
    if(bbox0 is None or bbox1 is None):
        return (bbox0, bbox1)

    x1_0, y1_0, x2_0, y2_0 = bbox0
    x1_1, y1_1, x2_1, y2_1 = bbox1

    lenX_0 = x2_0 - x1_0
    lenX_1 = x2_1 - x1_1
    lenY_0 = y2_0 - y1_0
    lenY_1 = y2_1 - y1_1


    centerX_0 = int((x1_0 + x2_0)*0.5)
    centerX_1 = int((x1_1 + x2_1) * 0.5)
    centerY_0 = int((y1_0 + y2_0) * 0.5)
    centerY_1 = int((y1_1 + y2_1) * 0.5)

    hlenX = int(max(lenX_0, lenX_1)/2)
    hlenY = int(max(lenY_0, lenY_1)/2)

    #extend around center
    x1_0 = centerX_0 - hlenX
    x2_0 = centerX_0 + hlenX
    x1_1 = centerX_1 - hlenX
    x2_1 = centerX_1 + hlenX
    y1_0 = centerY_0 - hlenY
    y2_0 = centerY_0 + hlenY
    y1_1 = centerY_1 - hlenY
    y2_1 = centerY_1 + hlenY


    # x2_0 = x2_0 - lenX_0 + max(lenX_0, lenX_1)
    # x2_1 = x2_1 - lenX_1 + max(lenX_0, lenX_1)
    # y2_0 = y2_0 - lenY_0 + max(lenY_0, lenY_1)
    # y2_1 = y2_1 - lenY_1 + max(lenY_0, lenY_1)

    bbox0 = [x1_0, y1_0, x2_0, y2_0 ]
    bbox1 = [x1_1, y1_1, x2_1, y2_1]

    if(not legal_bbox(bbox0, im0)):
        return (None, None)
    if (not legal_bbox(bbox1, im1)):
        return (None, None)

    return (bbox0,bbox1)



def cut_bbox(im,bbox):
    x1, y1, x2, y2 = bbox
    im1 = im[y1:y2,x1:x2].copy()
    return im1

def get_mask(img,roi):
    if (roi is None):
        mask = None
    else:
        mask = np.zeros_like(img)
        x1, y1, x2, y2 = roi
        mask[y1:y2, x1:x2] = 1
    return  mask




# argument parser
def getArgs():

   parser = argparse.ArgumentParser(
    description = '''Demo script showing various image alignment methods
                     including, phase correlation, feature based matching
                     and whole image based optimization.''',
    epilog = '''post bug reports to the github repository''')
    
   parser.add_argument('-im1',
                       '--image_1',
                       help = 'image to reference',
                       required = True)
                       
   parser.add_argument('-im2',
                       '--image_2',
                       help = 'image to match',
                       required = True)

   parser.add_argument('-m',
                       '--mode',
                       help = 'registation mode: translation, ecc or feature',
                       default = 'feature')

   parser.add_argument('-mf',
                       '--max_features',
                       help = 'maximum number of features to consider',
                       default = 5000)

   parser.add_argument('-fr',
                       '--feature_retention',
                       help = 'fraction of features to retain',
                       default = 0.15)

   parser.add_argument('-i',
                       '--iterations',
                       help = 'number of ecc iterations',
                       default = 5000)

   parser.add_argument('-te',
                       '--termination_eps',
                       help = 'ecc termination value',
                       default = 1e-8)

   return parser.parse_args()

def rotationAlign(im1, im2):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_red = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    height, width = im1_gray.shape[0:2]
    
    values = np.ones(360)
    
    for i in range(0,360):
      rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), i, 1)
      rot = cv2.warpAffine(im2_red, rotationMatrix, (width, height))
      values[i] = np.mean(im1_gray - rot)
    
    rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), np.argmin(values), 1)
    rotated = cv2.warpAffine(im2, rotationMatrix, (width, height))
    
    return rotated, rotationMatrix
      

# Enhanced Correlation Coefficient (ECC) Maximization
def eccAlign(im1,im2):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
     number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned, warp_matrix

# (ORB) feature based alignment      
def featureAlign(im1, im2):
  
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(max_features)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  list(matches).sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * feature_retention)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  #cv2.imwrite("matches.jpg", imMatches)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  return im1Reg, h

def eccAlign_boris(im1, im2, number_of_iterations = 1000, termination_eps = 1e-8, warp_mode = cv2.MOTION_EUCLIDEAN, bbox1 = None, bbox2 = None):

    # Convert images to grayscale
    im1_gray = im1
    im2_gray = im2

    # Find size of image1
    sz = im1.shape

    # Define the motion model


    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
     number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    mask2 = np.ones_like(im2_gray)
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray, im2_gray, warp_matrix, warp_mode, criteria, inputMask = mask2)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im1, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im1, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned, warp_matrix

def featureAlign_boris(im1Gray, im2Gray,max_features = 1000, min_matches = 5, max_matches = 50, save_matches = True,
                       transform = "Homography",
                       roi1 = None, roi2 = None ):

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)

    mask1 = get_mask(im1Gray,roi1)
    mask2 = get_mask(im2Gray, roi2)


    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, mask = mask1)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, mask = mask2)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1) #DESCRIPTOR_MATCHER_BRUTEFORCE_L1, DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    list(matches).sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    if(len(matches) > max_matches):
        matches = matches[:max_matches]
    if (len(matches) < min_matches):
        print("ERROR len(matches) < min_matches")

    # Draw top matches
    imMatches = cv2.drawMatches(im1Gray, keypoints1, im2Gray, keypoints2, matches, None)
    if(save_matches):
     cv2.imwrite("data/outputs/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    height, width = im2Gray.shape

    if(transform == "Homography"):
        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

         # Use homography

        im1Reg = cv2.warpPerspective(im1Gray, h, (width, height))
    else:

        h, _ = cv2.estimateAffine2D(points1, points2, method=cv2.RANSAC,
                                   ransacReprojThreshold=4)
        im1Reg = cv2.warpAffine(im1Gray, h, (width, height))

    return im1Reg, h


# FFT phase correlation
def translation(im0, im1):
    
    # Convert images to grayscale
    im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]

def translation_boris(im0, im1, bbox0 = None, bbox1 = None):
    # Convert images to grayscale
    # im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    bbox0, bbox1 = same_size_bboxes(bbox0, bbox1, im0, im1)
    if((bbox0 is not None) and (bbox1 is not None) ):
        tx_init = bbox0[0]-bbox1[0]
        ty_init = bbox0[1] - bbox1[1]
    else:
        tx_init = ty_init = 0

    if(bbox0 is not None):
        im0 = cut_bbox(im0,bbox0)
    if (bbox1 is not None):
        im1 = cut_bbox(im1, bbox1)

    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]

    H = np.eye(3, 3, dtype=np.float32)
    H[0,2] = -t1-tx_init
    H[1,2] = -t0-ty_init


    return H


if __name__ == '__main__':

  mode = "feature"#, "ecc", "rotation"
  image_1 = "/home/borisef/projects/align_images/data/767.jpg"
  image_2 = "/home/borisef/projects/align_images/data/768.jpg"

  # parse arguments
  #args = getArgs()
    
  # defaults feature values
  max_features = 1000
  feature_retention = 0.15
  
  # Specify the ECC number of iterations.
  number_of_iterations = 1000

  # Specify the ECC threshold of the increment
  # in the correlation coefficient between two iterations
  termination_eps = 1e-8

  # Read the images to be aligned
  im1 =  cv2.imread(image_1);
  im2 =  cv2.imread(image_2);

  # Switch between alignment modes
  if mode == "feature":
   # align and write to disk
   aligned, warp_matrix = featureAlign(im1, im2)
   cv2.imwrite("reg_image.jpg",
    aligned,
    [cv2.IMWRITE_JPEG_QUALITY, 90])
   print(warp_matrix)
  elif mode == "ecc":
   aligned, warp_matrix = eccAlign(im1, im2)
   cv2.imwrite("reg_image.jpg",
    aligned,
    [cv2.IMWRITE_JPEG_QUALITY, 90])
   print(warp_matrix)
  elif mode == "rotation":
   rotated, rotationMatrix = rotationAlign(im1, im2)
   cv2.imwrite("reg_image.jpg",
    rotated,
    [cv2.IMWRITE_JPEG_QUALITY, 90])
   print(rotationMatrix)
  else:
   warp_matrix = translation(im1, im2)
   print(warp_matrix)
