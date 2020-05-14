"""
IvLabs, VNIT
MONOCULAR VISUAL ODOMETRY ON KITTI DATASET

TEAM MEMBERS:
1. Arihant Gaur
2. Saurabh Kemekar
3. Aman Jain
"""


import cv2
import numpy as np

def feature_match(kp1,kp2,desc1,decs2):
    bf = cv2.BFMatcher()   # default parameter

    matches = bf.knnMatch(desc1,decs2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 *n.distance:
            good.append([m])

    pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return pts1,pts2

def KLT(image1, image2, trackpoints1):

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, trackpoints1, None, **lk_params)
    pttrackable = np.where(st == 1 )
    trackpoints1_KLT = trackpoints1[pttrackable[0]]
    trackpoints2_KLT = trackpoints2[pttrackable[0]]
    return trackpoints1_KLT, trackpoints2_KLT


def orb_features(img1,img2):

    orb = cv2.ORB_create()

    kp1,desc1 = orb.detectAndCompute(img1,None)
    kp2,desc2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck= True)


    pts1,pts2 = feature_match(kp1,kp2,desc1,desc2)

    return pts1,pts2

def fast_features(img1,img2):
    fast = cv2.FastFeatureDetector_create()
    kp1  = fast.detect(img1,None)

    kp1 = np.array([kp1[i].pt for i in range(len(kp1))],np.float32)
    pts1, pts2 = KLT(img1,img2,kp1)

    return pts1,pts2
