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

matchDiff = 1 # Minimum distance in KLT point correspondence
pixDiffThresh = 3 # Skip frame if pixel difference returned from KLT is less than the threshold.
featureThresh = 1000 # Minimum number of features required per frame if the pixel difference goes below the threshold


def FeatureDetection(img0gray, FeatureDetect):
	if FeatureDetect == 'FAST':
		featuredetect = cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True)
		kp0 = featuredetect.detect(img0gray)
		kp0 = np.array([kp0[idx].pt for idx in range(len(kp0))], dtype = np.float32)
	elif FeatureDetect == 'SIFT':
		featuredetect = cv2.xfeatures2d.SIFT_create()
		kp0, des0 = featuredetect.detectAndCompute(img0gray, None)
		kp0 = np.array([kp0[idx].pt for idx in range(len(kp0))], dtype = np.float32)
	elif FeatureDetect == 'SURF':
		featuredetect = cv2.xfeatures2d.SURF_create()
		kp0, des0 = featuredetect.detectAndCompute(img0gray, None)
		kp0 = np.array([kp0[idx].pt for idx in range(len(kp0))], dtype = np.float32)		
	return kp0
	
def FeatureTracking(image_ref, image_cur, px_ref):
	lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
	
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
	kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, **lk_params)

	d = abs(px_ref - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
	good = d < matchDiff  # Verify which features produced good results by the difference being less than the threshold

	if len(d) == 0:
		print('Error: No matches where made.')
	elif list(good).count(True) <= 5:  # If less than 5 good points, then the backtracked points are not used.
		print('Warning: No match was good. Returns the list without good point correspondence.')
		return kp1, kp2, 0

	# Considering good features
	n_kp1, n_kp2 = [], []
	for i, good_flag in enumerate(good):
		if good_flag:
			n_kp1.append(kp1[i])
			n_kp2.append(kp2[i])


	n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(n_kp2, dtype=np.float32)

	# Checks the movement between the pixel correspondences.
	d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

	# Returning the mean of all the movements
	diff_mean = np.mean(d)

	return n_kp1, n_kp2, diff_mean
	

