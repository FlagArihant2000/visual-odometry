"""
MONOCULAR VISUAL ODOMETRY ON KITTI DATASET

TEAM MEMBERS:
1. Arihant Gaur
2. Saurabh Kemekar
3. Aman Jain
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# PARAMETERS THAT CAN BE CHANGED
ImgLoc = '/home/arihant/visod/src/KITTI_sample/images/' # Images Location
GTLoc = '/home/arihant/visod/src/KITTI_sample/poses.txt' # Ground truth location file. Write None if you don't have it.
totImages = 151
FeatureDetect = 'SIFT' # FEATURE DETECTION METHOD ('FAST', 'SIFT', 'SURF', 'SHI-TOMASI')
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)) # Lucas Kanade Parameters for Optical Flow
matchDiff = 1 # Minimum distance in KLT point correspondence
pixDiffThresh = 3 # Skip frame if pixel difference returned from KLT is less than the threshold.
featureThresh = 1000 # Minimum number of features required per frame if the pixel difference goes below the threshold

K = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02], [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02], [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]]) # Camera Matrix of the form np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]).

OutlierRejection = False # Outlier Removal


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

def FeatureTracking(img0, img1, reference):
	kp1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, reference, None, **lk_params)
	kp0, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, kp1, None, **lk_params) # Backtracking to acquire good features
	
	d = abs(reference - kp0).reshape(-1, 2).max(-1) # Difference between feature points
	good = d < matchDiff
	if list(good).count(True) <= 5:
		return kp0, kp1, 0
	
	n_kp0 = []
	n_kp1 = []
	for i, good_flag in enumerate(good):
		if good_flag:
			n_kp0.append(kp0[i])
			n_kp1.append(kp1[i])
	n_kp0 = np.array(n_kp0, dtype = np.float32)
	n_kp1 = np.array(n_kp1, dtype = np.float32)
	diff = np.mean(abs(n_kp0 - n_kp1).reshape(-1, 2).max(-1))
	
	return kp0, kp1, diff
	
def Triangulation(R, t, kp0, kp1, K):
	P0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
	P0 = K.dot(P0)
	P1 = np.hstack((R, t))
	P1 = K.dot(P1)
	points1 = kp0.reshape(2, -1)
	points2 = kp1.reshape(2, -1)
	cloud = cv2.triangulatePoints(P0, P1, points1, points2).reshape(-1, 4)[:, :3]
	return cloud

def AbsoluteScale(groundTruth, i):
	T_mat = groundTruth[i-1].strip().split()
	x_p = float(T_mat[3])
	y_p = float(T_mat[7])
	z_p = float(T_mat[11])
	
	T_mat = groundTruth[i].strip().split()
	x = float(T_mat[3])
	y = float(T_mat[7])
	z = float(T_mat[11])
	
	Enorm = np.sqrt((x - x_p)**2 + (y - y_p)**2 + (z - z_p)**2)
	return Enorm
	
def RelativeScale(last_cloud, new_cloud):
	min_idx = min([new_cloud.shape[0], last_cloud.shape[0]])
	ratios = []  # List to obtain all the ratios of the distances
	for i in range(min_idx):
		if i > 0:
			Xk = new_cloud[i]
			p_Xk = new_cloud[i - 1]
			Xk_1 = last_cloud[i]
			p_Xk_1 = last_cloud[i - 1]

			if np.linalg.norm(p_Xk - Xk) != 0:
				ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))

	d_ratio = np.median(ratios) # Take the median of ratios list as the final ratio
	return d_ratio	
	
t = []
R = []

# Plotting values for absolute scale
t.append(tuple([[0], [0], [0]]))
R.append(tuple(np.zeros((3,3))))

i = 1
if GTLoc is not None:
	with open(GTLoc) as f:
		groundTruthValues = f.readlines()
		
		
while(1):
	img0 = cv2.imread(ImgLoc+str(i)+'.png') # First frame acquisition
	img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

	# FEATURE DETECTION
	kp0 = FeatureDetection(img0gray, FeatureDetect)	
	
	img1 = cv2.imread(ImgLoc+str(i+1)+'.png') # Second frame acquisition
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

	# FEATURE TRACKING
	kp0, kp1, diff = FeatureTracking(img0gray, img1gray, kp0)
	if diff < pixDiffThresh: # If pixel difference is not sufficient (almost no motion)
		i = i + 1
	else:
		i = i + 2
		break

# Essential matrix calculation
E, mask = cv2.findEssentialMat(kp1, kp0, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
if OutlierRejection:
	indx = np.where(mask == 1)
	kp0 = kp0[indx[0]]
	kp1 = kp1[indx[0]]

# Pose recovery	
_, R0, t0, mask = cv2.recoverPose(E, kp0, kp1, K)
t_curr = R0.dot(t0)
R_curr = R0
t.append(tuple(t_curr))
R.append(tuple(R_curr))

# Triangulation
Xnew = Triangulation(R0, t0, kp0, kp1, K)
#print(Xnew)
canvas = np.zeros((500,500,3), dtype = np.uint8)
while(i <= 151):
	#print(i)
	Ti = time.time()
	img0 = img1
	img0gray = img1gray
	Xold = Xnew
	kp0 = kp1
	
	img1 = cv2.imread(ImgLoc+str(i)+'.png') # Image acquisition
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	
	# Feature tracking
	kp0, kp1, diff = FeatureTracking(img0gray, img1gray, kp0)
	print(diff, kp0.shape[0])
	if diff < pixDiffThresh:
		if kp0.shape[0] < featureThresh:
			kp1 = FeatureDetection(img0gray, FeatureDetect)
			i = i + 1
			continue
	# Pose recovery
	E, mask = cv2.findEssentialMat(kp1, kp0, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
	if OutlierRejection:
		indx = np.where(mask == 1)
		kp0 = kp0[indx[0]]
		kp1 = kp1[indx[0]]
	_, R0, t0, mask = cv2.recoverPose(E, kp0, kp1, K)

	# Triangulation
	Xnew = Triangulation(R0, t0, kp0, kp1, K)
	#print(t_currR)
	if GTLoc is not None:
		scale1 = AbsoluteScale(groundTruthValues, i-1)
		t_curr = t_curr + scale1 * R_curr.dot(t0)
		R_curr = R0.dot(R_curr) 
		#tA.append(tuple(t_currA))
		#RA.append(tuple(R_currA))
	else:
		scale2 = RelativeScale(Xold, Xnew)
		t_curr = t_curr + scale1 * R_curr.dot(t0)
		R_curr = R0.dot(R_curr)			
		
	#scale2 = RelativeScale(Xold, Xnew)
	#t_currR = t_currR + scale2 * R_currR.dot(t0)
	#R_currR = R0.dot(R_currR) 
	#tR.append(tuple(t_currR))
	#RR.append(tuple(R_currR))
	#print(R0, t0)
	if kp0.shape[0] < featureThresh:
		kp1 = FeatureDetection(img1gray, FeatureDetect)
	#cv2.circle(canvas, (abs(t_currR[0]) + 200, abs(t_currR[2]) + 200), 2, (0,255,0), -1)
	cv2.circle(canvas, (abs(t_curr[0]) + 200, abs(t_curr[2]) + 200), 2, (0,0,255), -1)
	cv2.imshow('frame',img1)
	cv2.imshow('canvas',canvas)
	if cv2.waitKey(1) % 0xff == ord('q'):
		break
	i = i + 1
	Tf = time.time()
	#print(1/(Tf - Ti))

cv2.destroyAllWindows()
cv2.imshow('canvas',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
	
	
	
	
