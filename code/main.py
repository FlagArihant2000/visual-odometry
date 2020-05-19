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
import matplotlib.pyplot as plt
import time
import os
from  Visualization import *
from features import *

# PARAMETERS THAT CAN BE CHANGED
#ImgLoc = '/home/saurabh/Downloads/visual-odometry-master/KITTI_sample/images/'
ImgLoc = '/KITTI_sample/images/'
 # Images Location
GTLoc = False
totImages = len(os.listdir(ImgLoc))
FeatureDetect = 'FAST' # FEATURE DETECTION METHOD ('FAST', 'SIFT' and 'SURF')
 # Lucas Kanade Parameters for Optical Flow


K = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02], 
	      [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02], 
	      [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
	      


def Triangulation(R, t, kp0, kp1, K):
	P0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
	P0 = K.dot(P0)
	P1 = np.hstack((R, t))
	P1 = K.dot(P1)
	points1 = kp0.reshape(2, -1)
	points2 = kp1.reshape(2, -1)
	cloud = cv2.triangulatePoints(P0, P1, points1, points2).reshape(-1, 4)[:, :3]
	return cloud

def AbsoluteScale(groundTruth, last_id,curr_id):
    x_prev = ground_truth[last_id, 3]
    y_prev = ground_truth[last_id, 7]
    z_prev = ground_truth[last_id, 11]

    x = ground_truth[curr_id, 3]
    y = ground_truth[curr_id, 7]
    z = ground_truth[curr_id, 11]
    Enorm = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
    return Enorm
	
def RelativeScale(last_cloud, new_cloud):
	min_idx = min([new_cloud.shape[0],last_cloud.shape[0]])
	p_Xk = new_cloud[:min_idx]
	Xk = np.roll(p_Xk,shift = -3)
	p_Xk_1 = last_cloud[:min_idx]
	Xk_1 = np.roll(p_Xk_1,shift = -3)
	d_ratio = (np.linalg.norm(p_Xk_1 - Xk_1,axis = -1))/(np.linalg.norm(p_Xk - Xk,axis = -1))

	return np.median(d_ratio)
	
t = []
R = []
clahe = cv2.createCLAHE(clipLimit=5.0)
# Plotting values for absolute scale
t.append(tuple([[0], [0], [0]]))
R.append(tuple(np.zeros((3,3))))

i = 1
if GTLoc:
	#ground_truth = np.loadtxt('/home/saurabh/Downloads/visual-odometry-master/KITTI_sample/poses.txt')
	ground_truth = np.loadtxt('/KITTI_sample/poses.txt')	
		
while(1):
	img0 = cv2.imread(ImgLoc+str(i)+'.png') # First frame acquisition
	img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
#	img0gray = clahe.apply(img0gray)
	# FEATURE DETECTION
	kp0 = FeatureDetection(img0gray, FeatureDetect)
	print(len(kp0))	
	
	img1 = cv2.imread(ImgLoc+str(i+1)+'.png') # Second frame acquisition
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#	img1gray = clahe.apply(img1gray)
	# FEATURE TRACKING
	kp0, kp1, diff = FeatureTracking(img0gray, img1gray, kp0)
	print(diff)
	if diff < pixDiffThresh: # If pixel difference is not sufficient (almost no motion)
		i = i + 1
	else:
		i = i + 2
		break

# Essential matrix calculation
E, mask = cv2.findEssentialMat(kp1, kp0, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
kp0 = kp0[mask.ravel() ==1]
kp1 = kp1[mask.ravel() ==1]

# Pose recovery	
_, R0, t0, mask = cv2.recoverPose(E, kp0, kp1, K)
t_curr = R0.dot(t0)
R_curr = R0
t.append(tuple(t_curr))
R.append(tuple(R_curr))

# Triangulation
Xnew = Triangulation(R0, t0, kp0, kp1, K)
#print(Xnew)
canvas = np.zeros((700,1000,3), dtype = np.uint8)
a = 200
b = 200
while(i <= totImages):
	#print(i)
	Ti = time.time()
	img0 = img1
	img0gray = img1gray
	Xold = Xnew
	kp0 = kp1
	
	img1 = cv2.imread(ImgLoc+str(i)+'.png') # Image acquisition
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#	img1gray = clahe.apply(img1gray)
	
	# Feature tracking
	kp0, kp1, diff = FeatureTracking(img0gray, img1gray, kp0)
	#print(diff, kp0.shape[0])
	if diff < pixDiffThresh:
		if kp0.shape[0] < featureThresh:
			kp1 = FeatureDetection(img0gray, FeatureDetect)
			i = i + 1
			continue
	# Pose recovery
	E, mask = cv2.findEssentialMat(kp1, kp0, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
	kp0 = kp0[mask.ravel() ==1]
	kp1 = kp1[mask.ravel() ==1]
	_, R0, t0, mask = cv2.recoverPose(E, kp0, kp1, K)

	# Triangulation
	Xnew = Triangulation(R0, t0, kp0, kp1, K)
	#print(t_currR)
	if GTLoc:
		print('ground_truth')
		scale = - AbsoluteScale(ground_truth,i-2,i-1)
		plot_ground_truth(canvas,i-1,ground_truth[i,3]+a,ground_truth[i,11] + b)
	else:
		scale = RelativeScale(Xold, Xnew)

	t_curr = t_curr + scale * R_curr.dot(t0)
	R_curr = R_curr.dot(R0)			

	if kp0.shape[0] < featureThresh:
		kp1 = FeatureDetection(img1gray, FeatureDetect)
	
	canvas = plot_trajectory(canvas,i,int(t_curr[0]) +  a, int(t_curr[2])+b,t_curr,GTLoc)
	cv2.imshow('canvas',canvas)
	cv2.imshow('img',img1)

	if cv2.waitKey(1) % 0xff == ord('q'):
		break
	i = i + 1
	Tf = time.time()
	print(1/(Tf - Ti))

cv2.waitKey(0)
cv2.destroyAllWindows()
	
	
	
	
