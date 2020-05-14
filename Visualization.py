"""
IvLabs, VNIT
MONOCULAR VISUAL ODOMETRY ON KITTI DATASET

TEAM MEMBERS:
1. Arihant Gaur
2. Saurabh Kemekar
3. Aman Jain
"""

import numpy as np
import cv2
import math

def drawOpticalFlowField(img, ref_pts, cur_pts):
    for i, (new, old) in enumerate(zip(cur_pts, ref_pts)):
        x, y = old.ravel()
        v1 = tuple((new - old) * 2.5 + old)
        d_v = [new - old][0] * 0.75
        arrow_color = (0, 0, 255)
        arrow_t1 = rotateFunct([d_v], 0.5)
        arrow_t2 = rotateFunct([d_v], -0.5)
        tip1 = tuple(np.float32(np.array([x, y]) + arrow_t1)[0])
        tip2 = tuple(np.float32(np.array([x, y]) + arrow_t2)[0])
        cv2.line(img, v1, (x, y), (0, 0, 255), 2)
        cv2.line(img, (x, y), tip1, arrow_color, 2)
        cv2.line(img, (x, y), tip2, arrow_color, 2)
        cv2.circle(img, v1, 1, (0, 255, 0), -1)
    return img


def rotateFunct(pts_l, angle, degrees=False):
    if degrees == True:
        theta = math.radians(angle)
    else:
        theta = angle

    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    rot_pts = []
    for v in pts_l:
        v = np.array(v).transpose()
        v = R.dot(v)
        v = v.transpose()
        rot_pts.append(v)

    return rot_pts

def plot_tracjectory(window,i,x,y,t):

    text = "Frame no = {}    x = {}    y = {}    z = {}".format(i,t[0,0],t[1,0],t[2,0])
 #   test2 = 'No Inliers = {}'.format(n)
    cv2.rectangle(window, (0, 0), (950, 70), (0, 0, 0), cv2.FILLED)
    cv2.putText(window, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8);
    cv2.circle(window,(int(x),int(y)),3,(0,255,0))

    return window


def plot_ground_truth(window,i,ground_truth):
    x = ground_truth[i,3] + 200
    y = ground_truth[i,7]
    z = ground_truth[i,11]+ 450
    cv2.circle(window, (int(x), int(z)), 3, (0, 0, 255))

    return window

#ImgLoc = '/home/saurabh/Downloads/visual-odometry-master/src/KITTI_sample/images/'  # Images Location
#GTLoc = '/home/saurabh/Downloads/visual-odometry-master/src/KITTI_sample/poses.txt'  # Ground truth location file. Write None if you don't have it.
