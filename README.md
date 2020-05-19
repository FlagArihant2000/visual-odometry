# VISUAL ODOMETRY

### Introduction:
The project aims at understanding the working of visual odometry. An odometer is a small gadget that tells you how much distance it has travelled on the robot or car to which it is attached to. How it works internally varies with different types of gadgets and on the robot to which it is attached to. In robotics, a more generalized definition of odometry is considered, estimating the trajectory of the moving robot, along with the distance travelled. So, it is easy to see that for each position of the robot or car, there is an associated vector showing its pose in terms of cartesian coordinates and Euler angles. In this project, we aim at understanding at doing the same using a camera. So, given the input trajectory of the robot, we are required to construct its trajectory. If we use a single camera, it is known as Monocular Visual Odometry and if we are using two or more, then it is termed as Stereo Visual Odometry. The camera is assumed to be calibrated, that is, the intrinsic camera matrix is known.

### Methodology:
1. Input the first image and extract features from it.
2. Input the second image and determine the features using Kanade Lucas Tomasi (KLT) feature tracking algorithm.
3. Compute the essential matrix for the calculated point correspondences. 
4. Decompose the essential matrix for extracting the rotation and translation component.
5. Triangulate the feature points. Initialize a reference canonical matrix and define the projection matrix with the rotation and translation component obtained from step 4. Using this as well as the feature point correspondences, perform triangulation of points. 
6. Input the third image and track the features from the second image using KLT feature tracking algorithm.
7. Repeat step 3 to 5. 
8. To obtain the scale factor, acquire the point clouds from the current and the previous iteration and the distance between the two 3D point correspondences are calculated, followed by taking the median of all the distances obtained. 
9. Update the parameters accordingly. 
10. Plot the updated final translation component (x and z component).
11. Repeat step 6 to 10 till the video sequence is exhausted.

### Compile and Run:
```git clone https://github.com/FlagArihant2000/visual-odometry```
### Team Members:

1. Arihant Gaur
2. Saurabh Kemekar
3. Aman Jain
