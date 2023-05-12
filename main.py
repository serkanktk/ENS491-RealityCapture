#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Thu Feb  2 13:06:22 2023

This code performs reality capture (3D reconstruction) for a particular object in a video. The code first loads
the video using the cv2.VideoCapture method and retrieves the first frame. Then, it sets the termination criteria for feature
 detection using the cv2.goodFeaturesToTrack method and the parameters for optical flow calculation using the cv2.calcOpticalFlowPyrLK method.

In the while loop, the code reads the next frame and converts both frames to grayscale. It detects good features to track in the first frame, and
calculates the opticalflow between the first and second frames. The good points and their depth are appended to the arrays.
The first frame is then updated to be the second frame
for the next iteration. The loop continues until the end of the video or until the user presses the 'q' key.

After the video is processed, the 2D points are converted to 3D coordinates using the depth information.
Triangulation is performed using the cv2.triangulatePoints method to obtain a 3D point cloud, and the perspective transformation
 is performed using the cv2.perspectiveTransform method to obtain the final 3D points. The resulting point cloud is then
 plotted using the Open3D library.

@author: oercetin
"""
"""
point_cloud = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3,1)))), P2, p1.T, p2.T)
"""

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter1d

# Load the video and get the first frame
cap = cv2.VideoCapture("traf_video_Trim.mp4")
ret, frame1 = cap.read()

# Define the termination criteria for the feature detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Define the parameters for optical flow calculation
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some empty arrays to store the points and their corresponding depth
p1 = np.empty((0, 2))
p2 = np.empty((0, 2))
depths = np.empty((0))

# Loop through the video frame by frame
while (cap.isOpened()):
    ret, frame2 = cap.read()

    # Break if there is no more video
    if ret == False:
        break

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect good features to track in the first frame
    p_0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    # Draw circles at feature point locations on the original image
    for point in p_0:
        x, y = point.ravel()
        cv2.circle(frame1, (int(x), int(y)), 5, (0, 255, 0), -1)

    '''
    good features to track

    >>> a = np.zeros((2, 3, 4))
    array([[[ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]],

       [[ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]]])

    # a = np.zeros((2,1,2))
    array([[[ 0., 0.],
            [ 0., 0]]])

    The goodFeaturesToTrack function returns a NumPy array of shape (N, 1, 2), 
    where N is the number of feature points detected by the algorithm. 
    Each row in the array corresponds to a feature point and 
    contains the (x, y) coordinates of the point.
    '''

    # Calculate the optical flow between the first and second frames
    p_1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p_0, None, **lk_params)

    '''
p_1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p_0, None, **lk_params): 
This line calculates the optical flow between the first frame gray1 and the second frame gray2 using the Lucas-Kanade algorithm 
with pyramidal processing. The input feature points p_0 from the first frame are tracked in the second frame, 
and the resulting feature points in the second frame are stored in the NumPy array p_1. 
Additionally, the status of each tracked feature point (whether it was successfully tracked or not) 
is stored in the boolean NumPy array st, and the tracking error for each point is stored in the NumPy array err.

    The first array contains the new positions of the feature points computed by the
    Lucas-Kanade algorithm. Each row in this array corresponds to a
    feature point and contains the (x, y) coordinates of the new position.

    The second array contains a boolean value for each feature point,
    indicating whether the optical flow calculation was successful or not.
    A value of 1 indicates success, while a value of 0 indicates failure.
'''

    # Select only good points
    good_new = p_1[st == 1]
    good_old = p_0[st == 1]
    # Draw the good features on the frame
    """
    for point in good_old:
        point = (int(point[0]), int(point[1]))  # No need for additional index
        cv2.circle(frame1, point, 5, (0, 0, 255), -1)
"""
    '''
good_new = p_1[st == 1]: This line selects only the feature points that were successfully tracked 
in the second frame by indexing into the p_1 array using the boolean array st == 1, which contains 
True values for successfully tracked points and False values for unsuccessfully tracked points. 
The resulting feature points are stored in the NumPy array good_new.
good_old = p_0[st == 1]: This line selects the corresponding feature points from the first frame 
that correspond to the successfully tracked feature points in the second frame. It does this by indexing 
into the p_0 array using the same boolean array st == 1. The resulting feature points from the first frame 
are stored in the NumPy array good_old.

The end result is that good_old and good_new contain corresponding feature points from consecutive frames 
that can be used for further analysis or visualization, such as motion estimation, object tracking, or image registration.
'''

    # Calculate the depth of each point based on the motion of the camera
    # (this assumes a static camera, for a moving camera you would need to perform SfM)
    depth = np.abs(good_new - good_old)

    # Apply a moving average filter to the depth estimates
    window_size = 3  # or whatever window size you want
    depth = uniform_filter1d(depth, size=window_size, mode='reflect')


    # Append the good points and their depth to the arrays
    # p2 = np.append(p2, good_new, axis=0)
    # good_old = good_old[:, np.newaxis, :]
    p2 = np.vstack([p2, good_new])
    # p1 = np.append(p1, good_old, axis=0)
    p1 = np.vstack([p1, good_old])
    # depths = np.append(depths, depth[:,0].mean())
    depths = np.append(depths, np.full((good_new.shape[0],), depth[:, 0].mean()))

    # Resize the frame to a width of 640
    #cv2.imshow('Feature Points', cv2.resize(frame1, (1800, int(frame1.shape[0] * 1800 / frame1.shape[1]))))


    # Set the first frame to be the second frame for the next iteration
    frame1 = frame2.copy()
    # At the end of each loop iteration
    if len(good_old) > 0:
        # draw the good features on the frame
        for i, point in enumerate(good_old):
            point = (int(point[0]), int(point[1]))
            cv2.circle(frame1, point, 5, (0, 0, 255), -1)

            # draw the depth value next to the point
            depth = depths[i]
            cv2.putText(frame1, f'depth: {depth}', (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # resize and show the frame
        cv2.imshow('Feature Points', cv2.resize(frame1, (1800, int(frame1.shape[0] * 1800 / frame1.shape[1]))))

    # Break if the user presses the 'q' key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Convert the arrays to the correct format
p1 = np.round(p1).astype(int)
depths = depths[:, np.newaxis]

# Ensure p1 values do not exceed image dimensions
height, width, _ = frame1.shape
p1[:, 0] = np.clip(p1[:, 0], 0, width-1)
p1[:, 1] = np.clip(p1[:, 1], 0, height-1)

# Convert 2D points to 1D indices
indices = p1[:, 1] * width + p1[:, 0]

# Reshape the image to a 2D array of colors
colors = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
colors = colors.reshape(-1, 3)

# Create a point cloud
pcd = o3d.geometry.PointCloud()

# Set the points and colors of the point cloud
pcd.points = o3d.utility.Vector3dVector(np.hstack((p1, depths)))
# pcd.colors = o3d.utility.Vector3dVector(colors[indices])
pcd.colors = o3d.utility.Vector3dVector(colors[indices][:, ::-1] / 255)



# Show the point cloud
o3d.visualization.draw_geometries([pcd])






"""
# Convert the 2D points to 3D coordinates using the depth information
p3 = np.column_stack((p2, depths))

R = np.eye(3)
T = np.zeros((3, 1))
P2 = np.hstack((R, T))

# obtain a 3D point cloud
# point_cloud = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3,1)))), P2, p1.T, p2.T)

"""
"""
print(np.shape(np.hstack((np.eye(3), np.zeros((3, 1))))))
print(np.shape(P2))
print(np.shape(p1.T))
print(np.shape(p2.T))
print(np.hstack((np.eye(3), np.zeros((3, 1)))))
print(P2)
print(p1.T)
print(p2.T)

(3, 4)
(3, 4)
(2, 0)
(2, 0)
"""
"""
point_cloud = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3, 1)))), P2, p1.T, p2.T)
camera_matrix = P2

# projMatr1: np.hstack((np.eye(3), np.zeros((3,1))))
# projMatr2: P2
# projPoints1: p1.T
# projPoints2: p2.T


# Define the camera matrix
# camera_matrix = np.array([[1, 0, 0, 0],
#                           [0, 1, 0, 0],
#                           [0, 0, 1, 0]])

# Perform the perspective transformation to obtain the 3D points
# point_cloud = point_cloud / point_cloud[3]
# points_3d = np.dot(camera_matrix, point_cloud)
points_3d = cv2.perspectiveTransform(p3.reshape(-1, 1, 3), camera_matrix)

points_2d = points_3d.reshape(-1, 2)

points_3d = np.column_stack((points_2d, depths))
points_3d = points_3d.reshape(-1, 3)

# %%
# Plot the resulting point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_3d)
# o3d.visualization.draw_geometries([pcd])


# Load the video
cap = cv2.VideoCapture("traf_video.mp4")

# Read the first frame
ret, frame = cap.read()

# Get the height and width of the video
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Define the point cloud representation
points = points_2d

# Loop over all frames
while ret:
    # Draw the point cloud representation on the frame
    for point in points:
        point = (int(point[0]), int(point[1]))
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # Display the frame with the point cloud overlay
    cv2.imshow("Frame", frame)

    # Check if the 'q' key was pressed to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

'''

# My Part

import cv2
import numpy as np

# Load images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
img3 = cv2.imread('image3.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Detect features and extract descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

# Match features between images
matcher = cv2.BFMatcher()
matches1_2 = matcher.knnMatch(des1, des2, k=2)
matches1_3 = matcher.knnMatch(des1, des3, k=2)

# Apply ratio test to filter matches
good_matches1_2 = []
for m,n in matches1_2:
    if m.distance < 0.75*n.distance:
        good_matches1_2.append(m)

good_matches1_3 = []
for m,n in matches1_3:
    if m.distance < 0.75*n.distance:
        good_matches1_3.append(m)

# Get corresponding keypoints for good matches
src_pts1_2 = np.float32([kp1[m.queryIdx].pt for m in good_matches1_2]).reshape(-1,1,2)
dst_pts1_2 = np.float32([kp2[m.trainIdx].pt for m in good_matches1_2]).reshape(-1,1,2)

src_pts1_3 = np.float32([kp1[m.queryIdx].pt for m in good_matches1_3]).reshape(-1,1,2)
dst_pts1_3 = np.float32([kp3[m.trainIdx].pt for m in good_matches1_3]).reshape(-1,1,2)

# Compute essential matrix
E1_2, mask1_2 = cv2.findEssentialMat(src_pts1_2, dst_pts1_2, np.eye(3), method=cv2.RANSAC, threshold=1.0)
E1_3, mask1_3 = cv2.findEssentialMat(src_pts1_3, dst_pts1_3, np.eye(3), method=cv2.RANSAC, threshold=1.0)

# Recover pose and 3D points
_, R1_2, t1_2, mask1_2 = cv2.recoverPose(E1_2, src_pts1_2, dst_pts1_2, np.eye(3))
_, R1_3, t1_3, mask1_3 = cv2.recoverPose(E1_3, src_pts1_3, dst_pts1_3, np.eye(3))

# Triangulate points
proj_mat1 = np.hstack((np.eye(3), np.zeros((3,1))))
proj_mat2 = np.hstack((R1_2, t1_2))
proj_mat3 = np.hstack((R1_3, t1_3))

points4D = cv2.triangulatePoints(proj_mat1, proj_mat2, src_pts1_2, dst_pts1_2)
points4D /= points4D[3]
points3D_1_2 = points4D
"""
