# do not delete here -S
# import sys
# print("The virtual environment is in:")
# print(sys.executable)


import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter
import os
import torch
# import pafy
from time import time

# disparity map matrix: -->
depth_matrix = np.load('final_disparity.npy')
print(depth_matrix)
# exit()


# Load the video and get the first frame
cap = cv2.VideoCapture(r"StartToEnd.mp4")
ret, frame1 = cap.read()

# Create a FAST feature detector
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

# Storage for keypoints and depths
all_keypoints = []
all_matched_keypoints = []
all_depths = []
# As you detect and match keypoints between frames, store the indices of the keypoints in frame1 that have been matched. You'll need to store these for every frame.
all_matched_indices = []


# Feature matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
frame_idx = 0


while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Feature detection using FAST
    kp1 = fast.detect(gray1, None)
    kp2 = fast.detect(gray2, None)

    # Compute descriptors for the detected keypoints
    kp1, des1 = cv2.xfeatures2d.SIFT_create().compute(gray1, kp1)
    kp2, des2 = cv2.xfeatures2d.SIFT_create().compute(gray2, kp2)

    # Feature matching
    matches = bf.match(des1, des2)
    # When you concatenate to get matched_keypoints, do the same with the stored indices to get a list of indices of the concatenated keypoints array.
    matched_kpts_indices = [m.queryIdx for m in matches]
    all_matched_indices.append(matched_kpts_indices)

    """
    # Visualize matched keypoints
    matched_img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches, None)
    cv2.imshow('Matched Keypoints', matched_img)
    """
    # Store matched keypoints for later use
    matched_kpts = np.float32([kp1[m.queryIdx].pt for m in matches])
    all_matched_keypoints.append(matched_kpts)

    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to polar coordinates to get magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Extracting meaningful keypoints from the dense flow
    threshold_value = 20.0
    keypoints_y, keypoints_x = np.where(mag > threshold_value)
    keypoints = np.stack((keypoints_x, keypoints_y), axis=-1).astype(np.float32)
    all_keypoints.append(keypoints)

   


    int_keypoints = keypoints.astype(np.int32)
    current_depths = depth_matrix[frame_idx][int_keypoints[:, 1], int_keypoints[:, 0]]

    all_depths.append(current_depths)
    frame_idx += 1

    print(f"Detected {len(keypoints)} keypoints for this frame.")

    # Add delay for visualization and handle window closure
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Wait for 30ms and check if 'q' is pressed
        break

    frame1 = frame2.copy()

cap.release()
cv2.destroyAllWindows()


# Check if any keypoints were detected
if not all_keypoints:
    print("No keypoints detected throughout the video.")
    exit()


# Point cloud creation using keypoints from optical flow
keypoints = np.concatenate(all_keypoints)
depths = np.concatenate(all_depths)
depths = depths[:, np.newaxis]

height, width, _ = frame1.shape
keypoints = np.round(keypoints).astype(int)  # Ensuring keypoints are integers


keypoints[:, 0] = np.clip(keypoints[:, 0], 0, width-1)
keypoints[:, 1] = np.clip(keypoints[:, 1], 0, height-1)

indices = keypoints[:, 1] * width + keypoints[:, 0]
colors = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
colors = colors.reshape(-1, 3)

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(np.hstack((keypoints, depths)))
pcd1.colors = o3d.utility.Vector3dVector(colors[indices][:, ::-1] / 255)

o3d.visualization.draw_geometries([pcd1], window_name="Optical Flow Keypoints")
print(f"Number of optical flow keypoints being visualized: {len(keypoints)}")







# Concatenating matched keypoints and indices:
matched_keypoints = np.concatenate(all_matched_keypoints)
matched_indices = np.concatenate(all_matched_indices)

# Obtaining matched depths:
matched_depths = depths[matched_indices]


# Converting matched keypoints to float and constructing 3D points:
# Convert matched_keypoints to float, as o3d expects float values
matched_keypoints_float = matched_keypoints.astype(np.float32)
# Construct the 3D points for the matched keypoints
matched_3d_points = np.hstack((matched_keypoints_float, matched_depths))


# Creating the point cloud and visualizing:
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(matched_3d_points)

matched_colors_indices = matched_keypoints[:, 1] * width + matched_keypoints[:, 0]
matched_colors_indices = matched_colors_indices.astype(np.int32)
if np.any(matched_colors_indices < 0) or np.any(matched_colors_indices >= len(colors)):
    print("Invalid indices detected!")
    # Debug or handle the issue accordingly
    
#  the shapes and sizes of arrays
print(matched_keypoints.shape)
print(colors.shape)
print(matched_colors_indices.shape)





pcd2.colors = o3d.utility.Vector3dVector(colors[matched_colors_indices][:, ::-1] / 255)


o3d.visualization.draw_geometries([pcd2], window_name="Matched Keypoints Point Cloud")
print(f"Number of matched keypoints being visualized: {len(matched_keypoints)}")




# pip install open3d
# pip install opencv-python
# pip install scipy
# pip install numpy



"""
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
"""

"""
this is for me (Serkan):
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
you have to go to https://pytorch.org/get-started/locally/#windows-pip
and then find the correct PyTorch version for your hardware:



"""

"""
    -- Depth Display --

    # At the end of each loop iteration
    if len(good_old) > 0:
        # Draw the good features on the frame
        for i, point in enumerate(good_old):
            point = (int(point[0]), int(point[1]))
            cv2.circle(frame1, point, 5, (0, 0, 255), -1)

            # Draw the depth value next to the point
            depth = depths[i]
            cv2.putText(frame1, f'depth: {depth:.2f}', (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Resize and show the frame
        cv2.imshow('Feature Points', cv2.resize(frame1, (1800, int(frame1.shape[0] * 1800 / frame1.shape[1]))))
    
    # Break if the user presses the 'q' key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    """
