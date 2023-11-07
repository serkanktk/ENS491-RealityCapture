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

def load_model():
    """
    Loads Yolo5 model from pytorch hub.
    :return: Trained Pytorch model.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def get_video(file_path):
    """
    Opens the local video file using OpenCV.
    :return: opencv2 video capture object.
    """
    return cv2.VideoCapture(file_path)

def class_to_label(classes, x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    return classes[int(x)]

def score_frame(model, device, frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def get_object_coordinates(model, device, frame):
    results = score_frame(model, device, frame)
    labels, cord = results
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    
    allowed_labels = ["car", "truck", "bus"]
    valid_detections = []

    for i in range(len(labels)):
        if class_to_label(model.names, labels[i]) in allowed_labels:
            valid_detections.append(cord[i])

    if len(valid_detections) == 1:  # Only one valid detection
        row = valid_detections[0]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        
        # Calculate center and width/height
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1

        return x_center, y_center, width, height
    else:  # No valid detection or more than one valid detection
        return 10000, 10000, 10000, 10000


video_path = r"StartToEnd.mp4"
cap = get_video(video_path)

device = 'cpu'
model = load_model()

frame_data_list = []  # List to store the dictionary data for each frame
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x_center, y_center, width, height = get_object_coordinates(model, device, frame)

    # Create a dictionary for the current frame data
    frame_data = {
        "frame_number": frame_number,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "length": height
    }

    # Append the frame data to the list
    frame_data_list.append(frame_data)

    frame_number += 1

cap.release()


for data in frame_data_list:
    print(data)


# In data, frame_number starts from 0
# If there are more than 1 detected object or there is no any detected object in the frame
    # then it will return 10.000 (ten thousand) for the frame

# exit()

















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

    # getting the frame data
    frame_data = frame_data_list[frame_idx]
    x_center_of_frame = frame_data["x_center"]
    y_center_of_frame = frame_data["y_center"]
    width_of_boundingbox = frame_data["width"]
    length_of_boundingbox = frame_data["length"]

    x_min = x_center_of_frame - (width_of_boundingbox / 2)
    x_max = x_center_of_frame + (width_of_boundingbox / 2)
    y_min = y_center_of_frame - (length_of_boundingbox / 2)
    y_max = y_center_of_frame + (length_of_boundingbox / 2)

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

    # Filtering the matches and storing the results
    filtered_indices = []
    filtered_kpts = []
    for m in matches:
        pt = kp1[m.queryIdx].pt
        if (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max):
            filtered_indices.append(m.queryIdx)
            filtered_kpts.append(pt)

    # Append filtered indices and keypoints to their respective lists
    all_matched_indices.append(filtered_indices)
    all_matched_keypoints.append(np.float32(filtered_kpts))

    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to polar coordinates to get magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Extracting meaningful keypoints from the dense flow
    threshold_value = 20.0
    keypoints_y, keypoints_x = np.where(mag > threshold_value)
    keypoints = np.stack((keypoints_x, keypoints_y), axis=-1).astype(np.float32)
    
    # Filtering out the keypoints that fall inside the bounding box
    filtered_dense_kpts = [pt for pt in keypoints if (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max)]
    
    # Append the filtered keypoints to all_keypoints
    all_keypoints.append(np.float32(filtered_dense_kpts))


   


    int_filtered_kpts = np.array(filtered_dense_kpts).astype(np.int32)
    current_depths = depth_matrix[frame_idx][int_filtered_kpts[:, 1], int_filtered_kpts[:, 0]]

    all_depths.append(current_depths)
    frame_idx += 1

    print(f"Detected {len(keypoints)} keypoints for frame " + str(frame_idx))

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





# Concatenate all matched keypoints and their depths
matched_keypoints = np.concatenate(all_matched_keypoints)
depths = np.concatenate(all_depths)
depths = depths[:, np.newaxis]

# Filter matched keypoints and depths based on bounding box coordinates
inside_bbox_indices = np.where((matched_keypoints[:, 0] >= x_min) & 
                               (matched_keypoints[:, 0] <= x_max) &
                               (matched_keypoints[:, 1] >= y_min) &
                               (matched_keypoints[:, 1] <= y_max))[0]
inside_bbox_keypoints = matched_keypoints[inside_bbox_indices]
inside_bbox_depths = depths[inside_bbox_indices]

# Construct 3D points for the matched keypoints inside the bounding box
inside_bbox_3d_points = np.hstack((inside_bbox_keypoints, inside_bbox_depths))

# Create point cloud for visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(inside_bbox_3d_points)

inside_bbox_colors_indices = inside_bbox_keypoints[:, 1] * width + inside_bbox_keypoints[:, 0]
inside_bbox_colors_indices = inside_bbox_colors_indices.astype(np.int32)

# Ensuring no invalid indices
if np.any(inside_bbox_colors_indices < 0) or np.any(inside_bbox_colors_indices >= len(colors)):
    print("Invalid indices detected!")
    # Handle the issue accordingly
    
pcd.colors = o3d.utility.Vector3dVector(colors[inside_bbox_colors_indices][:, ::-1] / 255)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="Matched Keypoints Inside Bounding Box")
print(f"Number of matched keypoints inside the bounding box being visualized: {len(inside_bbox_keypoints)}")





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
