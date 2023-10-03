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

# camera calibration


"""
class ObjectDetection:
    
    Class implements Yolo5 model to make inferences on a local video using OpenCV.
    
    
    def __init__(self, file_path, out_file):
        
        Initializes the class with local video file path and output file.
        :param file_path: Path to the local video file on which prediction is made.
        :param out_file: A valid output file name.
        
        self._file_path = file_path
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)


    def get_video(self):
        
        Opens the local video file using OpenCV.
        :return: opencv2 video capture object.
        
        return cv2.VideoCapture(self._file_path)


    def load_model(self):
        
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model


    def score_frame(self, frame):
        
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels plotted on it.
        
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        allowed_labels = ["car", "truck", "bus"]  # Allowable labels
        for i in range(n):
            # Check if the detected class is in allowed labels
            if self.class_to_label(labels[i]) not in allowed_labels:
                continue
            row = cord[i]
            if row[4] >= 0.3:  # threshold value
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame


    def __call__(self):
        
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        
        player = self.get_video()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time() 
            ret, frame = player.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            out.write(frame)




# Create a new object and execute.
video_path = r"Dataset\Data1\video1.mp4"
detection = ObjectDetection(video_path, "data1_video1_output.avi")
detection()
"""


# Load the video and get the first frame
cap = cv2.VideoCapture(r"StartToEnd.mp4")
ret, frame1 = cap.read()







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
    threshold_value = 2.0
    keypoints_y, keypoints_x = np.where(mag > threshold_value)
    keypoints = np.stack((keypoints_x, keypoints_y), axis=-1).astype(np.float32)
    all_keypoints.append(keypoints)

    int_keypoints = keypoints.astype(np.int32)
    current_depths = mag[int_keypoints[:, 1], int_keypoints[:, 0]]
    all_depths.append(current_depths)

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
