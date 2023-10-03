from time import time
import torch
import cv2
import numpy as np


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a local video using OpenCV.
    """
    
    def __init__(self, file_path, out_file):
        """
        Initializes the class with local video file path and output file.
        :param file_path: Path to the local video file on which prediction is made.
        :param out_file: A valid output file name.
        """
        self._file_path = file_path
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)


    def get_video(self):
        """
        Opens the local video file using OpenCV.
        :return: opencv2 video capture object.
        """
        
        return cv2.VideoCapture(self._file_path)


    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model


    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels plotted on it.
        """
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
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
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