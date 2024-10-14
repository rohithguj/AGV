import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

class CameraSpecs:
    def __init__(self):
        self.intrinsics = np.array([[800, 0, 320],
                                    [0, 800, 240],
                                    [0, 0, 1]])

class Frontend:
    def __init__(self, cam_specs):
        self.cam_specs = cam_specs
        self.robot_path = []  # List to hold the robot's path
        self.orb = cv2.ORB_create()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def detect_objects(self, frame):
        results = self.model(frame)
        boxes = results.xyxy[0].cpu().numpy()  # Get the detections
        return boxes

    def track_robot_and_objects(self, obs):
        boxes = self.detect_objects(obs.frame)
        keypoints = self.orb.detect(obs.frame)  # Detect keypoints

        # Track the robot's position using the center of detected objects or keypoints
        robot_position = None
        
        # Use the center of the largest detected object as the robot's position
        if boxes.size > 0:
            largest_box = boxes[np.argmax(boxes[:, 4])]  # Get the box with highest confidence
            x1, y1, x2, y2, conf, class_id = largest_box
            robot_position = ((x1 + x2) / 2, (y1 + y2) / 2)

        # If no object detected, use keypoints (or the first keypoint as a fallback)
        if robot_position is None and len(keypoints) > 0:
            robot_position = (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))

        # Append the robot's position to the path
        if robot_position is not None:
            self.robot_path.append(robot_position)

        return keypoints, boxes

    def draw_robot_path(self, obs):
        for i in range(1, len(self.robot_path)):
            cv2.line(obs.frame, (int(self.robot_path[i-1][0]), int(self.robot_path[i-1][1])),
                         (int(self.robot_path[i][0]), int(self.robot_path[i][1])), (255, 0, 255), 2)

        # Print the path coordinates
        print(f'Robot path: {self.robot_path}')

class Observation:
    def __init__(self, frame):
        self.frame = frame

def main(video_file):
    cap = cv2.VideoCapture(video_file)
    cam_specs = CameraSpecs()
    frontend = Frontend(cam_specs)

    # Create a separate plot for the robot's path
    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots()
    ax.set_title('Robot Path Taken')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.axis('equal')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        obs = Observation(frame)
        keypoints, boxes = frontend.track_robot_and_objects(obs)
        frontend.draw_robot_path(obs)

        # Display the video frame
        cv2.imshow('Video', obs.frame)

        # Update the path plot in a separate window
        ax.clear()  # Clear the previous plot
        if len(frontend.robot_path) > 1:
            path = np.array(frontend.robot_path)
            ax.plot(path[:, 0], path[:, 1], marker='o', label='Robot Path')

        ax.set_title('Robot Path Taken')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.axis('equal')
        ax.legend()
        plt.pause(0.01)  # Pause to update the plot

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Show final plot after the loop ends
    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    video_file = "/home/zlabs/Desktop/test1.mp4"  # Change this to your video file path
    main(video_file)
