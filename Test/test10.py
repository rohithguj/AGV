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
        self.path = []
        self.object_paths = {}  # Dictionary to hold paths of detected objects
        self.orb = cv2.ORB_create()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def detect_objects(self, frame):
        results = self.model(frame)
        boxes = results.xyxy[0].cpu().numpy()  # Get the detections
        return boxes

    def track_objects_and_keypoints(self, obs):
        boxes = self.detect_objects(obs.frame)
        keypoints, descriptors = self.orb.detectAndCompute(obs.frame, None)

        # Plot detected objects and keypoints
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            cv2.rectangle(obs.frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{int(class_id)}: {conf:.2f}'
            cv2.putText(obs.frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Track the center of the bounding box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if int(class_id) not in self.object_paths:
                self.object_paths[int(class_id)] = []
            self.object_paths[int(class_id)].append(center)

        # Plot keypoints
        for kp in keypoints:
            cv2.circle(obs.frame, tuple(int(kp.pt[0]), int(kp.pt[1])), 5, (255, 0, 255), -1)

        return keypoints, boxes

    def draw_paths(self, obs):
        for object_id, path in self.object_paths.items():
            if len(path) > 1:
                for i in range(1, len(path)):
                    cv2.line(obs.frame, (int(path[i-1][0]), int(path[i-1][1])),
                             (int(path[i][0]), int(path[i][1])), (255, 0, 255), 2)

            # Print the path coordinates
            print(f'Object ID {object_id} path: {path}')

class Observation:
    def __init__(self, frame):
        self.frame = frame

def main(video_file):
    cap = cv2.VideoCapture(video_file)
    cam_specs = CameraSpecs()
    frontend = Frontend(cam_specs)

    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots()
    ax.set_title('Path Taken')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.axis('equal')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        obs = Observation(frame)
        keypoints, boxes = frontend.track_objects_and_keypoints(obs)
        frontend.draw_paths(obs)

        cv2.imshow('Video', obs.frame)

        # Update the path plot in real-time
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
