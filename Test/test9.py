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
        self.object_paths = {}  # Dictionary to hold paths of detected objects
        self.object_id_counter = 0  # ID counter for tracking
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def detect_objects(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img)
        return results.xyxy[0]  # Bounding boxes

    def track_objects_and_keypoints(self, obs):
        detections = self.detect_objects(obs.frame)
        current_ids = {}

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            object_id = self.object_id_counter  # Assign an ID to the object
            current_ids[object_id] = (x1 + x2) / 2, (y1 + y2) / 2  # Center position
            self.object_id_counter += 1

            cv2.rectangle(obs.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.model.names[int(cls)]}: {conf:.2f} ID: {object_id}"
            cv2.putText(obs.frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Store the object paths
        for object_id, position in current_ids.items():
            if object_id not in self.object_paths:
                self.object_paths[object_id] = []
            self.object_paths[object_id].append(position)

        return detections

    def draw_paths(self, frame):
        for object_id, positions in self.object_paths.items():
            if len(positions) > 1:
                points = np.array(positions, dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=False, color=(255, 255, 0), thickness=2)

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
        detections = frontend.track_objects_and_keypoints(obs)

        # Draw paths on the frame
        frontend.draw_paths(obs.frame)

        # Show the video with detected objects and paths
        cv2.imshow('Video', obs.frame)

        # Update the path plot in real-time
        path = np.array([pos for positions in frontend.object_paths.values() for pos in positions])
        if path.size > 0:
            ax.clear()  # Clear the previous plot
            ax.plot(path[:, 0], path[:, 1], marker='o', linestyle='-', color='orange')
            ax.set_title('Path Taken')
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.axis('equal')
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
