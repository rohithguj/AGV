import cv2
import numpy as np
import matplotlib.pyplot as plt

class CameraSpecs:
    def __init__(self):
        self.intrinsics = np.array([[800, 0, 320],
                                    [0, 800, 240],
                                    [0, 0, 1]])

class Frontend:
    def __init__(self, cam_specs):
        self.cam_specs = cam_specs
        self.prev_pose = np.eye(4)
        self.state = "INIT"
        self.path = []
        self.detector = self.setup_blob_detector()
        self.previous_blobs = []

    def setup_blob_detector(self):
        """ Set up the SimpleBlobDetector with parameters to reduce the number of blobs. """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 200  # Increase to filter out smaller blobs
        params.maxArea = 300  # Decrease to filter out larger blobs
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByColor = True
        return cv2.SimpleBlobDetector_create(params)

    def detect_blobs(self, gray_frame):
        """ Detect blobs in the given gray frame. """
        keypoints = self.detector.detect(gray_frame)
        return keypoints

    def track(self, obs):
        gray_frame = cv2.cvtColor(obs.frame, cv2.COLOR_BGR2GRAY)

        if self.state == "INIT":
            self.previous_blobs = self.detect_blobs(gray_frame)
            self.state = "TRACKING"
            if self.previous_blobs:
                self.path.append((self.previous_blobs[0].pt[0], self.previous_blobs[0].pt[1]))
            return {"init": True}, self.previous_blobs, None, None

        # Detect blobs in the current frame
        current_blobs = self.detect_blobs(gray_frame)
        if current_blobs:
            # Store blob positions
            blob_positions = [(kp.pt[0], kp.pt[1]) for kp in current_blobs]
            self.path.extend(blob_positions)

            # Calculate combinations of previous and current blobs
            if self.previous_blobs:
                for prev_blob in self.previous_blobs:
                    for curr_blob in current_blobs:
                        distance = np.linalg.norm(np.array(prev_blob.pt) - np.array(curr_blob.pt))
                        if distance < 100:  # Threshold distance for valid tracking
                            self.path.append((curr_blob.pt[0], curr_blob.pt[1]))

            self.previous_blobs = current_blobs
            return {"matches": len(current_blobs), "blobs": current_blobs}, current_blobs, None, None
        
        return {"matches": 0}, None, None, None

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
        result, kp, _, _ = frontend.track(obs)
        print(result)

        # Draw detected blobs
        if kp is not None:
            for keypoint in kp:
                x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
                cv2.circle(frame, (x, y), int(keypoint.size), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        # Update the path plot in real-time
        path = np.array(frontend.path)
        ax.clear()
        ax.plot(path[:, 0], path[:, 1], marker='o')
        ax.set_title('Path Taken')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.axis('equal')
        plt.pause(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    video_file = "/home/rohith-pt7726/AGV/Test/test1.mp4"  # Update with your video file path
    main(video_file)
