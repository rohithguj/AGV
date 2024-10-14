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
        self.prev_histogram = None
        self.translation_scale = 0.1  # Scale for translation movement

    def compute_histogram(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return histogram.astype(np.float32)

    def compare_histograms(self, hist1, hist2):
        if hist1 is not None and hist2 is not None:
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return 0

    def estimate_pose(self, frame):
        # This function estimates the new pose based on frame differences
        # You can refine this logic further as per your needs
        translation = np.array([self.translation_scale, 0, 0])  # Moving along the x-axis for simplicity
        return translation

    def track(self, obs):
        if self.state == "INIT":
            self.keyframe = obs.frame
            self.prev_histogram = self.compute_histogram(obs.frame)
            self.state = "TRACKING"
            self.path.append((0, 0))
            return {"init": True}, None, None, None, None
        else:
            current_histogram = self.compute_histogram(obs.frame)
            similarity = self.compare_histograms(self.prev_histogram, current_histogram)

            if similarity > 0.7:  # Threshold for similarity
                # Estimate pose change based on the current frame
                translation = self.estimate_pose(obs.frame)
                new_pose = self.prev_pose.copy()
                new_pose[0, 3] += translation[0]  # Update x position
                new_pose[1, 3] += translation[1]  # Update y position
                self.prev_pose = new_pose
                self.prev_histogram = current_histogram
                self.path.append((self.prev_pose[0, 3], self.prev_pose[1, 3]))
                return {"similarity": similarity, "pose": self.prev_pose}, None, None, None, None

            return {"similarity": similarity}, None, None, None, None

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
        result, _, _, _, _ = frontend.track(obs)
        print(result)

        cv2.imshow('Video', frame)

        # Update the path plot in real-time
        path = np.array(frontend.path)
        ax.clear()  # Clear the previous plot
        ax.plot(path[:, 0], path[:, 1], marker='o')
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
