import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

class CameraSpecs:
    def __init__(self):
        self.intrinsics = np.array([[800, 0, 320],
                                    [0, 800, 240],
                                    [0, 0, 1]])

class Frontend:
    def __init__(self, cam_specs, frame_skip=20, similarity_threshold=0.7):
        self.cam_specs = cam_specs
        self.prev_pose = np.eye(4)
        self.state = "INIT"
        self.path = []
        self.prev_frame = None
        self.frame_skip = frame_skip
        self.similarity_threshold = similarity_threshold

    def compare_frames(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2)

    def track(self, obs):
        if self.state == "INIT":
            self.prev_frame = obs.frame
            self.state = "TRACKING"
            self.path.append((0, 0))
            return {"init": True}, None, None, None, None
        else:
            similarity = self.compare_frames(self.prev_frame, obs.frame)

            if similarity < self.similarity_threshold:  # Process only if frames are sufficiently different
                # Update the pose; here we simulate motion as needed
                self.prev_pose[0, 3] += 0.1  # Simulating movement in x-direction
                self.path.append((self.prev_pose[0, 3], self.prev_pose[1, 3]))
                self.prev_frame = obs.frame  # Update the previous frame
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

    frame_count = 0  # To track frames for skipping

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        obs = Observation(frame)
        result, _, _, _, _ = frontend.track(obs)
        print(result)

        if frame_count % frontend.frame_skip == 0:  # Skip frames based on the configured skip rate
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

        frame_count += 1

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
