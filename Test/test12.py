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
        self.prev_gray = None
        self.prev_keypoints = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def track(self, obs):
        # Convert current frame to grayscale
        gray_frame = cv2.cvtColor(obs.frame, cv2.COLOR_BGR2GRAY)

        if self.state == "INIT":
            # Initialize keypoints in the first frame
            self.prev_gray = gray_frame
            self.prev_keypoints = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.3, minDistance=7)
            self.state = "TRACKING"
            self.path.append((0, 0))
            return {"init": True}, None, None, None

        # Calculate optical flow
        next_keypoints, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_frame, self.prev_keypoints, None, **self.lk_params)

        # Select good points
        good_new = next_keypoints[st == 1]
        good_old = self.prev_keypoints[st == 1]

        if len(good_new) >= 10:
            # Estimate translation based on good points
            translation = np.mean(good_new - good_old, axis=0)
            
            # Ensure the translation has three components
            new_pose = np.eye(4)
            new_pose[:3, 3] = np.array([translation[0], translation[1], 0])  # Add 0 for z-component

            # Update the pose
            self.prev_pose = self.prev_pose @ new_pose
            self.path.append((self.prev_pose[0, 3], self.prev_pose[1, 3]))

            # Update previous data for the next iteration
            self.prev_gray = gray_frame
            self.prev_keypoints = good_new.reshape(-1, 1, 2)

            return {"matches": len(good_new), "pose": self.prev_pose}, good_new, None, None

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

        # Draw tracked points
        if kp is not None:
            for point in kp:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

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
    video_file = "/home/rohith-pt7726/AGV/Test/test2.mp4"  # Change this to your video file path
    main(video_file)
