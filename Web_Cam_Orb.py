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
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_keypoints = None
        self.prev_descriptors = None

    def track(self, obs):
        try:
            gray_frame = cv2.cvtColor(obs.frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error converting frame to grayscale: {e}")
            return {"matches": 0}, None, None, None
        
        try:
            kp, des = self.orb.detectAndCompute(gray_frame, None)
        except Exception as e:
            print(f"Error detecting keypoints: {e}")
            return {"matches": 0}, None, None, None
        
        if len(kp) == 0:  # Check if no keypoints are detected
            print("No keypoints detected!")
            return {"matches": 0}, None, None, None

        if self.state == "INIT":
            self.prev_keypoints = kp
            self.prev_descriptors = des
            self.state = "TRACKING"
            self.path.append((0, 0))
            return {"init": True}, None, None, None

        # Match descriptors
        matches = self.bf.match(self.prev_descriptors, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter matches based on distance
        good_matches = [m for m in matches if m.distance < 50]  # Adjust threshold as needed

        if len(good_matches) >= 10:
            # Extract locations of good matches
            points1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
            points2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

            try:
                # Estimate essential matrix
                E, mask = cv2.findEssentialMat(points2, points1, self.cam_specs.intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is None:
                    print("Essential matrix is None!")
                    return {"matches": 0}, None, None, None
                
                _, R, t, mask = cv2.recoverPose(E, points2, points1, self.cam_specs.intrinsics)

                # Update pose
                translation = t.flatten()
                new_pose = np.eye(4)
                new_pose[:3, :3] = R
                new_pose[:3, 3] = translation

                self.prev_pose = self.prev_pose @ new_pose
                self.path.append((self.prev_pose[0, 3], self.prev_pose[1, 3]))

                self.prev_keypoints = kp
                self.prev_descriptors = des

                return {"matches": len(good_matches), "pose": self.prev_pose}, kp, None, None
            except Exception as e:
                print(f"Error during essential matrix estimation: {e}")
                return {"matches": 0}, None, None, None

        return {"matches": 0}, None, None, None

class Observation:
    def __init__(self, frame):
        self.frame = frame

def main():
    cap = cv2.VideoCapture(0)  # Use webcam by passing 0 to VideoCapture
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    cam_specs = CameraSpecs()
    frontend = Frontend(cam_specs)

    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots()
    ax.set_title('Path Taken')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.axis('equal')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        obs = Observation(frame)
        result, kp, _, _ = frontend.track(obs)
        print(result)

        # Draw matches (optional)
        if kp is not None:
            cv2.drawKeypoints(frame, kp, frame)

        cv2.imshow('Webcam Feed', frame)

        # Update the path plot in real-time
        try:
            path = np.array(frontend.path)
            ax.clear()  # Clear the previous plot
            ax.plot(path[:, 0], path[:, 1], marker='o')
            ax.set_title('Path Taken')
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.axis('equal')
            plt.pause(0.01)  # Pause to update the plot
        except Exception as e:
            print(f"Error during plotting: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Show final plot after the loop ends
    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    main()
