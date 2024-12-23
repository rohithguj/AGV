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
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.prev_keypoints = None
        self.prev_descriptors = None

    def custom_feature_mapping(self, kp, des, num_features=50):
        """ Select unique and strong keypoints based on their response. """
        if des is not None:
            idx = np.argsort([-k.response for k in kp])[:num_features]
            selected_kp = [kp[i] for i in idx]
            selected_des = des[idx]
            return selected_kp, selected_des
        return [], None

    def refresh_keypoints(self, gray_frame):
        """ Compute new keypoints and descriptors if matches are low. """
        kp, des = self.sift.detectAndCompute(gray_frame, None)
        return self.custom_feature_mapping(kp, des)

    def track(self, obs):
        gray_frame = cv2.cvtColor(obs.frame, cv2.COLOR_BGR2GRAY)

        if self.state == "INIT":
            self.prev_keypoints, self.prev_descriptors = self.sift.detectAndCompute(gray_frame, None)
            self.prev_keypoints, self.prev_descriptors = self.custom_feature_mapping(self.prev_keypoints, self.prev_descriptors)
            self.state = "TRACKING"
            self.path.append((0, 0))
            return {"init": True}, None, None, None

        # Match descriptors
        if self.prev_descriptors is not None:
            kp, des = self.sift.detectAndCompute(gray_frame, None)
            kp, des = self.custom_feature_mapping(kp, des)

            matches = self.bf.match(self.prev_descriptors, des)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) >= 10:
                points1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

                # RANSAC to estimate essential matrix
                E, mask = cv2.findEssentialMat(points2, points1, self.cam_specs.intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, points2, points1, self.cam_specs.intrinsics)

                    # Update pose
                    translation = t.flatten()
                    new_pose = np.eye(4)
                    new_pose[:3, :3] = R
                    new_pose[:3, 3] = np.array([translation[0], translation[1], 0])

                    self.prev_pose = self.prev_pose @ new_pose
                    self.path.append((self.prev_pose[0, 3], self.prev_pose[1, 3]))

                    # Update previous keypoints and descriptors
                    self.prev_keypoints = kp
                    self.prev_descriptors = des

                    return {"matches": len(good_matches), "pose": self.prev_pose}, kp, None, None
            
            # If no good matches found, refresh keypoints
            if len(good_matches) < 10:
                print("Low matches, refreshing keypoints...")
                self.prev_keypoints, self.prev_descriptors = self.refresh_keypoints(gray_frame)
                return {"matches": 0}, None, None, None

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
            for keypoint in kp:
                x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

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
    video_file = "/home/rohith-pt7726/Desktop/AGV/Test/test1.mp4"  # Update with your video file path
    main(video_file)
