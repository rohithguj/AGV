import cv2
import numpy as np
import matplotlib.pyplot as plt

class CameraSpecs:
    def __init__(self):
        self.intrinsics = np.array([[800, 0, 320],
                                    [0, 800, 240],
                                    [0, 0, 1]])

class OrbBasedFeatureMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def match(self, keyframe, frame):
        kp1, des1 = self.orb.detectAndCompute(keyframe, None)
        kp2, des2 = self.orb.detectAndCompute(frame, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return matches, kp1, kp2

class Frontend:
    def __init__(self, cam_specs):
        self.matcher = OrbBasedFeatureMatcher()
        self.cam_specs = cam_specs
        self.prev_pose = np.eye(4)
        self.state = "INIT"
        self.path = []

    def track(self, obs):
        if self.state == "INIT":
            self.keyframe = obs.frame
            self.state = "TRACKING"
            self.path.append((0, 0))
            return {"init": True}, None, None, None, None
        else:
            matches, kp1, kp2 = self.matcher.match(self.keyframe, obs.frame)
            if len(matches) > 10:
                self.keyframe = obs.frame
                
                points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                E, mask = cv2.findEssentialMat(points2, points1, self.cam_specs.intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, points2, points1, self.cam_specs.intrinsics)

                    new_pose = np.eye(4)
                    new_pose[:3, :3] = R
                    new_pose[:3, 3] = t.flatten()
                    
                    self.prev_pose = self.prev_pose @ new_pose
                    self.path.append((self.prev_pose[0, 3], self.prev_pose[1, 3]))

                    return {"matches": len(matches), "pose": self.prev_pose}, kp1, kp2, mask, matches
            return {"matches": 0}, None, None, None, None

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
        result, kp1, kp2, mask, matches = frontend.track(obs)
        print(result)

        if kp1 is not None and kp2 is not None and mask is not None and matches is not None:
            for i, m in enumerate(mask):
                if m:
                    pt1 = tuple(np.round(kp1[matches[i].queryIdx].pt).astype(int))
                    pt2 = tuple(np.round(kp2[matches[i].trainIdx].pt).astype(int))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                    cv2.circle(frame, pt1, 5, (0, 0, 255), -1)
                    cv2.circle(frame, pt2, 5, (255, 0, 0), -1)

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
