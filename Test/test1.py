import cv2
import numpy as np
import matplotlib.pyplot as plt

class CameraSpecs:
    def __init__(self):
        # Example intrinsic parameters
        self.intrinsics = np.array([[800, 0, 320],
                                    [0, 800, 240],
                                    [0, 0, 1]])

class OrbBasedFeatureMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def match(self, keyframe, frame):
        # Detect keypoints and compute descriptors
        kp1, des1 = self.orb.detectAndCompute(keyframe, None)
        kp2, des2 = self.orb.detectAndCompute(frame, None)

        # Match descriptors using BFMatcher
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
            self.path.append((0, 0))  # Start at origin
            return {"init": True}
        else:
            matches, kp1, kp2 = self.matcher.match(self.keyframe, obs.frame)
            if len(matches) > 10:  # Arbitrary threshold
                self.keyframe = obs.frame  # Update keyframe
                
                # Estimate motion based on matches
                points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Find essential matrix
                E, mask = cv2.findEssentialMat(points2, points1, self.cam_specs.intrinsics)
                _, R, t, mask = cv2.recoverPose(E, points2, points1, self.cam_specs.intrinsics)
                
                # Create a 4x4 transformation matrix from R and t
                new_pose = np.eye(4)
                new_pose[:3, :3] = R  # Set rotation part
                new_pose[:3, 3] = t.flatten()  # Set translation part
                
                # Update pose
                self.prev_pose = self.prev_pose @ new_pose  # Update pose
                self.path.append((self.prev_pose[0, 3], self.prev_pose[1, 3]))  # Store position
                
                return {"matches": len(matches), "pose": self.prev_pose}
            else:
                return {"matches": 0}

class Observation:
    def __init__(self, frame):
        self.frame = frame

def main(video_file):
    cap = cv2.VideoCapture(video_file)
    cam_specs = CameraSpecs()
    frontend = Frontend(cam_specs)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        obs = Observation(frame)
        result = frontend.track(obs)
        print(result)

        # Display the current frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot the path
    path = np.array(frontend.path)
    plt.plot(path[:, 0], path[:, 1], marker='o')
    plt.title('Path Taken')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    video_file =  "/home/zlabs/Desktop/test1.mp4" # Change this to your video file path
    main(video_file)
