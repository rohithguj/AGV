import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class CameraSpecs:
    def __init__(self):
        self.intrinsics = np.array([[800, 0, 320],
                                    [0, 800, 240],
                                    [0, 0, 1]])

class Frontend:
    def __init__(self, cam_specs, vocab_size=100):
        self.cam_specs = cam_specs
        self.prev_pose = np.eye(4)
        self.state = "INIT"
        self.path = []
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.keyframes = []
        self.keyframe_threshold = 10
        self.vocab_size = vocab_size
        self.vocabulary = None
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.orb, cv2.KMeans())
        self.bow_trainer = cv2.BOWKMeansTrainer(vocab_size)
        self.descriptors_list = []

    def train_bow(self):
        if self.descriptors_list:
            self.vocabulary = self.bow_trainer.cluster()
            self.bow_extractor.setVocabulary(self.vocabulary)

    def track(self, obs):
        gray_frame = cv2.cvtColor(obs.frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray_frame, None)

        # Store descriptors for BoW training
        if des is not None:
            self.descriptors_list.append(des)
        
        # Train vocabulary if enough descriptors are collected
        if len(self.descriptors_list) >= 10:
            self.train_bow()

        if self.state == "INIT":
            self.prev_keypoints = kp
            self.prev_descriptors = des
            self.keyframes.append((self.prev_pose, self.prev_keypoints, self.prev_descriptors))
            self.state = "TRACKING"
            self.path.append((0, 0))
            return {"init": True}, None, None, None

        # Get BoW histogram
        bow_hist = self.bow_extractor.compute(gray_frame, kp)

        if self.state == "TRACKING":
            # Match descriptors with the previous frame's BoW histogram
            matches = self.bf.match(self.prev_descriptors, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) >= self.keyframe_threshold:
                points1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

                # Geometric verification using RANSAC
                E, mask = cv2.findEssentialMat(points2, points1, self.cam_specs.intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, points2, points1, self.cam_specs.intrinsics)
                    translation = t.flatten()
                    new_pose = np.eye(4)
                    new_pose[:3, :3] = R
                    new_pose[:3, 3] = translation
                    self.prev_pose = self.prev_pose @ new_pose
                    self.path.append((self.prev_pose[0, 3], self.prev_pose[1, 3]))
                    self.keyframes.append((self.prev_pose, self.prev_keypoints, self.prev_descriptors))

                    # Detect loops
                    self.detect_loop(obs, kp, des)

                    self.prev_keypoints = kp
                    self.prev_descriptors = des

                    return {"matches": len(good_matches), "pose": self.prev_pose}, kp, None, None

        return {"matches": 0}, None, None, None

    def detect_loop(self, obs, kp, des):
        for i, (pose, kpts, desc) in enumerate(self.keyframes):
            matches = self.bf.match(desc, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) >= self.keyframe_threshold:
                # Geometric verification using RANSAC
                points1 = np.float32([kpts[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

                E, mask = cv2.findEssentialMat(points2, points1, self.cam_specs.intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None:
                    print(f"Loop detected with keyframe {i}!")
                    self.optimize_graph(i)  # Optimize the graph when a loop is detected
                    break

    def optimize_graph(self, loop_index):
        # Placeholder for pose graph optimization
        print("Optimizing graph...")

class Observation:
    def __init__(self, frame):
        self.frame = frame

def main(video_file):
    cap = cv2.VideoCapture(video_file)
    cam_specs = CameraSpecs()
    frontend = Frontend(cam_specs)

    plt.ion()
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

        if kp is not None:
            cv2.drawKeypoints(frame, kp, frame)

        cv2.imshow('Video', frame)

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
    video_file = "/home/rohith-pt7726/AGV/Test/test2.mp4"  # Change this to your video file path
    main(video_file)
