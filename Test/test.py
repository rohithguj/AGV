import cv2
import numpy as np
import attr

@attr.s(auto_attribs=True)
class CameraSpecs:
    intrinsics: np.ndarray
    extrinsics: np.ndarray

# Example intrinsics for a typical camera (adjust as necessary)
intrinsics = np.array([[800, 0, 320],
                       [0, 800, 240],
                       [0, 0, 1]])
extrinsics = np.eye(4)

cam_specs = CameraSpecs(intrinsics=intrinsics, extrinsics=extrinsics)

class OrbBasedFeatureMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def match(self, frame1, frame2):
        kp1, des1 = self.orb.detectAndCompute(frame1, None)
        kp2, des2 = self.orb.detectAndCompute(frame2, None)
        
        if des1 is None or des2 is None:
            return [], None, None
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return matches, kp1, kp2

@attr.s(auto_attribs=True)
class Observation:
    frame: np.ndarray

class Frontend:
    def __init__(self, cam_specs):
        self.cam_specs = cam_specs
        self.state = "INIT"
        self.matcher = OrbBasedFeatureMatcher()
        self.keyframe = None

    def track(self, obs):
        if self.state == "INIT":
            self.keyframe = obs.frame
            self.state = "TRACKING"
            return {"init": True}
        else:
            matches, kp1, kp2 = self.matcher.match(self.keyframe, obs.frame)
            if len(matches) > 10:  # Arbitrary threshold
                self.keyframe = obs.frame  # Update keyframe
                return {"matches": len(matches)}
            else:
                return {"matches": 0}

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frontend = Frontend(cam_specs)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error.")
            break

        # Convert frame to grayscale for feature detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create an Observation
        obs = Observation(gray_frame)

        # Track using the Frontend
        result = frontend.track(obs)

        # Show results or feedback
        cv2.imshow('vSLAM', frame)
        print(result)  # Print tracking results

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "/home/zlabs/Desktop/test1.mp4"  # Update with your video file path
    main(video_file)
