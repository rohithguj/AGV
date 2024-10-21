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
        self.path = []
        self.prev_frame = None
        self.prev_points = None

    def track(self, obs):
        gray_frame = cv2.cvtColor(obs.frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for better feature detection
        binary_frame = cv2.adaptiveThreshold(gray_frame, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 
                                             11, 2)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the color frame and track the robot's center
        color_frame_with_contours = obs.frame.copy()
        centers = []  # Store all centers for this frame
        
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Filter out small contours
                cv2.drawContours(color_frame_with_contours, [contour], -1, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    centers.append((center_x, center_y))  # Add center position

        # Track using Optical Flow if previous points exist
        if self.prev_frame is not None and self.prev_points is not None and centers:
            # Calculate optical flow for the previous points
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_frame, self.prev_points, None)
            
            # Ensure new_points and status are valid
            if new_points is not None and status is not None:
                good_new = new_points[status.flatten() == 1]  # Flatten status for boolean indexing
                good_old = self.prev_points[status.flatten() == 1]
                
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(color_frame_with_contours, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)
                    cv2.circle(color_frame_with_contours, (int(a), int(b)), 5, (0, 0, 255), -1)
                    self.path.append((int(a), int(b)))  # Track the path

        # Update previous frame and points
        self.prev_frame = gray_frame.copy()
        self.prev_points = np.array(centers, dtype=np.float32) if centers else None

        return contours, binary_frame, color_frame_with_contours

    def plot_path(self):
        """ Plot the robot's path in a separate window. """
        if not self.path:
            return

        path_np = np.array(self.path)
        plt.figure()
        plt.title('Robot Path')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.plot(path_np[:, 0], path_np[:, 1], marker='o', color='blue', markersize=3)
        plt.axis('equal')
        plt.show()

class Observation:
    def __init__(self, frame):
        self.frame = frame

def main(video_file):
    cap = cv2.VideoCapture(video_file)
    cam_specs = CameraSpecs()
    frontend = Frontend(cam_specs)

    plt.ion()  # Turn on interactive mode for real-time plotting
    plt.figure(figsize=(12, 6))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        obs = Observation(frame)
        contours, binary_frame, color_frame_with_contours = frontend.track(obs)

        # Display the color frame with the robot path
        cv2.imshow('Color Frame with Robot Path', color_frame_with_contours)

        # Display the binary frame
        cv2.imshow('Binary Frame', binary_frame)

        # Plot the path in real-time
        if frontend.path:
            path_np = np.array(frontend.path)
            plt.clf()  # Clear the previous plot
            plt.title('Robot Path')
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.plot(path_np[:, 0], path_np[:, 1], marker='o', color='blue', markersize=3)
            plt.axis('equal')
            plt.pause(0.01)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    video_file = "/home/rohith-pt7726/AGV/Test/test2.mp4"  # Update with your video file path
    main(video_file)
