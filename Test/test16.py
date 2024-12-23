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
        self.prev_center = None  # Previous center position of the robot

    def track(self, obs):
        gray_frame = cv2.cvtColor(obs.frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to binary image
        _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the color frame and track the robot's center
        color_frame_with_contours = obs.frame.copy()
        center = None  # Reset center for the current frame
        
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Filter out small contours
                cv2.drawContours(color_frame_with_contours, [contour], -1, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    center = (center_x, center_y)  # Current center position
                    self.path.append(center)  # Track the center point
                    cv2.circle(color_frame_with_contours, (center_x, center_y), 5, (0, 0, 255), -1)  # Mark center

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
