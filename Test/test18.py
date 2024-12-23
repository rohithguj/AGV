import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

class CameraSpecs:
    def __init__(self):
        self.intrinsics = np.array([[800, 0, 320],
                                    [0, 800, 240],
                                    [0, 0, 1]])

class Frontend:
    def __init__(self, cam_specs):
        self.cam_specs = cam_specs
        self.path = []  # List of robot positions
        self.contour_colors = {}  # To store colors of each contour
        self.prev_contours = []  # To store contours from the previous frame
        self.contour_id_counter = 0  # Counter for assigning unique IDs to contours

    def generate_random_color(self):
        """Generate a random color for a contour."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def track(self, obs):
        gray_frame = cv2.cvtColor(obs.frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for better feature detection
        binary_frame = cv2.adaptiveThreshold(gray_frame, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 
                                             11, 2)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and track robot's center
        color_frame_with_contours = obs.frame.copy()
        centers = []  # Store all centers for this frame
        matched_colors = {}
        updated_contours = []

        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Filter out small contours
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    centers.append((center_x, center_y))  # Add center position

                    # Match with previous contours
                    best_match_id = None
                    best_similarity = float('inf')
                    for prev_contour in self.prev_contours:
                        if cv2.contourArea(prev_contour) > 200:  # Consider only significant contours
                            similarity = cv2.matchShapes(contour, prev_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                            if similarity < best_similarity:
                                best_similarity = similarity
                                best_match_id = id(prev_contour)  # Use the ID of the previous contour

                    # Initialize contour_id for new contours
                    contour_id = None

                    # If a match was found, update the contour
                    if best_match_id is not None and best_similarity < 0.1:  # Similarity threshold
                        updated_contours.append(contour)  # Keep the updated contour
                        matched_colors[best_match_id] = self.contour_colors.get(best_match_id, self.generate_random_color())
                    else:
                        # If no match was found, assign a new random color
                        contour_id = self.contour_id_counter
                        self.contour_id_counter += 1
                        matched_colors[contour_id] = self.generate_random_color()
                        updated_contours.append(contour)

                    # Draw filled contour with the matched or new color
                    color_to_use = matched_colors.get(best_match_id) if best_match_id is not None else matched_colors.get(contour_id)
                    cv2.drawContours(color_frame_with_contours, [contour], -1, color_to_use, thickness=cv2.FILLED)

                    # Mark center
                    cv2.circle(color_frame_with_contours, (center_x, center_y), 5, (0, 0, 255), -1)

        # Update path with current centers
        for center in centers:
            self.path.append(center)

        # Update previous contours for the next frame
        self.prev_contours = updated_contours

        # Update contour colors for the current frame
        self.contour_colors.update(matched_colors)

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
