import cv2
import argparse
import numpy as np
import time
from ultralytics import YOLO  # Ensure ultralytics YOLO is correctly installed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time human detection using YOLO")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int,
                        help="Resolution of the webcam feed")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load the YOLO model, make sure the model file is in the correct path
    model = YOLO("yolov8l.pt")

    capture_interval = 5  # Interval between detections in seconds
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no frame is captured

        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            # Process the frame through YOLO
            results = model(frame)
            human_count = sum(1 for result in results if result.label == 0)  # Count 'person' detections

            # Log the count to a file
            with open("human_count.txt", "a") as file:
                file.write(f"Number of people detected: {human_count}\n")

            print(f"Number of people detected: {human_count}")  # Optional: Print count to console
            last_capture_time = current_time  # Update time of last processed frame

        cv2.imshow("Live Detection", frame)  # Show the current video frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

