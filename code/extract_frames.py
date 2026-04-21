import cv2
import os

# Input video path
video_path = "../dataset/videos/road1.mp4"

# Output folder
output_folder = "../dataset/frames/"

# Create folder if not exists
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save every frame
    if count % 5 == 0:
        cv2.imwrite(f"{output_folder}frame_{count}.jpg", frame)

    count += 1

print(f"Extracted {count} frames")

cap.release()