import cv2
import os
import numpy as np

input_folder = "../dataset/frames/"
output_folder = "../dataset/annotations/"

os.makedirs(output_folder, exist_ok=True)

def simple_lane_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Only keep lower half (road area)
    h, w = thresh.shape
    mask = np.zeros_like(thresh)
    mask[int(h*0.5):h, :] = thresh[int(h*0.5):h, :]

    return mask

count = 0

for file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)

    mask = simple_lane_mask(img)

    cv2.imwrite(os.path.join(output_folder, file), mask)
    count += 1

print(f"Generated {count} labels")