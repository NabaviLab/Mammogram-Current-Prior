import os
import cv2
import numpy as np

def remove_artifacts(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.ndim > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    thresh1 = cv2.threshold(gray, 0, 65535, cv2.THRESH_BINARY)[1]
    thresh1 = 65535 - thresh1
    count_cols = np.count_nonzero(thresh1, axis=0)
    first_x = np.where(count_cols > 0)[0][0]
    last_x = np.where(count_cols > 0)[0][-1]
    count_rows = np.count_nonzero(thresh1, axis=1)
    first_y = np.where(count_rows > 0)[0][0]
    last_y = np.where(count_rows > 0)[0][-1]
    crop = img[first_y:last_y+1, first_x:last_x+1]
    thresh2 = thresh1[first_y:last_y+1, first_x:last_x+1]
    thresh2 = 65535 - thresh2
    contours, _ = cv2.findContours(thresh2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(thresh2, dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, -1)
    result = crop.copy()
    if result.ndim == 2 or result.shape[2] == 1:
        result[mask == 0] = 0
    else:  # Color image
        result[mask == 0] = (0, 0, 0)
    return result

def process_images(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.tif'):
            file_path = os.path.join(source_folder, filename)
            result = remove_artifacts(file_path)
            destination_file_path = os.path.join(destination_folder, filename)
            cv2.imwrite(destination_file_path, result, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            print(f"Processed and saved: {destination_file_path}")

source_folder = ''
destination_folder = ''

process_images(source_folder, destination_folder)
