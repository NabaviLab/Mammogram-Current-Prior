import cv2
import os
import numpy as np

def find_breast_contour(image):
    _, thresh = cv2.threshold(image, 1, 65535, cv2.THRESH_BINARY)
    thresh_8bit = np.uint8(thresh / 256)
    contours, _ = cv2.findContours(thresh_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def process_image(file_path, filename):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    contour = find_breast_contour(image)
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]
    
    size = max(cropped.shape[0], cropped.shape[1])
    square = np.zeros((size, size), dtype=np.uint16)
    
    if 'LEFT' in filename or 'LCC' in filename:  # Left breast images
        x_offset = 0
    else:
        x_offset = size - cropped.shape[1]
    
    y_offset = (size - cropped.shape[0]) // 2
    square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
    
    final_image = cv2.resize(square, (1024, 1024), interpolation=cv2.INTER_AREA)
    return final_image

input_folder = ''
output_folder = ''

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if any(tag in filename for tag in ['LEFT', 'LCC', 'RIGHT', 'RCC']):
        file_path = os.path.join(input_folder, filename)
        processed_image = process_image(file_path, filename)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed_image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        print(f'Processed and saved: {filename}')
