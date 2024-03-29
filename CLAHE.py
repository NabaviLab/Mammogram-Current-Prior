import cv2
import numpy as np
import os
from skimage import io

def apply_clahe_to_folder(input_folder, output_folder, clip_limit=2.0, grid_size=(8,8)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            img = io.imread(input_path)
            
            if img.ndim > 2:
                print(f"Skipping {filename}, as it is not a grayscale image.")
                continue

            img_normalized = img.astype(np.float32) / 65535.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            clahe_img = clahe.apply((img_normalized * 255).astype(np.uint8))
            clahe_img_16bit = (clahe_img.astype(np.float32) / 255 * 65535).astype(np.uint16)
            
            io.imsave(output_path, clahe_img_16bit)

apply_clahe_to_folder('', '')
