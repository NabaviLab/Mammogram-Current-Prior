import cv2
import os
import numpy as np

def convert_images_to_rgb_opencv(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        try:
            img = cv2.imread(filepath)
            if img is None:
                print(f"Failed to load {filename}, skipping.")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            save_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.tif')
            cv2.imwrite(save_path, img_rgb)
            print(f"Processed and saved: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

input_folder = ''
output_folder = ''

convert_images_to_rgb_opencv(input_folder, output_folder)




