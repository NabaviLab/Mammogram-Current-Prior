import os
import numpy as np
import pydicom
from PIL import Image

def convert_and_normalize_dcm_to_tif(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.dcm'):
            file_path = os.path.join(source_folder, filename)
            try:
                ds = pydicom.dcmread(file_path)
                image_array = ds.pixel_array.astype(float)
                normalized_image = ((image_array - image_array.min()) * (65535 / (image_array.max() - image_array.min()))).astype(np.uint16)
                img = Image.fromarray(normalized_image, mode='I;16')
                new_filename = os.path.splitext(filename)[0] + '.tif'
                img.save(os.path.join(destination_folder, new_filename), 'TIFF', compression='tiff_deflate')
                
                print(f"Converted and saved: {new_filename}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

source_folder = ''
destination_folder = ''

convert_and_normalize_dcm_to_tif(source_folder, destination_folder)
