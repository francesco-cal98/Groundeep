
import torch
import numpy as np 
import matplotlib.pyplot as plt 
import pickle as pkl
import cv2
import os

input_path = "/home/student/Desktop/Groundeep/circle_dataset/circle_dataset_300x300.npz"
output_path = "/home/student/Desktop/Groundeep/circle_dataset/circle_dataset_100x100.npz"
# Load data
data = np.load(input_path)
original_data = data['images']  # shape: (n_images, 200*200)

#original_data = D['D'].T  # likely shape (n_images, 200*200)
n_images = original_data.shape[0]

# Preallocate output array in float32
D_redrawn = np.zeros((n_images, 10000), dtype=np.uint8)

# Resize one image at a time to avoid memory explosion
for i in range(n_images):
    img_flat = original_data[i]
    img = img_flat.reshape(300, 300).astype(np.uint8)  # Convert only now
    resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)  # still float32
    D_redrawn[i] = resized.flatten()


# Save
np.savez(
    output_path,
    D=D_redrawn,
    N_list=data['numerosity'],
    cumArea_list=data['cumulative_area'],
    FA_list=data['field_radius'],
    CH_list=data['convex_hull'],
)

print("✅ Done resizing and saving. Final shape:", D_redrawn.shape)
