
import torch
import numpy as np 
import matplotlib.pyplot as plt 
import pickle as pkl
import cv2
import os

input_path = "/home/student/Desktop/Groundeep/training_tensors/uniform/NumStim_1to40_200x200_TR_uniform.pkl"
output_path = "/home/student/Desktop/Groundeep/training_tensors/uniform/NumStim_1to40_100x100_TR_uniform.npz"

# Load data
with open(input_path, 'rb') as f:
    D = pkl.load(f)


original_data = D['D'].T  # likely shape (n_images, 200*200)
n_images = original_data.shape[0]

# Preallocate output array in float32
D_redrawn = np.zeros((n_images, 10000), dtype=np.float32)

# Resize one image at a time to avoid memory explosion
for i in range(n_images):
    img_flat = original_data[i]
    img = img_flat.reshape(200, 200).astype(np.float32)  # Convert only now
    resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)  # still float32
    D_redrawn[i] = resized.flatten()


# Save
np.savez(
    output_path,
    D=D_redrawn,
    N_list=D['N_list']
)

print("✅ Done resizing and saving. Final shape:", D_redrawn.shape)
