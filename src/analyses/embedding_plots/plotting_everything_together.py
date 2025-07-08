import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

# ==== Configuration ====
image_dir = "/home/student/Desktop/Groundeep/outputs/images_combined/2D_PCA/uniform"
output_path = os.path.join(image_dir, "all_PCA_Uniform.jpg")

# ==== Collect and sort image paths ====
image_files = glob(os.path.join(image_dir, "*.jpg"))
image_paths = {}

for img_path in image_files:
    filename = os.path.basename(img_path)
    
    # Extract layer sizes at the beginning of the filename
    match = re.match(r"idbn_trained_\w+_(\d+)_(\d+)\.jpg", filename)
    if match:
        layer1, layer2 = int(match.group(1)), int(match.group(2))
        image_paths[(layer1, layer2)] = img_path
    else:
        print(f"⚠️ Skipped {filename}: could not extract layer sizes.")

# ==== Determine grid size ====
row_sizes = sorted(set(k[0] for k in image_paths))
col_sizes = sorted(set(k[1] for k in image_paths))
rows, cols = len(row_sizes), len(col_sizes)

# ==== Create grid figure ====
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

for i, row_val in enumerate(row_sizes):
    for j, col_val in enumerate(col_sizes):
        ax = axes[i, j] if rows > 1 else axes[j]
        key = (row_val, col_val)
        if key in image_paths:
            img = Image.open(image_paths[key])
            ax.imshow(img)
            ax.set_title(f"{row_val} → {col_val}", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=12)
        ax.axis('off')

fig.suptitle("PCA Uniform Grid by Hidden Layer Sizes", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ Grid saved to: {output_path}")
