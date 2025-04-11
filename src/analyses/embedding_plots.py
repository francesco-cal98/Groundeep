import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from matplotlib.image import imread
from PIL import Image
from glob import glob
import sys
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root) 
sys.path.append(current_dir) # Add the project root to sys.path

from src.analyses.embedding_analysis import Embedding_analysis  # Replace with actual import if needed

# Paths
data_path = "/home/student/Desktop/Groundeep/training_tensors/uniform/"
data_file = "NumStim_1to40_100x100_TR_uniform.npz"
network_dir = "/home/student/Desktop/Groundeep/networks/uniform/"
output_dir = "/home/student/Desktop/Groundeep/outputs/images_uniform"
os.makedirs(output_dir, exist_ok=True)

def plot_embeddings(emb, y, title="", colormap="viridis", save_path=None):
    print(f"Plotting embeddings for {title}...")
    fig, ax = plt.subplots(figsize=(10, 5))
    num_classes = len(np.unique(y))
    cmap = get_cmap(colormap, num_classes)
    norm = plt.Normalize(vmin=min(y), vmax=max(y))
    
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap, norm=norm, s=40)
    cbar = plt.colorbar(sc, ax=ax, ticks=np.linspace(min(y), max(y), num_classes // 5))
    cbar.set_label("Class Index")
    ax.set_title(title)

    if save_path:
        print(f"Saving plot to {save_path}...")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Process all network files
print("Loading network models and generating plots...")
pkl_files = glob(os.path.join(network_dir, "*.pkl"))
image_paths = []

for pkl_path in pkl_files:
    arch_name = os.path.splitext(os.path.basename(pkl_path))[0]
    img_path = os.path.join(output_dir, f"{arch_name}.jpg")

    # Skip if already processed
    if os.path.exists(img_path):
        print(f"Skipping {arch_name} — already processed.")
        image_paths.append(img_path)
        continue

    print(f"Processing network architecture: {arch_name}")
    
    analyser = Embedding_analysis(data_path, data_file, pkl_path)
    embeddings, labels = analyser._get_encodings()
    embeddings = np.array(embeddings, dtype=np.float64)

    mds = MDS(n_components=2, max_iter=100, n_jobs=1, dissimilarity='euclidean', random_state=42)
    emb = mds.fit_transform(embeddings)

    plot_embeddings(emb, labels, title=arch_name, save_path=img_path)
    image_paths.append(img_path)


print(f"Generated {len(image_paths)} individual plots. Now creating the final combined plot...")

# Create a final combined plot
cols = 3
rows = int(np.ceil(len(image_paths) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))

for idx, img_path in enumerate(image_paths):
    row, col = divmod(idx, cols)
    ax = axes[row, col] if rows > 1 else axes[col]
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(os.path.basename(img_path), fontsize=10)
    ax.axis('off')

# Turn off any unused subplots
for idx in range(len(image_paths), rows * cols):
    row, col = divmod(idx, cols)
    ax = axes[row, col] if rows > 1 else axes[col]
    ax.axis('off')

final_plot_path = os.path.join(output_dir, "all_embeddings.jpg")
print(f"Saving final combined plot to {final_plot_path}...")
fig.tight_layout()
fig.savefig(final_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print("All plots generated successfully!")
