import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
import sys

# Optional: add project paths
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from src.analyses.embedding_analysis import Embedding_analysis  # Your analysis class

# Paths
data_path = "/home/student/Desktop/Groundeep/training_tensors/uniform/"
data_file = "NumStim_1to40_100x100_TR_uniform.npz"
pkl_path = "/home/student/Desktop/Groundeep/networks/uniform/idbn/idbn_trained_uniform_500_1500.pkl"  # <- Pick one
output_dir = "/home/student/Desktop/Groundeep/outputs/images_uniform/2D_MDS_radius"
os.makedirs(output_dir, exist_ok=True)

# Load embeddings
analyser = Embedding_analysis(data_path, data_file, pkl_path)
embeddings, labels = analyser._get_encodings()
embeddings = np.array(embeddings, dtype=np.float64)

# 2D MDS
print("Running 2D MDS...")
mds = MDS(n_components=2, max_iter=300, n_jobs=1, dissimilarity='euclidean', random_state=42)
emb = mds.fit_transform(embeddings)

# Radius: You could use distance from origin or from centroid
centroid = np.mean(emb, axis=0)
radii = np.linalg.norm(emb - centroid, axis=1)

# Plot
print("Plotting 2D MDS with radius coloring...")
fig, ax = plt.subplots(figsize=(10, 6))
num_classes = len(np.unique(labels))
cmap = get_cmap("viridis", num_classes)
#orm = Normalize(vmin=min(labels), vmax=max(labels))

# You can also use `radii` for marker sizes or as color if desired
scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap, s=radii * 5, alpha=0.8)
cbar = plt.colorbar(scatter, ax=ax, ticks=np.linspace(min(labels), max(labels), num_classes // 5))
cbar.set_label("Class Index")
ax.set_title("2D MDS of Embeddings (Colored by Class, Size by Radius)")
ax.set_xlabel("MDS Dimension 1")
ax.set_ylabel("MDS Dimension 2")

# Save
save_path = os.path.join(output_dir, "2D_MDS_single_network.jpg")
print(f"Saving to {save_path}...")
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print("Done!")
