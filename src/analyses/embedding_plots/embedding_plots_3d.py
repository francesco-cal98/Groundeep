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

# ==== USER SETTINGS ====
arch_name = "idbn_trained_zipfian_500_1500.pkl"  # <-- Replace with your filename
data_path = "/home/student/Desktop/Groundeep/training_tensors/zipfian/"
data_file = "NumStim_1to40_100x100_TR_zipfian.npz"
pkl_path = f"/home/student/Desktop/Groundeep/networks/zipfian/{arch_name}"
output_path = f"/home/student/Desktop/Groundeep/outputs/images_zipfian/3D_MDS/{arch_name.replace('.pkl', '.jpg')}"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ==== FUNCTION TO PLOT EMBEDDING ====
def plot_3d_embeddings(embeddings, labels, title="", colormap="viridis", save_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    num_classes = len(np.unique(labels))
    cmap = get_cmap(colormap, num_classes)
    norm = plt.Normalize(vmin=min(labels), vmax=max(labels))

    sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
                    c=labels, cmap=cmap, norm=norm, s=40)
    
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Class Index")

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")

    if save_path:
        print(f"Saving plot to: {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==== LOAD & PROCESS ARCHITECTURE ====
print(f"Processing architecture: {arch_name}")
analyser = Embedding_analysis(data_path, data_file, pkl_path)
embeddings, labels = analyser._get_encodings()
embeddings = np.array(embeddings, dtype=np.float64)

# ==== APPLY MDS ====
mds = MDS(n_components=3, max_iter=300, n_jobs=1, dissimilarity='euclidean', random_state=42)
emb_3d = mds.fit_transform(embeddings)

# ==== PLOT & SAVE ====
plot_3d_embeddings(emb_3d, labels, title=arch_name.replace('.pkl', ''), save_path=output_path)

print("Done! ✅")
