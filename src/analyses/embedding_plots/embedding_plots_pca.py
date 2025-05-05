import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
from PIL import Image
from glob import glob
import sys

# Impostazioni dei percorsi
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from src.analyses.embedding_analysis import Embedding_analysis

# Percorsi
data_path = "/home/student/Desktop/Groundeep/training_tensors/zipfian/"
data_file = "NumStim_1to40_100x100_TR_zipfian.npz"
network_dir = "/home/student/Desktop/Groundeep/networks/zipfian"
output_dir = "/home/student/Desktop/Groundeep/outputs/images_zipfian/2D_pca"
os.makedirs(output_dir, exist_ok=True)

def plot_embeddings(emb, y, title="", colormap="viridis", save_path=None, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    num_classes = len(np.unique(y))
    cmap = get_cmap(colormap, num_classes)
    norm = plt.Normalize(vmin=min(y), vmax=max(y))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap, norm=norm, s=40)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Class Index")
    ax.set_title(title)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Step 1: Esegui PCA per ogni rete e memorizza i limiti
pkl_files = glob(os.path.join(network_dir, "*.pkl"))
embeddings_data = []
x_lims, y_lims = [], []

for pkl_path in pkl_files:
    arch_name = os.path.splitext(os.path.basename(pkl_path))[0]
    analyser = Embedding_analysis(data_path, data_file, pkl_path)
    embeddings, labels = analyser._get_encodings()
    embeddings = np.array(embeddings, dtype=np.float64)

    # PCA per rete
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    # Memorizza i limiti per ogni rete
    x_lims.append((emb_2d[:, 0].min(), emb_2d[:, 0].max()))
    y_lims.append((emb_2d[:, 1].min(), emb_2d[:, 1].max()))

    embeddings_data.append((arch_name, emb_2d, labels))

# Step 2: Calcola i limiti globali (minimo e massimo) per ogni asse
x_min = min([lim[0] for lim in x_lims])
x_max = max([lim[1] for lim in x_lims])
y_min = min([lim[0] for lim in y_lims])
y_max = max([lim[1] for lim in y_lims])

# Step 3: Plot con limiti globali uguali per tutti
image_paths = []
for arch_name, emb_2d, labels in embeddings_data:
    img_path = os.path.join(output_dir, f"{arch_name}.jpg")
    plot_embeddings(emb_2d, labels, title=arch_name, save_path=img_path,
                    xlim=(x_min, x_max), ylim=(y_min, y_max))
    image_paths.append(img_path)

# Step 4: Crea il plot finale con tutte le immagini
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

# Nascondi gli assi inutilizzati
for idx in range(len(image_paths), rows * cols):
    row, col = divmod(idx, cols)
    ax = axes[row, col] if rows > 1 else axes[col]
    ax.axis('off')

final_path = os.path.join(output_dir, "all_embeddings_PCA_aligned_uniform.jpg")
fig.tight_layout()
fig.savefig(final_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print("✅ PCA fatta per ogni rete con limiti comuni per tutti i plot.")
