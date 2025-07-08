import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.morphology import convex_hull_image
from collections import defaultdict
from tqdm import tqdm
import scipy.stats

# --- CONFIGURAZIONE ---
disp_size = 200
num_levels = np.linspace(4, 40, 37, dtype=int)
min_dist = 4
target_per_level = 2000
output_dir = "circle_dataset"
os.makedirs(output_dir, exist_ok=True)

# Independent parameter sampling to break correlations
def sample_parameters(N):
    # Sample each parameter independently, not based on N
    ca = np.random.uniform(800, 6000)  # Fixed range regardless of N
    fr = np.random.uniform(30, 100)    # Fixed range regardless of N
    
    # Add noise to break systematic relationships
    ca *= np.random.uniform(0.7, 1.5)
    fr *= np.random.uniform(0.8, 1.3)
    
    return ca, fr

def generate_circle_stimulus(N, total_area, field_radius, disp_size, min_dist=4, max_attempts=100):
    img = np.zeros((disp_size, disp_size), dtype=np.uint8)
    centers = []
    radius = np.sqrt(total_area / (N * np.pi))
    
    # KEY FIX: Control convex hull through PLACEMENT STRATEGY, not post-scaling
    # Target convex hull area sampled independently of N
    target_ch_area = np.random.uniform(2000, 12000)  # Independent of N
    
    # Calculate required spread to achieve target convex hull
    # Rough approximation: convex_hull ≈ π * effective_radius²
    effective_radius = np.sqrt(target_ch_area / np.pi)
    max_spread = min(effective_radius * 0.8, field_radius * 0.9)
    
    dynamic_min_dist = max(1, min_dist * (0.5 if N > 25 else 1.0))
    
    # Place circles to fill the target convex hull area
    placement_attempts = 0
    max_total_attempts = min(max_attempts * 3, 500)  # Cap total attempts
    while len(centers) < N and placement_attempts < max_total_attempts:
        # Strategic placement - simplified to avoid getting stuck
        if len(centers) < 2:
            # First circles: establish hull
            r = np.random.uniform(max_spread * 0.5, max_spread)
        else:
            # Mix of boundary and interior
            if np.random.random() < 0.3:
                r = np.random.uniform(max_spread * 0.6, max_spread)  # boundary
            else:
                r = np.random.uniform(0, max_spread * 0.7)  # interior
            
        theta = np.random.uniform(0, 2 * np.pi)
        cx = int(disp_size // 2 + r * np.cos(theta))
        cy = int(disp_size // 2 + r * np.sin(theta))

        # Boundary check
        if (cx < radius or cx >= disp_size - radius or 
            cy < radius or cy >= disp_size - radius):
            placement_attempts += 1
            continue

        # Distance check
        if all(np.hypot(cx - x, cy - y) > 2 * radius + dynamic_min_dist 
               for x, y in centers):
            centers.append((cx, cy))
            rr, cc = disk((cy, cx), radius, shape=img.shape)
            img[rr, cc] = 1
            
        placement_attempts += 1

    if len(centers) < max(2, N * 0.5):  # More lenient threshold
        return None, None, None, None, None

    # Calculate final metrics
    ch_area = convex_hull_image(img).sum()
    actual_N = len(centers)
    
    # CRITICAL: Vary item size to control density independently
    # If convex hull is larger than expected, make circles smaller to reduce density
    # If convex hull is smaller than expected, make circles larger to increase density
    ch_ratio = ch_area / target_ch_area
    size_adjustment = np.random.uniform(0.7, 1.4) / np.sqrt(ch_ratio)
    adjusted_radius = radius * size_adjustment
    
    # Redraw with adjusted size
    img.fill(0)
    adjusted_area = 0
    for cx, cy in centers:
        rr, cc = disk((cy, cx), adjusted_radius, shape=img.shape)
        img[rr, cc] = 1
        adjusted_area += np.pi * adjusted_radius**2
    
    # Recalculate final convex hull
    ch_area = convex_hull_image(img).sum()
    density = actual_N / ch_area if ch_area > 0 else 0
    
    return img, ch_area, centers, density, adjusted_radius

def generate_balanced_batch(num_levels, target_count, disp_size):
    data, images = [], []
    
    # Track parameters to ensure variety
    param_history = {'ca': [], 'fr': [], 'ch': [], 'density': []}
    
    for N in num_levels:
        count = 0
        attempts = 0
        
        while count < target_count and attempts < 1000:  # Cap attempts per N
            ca, fr = sample_parameters(N)
            
            # Encourage parameter diversity
            if len(param_history['ca']) > 50:
                # Bias towards unexplored parameter space
                ca_mean = np.mean(param_history['ca'][-50:])
                if abs(ca - ca_mean) < 1000:  # Too similar to recent values
                    ca = np.random.uniform(800, 6000)  # Resample
                    
                fr_mean = np.mean(param_history['fr'][-50:])
                if abs(fr - fr_mean) < 15:
                    fr = np.random.uniform(30, 100)

            result = generate_circle_stimulus(N, ca, fr, disp_size)
            img, ch, centers, density, size = result
            
            if img is not None and len(centers) >= max(2, N * 0.5):  # More lenient
                # Update actual N to number of circles placed
                actual_N = len(centers)
                data.append((actual_N, ca, fr, ch, density, size))
                images.append(img)
                
                # Track parameter history
                param_history['ca'].append(ca)
                param_history['fr'].append(fr)
                param_history['ch'].append(ch)
                param_history['density'].append(density)
                
                count += 1
            attempts += 1
            
    return data, images

def check_correlations(data_arr, threshold=0.25):
    N_arr, CA_arr, FR_arr, CH_arr, dens_arr, size_arr = data_arr.T
    correlations = {
        'Convex Hull': scipy.stats.pearsonr(CH_arr, N_arr)[0],
        'Cumulative Area': scipy.stats.pearsonr(CA_arr, N_arr)[0],
        'Field Radius': scipy.stats.pearsonr(FR_arr, N_arr)[0],
        'Density': scipy.stats.pearsonr(dens_arr, N_arr)[0],
        'Item Size': scipy.stats.pearsonr(size_arr, N_arr)[0]
    }
    
    print("Correlations with Numerosity:")
    for k, v in correlations.items():
        status = "✓" if abs(v) < threshold else "✗"
        print(f"  {k}: {v:.3f} {status}")
    
    return all(abs(v) < threshold for v in correlations.values())

def save_dataset(data, images, filename):
    data = np.array(data)
    np.savez_compressed(
        filename,
        images=np.array(images),
        numerosity=data[:, 0],
        cumulative_area=data[:, 1],
        field_radius=data[:, 2],
        convex_hull=data[:, 3],
        density=data[:, 4],
        item_size=data[:, 5]
    )
    print(f"Dataset saved: {filename}")

def plot_correlations(data, filename):
    N_arr, CA_arr, FR_arr, CH_arr, dens_arr, size_arr = data.T
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    plots = [
        (N_arr, CH_arr, "Numerosity vs Convex Hull"),
        (N_arr, CA_arr, "Numerosity vs Cumulative Area"),
        (N_arr, FR_arr, "Numerosity vs Field Radius"),
        (N_arr, dens_arr, "Numerosity vs Density"),
        (N_arr, size_arr, "Numerosity vs Item Size"),
        (CH_arr, dens_arr, "Convex Hull vs Density")
    ]
    
    for i, (x, y, title) in enumerate(plots):
        axes[i].scatter(x, y, alpha=0.3, s=1)
        axes[i].set_title(f"{title}\nr={scipy.stats.pearsonr(x,y)[0]:.3f}")
        axes[i].set_xlabel(title.split(' vs ')[0])
        axes[i].set_ylabel(title.split(' vs ')[1])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    print("Starting dataset generation...")
    start_time = time.time()
    
    # Generate data in batches
    batch_size = 10
    all_data, all_images = [], []
    
    with tqdm(total=len(num_levels) * target_per_level, desc="Generating") as pbar:
        for i in range(0, len(num_levels), batch_size):
            batch_levels = num_levels[i:i+batch_size]
            data_batch, img_batch = generate_balanced_batch(batch_levels, target_per_level, disp_size)
            
            all_data.extend(data_batch)
            all_images.extend(img_batch)
            pbar.update(len(data_batch))

    # Final correlation check
    data_arr = np.array(all_data)
    print(f"\nGenerated {len(all_data)} stimuli")
    is_valid = check_correlations(data_arr)

    # Save results
    final_file = os.path.join(output_dir, "circle_dataset.npz")
    save_dataset(all_data, all_images, final_file)
    plot_correlations(data_arr, os.path.join(output_dir, "correlations.png"))

    print(f"\nDataset valid: {'✓' if is_valid else '✗'}")
    print(f"Time: {(time.time() - start_time)/60:.1f} minutes")