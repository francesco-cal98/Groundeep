import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.morphology import convex_hull_image
from multiprocessing import Pool, cpu_count
import scipy.stats

# --- CONFIG ---
disp_size = 300 # Questo valore sarà sovrascritto dalla funzione generate_dataset
num_levels = np.concatenate([
    np.array([1, 2, 3]),
    np.linspace(4, 32, 29, dtype=int)
])
target_per_level = 1500
output_dir = "circle_dataset" # Questo sarà sovrascritto per il 100x100
os.makedirs(output_dir, exist_ok=True) # Questa riga può essere rimossa o lasciata, verrà ricreata dalla funzione

def generate_simple_stimulus(N):
    img = np.zeros((disp_size, disp_size), dtype=np.uint8)

    cumulative_area = np.random.uniform(3000, 12000)
    field_radius = np.random.uniform(50, 100)
    item_radius = np.random.uniform(3, 12)

    center_x, center_y = disp_size // 2, disp_size // 2
    placement_radius = min(field_radius, (disp_size // 2) - item_radius - 5)

    centers = []
    attempts = 0
    max_attempts = 5000

    while len(centers) < N and attempts < max_attempts:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, placement_radius)
        cx = int(center_x + r * np.cos(angle))
        cy = int(center_y + r * np.sin(angle))

        if (cx < item_radius or cx >= disp_size - item_radius or
            cy < item_radius or cy >= disp_size - item_radius):
            attempts += 1
            continue

        min_dist = item_radius * 2.1
        if all(np.hypot(cx - x, cy - y) >= min_dist for x, y in centers):
            centers.append((cx, cy))

        attempts += 1

    for cx, cy in centers:
        rr, cc = disk((cy, cx), item_radius, shape=img.shape)
        img[rr, cc] = 255

    actual_N = len(centers)
    if actual_N < max(1, N * 0.7):
        return None

    try:
        ch_area = convex_hull_image(img > 0).sum()
    except:
        ch_area = actual_N * np.pi * item_radius**2 * 2 if N <= 3 else np.pi * min(field_radius, 100)**2

    actual_cumulative_area = actual_N * np.pi * item_radius**2
    density = actual_N / ch_area if ch_area > 0 else 0

    return (img, actual_N, actual_cumulative_area, field_radius, ch_area, density, item_radius)

def generate_level_fast(N):
    data = []
    images = []
    count = 0
    attempts = 0
    max_total_attempts = 20000

    while count < target_per_level and attempts < max_total_attempts:
        result = generate_simple_stimulus(N)
        if result is not None:
            img, actual_N, actual_ca, field_r, ch_area, density, radius = result
            data.append([actual_N, actual_ca, field_r, ch_area, density, radius])
            images.append(img)
            count += 1

            if count % 100 == 0:
                print(f"Level {N}: {count}/{target_per_level}")
        attempts += 1

    print(f"Level {N}: {count} samples in {attempts} attempts")
    return N, data, images

def parallel_generate_fast(levels):
    cores = min(cpu_count(), 14)
    print(f"Using {cores} cores...")
    with Pool(cores) as pool:
        results = pool.map(generate_level_fast, levels)

    all_data = []
    all_images = []

    for level, data, imgs in results:
        all_data.extend(data)
        all_images.extend(imgs)
        print(f"Collected level {level}: {len(data)} samples")

    return all_data, all_images

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

    return correlations, all(abs(v) < threshold for v in correlations.values())

def save_dataset(data, images, filename):
    data = np.array(data)
    # Appiattisci ogni immagine
    flattened_images = np.array([img.flatten() for img in images])

    np.savez_compressed(
        filename,
        D=flattened_images, # Ora D conterrà le immagini vettorizzate
        N_list=data[:, 0],
        cumArea_list=data[:, 1],
        FA_list=data[:, 2],
        CH_list=data[:, 3],
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
        r, p = scipy.stats.pearsonr(x, y)
        axes[i].set_title(f"{title}\nr={r:.3f}")
        axes[i].set_xlabel(title.split(' vs ')[0])
        axes[i].set_ylabel(title.split(' vs ')[1])

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def generate_dataset(size, output_directory):
    global disp_size, output_dir
    disp_size = size
    output_dir = output_directory
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    print(f"Generating {size}x{size} dataset...")
    print(f"Levels: {len(num_levels)} total (1,2,3,4-40)")
    print(f"Target: {target_per_level} samples × {len(num_levels)} levels = {target_per_level * len(num_levels)} total")

    data, images = parallel_generate_fast(num_levels)

    print(f"\nGenerated {len(data)} samples")

    data_arr = np.array(data)
    correlations, is_valid = check_correlations(data_arr)

    dataset_filename = f"circle_dataset_{size}x{size}_v1.npz"
    correlations_filename = f"correlations_{size}x{size}_v1.png"

    save_dataset(data, images, os.path.join(output_dir, dataset_filename))
    plot_correlations(data_arr, os.path.join(output_dir, correlations_filename))

    print(f"\nDataset valid: {'✓' if is_valid else '✗'}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")

    print("\n" + "="*60)
    print(f"FINAL {size}x{size} DATASET SUMMARY")
    print("="*60)

    N_values = data_arr[:, 0]
    total_samples = len(data_arr)

    print(f"Total samples generated: {total_samples:,}")
    print(f"Target samples: {target_per_level * len(num_levels):,}")
    print(f"Success rate: {(total_samples / (target_per_level * len(num_levels))) * 100:.1f}%")

    print(f"\nSamples per level:")
    level_counts = {}
    for level in sorted(set(N_values)):
        count = np.sum(N_values == level)
        level_counts[int(level)] = count
        print(f"  Level {int(level):2d}: {count:4d} samples")

    print(f"\nLevel completion rates:")
    for level, count in level_counts.items():
        completion = (count / target_per_level) * 100
        status = "✓" if completion >= 90 else "⚠" if completion >= 70 else "✗"
        print(f"  Level {level:2d}: {completion:5.1f}% {status}")

    CA_arr, FR_arr, CH_arr, dens_arr, size_arr = data_arr[:, 1], data_arr[:, 2], data_arr[:, 3], data_arr[:, 4], data_arr[:, 5]

    print(f"\nData quality metrics:")
    print(f"  Cumulative Area: {CA_arr.min():.0f} - {CA_arr.max():.0f} (mean: {CA_arr.mean():.0f})")
    print(f"  Field Radius: {FR_arr.min():.1f} - {FR_arr.max():.1f} (mean: {FR_arr.mean():.1f})")
    print(f"  Convex Hull: {CH_arr.min():.0f} - {CH_arr.max():.0f} (mean: {CH_arr.mean():.0f})")
    print(f"  Density: {dens_arr.min():.4f} - {dens_arr.max():.4f} (mean: {dens_arr.mean():.4f})")
    print(f"  Item Size: {size_arr.min():.1f} - {size_arr.max():.1f} (mean: {size_arr.mean():.1f})")

    print(f"\nFinal correlations with numerosity:")
    for name, corr in correlations.items():
        status = "✓ GOOD" if abs(corr) < 0.25 else "⚠" if abs(corr) < 0.5 else "✗ BAD"
        print(f"  {name:15s}: {corr:6.3f} {status}")

    print("\n" + "="*60)
    print(f"{size}x{size} DATASET GENERATION COMPLETE!")
    print("="*60)

    print(f"\nExample images saved as {size}x{size} uint8 arrays")
    print(f"Values: 0 (black background), 255 (white circles)")
    print(f"File: {os.path.join(output_dir, dataset_filename)}")

    return data_arr, images, correlations

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    print("\n\nGENERATING 100x100 VERSION")
    print("=" * 50)
    data_100, images_100, correlations_100 = generate_dataset(100, "circle_dataset_100x100")

    print("\n" + "="*60)
    print("FINAL SUMMARY: 100x100 Dataset")
    print("="*60)
    print(f"Total samples generated: {len(data_100)} samples")

    print(f"\n100x100 Correlations:")
    for name, corr in correlations_100.items():
        print(f"  {name}: {corr:.3f}")

    print(f"\nFiles saved:")
    print(f"  100x100: circle_dataset_100x100/circle_dataset_100x100.npz")