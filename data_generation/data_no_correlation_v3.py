import os
import numpy as np
from skimage.draw import disk
from skimage.morphology import convex_hull_image
from skimage.transform import resize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
import scipy.stats

# ==========================================================
# CONFIGURAZIONE
# ==========================================================
GLOBAL_IMG_SIZE = 100
GLOBAL_NUM_LEVELS = np.arange(1, 33)  # N da 1 a 32

ITEM_RADIUS_RANGE = (3, 10)
FIELD_RADIUS_RANGE = (15, 60)

# range target iniziali (poi si allargano se serve)
CUMAREA_TARGET = (200, 1200)
CH_TARGET = (0.2, 0.3)

MAX_PLACEMENT_ATTEMPTS = 5000
MAX_TOTAL_ATTEMPTS_FACTOR = 10000

# ==========================================================
# GENERAZIONE DI UN SINGOLO STIMOLO
# ==========================================================
def generate_stimulus(N_target, disp_size=GLOBAL_IMG_SIZE):
    img = np.zeros((disp_size, disp_size), dtype=np.float32)

    field_radius = np.random.uniform(*FIELD_RADIUS_RANGE)
    item_radius = np.random.uniform(*ITEM_RADIUS_RANGE)

    center = disp_size // 2
    placement_radius_max = min(field_radius, center - item_radius - 1)
    if placement_radius_max <= 0:
        return None

    centers = []
    attempts = 0
    while len(centers) < N_target and attempts < MAX_PLACEMENT_ATTEMPTS:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, placement_radius_max ** 2))
        cx = int(round(center + r * np.cos(angle)))
        cy = int(round(center + r * np.sin(angle)))

        if (cx < 0 or cx >= disp_size or cy < 0 or cy >= disp_size):
            attempts += 1
            continue

        min_dist = 2 * item_radius + 1.0
        if all(np.hypot(cx - x, cy - y) >= min_dist for x, y in centers):
            centers.append((cx, cy))
        attempts += 1

    if len(centers) < N_target:
        return None

    int_radius = max(1, int(round(item_radius)))
    for cx, cy in centers:
        rr, cc = disk((cy, cx), int_radius, shape=img.shape)
        img[rr, cc] = 1.0

    img_resized = resize(img, (disp_size, disp_size),
                         anti_aliasing=False, preserve_range=True)

    # features
    cum_area = np.sum(img_resized > 0.5)
    try:
        ch = convex_hull_image(img_resized > 0.5).sum()
    except Exception:
        ch = cum_area
    density = cum_area / ch if ch > 0 else 0.0

    # porta immagine in 0–255
    img_resized = (img_resized * 255).astype(np.uint8)

    return img_resized, N_target, cum_area, density

# ==========================================================
# GENERAZIONE ADATTIVA PER UN LIVELLO (N)
# ==========================================================
def generate_level(args):
    N_level, samples_per_level = args
    images, data = [], []

    # vincoli iniziali
    cumarea_low, cumarea_high = CUMAREA_TARGET
    ch_low, ch_high = CH_TARGET

    tolerance_step_area = 100   # ogni volta allarghiamo ±100
    tolerance_step_ch = 0.05    # allarghiamo 0.05 in su e in giù

    attempts = 0
    max_attempts = samples_per_level * MAX_TOTAL_ATTEMPTS_FACTOR

    while len(images) < samples_per_level and attempts < max_attempts:
        result = generate_stimulus(N_level)
        attempts += 1
        if result is None:
            continue

        img, N, ca, dens = result

        # convex hull stimato via densità inversa
        ch_val = ca / dens if dens > 0 else ca

        # controlla vincoli
        if cumarea_low <= ca <= cumarea_high and ch_low <= (ch_val / (GLOBAL_IMG_SIZE**2)) <= ch_high:
            images.append(img.flatten())
            data.append([N, ca, ch_val, dens])

        # se dopo troppi tentativi non trovo nulla → rilassa vincoli
        if attempts % (MAX_PLACEMENT_ATTEMPTS * 5) == 0 and len(images) < samples_per_level:
            cumarea_low = max(50, cumarea_low - tolerance_step_area)
            cumarea_high = cumarea_high + tolerance_step_area
            ch_low = max(0.0, ch_low - tolerance_step_ch)
            ch_high = min(1.0, ch_high + tolerance_step_ch)

    return N_level, data, images

# ==========================================================
# GENERAZIONE DATASET COMPLETO
# ==========================================================
def generate_dataset(output_dir="stimuli_dataset_adaptive", samples_per_level=1000):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    tasks = [(N, samples_per_level) for N in GLOBAL_NUM_LEVELS]

    all_data, all_images = [], []

    with Pool(processes=min(cpu_count(), 12)) as pool:
        for N, data, images in tqdm(pool.imap_unordered(generate_level, tasks), total=len(tasks)):
            all_data.extend(data)
            all_images.extend(images)

    all_data = np.array(all_data, dtype=np.float32)
    all_images = np.array(all_images, dtype=np.uint8)

    np.savez_compressed(os.path.join(output_dir, "stimuli_dataset.npz"),
                        D=all_images,
                        N_list=all_data[:, 0],
                        cumArea_list=all_data[:, 1],
                        CH_list=all_data[:, 2],
                        density=all_data[:, 3])

    # Correlazioni (solo N > 5)
    mask = all_data[:, 0] > 5
    N_arr = all_data[mask, 0]
    print("\nCorrelazioni con N (N>5):")
    names = ["cumArea", "CH", "density"]
    for i, name in enumerate(names, start=1):
        try:
            r = scipy.stats.pearsonr(N_arr, all_data[mask, i])[0]
        except Exception:
            r = np.nan
        print(f"  {name:8s}: {r:.3f}")

    return os.path.join(output_dir, "stimuli_dataset.npz")


if __name__ == "__main__":
    fn = generate_dataset(samples_per_level=1500)
    print("Dataset file:", fn)
