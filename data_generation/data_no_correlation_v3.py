import os
import math
import random
import time
import numpy as np
from skimage.draw import disk
from skimage.morphology import convex_hull_image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
import scipy.stats

# -----------------------------
# CONFIGURAZIONE (modificabile)
# -----------------------------
GLOBAL_IMG_SIZE = 100
GLOBAL_NUM_LEVELS = np.arange(1, 33)  # N=1..32

ITEM_RADIUS_RANGE = (3.0, 10.0)       # raggio in pixel (float)
FIELD_RADIUS_RANGE = (15.0, 60.0)     # raggio campo
CUMAREA_TARGET_RANGE = (200.0, 1200.0) # target cumArea indipendente da N
CH_RATIO_ACCEPT = (0.10, 0.60)        # accettiamo CH ratio in questo range (soft filter)

SAMPLES_PER_LEVEL = 2000               # target per classe (puoi abbassare per debug)
UNIQUE_LAYOUTS_PER_LEVEL = 500         # numero di layout diversi da cui generare jitter variants
MAX_JITTER_SHIFT = 1.5                 # pixels, jitter per augmentazione (può essere molto piccolo)
MAX_PLACEMENT_ATTEMPTS = 5000
MAX_TOTAL_ATTEMPTS_FACTOR = 8000       # massima soglia di tentativi per livello
MIN_SPACING_FACTOR = 1.0               # moltiplicatore distanza minima relativa a 2*radius

SAVE_DIR = "stimuli_dataset_final_hybrid"
CHECKPOINT_LEVEL_DIR = os.path.join(SAVE_DIR, "levels")

# -----------------------------
# UTILITIES
# -----------------------------
def _compute_allowed_scale_for_disk(cx, cy, radius, centers, img_size):
    """
    Calcola il massimo fattore di scala per un disco tale che:
     - il disco rimanga dentro i bordi,
     - non superi la metà della distanza verso i centri vicini (per evitare overlap).
    Ritorna max_scale (>=0).
    """
    # distanza al bordo
    dist_left = cx - 0
    dist_right = img_size - 1 - cx
    dist_top = cy - 0
    dist_bottom = img_size - 1 - cy
    max_radius_border = min(dist_left, dist_right, dist_top, dist_bottom)
    if max_radius_border <= 0:
        return 0.0

    # distanza ai centri vicini
    min_dist_to_center = float('inf')
    for (ox, oy, orad) in centers:
        if ox == cx and oy == cy:
            continue
        d = math.hypot(cx - ox, cy - oy)
        # per non far collidere: scaled_radius + other_scaled_radius <= d - safety_margin
        # assumiamo scaling uniforme su tutti i dischi, quindi limite per questo disco è (d - 2*radius)/2 + radius approx
        min_dist_to_center = min(min_dist_to_center, d)

    # se non ci sono altri centri, min_dist_to_center rimane inf -> non vincola
    if min_dist_to_center == float('inf'):
        max_radius_centers = max_radius_border
    else:
        # vogliamo scaled_radius <= (min_dist_to_center - safety) / 2
        max_radius_centers = max(0.0, (min_dist_to_center - MIN_SPACING_FACTOR) / 2.0)

    # massimo radius possibile = min(max_radius_border, max_radius_centers)
    max_radius_possible = min(max_radius_border, max_radius_centers)

    # allowed scale is ratio of max_possible to current radius
    if radius <= 0:
        return 0.0
    return max_radius_possible / radius

def _draw_disks_binary(img_size, centers_with_radii):
    """
    Disegna dischi binari su un'immagine float in {0,1}.
    centers_with_radii: list of (cx, cy, r_int) where r_int is integer radius >=1
    """
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    for cx, cy, r in centers_with_radii:
        rr, cc = disk((cy, cx), r, shape=img.shape)
        img[rr, cc] = 1
    return img

# -----------------------------
# GENERAZIONE SINGOLO LAYOUT BASE
# -----------------------------
def generate_layout_base(N_target, img_size=GLOBAL_IMG_SIZE):
    """
    Genera centers e radii (float radii) senza scaling finale.
    Ritorna centers list: [(cx, cy, r_float), ...] oppure None se non possibile.
    """
    # sample field and base radius
    field_radius = np.random.uniform(*FIELD_RADIUS_RANGE)
    item_radius = np.random.uniform(*ITEM_RADIUS_RANGE)

    center = img_size // 2
    placement_radius_max = min(field_radius, center - item_radius - 1)
    if placement_radius_max <= 0:
        return None

    centers = []
    attempts = 0
    while len(centers) < N_target and attempts < MAX_PLACEMENT_ATTEMPTS:
        angle = np.random.uniform(0, 2 * math.pi)
        rpos = math.sqrt(np.random.uniform(0, placement_radius_max ** 2))
        cx = int(round(center + rpos * math.cos(angle)))
        cy = int(round(center + rpos * math.sin(angle)))

        if cx < 0 or cx >= img_size or cy < 0 or cy >= img_size:
            attempts += 1
            continue

        # enforce minimal non-overlap using current item_radius
        if all(math.hypot(cx - ox, cy - oy) >= (2.0 * item_radius + MIN_SPACING_FACTOR) for ox, oy, _ in centers):
            centers.append((cx, cy, item_radius))
        attempts += 1

    if len(centers) < N_target:
        return None

    return centers  # float radii

# -----------------------------
# PROVA A SCALARE I RAGGI PER RAGGIUNGERE CUMAREA TARGET
# -----------------------------
def try_scale_to_cumarea(centers, img_size, cum_area_target):
    """
    Prova a scalare uniformemente i radii dei dischi per ottenere approx cum_area_target.
    - centers: list of (cx, cy, r_float)
    Ritorna (img_bin_uint8, centers_scaled_int, cum_area) oppure None se non possibile.
    """
    # disegna con r_int = round(r_float)
    centers_int = [(cx, cy, max(1, int(round(r)))) for cx, cy, r in centers]
    img0 = _draw_disks_binary(img_size, centers_int).astype(np.float32)
    current_area = float(img0.sum())
    if current_area <= 0:
        return None

    # desideriamo scale_factor such that area scales ~ scale_factor^2 (area ~ r^2)
    desired_scale = math.sqrt(cum_area_target / (current_area + 1e-9))

    # ma dobbiamo limitare desired_scale in base ai max_scale_allowed di *ogni* disco
    max_scales = []
    for (cx, cy, r) in centers:
        ms = _compute_allowed_scale_for_disk(cx, cy, r, centers, img_size)
        max_scales.append(ms if ms > 0 else 0.0)
    global_max_scale = min(max_scales) if max_scales else 0.0

    # allow also slight relaxation: if global_max_scale==0 then fail
    if global_max_scale <= 0:
        return None

    applied_scale = min(desired_scale, global_max_scale * 0.999)  # small safety margin

    # apply scale to radii and redraw with integer radii
    centers_scaled = []
    for (cx, cy, r) in centers:
        r_scaled = max(1.0, r * applied_scale)
        centers_scaled.append((cx, cy, max(1, int(round(r_scaled)))))

    img_scaled = _draw_disks_binary(img_size, centers_scaled)
    cum_area = float(img_scaled.sum())

    # If area far from target ( >10% ), we consider fail (to keep quality)
    if abs(cum_area - cum_area_target) / (cum_area_target + 1e-9) > 0.12:
        return None

    return img_scaled.astype(np.uint8), centers_scaled, cum_area

# -----------------------------
# GENERA VARIANTI PER AUGMENTAZIONE (jitter)
# -----------------------------
def expand_with_jitter(base_centers, n_variants, img_size):
    """
    Genera n_variants immagini dalla stessa base centers aggiungendo piccolissimi jitter.
    Il jitter è limitato per non violare i vincoli di bordo/overlap.
    Ritorna lista di (img_uint8_flat, centers_variant, cum_area)
    """
    results = []
    base_radii = [r for (_,_,r) in base_centers]

    for k in range(n_variants):
        # small jitter of centers: gaussiana clipped
        new_centers = []
        ok = True
        for (cx, cy, r) in base_centers:
            dx = np.random.normal(scale=MAX_JITTER_SHIFT)
            dy = np.random.normal(scale=MAX_JITTER_SHIFT)
            ncx = int(round(cx + dx))
            ncy = int(round(cy + dy))
            # clip to keep margin for radius
            ncx = max(int(r+1), min(img_size - int(r) - 2, ncx))
            ncy = max(int(r+1), min(img_size - int(r) - 2, ncy))
            new_centers.append((ncx, ncy, r))

        # quick overlap check
        for i,(cx,cy,r) in enumerate(new_centers):
            for j,(ox,oy,orad) in enumerate(new_centers):
                if i==j: continue
                if math.hypot(cx-ox, cy-oy) < (r+orad - 0.5):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        img = _draw_disks_binary(img_size, new_centers).astype(np.uint8)
        results.append((img.flatten(), new_centers, float(img.sum())))
    return results

# -----------------------------
# GENERA UN LIVELLO (workflow ibrido)
# -----------------------------
def generate_level(args):
    N_level, samples_per_level, unique_layouts_target = args
    rng = random.Random()  # local RNG
    img_size = GLOBAL_IMG_SIZE

    images = []
    data = []

    # pre-check directories
    os.makedirs(CHECKPOINT_LEVEL_DIR, exist_ok=True)

    # we'll try to find 'unique_layouts_target' base layouts and then expand each
    base_layouts = []
    attempts = 0
    max_attempts = samples_per_level * MAX_TOTAL_ATTEMPTS_FACTOR

    # Phase A: collect base layouts (unique)
    while len(base_layouts) < unique_layouts_target and attempts < max_attempts:
        attempts += 1
        layout = generate_layout_base(N_level, img_size=img_size)
        if layout is None:
            continue

        # pick a cum_area target independent from N
        cum_target = float(np.random.uniform(*CUMAREA_TARGET_RANGE))

        scaled = try_scale_to_cumarea(layout, img_size, cum_target)
        if scaled is None:
            continue
        img_scaled, centers_scaled, cum_area = scaled

        # compute convex hull ratio (normalized by image area)
        try:
            ch = convex_hull_image(img_scaled > 0).sum()
        except Exception:
            ch = cum_area
        ch_ratio = ch / (img_size * img_size)

        # filter CH ratio softly
        if not (CH_RATIO_ACCEPT[0] <= ch_ratio <= CH_RATIO_ACCEPT[1]):
            continue

        # accept base layout
        base_layouts.append((centers_scaled, cum_area, ch_ratio))

    # If not enough unique layouts found, we will still attempt to expand what we have
    if len(base_layouts) == 0:
        return N_level, data, images  # empty

    # Phase B: expand via jitter to reach samples_per_level
    idx = 0
    while len(images) < samples_per_level and idx < len(base_layouts):
        centers_scaled, base_cum, ch_ratio = base_layouts[idx]
        # use jitter to generate many variants from this base
        # compute how many variants we need from this base
        remaining = samples_per_level - len(images)
        # try to get a chunk up to remaining but not exceed a limit per base
        n_var = min(remaining, max(50, int(remaining / (len(base_layouts) - idx + 1))))
        variants = expand_with_jitter(centers_scaled, n_var, img_size)
        # if expand returns fewer variants, accept them
        for img_flat, centers_var, cum_area in variants:
            # check ch ratio roughly
            try:
                chv = convex_hull_image(img_flat.reshape(img_size, img_size) > 0).sum()
            except Exception:
                chv = cum_area
            ch_ratio_v = chv / (img_size * img_size)
            if not (CH_RATIO_ACCEPT[0] <= ch_ratio_v <= CH_RATIO_ACCEPT[1]):
                continue
            images.append(img_flat.astype(np.uint8))
            data.append([N_level, float(cum_area), float(chv), float(cum_area / chv) if chv>0 else 0.0])
        idx += 1

    # If still not enough, try a last-resort brute-force generation (relaxed constraints)
    relax_attempts = 0
    while len(images) < samples_per_level and attempts < max_attempts and relax_attempts < 10000:
        attempts += 1
        relax_attempts += 1
        layout = generate_layout_base(N_level, img_size=img_size)
        if layout is None:
            continue
        # try a cum target but allow wider tolerance by increasing allowed scale
        cum_target = float(np.random.uniform(*CUMAREA_TARGET_RANGE))
        scaled = try_scale_to_cumarea(layout, img_size, cum_target)
        if scaled is None:
            continue
        img_scaled, centers_scaled, cum_area = scaled
        try:
            ch = convex_hull_image(img_scaled > 0).sum()
        except Exception:
            ch = cum_area
        ch_ratio = ch / (img_size * img_size)
        if not (CH_RATIO_ACCEPT[0] <= ch_ratio <= CH_RATIO_ACCEPT[1]):
            continue
        images.append(img_scaled.flatten().astype(np.uint8))
        data.append([N_level, float(cum_area), float(ch), float(cum_area / ch) if ch>0 else 0.0])

    # checkpoint save for this level
    level_fn = os.path.join(CHECKPOINT_LEVEL_DIR, f"level_{int(N_level)}_samples_{len(images)}.npz")
    np.savez_compressed(level_fn, D=np.array(images, dtype=np.uint8), meta=np.array(data, dtype=np.float32))

    return N_level, data, images

# -----------------------------
# GENERA DATASET COMPLETO (parallel)
# -----------------------------
def generate_dataset(output_dir=SAVE_DIR, samples_per_level=SAMPLES_PER_LEVEL, unique_layouts=UNIQUE_LAYOUTS_PER_LEVEL):
    if os.path.exists(output_dir):
        print(f"Cleaning existing folder {output_dir} ...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_LEVEL_DIR, exist_ok=True)

    tasks = [(int(N), int(samples_per_level), int(unique_layouts)) for N in GLOBAL_NUM_LEVELS]
    all_data = []
    all_images = []

    nproc = max(1, cpu_count() - 0)
    print(f"Using {nproc} processes (cpu_count={cpu_count()})")
    start_time = time.time()
    with Pool(processes=nproc) as pool:
        for N, data, images in tqdm(pool.imap_unordered(generate_level, tasks), total=len(tasks)):
            all_data.extend(data)
            all_images.extend(images)

    all_data = np.array(all_data, dtype=np.float32)
    all_images = np.array(all_images, dtype=np.uint8)

    out_fn = os.path.join(output_dir, "stimuli_dataset_hybrid.npz")
    np.savez_compressed(out_fn,
                        D=all_images,
                        N_list=all_data[:, 0] if all_data.size else np.array([]),
                        cumArea_list=all_data[:, 1] if all_data.size else np.array([]),
                        CH_list=all_data[:, 2] if all_data.size else np.array([]),
                        density_list=all_data[:, 3] if all_data.size else np.array([]))
    elapsed = time.time() - start_time
    print(f"Saved dataset to {out_fn} (generated {len(all_images)} images) in {elapsed/60:.1f} min")

    # Correlazioni su N>5
    if all_data.size:
        mask = all_data[:,0] > 5
        N_arr = all_data[mask, 0]
        print("\nCorrelazioni con N (N>5):")
        names = ["cumArea", "CH", "density"]
        for i, name in enumerate(names, start=1):
            try:
                r = scipy.stats.pearsonr(N_arr, all_data[mask, i])[0]
            except Exception:
                r = np.nan
            print(f"  {name:8s}: {r:.3f}")

    return out_fn

# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    # esempio: lascia girare tutta la notte
    fn = generate_dataset(samples_per_level=SAMPLES_PER_LEVEL, unique_layouts=UNIQUE_LAYOUTS_PER_LEVEL)
    print("Dataset file:", fn)
