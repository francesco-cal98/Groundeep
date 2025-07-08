import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from multiprocessing import Pool, cpu_count, current_process
import scipy.stats
import random
import cv2
import shutil
from skimage.draw import disk
from skimage.transform import resize
from numba import jit, types
from numba.typed import List
from tqdm import tqdm # Importa tqdm

# --- CONFIGURAZIONE PRINCIPALE ---
DISP_SIZE = 200
FINAL_OUTPUT_SIZE = 100
NUM_LEVELS = np.concatenate([np.array([1, 2, 3, 4]), np.linspace(5, 32, 28, dtype=int)])
TARGET_PER_LEVEL = 1500
OUTPUT_DIR = "circle_dataset_200x200_to_100x100_targeted_generation_v2_optimized_faster" # Cambiato il nome per distinguere

# Parametri di generazione e "micro-filtraggio"
MAX_STIMULUS_ATTEMPTS = 150
MAX_PACKING_ITERATIONS = 1000

# Correlazione minima N per calcoli statistici
MIN_N_FOR_CORRELATION = 5

# --- Range Target per le Feature Disentangled da N (per N >= 5) ---
# Questi range sono pensati per la generazione a 200x200
TARGET_CUMULATIVE_AREA_RANGE = (1000, 10000)
TARGET_FIELD_RADIUS_RANGE = (60, 95)

# Tollaranze per la validazione delle feature sul singolo stimolo (per N >= 5)
TOLERANCE_FEATURE_MATCH = 0.50

# Tolleranza dinamica per N elevati
HIGH_N_THRESHOLD_TOLERANCE = 25
TOLERANCE_HIGH_N_MULTIPLIER = 1.2

# Target per livello ridotto per N molto alti
HIGH_N_THRESHOLD_TARGET = 25
TARGET_PER_LEVEL_HIGH_N = 1000

# Range per la generazione "libera" (per N < 5)
FREE_GEN_ITEM_RADIUS_RANGE = (8, 40)
FREE_GEN_FIELD_RADIUS_FACTOR_RANGE = (0.8, 1.5)

# Range accettabili globali per tutte le feature (per scartare stimoli insensati)
# Questi range sono pensati per le feature calcolate sulle immagini 200x200
FINAL_FEATURE_VALUE_RANGES_VALIDATION = {
    'cumulative_area': (100, 12000),
    'field_radius': (10, 100),
    'convex_hull': (20, 12000),
    'density': (5e-05, 0.2),
    'item_size': (3.14, 800)
}

# Directory temporanea per le immagini generate prima della selezione finale
TEMP_IMG_DIR = os.path.join(OUTPUT_DIR, "temp_stimulus_images")

# Creazione delle directory di output
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

print(f"Dataset will be saved in: {OUTPUT_DIR}")
print(f"Temporary images will be stored in: {TEMP_IMG_DIR}")
print(f"Target CA range (N>=5): {TARGET_CUMULATIVE_AREA_RANGE}, Target Field Radius range (N>=5): {TARGET_FIELD_RADIUS_RANGE}")
print(f"Correlations for N >= {MIN_N_FOR_CORRELATION}")


# --- FUNZIONE DI PACKING BASATA SU REPULSIONE CON GRIGLIA ---
@jit(nopython=True, fastmath=True)
def attempt_circle_packing(N_target, draw_radius, target_field_radius, disp_size, max_packing_iterations, repulsion_strength=2.5, attraction_strength=0.01):
    adjusted_target_field_radius = min(target_field_radius, (disp_size / 2) - draw_radius * 2.5 - 5)
    if adjusted_target_field_radius < draw_radius * 1.0: return np.array([[-1.0, -1.0]])
    if adjusted_target_field_radius <= 0: adjusted_target_field_radius = 1

    max_local_center_offset = (disp_size / 2) - (draw_radius * 1.5)
    if max_local_center_offset < 0:
        max_local_center_offset = 0

    global_offset_x = np.random.uniform(-max_local_center_offset, max_local_center_offset)
    global_offset_y = np.random.uniform(-max_local_center_offset, max_local_center_offset)

    centers = np.empty((N_target, 2), dtype=np.float64)
    for i in range(N_target):
        distance = adjusted_target_field_radius * np.random.rand()**0.5
        angle = np.random.uniform(0, 2 * np.pi)
        px_local = distance * np.cos(angle)
        py_local = distance * np.sin(angle)
        centers[i] = np.array([px_local, py_local])

    min_dist_for_overlap = (draw_radius * 2 + 1.5)

    cell_size = draw_radius * 2.5
    if cell_size <= 0: cell_size = 1
    num_cols = int(np.ceil(disp_size / cell_size))
    num_rows = int(np.ceil(disp_size / cell_size))

    if num_cols == 0: num_cols = 1
    if num_rows == 0: num_rows = 1

    grid_cells_indices = List.empty_list(List.empty_list(types.int64))
    for _ in range(num_rows * num_cols):
        grid_cells_indices.append(List.empty_list(types.int64))

    for iteration in range(max_packing_iterations):
        moved = False
        forces = np.zeros_like(centers)

        for i in range(num_rows * num_cols):
            grid_cells_indices[i].clear()

        for k in range(N_target):
            center = centers[k]
            col_val = (center[0] / cell_size + (disp_size / (2 * cell_size)))
            row_val = (center[1] / cell_size + (disp_size / (2 * cell_size)))

            col = np.int64(max(0, min(col_val, num_cols - 1)))
            row = np.int64(max(0, min(row_val, num_rows - 1)))

            grid_cells_indices[row * num_cols + col].append(k)

        for i in range(N_target):
            current_col_val = (centers[i, 0] / cell_size + (disp_size / (2 * cell_size)))
            current_row_val = (centers[i, 1] / cell_size + (disp_size / (2 * cell_size)))

            current_col = np.int64(max(0, min(current_col_val, num_cols - 1)))
            current_row = np.int64(max(0, min(current_row_val, num_rows - 1)))

            for r_offset in [-1, 0, 1]:
                for c_offset in [-1, 0, 1]:
                    neighbor_row = current_row + r_offset
                    neighbor_col = current_col + c_offset

                    if 0 <= neighbor_row < num_rows and 0 <= neighbor_col < num_cols:
                        cell_idx = neighbor_row * num_cols + neighbor_col
                        for j in grid_cells_indices[cell_idx]:
                            if i == j: continue
                            if i > j: continue

                            diff = centers[i] - centers[j]
                            dist_squared = diff[0]**2 + diff[1]**2

                            if dist_squared < min_dist_for_overlap**2 and dist_squared > 1e-6:
                                dist = np.sqrt(dist_squared)
                                overlap_amount = min_dist_for_overlap - dist
                                if overlap_amount > 0:
                                    direction = diff / dist
                                    force_magnitude = repulsion_strength * (overlap_amount / min_dist_for_overlap)
                                    forces[i] += direction * force_magnitude
                                    forces[j] -= direction * force_magnitude
                                    moved = True

        if N_target > 1:
            current_cluster_centroid = np.sum(centers, axis=0) / N_target
            for i in range(N_target):
                vec_to_centroid = current_cluster_centroid - centers[i]
                dist_to_centroid = np.sqrt(vec_to_centroid[0]**2 + vec_to_centroid[1]**2)

                if dist_to_centroid > adjusted_target_field_radius * 1.3:
                    forces[i] += vec_to_centroid * attraction_strength
                    moved = True

        centers += forces

        if not moved and iteration > 0:
            break

    final_centers_on_display = centers + np.array([disp_size/2 + global_offset_x, disp_size/2 + global_offset_y])

    for i in range(N_target):
        for j in range(i + 1, N_target):
            dist = np.sqrt((final_centers_on_display[i, 0] - final_centers_on_display[j, 0])**2 +
                           (final_centers_on_display[i, 1] - final_centers_on_display[j, 1])**2)
            if dist < (draw_radius * 2 + 1.5 - 0.5):
                return np.array([[-1.0, -1.0]])

    min_coord_display = draw_radius + 0.5
    max_coord_display = disp_size - draw_radius - 0.5

    for k in range(N_target):
        cx, cy = final_centers_on_display[k]
        if not (min_coord_display <= cx < max_coord_display and
                min_coord_display <= cy < max_coord_display):
            return np.array([[-1.0, -1.0]])

    return final_centers_on_display

# --- GENERATORE DI STIMOLI MIRATO (Diventa la funzione per il Pool) ---
# MODIFICA: Accetta una tupla `task_args` per compatibilità con imap_unordered
def generate_single_stimulus_task(task_args):
    N_target, current_disp_size = task_args # Scompattazione degli argomenti

    current_tolerance = TOLERANCE_FEATURE_MATCH
    if N_target >= HIGH_N_THRESHOLD_TOLERANCE:
        current_tolerance *= TOLERANCE_HIGH_N_MULTIPLIER

    for stimulus_attempt in range(MAX_STIMULUS_ATTEMPTS):
        draw_radius = 0
        target_field_radius_for_placement = 0

        if N_target < MIN_N_FOR_CORRELATION:
            draw_radius = random.randint(*FREE_GEN_ITEM_RADIUS_RANGE)

            min_field_radius_for_packing_heuristic = draw_radius * (N_target**0.3) * 2.0

            target_field_radius_for_placement = random.uniform(
                min_field_radius_for_packing_heuristic * FREE_GEN_FIELD_RADIUS_FACTOR_RANGE[0],
                min_field_radius_for_packing_heuristic * FREE_GEN_FIELD_RADIUS_FACTOR_RANGE[1]
            )
            max_allowed_field_radius = (current_disp_size / 2) - draw_radius - 5
            target_field_radius_for_placement = min(target_field_radius_for_placement, max_allowed_field_radius)

            if target_field_radius_for_placement <= 0: continue

        else:
            target_cumulative_area = np.random.uniform(*TARGET_CUMULATIVE_AREA_RANGE)
            target_field_radius_for_placement = np.random.uniform(*TARGET_FIELD_RADIUS_RANGE)

            item_size_target = target_cumulative_area / N_target
            item_radius_target = np.sqrt(item_size_target / np.pi)
            draw_radius = max(1, int(round(item_radius_target)))

            if not (FINAL_FEATURE_VALUE_RANGES_VALIDATION['item_size'][0] <= item_size_target <= FINAL_FEATURE_VALUE_RANGES_VALIDATION['item_size'][1]):
                continue

            if draw_radius * 2 > current_disp_size * 0.9:
                continue

            min_field_radius_for_packing_heuristic = item_radius_target * (N_target**0.35) * 1.5
            target_field_radius_for_placement = max(target_field_radius_for_placement, min_field_radius_for_packing_heuristic)

            max_allowed_field_radius = (current_disp_size / 2) - draw_radius * 2.5 - 5
            target_field_radius_for_placement = min(target_field_radius_for_placement, max_allowed_field_radius)
            if target_field_radius_for_placement <= 0: continue

        centers_float = attempt_circle_packing(N_target, draw_radius, target_field_radius_for_placement, current_disp_size, MAX_PACKING_ITERATIONS)

        if centers_float is None or (centers_float.shape[0] == 1 and centers_float[0,0] == -1.0 and centers_float[0,1] == -1.0):
             continue

        centers = centers_float.astype(np.int64)

        img = np.zeros((current_disp_size, current_disp_size), dtype=np.uint8)
        actual_centers_on_img = []

        min_coord_img = draw_radius
        max_coord_img = current_disp_size - draw_radius

        for center_coords in centers:
            cx, cy = int(center_coords[0]), int(center_coords[1])
            if (cx >= min_coord_img and cx < max_coord_img and
                cy >= min_coord_img and cy < max_coord_img):
                rr, cc = disk((cy, cx), draw_radius, shape=img.shape)

                # Assegna il valore 255 (bianco) ai pixel del cerchio nell'immagine
                img[rr, cc] = 255

                actual_centers_on_img.append((cx, cy))

            else:
                continue

        actual_N = len(actual_centers_on_img)
        if actual_N != N_target:
            continue

        # --- MODIFICA CHIAVE QUI: Binarizza l'immagine prima di calcolare le feature e salvarla ---
        img = np.where(img > 0, 255, 0).astype(np.uint8)
        # --- FINE MODIFICA CHIAVE ---

        actual_item_size = (draw_radius ** 2) * np.pi
        actual_cumulative_area = actual_N * actual_item_size

        max_dist_from_center = 0
        if actual_centers_on_img:
            centers_np = np.array(actual_centers_on_img, dtype=np.float64)
            avg_cx, avg_cy = np.mean(centers_np[:,0]), np.mean(centers_np[:,1])

            if N_target > 0:
                max_dist_from_center = np.max(np.sqrt((centers_np[:,0] - avg_cx)**2 + (centers_np[:,1] - avg_cy)**2))

        actual_field_radius = max_dist_from_center + (draw_radius if actual_N == 1 else np.mean([np.sqrt(actual_item_size / np.pi)]))
        if actual_field_radius == 0 and actual_N > 0: actual_field_radius = 1

        ch_area = convex_hull_image(img > 0).sum() if np.any(img > 0) else 0

        if ch_area == 0 and actual_N > 0:
            if actual_N == 1:
                ch_area = actual_item_size
            elif actual_N == 2:
                dist_centers = np.linalg.norm(np.array(actual_centers_on_img[0]) - np.array(actual_centers_on_img[1]))
                ch_area = 2 * actual_item_size + 2 * draw_radius * dist_centers
            else:
                ch_area = np.pi * actual_field_radius**2
                if ch_area == 0: ch_area = 1

        density = actual_N / ch_area if ch_area > 0 else 0

        if N_target >= MIN_N_FOR_CORRELATION:
            target_cumulative_area_val = target_cumulative_area

            if not (abs(actual_cumulative_area - target_cumulative_area_val) / target_cumulative_area_val <= current_tolerance and
                    abs(actual_field_radius - target_field_radius_for_placement) / target_field_radius_for_placement <= current_tolerance):
                continue

        if not (FINAL_FEATURE_VALUE_RANGES_VALIDATION['cumulative_area'][0] <= actual_cumulative_area <= FINAL_FEATURE_VALUE_RANGES_VALIDATION['cumulative_area'][1] and
                FINAL_FEATURE_VALUE_RANGES_VALIDATION['field_radius'][0] <= actual_field_radius <= FINAL_FEATURE_VALUE_RANGES_VALIDATION['field_radius'][1] and
                FINAL_FEATURE_VALUE_RANGES_VALIDATION['convex_hull'][0] <= ch_area <= FINAL_FEATURE_VALUE_RANGES_VALIDATION['convex_hull'][1] and
                FINAL_FEATURE_VALUE_RANGES_VALIDATION['density'][0] <= density <= FINAL_FEATURE_VALUE_RANGES_VALIDATION['density'][1] and
                FINAL_FEATURE_VALUE_RANGES_VALIDATION['item_size'][0] <= actual_item_size <= FINAL_FEATURE_VALUE_RANGES_VALIDATION['item_size'][1]):
            continue

        temp_filepath = os.path.join(TEMP_IMG_DIR, f"N{N_target}_stim_{current_process().pid}_{int(time.time()*1000000)}_{stimulus_attempt}.png")
        cv2.imwrite(temp_filepath, img)

        return (N_target, actual_cumulative_area, actual_field_radius, ch_area, density, actual_item_size, temp_filepath)

    return None

# --- FUNZIONI DI SUPPORTO ---

def parallel_generate_pool_new(levels, current_disp_size):
    all_tasks = []
    total_target_count = 0
    for N_level in levels:
        current_target_per_level = TARGET_PER_LEVEL
        if N_level >= HIGH_N_THRESHOLD_TARGET:
            current_target_per_level = TARGET_PER_LEVEL_HIGH_N

        attempts_multiplier = 0 # Inizializza a 0 per sovrascriverlo con logica condizionale
        if N_level < MIN_N_FOR_CORRELATION:
            attempts_multiplier = 2 # RIDOTTO (era 5)
        elif N_level >= HIGH_N_THRESHOLD_TARGET:
            attempts_multiplier = 5 # RIDOTTO (era 20)
        else: # Per i livelli tra MIN_N_FOR_CORRELATION e HIGH_N_THRESHOLD_TARGET (es. 5-24)
            attempts_multiplier = 3 # Rimane 3

        num_tasks_to_create = int(current_target_per_level * attempts_multiplier)
        total_target_count += current_target_per_level

        print(f"N={N_level}: Creating {num_tasks_to_create} generation tasks to achieve target {current_target_per_level} (Multiplier: {attempts_multiplier}x).")
        for _ in range(num_tasks_to_create):
            all_tasks.append((N_level, current_disp_size)) # Aggiunge la tupla

    random.shuffle(all_tasks)

    print(f"\nTotal tasks created: {len(all_tasks)}. Total target samples for dataset: {total_target_count}.")

    final_data_by_level = {n: [] for n in levels}
    final_image_paths_by_level = {n: [] for n in levels}

    num_processes = cpu_count()
    if num_processes < 1:
        num_processes = 1
    print(f"Utilizing {num_processes} processes for generation.")

    with Pool(processes=num_processes) as pool:
        # Usa tqdm con imap_unordered per la progress bar
        results_iterator = tqdm(
            pool.imap_unordered(generate_single_stimulus_task, all_tasks),
            total=len(all_tasks),
            desc="Generating stimuli"
        )

        for result in results_iterator: # `i` non è più necessario qui, tqdm gestisce il conteggio
            if result is not None:
                N_level, actual_cumulative_area, field_radius_actual, ch_area, density, item_size, temp_filepath = result
                # Assicurati di non aggiungere più del necessario per ogni livello
                if len(final_data_by_level[N_level]) < (TARGET_PER_LEVEL_HIGH_N if N_level >= HIGH_N_THRESHOLD_TARGET else TARGET_PER_LEVEL):
                    final_data_by_level[N_level].append((N_level, actual_cumulative_area, field_radius_actual, ch_area, density, item_size))
                    final_image_paths_by_level[N_level].append(temp_filepath)
    print("\nAll generation tasks processed.") # Messaggio di completamento per la generazione

    all_final_data = []
    all_final_image_paths = []

    for n in sorted(levels):
        current_target_count = TARGET_PER_LEVEL
        if n >= HIGH_N_THRESHOLD_TARGET:
            current_target_count = TARGET_PER_LEVEL_HIGH_N

        data_for_n = final_data_by_level[n]
        paths_for_n = final_image_paths_by_level[n]

        if len(data_for_n) >= current_target_count:
            combined = list(zip(data_for_n, paths_for_n))
            selected_combined = random.sample(combined, current_target_count)
            all_final_data.extend([s[0] for s in selected_combined])
            all_final_image_paths.extend([s[1] for s in selected_combined])
            print(f"Level {n}: Selected {len(selected_combined)} samples (from {len(data_for_n)} generated).")
        else:
            all_final_data.extend(data_for_n)
            all_final_image_paths.extend(paths_for_n)
            print(f"Level {n}: WARNING - Only {len(data_for_n)} samples generated, less than target {current_target_count}. Cannot fulfill target for this level.")
            # Non duplichiamo se non abbiamo abbastanza, per evitare di alterare le statistiche se il sample pool è troppo piccolo
            # Se la generazione è stata insufficiente, qui potresti voler generare di più in un secondo step, ma per ora il codice è così.

    return all_final_data, all_final_image_paths


def calculate_correlations(data_arr, min_N_filter):
    filtered_data_arr = data_arr[data_arr[:, 0] >= min_N_filter]
    if filtered_data_arr.shape[0] < 2: return {k: np.nan for k in ['CumulativeArea', 'ConvexHull', 'Density', 'ItemSize', 'FieldRadius']}

    N_arr = filtered_data_arr[:, 0]
    correlations = {}
    feature_indices_in_array = {
        'CumulativeArea': 1,
        'FieldRadius': 2,
        'ConvexHull': 3,
        'Density': 4,
        'ItemSize': 5
    }

    if len(np.unique(N_arr)) <= 1:
        return {k: np.nan for k in feature_indices_in_array.keys()}

    for name, idx in feature_indices_in_array.items():
        feature_arr = filtered_data_arr[:, idx]
        if len(np.unique(feature_arr)) > 1 and len(np.unique(N_arr)) > 1:
            correlations[name] = scipy.stats.pearsonr(feature_arr, N_arr)[0]
        else:
            correlations[name] = np.nan
    return correlations

# --- SAVE E PLOT ---
def save_dataset(data, images, filename, final_output_size):
    if not data:
        print(f"No data to save for {filename}.")
        return

    resized_images = []
    # Usiamo tqdm anche qui per il progresso del ridimensionamento
    for img in tqdm(images, desc="Resizing and binarizing images"):
        # Ridimensiona l'immagine (può introdurre anti-aliasing)
        img_resized = resize(img, (final_output_size, final_output_size), anti_aliasing=True)
        # --- MODIFICA CHIAVE QUI: Binarizza nuovamente l'immagine dopo il ridimensionamento ---
        # Si assume che i pixel "on" debbano essere >= un certo valore (e.g., metà di 255)
        # o semplicemente tutti i valori > 0 diventano 255.
        img_binary = np.where(img_resized > (255 / 2), 255, 0).astype(np.uint8)
        # --- FINE MODIFICA CHIAVE ---
        resized_images.append(img_binary)

    data = np.array(data)
    images_flat = np.array(resized_images).reshape(len(resized_images), -1)

    N_list = data[:, 0] if data.size > 0 else np.array([])
    cumArea_list = data[:, 1] if data.size > 0 else np.array([])
    FA_list = data[:, 2] if data.size > 0 else np.array([])
    CH_list = data[:, 3] if data.size > 0 else np.array([])
    density = data[:, 4] if data.size > 0 else np.array([])
    item_size = data[:, 5] if data.size > 0 else np.array([])

    np.savez_compressed(filename, D=images_flat, N_list=N_list, cumArea_list=cumArea_list,
                         FA_list=FA_list, CH_list=CH_list, density=density, item_size=item_size)
    print(f"✅ Dataset saved: {filename}")

def plot_correlations(data, filename, min_N_filter):
    if len(data) < 2:
        print(f"Not enough data to plot correlations for {filename}.")
        return
    filtered_data = data[data[:, 0] >= min_N_filter]
    if len(filtered_data) < 2:
        print(f"Not enough filtered data (N>={min_N_filter}) to plot correlations for {filename}.")
        return

    N_arr = filtered_data[:, 0]
    CA_arr = filtered_data[:, 1]
    FR_arr = filtered_data[:, 2]
    CH_arr = filtered_data[:, 3]
    dens_arr = filtered_data[:, 4]
    size_arr = filtered_data[:, 5]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plots_data = [
        (N_arr, CA_arr, "Numerosity vs CumulativeArea"),
        (N_arr, CH_arr, "Numerosity vs ConvexHull"),
        (N_arr, dens_arr, "Numerosity vs Density"),
        (N_arr, size_arr, "Numerosity vs ItemSize"),
        (N_arr, FR_arr, "Numerosity vs FieldRadius"),
        (CH_arr, dens_arr, "ConvexHull vs Density")
    ]
    for i, (x, y, title) in enumerate(plots_data):
        ax = axes.flatten()[i]
        if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            r, _ = scipy.stats.pearsonr(x, y)
        else:
            r = np.nan
        ax.scatter(x, y, alpha=0.3, s=10)
        ax.set_title(f"{title}\n(N>={min_N_filter}) r={r:.3f}")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# --- FUNZIONE PRINCIPALE ---
def generate_dataset_main_function(generation_size, final_output_size, output_directory):
    global DISP_SIZE, OUTPUT_DIR, TEMP_IMG_DIR
    DISP_SIZE = generation_size
    OUTPUT_DIR = output_directory
    TEMP_IMG_DIR = os.path.join(OUTPUT_DIR, "temp_stimulus_images")

    if os.path.exists(TEMP_IMG_DIR):
        try:
            shutil.rmtree(TEMP_IMG_DIR)
            print(f"Cleaned up existing temporary image directory: {TEMP_IMG_DIR}")
        except OSError as e:
            print(f"Error cleaning up temporary directory {TEMP_IMG_DIR}: {e}. Please ensure it's not in use.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_IMG_DIR, exist_ok=True)

    start_time = time.time()
    print(f"Generating {generation_size}x{generation_size} dataset (then resizing to {final_output_size}x{final_output_size}) with {len(NUM_LEVELS)} levels. Target samples per level: {TARGET_PER_LEVEL} (reduced to {TARGET_PER_LEVEL_HIGH_N} for N >= {HIGH_N_THRESHOLD_TARGET}).")

    all_final_data, all_final_image_paths = parallel_generate_pool_new(NUM_LEVELS, generation_size)

    final_images_raw = []
    print("\nLoading and processing temporary images...")
    # Usa tqdm anche per il caricamento delle immagini temporanee
    for img_path in tqdm(all_final_image_paths, desc="Loading temp images"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            final_images_raw.append(img)
        else:
            print(f"ERROR: Could not load image from {img_path}. Skipping.")
        try:
            os.remove(img_path)
        except OSError as e:
            print(f"Error removing temporary file {img_path}: {e}")

    if os.path.exists(TEMP_IMG_DIR):
        try:
            shutil.rmtree(TEMP_IMG_DIR)
            print(f"Cleaned up temporary image directory: {TEMP_IMG_DIR}")
        except OSError as e:
            print(f"Error cleaning up temporary directory {TEMP_IMG_DIR}: {e}. It might not be empty or still in use.")

    print(f"\nFinal dataset size: {len(all_final_data)} samples.")

    final_data_arr = np.array(all_final_data) if all_final_data else np.array([])

    feature_column_map = {
        'CumulativeArea': 1,
        'FieldRadius': 2,
        'ConvexHull': 3,
        'Density': 4,
        'ItemSize': 5
    }

    final_correlations = {}
    if final_data_arr.shape[0] > 1:
        N_for_corr = final_data_arr[final_data_arr[:,0] >= MIN_N_FOR_CORRELATION, 0]
        if len(N_for_corr) > 1 and len(np.unique(N_for_corr)) > 1:
            for name, col_idx in feature_column_map.items():
                feature_values_for_corr = final_data_arr[final_data_arr[:,0] >= MIN_N_FOR_CORRELATION, col_idx]
                if len(np.unique(feature_values_for_corr)) > 1:
                    final_correlations[name] = scipy.stats.pearsonr(feature_values_for_corr, N_for_corr)[0]
                else:
                    final_correlations[name] = np.nan
        else:
            print("Not enough varied N values (>= MIN_N_FOR_CORRELATION) to calculate final correlations.")
    else:
        print("Not enough data to calculate final correlations.")

    dataset_filename = os.path.join(OUTPUT_DIR, f"circle_dataset_{final_output_size}x{final_output_size}_targeted_correlations.npz")
    correlations_filename = os.path.join(OUTPUT_DIR, f"correlations_{final_output_size}x{final_output_size}_targeted.png")

    save_dataset(all_final_data, final_images_raw, dataset_filename, final_output_size)
    plot_correlations(final_data_arr, correlations_filename, MIN_N_FOR_CORRELATION)

    print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Final correlations (for N >= {MIN_N_FOR_CORRELATION}):")
    print(f"  CumulativeArea: {final_correlations.get('CumulativeArea', np.nan):6.3f} (Target: ~0)")
    print(f"  ConvexHull:     {final_correlations.get('ConvexHull', np.nan):6.3f} (Target: ~0)")
    print(f"  Density:        {final_correlations.get('Density', np.nan):6.3f} (Expected: >0, consequence of CH disentanglement)")
    print(f"  ItemSize:       {final_correlations.get('ItemSize', np.nan):6.3f} (Expected: <0, consequence of CA disentanglement)")
    print(f"  FieldRadius:    {final_correlations.get('FieldRadius', np.nan):6.3f} (Expected: varies, consequence of CH disentanglement)")

    return final_data_arr, final_images_raw, final_correlations

# --- ESECUZIONE DEL CODICE ---
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    generate_dataset_main_function(DISP_SIZE, FINAL_OUTPUT_SIZE, OUTPUT_DIR)