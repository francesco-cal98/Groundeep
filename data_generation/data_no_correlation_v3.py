import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.morphology import convex_hull_image
from multiprocessing import Pool, cpu_count
import scipy.stats
from tqdm import tqdm
import shutil
from skimage.transform import resize

# --- CONFIGURAZIONE GLOBALE (PARAMETRI DI BASE DEL SIMULATORE) ---
GLOBAL_DISP_SIZE_GEN = 100 # Dimensione di generazione interna, ORA è 100x100
GLOBAL_FINAL_OUTPUT_SIZE = 100 # Dimensione dell'immagine finale salvata (coincide con quella di generazione)

# Range di N per la generazione (utilizzati in entrambe le fasi)
GLOBAL_NUM_LEVELS_TO_GENERATE = np.concatenate([
    np.array([1, 2, 3]),
    np.linspace(4, 32, 29, dtype=int)
])

# --- OBIETTIVI DI CORRELAZIONE ---
TARGET_CORRELATION_CH = 0.2     # La correlazione r(N, CH) che vogliamo ottenere (circa)
TARGET_CORRELATION_DENSITY_MAX = 0.4 # La correlazione r(N, Density) deve essere sotto questo valore

# --- PARAMETRI DI CONTROLLO DEL PROCESSO DI GENERAZIONE ---
GLOBAL_MIN_SPACING_DOTS = 1.1 # Minima distanza tra i centri (circa metà di prima)
GLOBAL_MAX_DOT_PLACEMENT_ATTEMPTS_PER_STIMULUS = 5000 # Tentativi massimi per posizionare tutti i cerchi in un singolo stimolo
# GLOBAL_MIN_PLACEMENT_RADIUS_OFFSET rimosso per permettere punti al centro

# Fattore di tentativi per la generazione finale (da 100 in su, a seconda di quanto è difficile generare)
GLOBAL_MAX_TOTAL_ATTEMPTS_PER_N_LEVEL_FACTOR = 1000

# Tolleranza per le correlazioni sui plot finali (solo per visualizzazione)
GLOBAL_CORRELATION_THRESHOLD = 0.25

# --- PARAMETRI PER LA FASE ESPLORATIVA / CALIBRAZIONE ---
# Questi range devono essere proporzionati alla nuova GLOBAL_DISP_SIZE_GEN (100)
EXPLORATION_CUM_AREA_RANGE = (500, 10000)
EXPLORATION_FIELD_RADIUS_RANGE = (5, 75)
EXPLORATION_ITEM_RADIUS_RANGE = (3, 10) # Aumentato il range del raggio come richiesto

EXPLORATION_NUM_SAMPLES_PER_N_LEVEL = 200 # Meno campioni per una calibrazione più veloce
EXPLORATION_MAX_TOTAL_ATTEMPTS_FACTOR = 500 # Molti più tentativi per trovare i campioni validi per la calibrazione

# --- FUNZIONE PRINCIPALE DI GENERAZIONE DI UN SINGOLO STIMOLO ---
def generate_simple_stimulus_core(N_target, current_disp_size_gen,
                                  cum_area_range, field_radius_range, item_radius_range):
    """
    Genera un singolo stimolo con N_target punti, utilizzando parametri casuali
    specificati dai range passati. Questa è la funzione di base.
    """
    img = np.zeros((current_disp_size_gen, current_disp_size_gen), dtype=np.uint8)

    cumulative_area_rand = np.random.uniform(*cum_area_range)
    field_radius_rand = np.random.uniform(*field_radius_range)
    item_radius_rand = np.random.uniform(*item_radius_range)

    center_x, center_y = current_disp_size_gen // 2, current_disp_size_gen // 2

    # Il raggio massimo per il piazzamento dei centri dei punti
    # Deve lasciare spazio per il raggio del punto stesso e lo spazio minimo
    placement_radius_max = min(field_radius_rand, (current_disp_size_gen // 2) - item_radius_rand - GLOBAL_MIN_SPACING_DOTS)

    # Se non c'è spazio sufficiente per posizionare il punto, fallisci
    # Questo controllo è importante per evitare np.sqrt(negativo) o divisioni per zero implicite
    if placement_radius_max <= 0:
        return None

    centers = []
    attempts = 0

    while len(centers) < N_target and attempts < GLOBAL_MAX_DOT_PLACEMENT_ATTEMPTS_PER_STIMULUS:
        angle = np.random.uniform(0, 2 * np.pi)
        # MODIFICATO: r_placement campionata in modo da distribuire uniformemente i punti
        # sull'AREA del cerchio, non sul raggio, permettendo al centro di essere una posizione valida
        r_placement = np.sqrt(np.random.uniform(0, placement_radius_max**2))

        cx = int(center_x + r_placement * np.cos(angle))
        cy = int(center_y + r_placement * np.sin(angle))

        # Controlla che il punto non esca dai bordi dell'immagine
        if (cx < item_radius_rand + 1 or cx >= current_disp_size_gen - item_radius_rand - 1 or
            cy < item_radius_rand + 1 or cy >= current_disp_size_gen - item_radius_rand - 1):
            attempts += 1
            continue

        min_dist_between_centers = item_radius_rand * 2 + GLOBAL_MIN_SPACING_DOTS
        if all(np.hypot(cx - x, cy - y) >= min_dist_between_centers for x, y in centers):
            centers.append((cx, cy))

        attempts += 1

    # Se non riusciamo a posizionare abbastanza punti, consideriamo lo stimolo fallito
    if len(centers) < max(1, N_target * 0.7):
        return None

    for cx, cy in centers:
        rr, cc = disk((cy, cx), item_radius_rand, shape=img.shape)
        img[rr, cc] = 255

    # --- Calcolo delle feature emerse ---
    actual_N = len(centers)
    actual_cumulative_area = np.sum(img > 0)

    try:
        ch_area = convex_hull_image(img > 0).sum()
    except Exception:
        ch_area = actual_cumulative_area if actual_N > 0 else 0

    actual_field_radius = 0
    if actual_N > 0:
        centers_np = np.array(centers, dtype=np.float64)
        avg_cx, avg_cy = np.mean(centers_np[:,0]), np.mean(centers_np[:,1])
        max_dist_from_mean_center = np.max(np.sqrt((centers_np[:,0] - avg_cx)**2 + (centers_np[:,1] - avg_cy)**2))
        actual_field_radius = max_dist_from_mean_center + item_radius_rand

    if actual_field_radius == 0 and actual_N > 0: actual_field_radius = 1

    density = actual_cumulative_area / ch_area if ch_area > 0 else 0

    actual_item_size = (item_radius_rand ** 2) * np.pi

    # Ritorna anche i parametri randomici usati per questo stimolo
    return (img, actual_N, actual_cumulative_area, actual_field_radius,
            ch_area, density, actual_item_size,
            cumulative_area_rand, field_radius_rand, item_radius_rand)


# --- FUNZIONE PER GENERARE UN INTERO LIVELLO DI N (NESSUN FILTRO CH/DENSITÀ SUI VALORI ASSOLUTI) ---
def generate_level_task_no_value_filters(task_args):
    """
    Funzione wrapper per la generazione parallela di un livello N.
    Non applica filtri sui valori assoluti di CH o Densità, ma solo controlli di qualità base.
    Ritorna anche i parametri casuali usati se richiesto per l'esplorazione.
    """
    N_level, current_disp_size_gen, target_per_level_local, \
    cum_area_range, field_radius_range, item_radius_range, \
    max_total_attempts_factor, return_generation_params = task_args

    data_for_level = []
    images_for_level = []
    generation_params_for_level = [] # Nuova lista per i parametri generativi
    count = 0
    attempts = 0
    max_total_attempts = target_per_level_local * max_total_attempts_factor

    while count < target_per_level_local and attempts < max_total_attempts:
        result = generate_simple_stimulus_core(N_level, current_disp_size_gen,
                                               cum_area_range, field_radius_range, item_radius_range)

        if result is not None:
            img, actual_N, actual_ca, field_r, ch_area, density, item_s, \
            rand_cum_area, rand_field_r, rand_item_r = result

            # Controlli di qualità base, NON filtri su CH/Densità target values
            # Ho leggermente abbassato le soglie per CA e CH per accomodare punti più grandi su 100x100
            if (actual_ca > 50 and field_r > 3 and ch_area > 50 and density > 0.0001 and item_s > 1):
                data_for_level.append([actual_N, actual_ca, field_r, ch_area, density, item_s])
                images_for_level.append(img)
                if return_generation_params:
                    generation_params_for_level.append([rand_cum_area, rand_field_r, rand_item_r])
                count += 1
        attempts += 1

    return N_level, data_for_level, images_for_level, generation_params_for_level


# --- FUNZIONE PER L'ESPLORAZIONE E IL SUGGERIMENTO DEI RANGE ---
def explore_and_suggest_ranges(disp_size_gen, num_levels,
                               exploration_target_samples_per_n,
                               exploration_max_attempts_factor,
                               initial_cum_area_range,
                               initial_field_radius_range,
                               initial_item_radius_range):
    """
    Esegue una fase esplorativa per suggerire range più stretti per i parametri di generazione,
    che rendano la generazione efficiente in generale. Non ottimizza per le correlazioni.
    """
    print("\n--- Starting Exploration Phase to Suggest Optimal Generation Ranges ---")
    print(f"Using initial wide ranges: CumArea={initial_cum_area_range}, FieldR={initial_field_radius_range}, ItemR={initial_item_radius_range}")
    print(f"Targeting {exploration_target_samples_per_n} samples per N level for exploration.")

    tasks = [(n, disp_size_gen, exploration_target_samples_per_n,
              initial_cum_area_range, initial_field_radius_range, initial_item_radius_range,
              exploration_max_attempts_factor, True) # return_generation_params = True
             for n in num_levels]

    all_accepted_gen_params = []

    num_processes = min(cpu_count(), 14)
    print(f"Utilizing {num_processes} processes for exploration.")

    with Pool(processes=num_processes) as pool:
        results_iterator = tqdm(
            pool.imap_unordered(generate_level_task_no_value_filters, tasks),
            total=len(tasks),
            desc="Exploring N levels"
        )
        for N_level, data_for_level, images_for_level, gen_params_for_level in results_iterator:
            all_accepted_gen_params.extend(gen_params_for_level)

    print(f"\nExploration phase completed. Total accepted samples for analysis: {len(all_accepted_gen_params)}")

    if not all_accepted_gen_params:
        print("Warning: No stimuli met the basic quality criteria during exploration. Cannot suggest ranges.")
        return initial_cum_area_range, initial_field_radius_range, initial_item_radius_range

    accepted_gen_params_np = np.array(all_accepted_gen_params)

    # Calcola i percentili per suggerire i nuovi range
    p_low = 5
    p_high = 95

    suggested_cum_area_range = (np.percentile(accepted_gen_params_np[:, 0], p_low),
                                np.percentile(accepted_gen_params_np[:, 0], p_high))
    suggested_field_radius_range = (np.percentile(accepted_gen_params_np[:, 1], p_low),
                                    np.percentile(accepted_gen_params_np[:, 1], p_high))
    suggested_item_radius_range = (np.percentile(accepted_gen_params_np[:, 2], p_low),
                                   np.percentile(accepted_gen_params_np[:, 2], p_high))

    print("\n--- Suggested Optimal Generation Ranges ---")
    print(f"Cumulative Area Factor Range: ({suggested_cum_area_range[0]:.0f}, {suggested_cum_area_range[1]:.0f})")
    print(f"Field Radius Factor Range:    ({suggested_field_radius_range[0]:.1f}, {suggested_field_radius_range[1]:.1f})")
    print(f"Item Radius Factor Range:     ({suggested_item_radius_range[0]:.1f}, {suggested_item_radius_range[1]:.1f})")
    print("These ranges are optimized for general generation efficiency (fewer rejections).")
    print("Correlations with N are an emergent property and might require manual tuning via these ranges.")

    # Arrotonda per praticità
    suggested_cum_area_range = (int(suggested_cum_area_range[0]), int(suggested_cum_area_range[1]))
    suggested_field_radius_range = (round(suggested_field_radius_range[0]), round(suggested_field_radius_range[1]))
    suggested_item_radius_range = (round(suggested_item_radius_range[0]), round(suggested_item_radius_range[1]))

    return suggested_cum_area_range, suggested_field_radius_range, suggested_item_radius_range


# --- FUNZIONI DI SUPPORTO E PLOTTING ---

def check_correlations(data_arr, min_N_for_correlation=GLOBAL_NUM_LEVELS_TO_GENERATE[0]):
    """
    Calcola e stampa le correlazioni di Pearson tra Numerosità e altre feature.
    """
    if data_arr.shape[0] < 2:
        print("Not enough data to check correlations.")
        return {}, False

    filtered_data_arr = data_arr[data_arr[:, 0] >= min_N_for_correlation]

    if filtered_data_arr.shape[0] < 2 or len(np.unique(filtered_data_arr[:, 0])) <= 1:
        print(f"Not enough varied N values (N >= {min_N_for_correlation}) to calculate correlations. Need at least 2 unique Ns.")
        return {k: np.nan for k in ['Cumulative Area', 'Convex Hull', 'Field Radius', 'Density', 'Item Size']}, False

    N_arr = filtered_data_arr[:, 0]
    CA_arr = filtered_data_arr[:, 1]
    FR_arr = filtered_data_arr[:, 2]
    CH_arr = filtered_data_arr[:, 3]
    dens_arr = filtered_data_arr[:, 4]
    size_arr = filtered_data_arr[:, 5]

    correlations = {}
    feature_arrays = {
        'Cumulative Area': CA_arr,
        'Field Radius': FR_arr,
        'Convex Hull': CH_arr,
        'Density': dens_arr,
        'Item Size': size_arr
    }

    print(f"Correlations with Numerosity (for N >= {min_N_for_correlation}):")
    for name, arr in feature_arrays.items():
        if len(np.unique(arr)) > 1 and len(np.unique(N_arr)) > 1:
            corr_val = scipy.stats.pearsonr(arr, N_arr)[0]
        else:
            corr_val = np.nan

        correlations[name] = corr_val
        print(f"   {name}: {corr_val:.3f}")

    is_valid_overall = True
    return correlations, is_valid_overall

def save_dataset(data, images, filename, final_output_size, disp_size_gen):
    """
    Salva il dataset in formato .npz. Se disp_size_gen è diverso da final_output_size, ridimensiona.
    """
    if not data:
        print(f"No data to save for {filename}.")
        return

    processed_images = []

    # Condizione per evitare il ridimensionamento se le dimensioni sono già uguali
    if disp_size_gen == final_output_size:
        print(f"Images already at target size {final_output_size}x{final_output_size}. Skipping resize step.")
        for img_arr in tqdm(images, desc=f"Processing images (no resize)"):
            img_binary = np.where(img_arr > 0.5, 255, 0).astype(np.uint8) # Assicurati che siano binarie e uint8
            processed_images.append(img_binary.flatten())
    else:
        for img_arr in tqdm(images, desc=f"Resizing images to {final_output_size}x{final_output_size}"):
            img_float = img_arr.astype(np.float32) / 255.0
            img_resized = resize(img_float, (final_output_size, final_output_size), anti_aliasing=True)
            img_binary = np.where(img_resized > 0.5, 255, 0).astype(np.uint8)
            processed_images.append(img_binary.flatten())

    data_arr = np.array(data)
    images_flat = np.array(processed_images)

    np.savez_compressed(filename, D=images_flat, N_list=data_arr[:, 0], cumArea_list=data_arr[:, 1],
                         FA_list=data_arr[:, 2], CH_list=data_arr[:, 3], density=data_arr[:, 4], item_size=data_arr[:, 5])
    print(f"✅ Dataset saved: {filename}")


def plot_correlations(data_arr, filename, min_N_filter=GLOBAL_NUM_LEVELS_TO_GENERATE[0]):
    """
    Genera un plot delle correlazioni tra Numerosità e altre feature.
    """
    if data_arr.shape[0] < 2:
        print(f"Not enough data to plot correlations for {filename}.")
        return

    filtered_data = data_arr[data_arr[:, 0] >= min_N_filter]
    if filtered_data.shape[0] < 2 or len(np.unique(filtered_data[:,0])) < 2:
        print(f"Not enough filtered data (N>={min_N_filter}, or not enough unique Ns) to plot correlations for {filename}.")
        return

    N_arr = filtered_data[:, 0]
    CA_arr = filtered_data[:, 1]
    FR_arr = filtered_data[:, 2]
    CH_arr = filtered_data[:, 3]
    dens_arr = filtered_data[:, 4]
    size_arr = filtered_data[:, 5]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plots_data = [
        (N_arr, CA_arr, "Numerosity vs Cumulative Area"),
        (N_arr, CH_arr, "Numerosity vs Convex Hull"),
        (N_arr, dens_arr, "Numerosity vs Density"),
        (N_arr, size_arr, "Numerosity vs Item Size"),
        (N_arr, FR_arr, "Numerosity vs Field Radius"),
        (CH_arr, dens_arr, "Convex Hull vs Density")
    ]
    for i, (x, y, title) in enumerate(plots_data):
        ax = axes.flatten()[i]
        if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            r, _ = scipy.stats.pearsonr(x, y)
        else:
            r = np.nan
        ax.scatter(x, y, alpha=0.3, s=10)
        ax.set_title(f"{title}\n(N>={min_N_filter}) r={r:.3f}")
        ax.set_xlabel(title.split(' vs ')[0])
        ax.set_ylabel(title.split(' vs ')[1])
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# --- FUNZIONE PRINCIPALE DI GENERAZIONE DEL DATASET CON I RANGE CALIBRATI ---
def generate_dataset_fast_no_value_filters(
    gen_size,
    final_output_size,
    output_directory,
    num_levels_to_generate,
    target_samples_per_n_level,
    calibrated_cum_area_range,
    calibrated_field_radius_range,
    calibrated_item_radius_range
):
    """
    Funzione principale per la generazione del dataset, usando i range di generazione calibrati.
    Non applica filtri sui valori assoluti di CH/Densità.
    """

    current_output_dir = os.path.join(output_directory)
    if os.path.exists(current_output_dir):
        try:
            shutil.rmtree(current_output_dir)
            print(f"Cleaned up existing output directory: {current_output_dir}")
        except OSError as e:
            print(f"Error cleaning up directory {current_output_dir}: {e}. Please ensure it's not in use.")
    os.makedirs(current_output_dir, exist_ok=True)
    print(f"Output will be saved in: {current_output_dir}")

    start_time = time.time()
    print(f"\n--- Starting Full Dataset Generation with Calibrated Ranges ({final_output_size}x{final_output_size}) ---")
    print(f"Using calibrated ranges: CumArea={calibrated_cum_area_range}, FieldR={calibrated_field_radius_range}, ItemR={calibrated_item_radius_range}")
    print(f"Targeting {target_samples_per_n_level} samples per N level.")
    print("NOTE: No absolute value filtering on CH/Density. Correlations will be an emergent property.")

    # I task per la generazione completa non devono ritornare i parametri di generazione
    tasks = [(n, gen_size, target_samples_per_n_level,
              calibrated_cum_area_range, calibrated_field_radius_range, calibrated_item_radius_range,
              GLOBAL_MAX_TOTAL_ATTEMPTS_PER_N_LEVEL_FACTOR, False)
             for n in num_levels_to_generate]

    all_data = []
    all_images_raw = []

    num_processes = min(cpu_count(), 14)
    print(f"Utilizing {num_processes} processes for generation.")

    with Pool(processes=num_processes) as pool:
        results_iterator = tqdm(
            pool.imap_unordered(generate_level_task_no_value_filters, tasks),
            total=len(tasks),
            desc="Generating N levels"
        )
        for N_level, data_for_level, images_for_level, _ in results_iterator:
            all_data.extend(data_for_level)
            all_images_raw.extend(images_for_level)

    print(f"\nFull generation phase completed. Total successfully generated stimuli: {len(all_data)}")

    # --- NUOVA LOGICA: SUBSAMPLING PER BILANCIARE I LIVELLI DI N ---
    print("\n--- Balancing N levels by subsampling to the minimum count ---")
    data_by_N = {n: [] for n in num_levels_to_generate}
    images_by_N = {n: [] for n in num_levels_to_generate}

    # Raggruppa i dati e le immagini per livello di N
    for i, (n_val, *_) in enumerate(all_data):
        if n_val in data_by_N: # Assicurati che N_val sia un livello che ci interessa
            data_by_N[n_val].append(all_data[i])
            images_by_N[n_val].append(all_images_raw[i])

    min_samples_per_level = float('inf')
    for n_level in num_levels_to_generate:
        if n_level in data_by_N:
            min_samples_per_level = min(min_samples_per_level, len(data_by_N[n_level]))
    
    if min_samples_per_level == float('inf') or min_samples_per_level == 0:
        print("Warning: No samples generated for any N level or min samples is 0. Cannot subsample.")
        final_data_arr = np.array([])
        all_images_raw_subsampled = []
    else:
        print(f"Minimum samples found for any N level: {min_samples_per_level}")
        subsampled_data = []
        subsampled_images = []

        for n_level in num_levels_to_generate:
            current_level_data = data_by_N.get(n_level, [])
            current_level_images = images_by_N.get(n_level, [])

            if len(current_level_data) > min_samples_per_level:
                # Seleziona casualmente gli indici
                indices = np.random.choice(len(current_level_data), min_samples_per_level, replace=False)
                subsampled_data.extend([current_level_data[i] for i in indices])
                subsampled_images.extend([current_level_images[i] for i in indices])
            else:
                subsampled_data.extend(current_level_data)
                subsampled_images.extend(current_level_images)
        
        final_data_arr = np.array(subsampled_data)
        all_images_raw_subsampled = subsampled_images
    # --- FINE LOGICA DI SUBSAMPLING ---


    final_correlations, _ = check_correlations(final_data_arr, min_N_for_correlation=GLOBAL_NUM_LEVELS_TO_GENERATE[0])

    dataset_filename = os.path.join(current_output_dir, f"circle_dataset_{final_output_size}x{final_output_size}_corr_tuned_v12.npz")
    correlations_filename = os.path.join(current_output_dir, f"correlations_{final_output_size}x{final_output_size}_corr_tuned_v12.png")

    # Passa i dati subsamplati a save_dataset
    save_dataset(final_data_arr.tolist(), all_images_raw_subsampled, dataset_filename, final_output_size, gen_size)
    plot_correlations(final_data_arr, correlations_filename, min_N_filter=GLOBAL_NUM_LEVELS_TO_GENERATE[0])

    print(f"\nTotal process time (Exploration + Full Generation): {(time.time() - start_time) / 60:.1f} minutes")
    print(f"\nFinal correlations (for N >= {GLOBAL_NUM_LEVELS_TO_GENERATE[0]}):")
    for name, corr in final_correlations.items():
        print(f"   {name:15s}: {corr:6.3f}")

    print("\n" + "="*70)
    print(f"FINAL DATASET SUMMARY ({final_output_size}x{final_output_size})")
    print("="*70)

    N_values = final_data_arr[:, 0] if final_data_arr.size > 0 else np.array([])
    total_samples = len(final_data_arr)

    print(f"Total samples generated (after subsampling): {total_samples:,}")
    print(f"Target samples per level (before subsampling): {target_samples_per_n_level:,}")
    if min_samples_per_level != float('inf'):
        print(f"Actual samples per level (after subsampling): {min_samples_per_level:,}")
    else:
        print("Actual samples per level (after subsampling): N/A (no samples generated)")

    print(f"\nSamples per N level (after subsampling):")
    level_counts = {}
    for level in sorted(set(N_values)):
        count = np.sum(N_values == level)
        level_counts[int(level)] = count
        print(f"  Level {int(level):2d}: {count:4d} samples")

    if total_samples > 0:
        CA_arr, FR_arr, CH_arr, dens_arr, size_arr = final_data_arr[:, 1], final_data_arr[:, 2], final_data_arr[:, 3], final_data_arr[:, 4], final_data_arr[:, 5]

        print(f"\nData quality metrics (ranges after generation):")
        # CH normalizzata ora rispetto a 100x100
        ch_normalizzata_effettiva = CH_arr / (gen_size * gen_size)

        print(f"  Normalized Convex Hull: {ch_normalizzata_effettiva.min():.2f} - {ch_normalizzata_effettiva.max():.2f} (mean: {ch_normalizzata_effettiva.mean():.2f})")
        print(f"  Density: {dens_arr.min():.4f} - {dens_arr.max():.4f} (mean: {dens_arr.mean():.4f})")

        print(f"  Cumulative Area: {CA_arr.min():.0f} - {CA_arr.max():.0f} (mean: {CA_arr.mean():.0f})")
        print(f"  Field Radius: {FR_arr.min():.1f} - {FR_arr.max():.1f} (mean: {FR_arr.mean():.1f})")
        print(f"  Item Size: {size_arr.min():.1f} - {size_arr.max():.1f} (mean: {size_arr.mean():.1f})")

    # --- GUIDA PER IL TUNING MANUALE DELLE CORRELAZIONI ---
    print("\n" + "="*70)
    print("GUIDA PER IL TUNING DELLE CORRELAZIONI CON N")
    print("Le correlazioni (r) sono proprietà dell'intero dataset e non possono essere filtrate stimolo per stimolo.")
    print("Per raggiungere i tuoi obiettivi di correlazione con N (CH r~0.2, Density r<0.4), potresti dover")
    print("eseguire il programma più volte, modificando i range esplorativi iniziali per influenzare i range calibrati.")
    print("\nEcco alcuni suggerimenti basati sui risultati ottenuti:")

    current_corr_ch = final_correlations.get('Convex Hull', np.nan)
    current_corr_density = final_correlations.get('Density', np.nan)

    print(f"\nCorrelazione attuale r(N, Convex Hull): {current_corr_ch:.3f} (Obiettivo: ~{TARGET_CORRELATION_CH:.1f})")
    if not np.isnan(current_corr_ch):
        if abs(current_corr_ch - TARGET_CORRELATION_CH) < 0.05:
            print("  -> Correlazione CH con N vicina all'obiettivo! Ottimo lavoro!")
        elif current_corr_ch < TARGET_CORRELATION_CH:
            print("  -> Per AUMENTARE r(N, Convex Hull) verso 0.2:")
            print("     - Prova ad AUMENTARE leggermente `EXPLORATION_FIELD_RADIUS_RANGE` (es. `(10, 80)` o più ampio).")
            print("     - Questo tenderà a distribuire più i punti per N maggiori, aumentando CH con N.")
        else: # current_corr_ch > TARGET_CORRELATION_CH
            print("  -> Per DIMINUIRE r(N, Convex Hull) verso 0.2 (se troppo alta):")
            print("     - Prova a RIDURRE leggermente `EXPLORATION_FIELD_RADIUS_RANGE` (es. `(5, 60)` o più stretto).")
            print("     - Questo tenderà a mantenere CH più compatta per N maggiori.")

    print(f"\nCorrelazione attuale r(N, Density): {current_corr_density:.3f} (Obiettivo: <{TARGET_CORRELATION_DENSITY_MAX:.1f})")
    if not np.isnan(current_corr_density):
        if current_corr_density < TARGET_CORRELATION_DENSITY_MAX:
            print("  -> Correlazione Densità con N sotto l'obiettivo! Molto bene!")
        else: # current_corr_density >= TARGET_CORRELATION_DENSITY_MAX
            print("  -> Per DIMINUIRE r(N, Density) sotto 0.4 (se troppo alta):")
            print("     - Prova ad AUMENTARE `EXPLORATION_FIELD_RADIUS_RANGE` (punti più sparsi -> CH maggiore -> densità minore).")
            print("     - Prova a DIMINUIRE `EXPLORATION_ITEM_RADIUS_RANGE` (item più piccoli -> CA minore -> densità minore).")
            print("     - Questi cambiamenti dovrebbero ridurre la tendenza della densità ad aumentare con N.")

    print("\nRicorda: Modifica i range in `EXPLORATION_..._RANGE` all'inizio del codice e riavvia.")
    print("Potrebbero volerci alcuni tentativi per trovare la combinazione ideale.")
    print("="*70)

    return final_data_arr, all_images_raw_subsampled, final_correlations

# --- ESECUZIONE DEL PROGRAMMA PRINCIPALE ---
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support() # Necessario per Windows per il multiprocessing

    print("\n\nSTARTING DATASET GENERATION PROCESS (VERSION 12 - CORRELATION TUNING ASSISTED)")
    print("=" * 70)

    # Definizione del target_samples_per_n_level per la generazione FINALE
    GLOBAL_TARGET_SAMPLES_PER_N_LEVEL = 2500 # Puoi aggiustare questo valore

    # FASE 1: Esplorazione e Calibrazione dei Range
    suggested_cum_area, suggested_field_radius, suggested_item_radius = \
        explore_and_suggest_ranges(
            disp_size_gen=GLOBAL_DISP_SIZE_GEN,
            num_levels=GLOBAL_NUM_LEVELS_TO_GENERATE,
            exploration_target_samples_per_n=EXPLORATION_NUM_SAMPLES_PER_N_LEVEL,
            exploration_max_attempts_factor=EXPLORATION_MAX_TOTAL_ATTEMPTS_FACTOR,
            initial_cum_area_range=EXPLORATION_CUM_AREA_RANGE,
            initial_field_radius_range=EXPLORATION_FIELD_RADIUS_RANGE,
            initial_item_radius_range=EXPLORATION_ITEM_RADIUS_RANGE
        )

    # FASE 2: Generazione completa del Dataset con i Range Calibrati
    # Puoi opzionalmente modificare i range suggeriti qui prima di passarli alla funzione
    # Esempio:
    # suggested_cum_area = (int(suggested_cum_area[0] * 0.9), int(suggested_cum_area[1] * 1.1))

    data_final, images_final, correlations_final = \
        generate_dataset_fast_no_value_filters(
            gen_size=GLOBAL_DISP_SIZE_GEN,
            final_output_size=GLOBAL_FINAL_OUTPUT_SIZE,
            output_directory="circle_dataset_corr_tuned_v12",
            num_levels_to_generate=GLOBAL_NUM_LEVELS_TO_GENERATE,
            target_samples_per_n_level=GLOBAL_TARGET_SAMPLES_PER_N_LEVEL,
            calibrated_cum_area_range=suggested_cum_area,
            calibrated_field_radius_range=suggested_field_radius,
            calibrated_item_radius_range=suggested_item_radius
        )

    print("\n" + "="*70)
    print("PROGRAM EXECUTION COMPLETE!")
    print("="*70)