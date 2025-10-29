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
GLOBAL_NUM_LEVELS = np.arange(20, 33)  # N da 1 a 32

ITEM_RADIUS_RANGE = (3, 10)      # raggio minimo/massimo per singolo dot (px)
ITEM_SIZE_CV = 0.25              # <== NUOVO: coefficiente di variazione intra-immagine delle size (0 = tutte uguali)

# range target iniziali (poi si allargano se serve)
CUMAREA_TARGET = (200, 1200)
CH_TARGET = (0.2, 0.3)  # su CH normalizzata (CH / img_area)

# tentativi (restano, ma ora usiamo una strategia che impedisce di uscire con quota incompleta)
MAX_PLACEMENT_ATTEMPTS = 2000

# Nuvola mobile: cerchio di posizionamento come frazione del semilato (50px)
PLACEMENT_RADIUS_RATIO_RANGE = (0.40, 0.90)

# Opzioni extra
AREA_BUDGET = True    # scala i raggi per fissare la cumArea target per immagine (indipendente da N)
LOG_CSV = True        # salva progressivi in stimuli_dataset_adaptive/progress.csv

# RNG dedicato (opzionale)
_RNG = np.random.default_rng()

# ==========================================================
# UTILS
# ==========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def sample_placement_circle(disp_size, max_item_radius):
    """
    Cerchio di posizionamento MOBILE:
    - Per evitare il fallback al centro, calcoliamo il massimo raggio consentito
      in funzione del margine: r_place_max = half - margin - 1.
    - Poi scegliamo r_place uniformemente in [r_min, r_place_max].
    - Il centro (cx, cy) è uniforme nel range [r_place + margin, disp_size - r_place - margin],
      che è sempre valido per come abbiamo scelto r_place.
    Ritorna (cx, cy, r_place_eff) dove r_place_eff ha già tolto max_item_radius.
    """
    half = disp_size / 2.0
    margin = max_item_radius + 2.0
    r_place_max = max(5.0, half - margin - 1.0)  # garantisce spazio per traslare
    r_place_min = max(5.0, 0.40 * r_place_max)   # evita cerchi troppo piccoli

    # se per qualche motivo il margine “mangia” tutto, fai un raggio minimo
    if r_place_max <= r_place_min:
        r_place_max = r_place_min + 1.0

    # estrai r_place in modo che ci sia SEMPRE spazio per muovere il centro
    r_place = np.random.uniform(r_place_min, r_place_max)

    lo = r_place + margin
    hi = disp_size - r_place - margin
    # ora hi > lo è garantito dalla scelta di r_place
    cx = np.random.uniform(lo, hi)
    cy = np.random.uniform(lo, hi)

    r_place_eff = max(3.0, r_place - max_item_radius - 1.0)
    return cx, cy, r_place_eff

def _apply_area_budget(radii, target_range):
    """
    Scala i raggi per portare la somma delle aree nel range target (uniforme nel range),
    clampando in ITEM_RADIUS_RANGE. Un micro-pass finale se abbiamo clippato molto.
    """
    area_now = np.pi * np.sum(radii ** 2)
    if area_now <= 0:
        return radii
    area_tgt = np.random.uniform(*target_range)
    s = np.sqrt(max(area_tgt, 1e-6) / area_now)
    r = np.clip(radii * s, ITEM_RADIUS_RANGE[0], ITEM_RADIUS_RANGE[1])

    area_now2 = np.pi * np.sum(r ** 2)
    if area_tgt > 0 and abs(area_now2 - area_tgt) / area_tgt > 0.25:
        s2 = np.sqrt(area_tgt / max(area_now2, 1e-6))
        r = np.clip(r * s2, ITEM_RADIUS_RANGE[0], ITEM_RADIUS_RANGE[1])
    return r

# ==========================================================
# GENERAZIONE DI UN SINGOLO STIMOLO
# ==========================================================
def generate_stimulus(N_target, disp_size=GLOBAL_IMG_SIZE):
    """
    - Posizionamento in un CERCHIO MOBILE (centro random) => nuvola davvero “jitterata”
    - AREA_BUDGET per rendere cumArea ~ indipendente da N
    - NUOVO: variazione intra-immagine delle size (ITEM_SIZE_CV > 0)
    - Ritorna anche mean/std dei raggi item
    """
    img = np.zeros((disp_size, disp_size), dtype=np.float32)

    # raggio base "mu" attorno al quale campionare le size
    base_radius = np.random.uniform(*ITEM_RADIUS_RANGE)

    # cerchio MOBILE di posizionamento (usiamo base_radius per evitare uscita)
    cx_circ, cy_circ, r_place_eff = sample_placement_circle(disp_size, base_radius)
    placement_radius_max = r_place_eff
    if placement_radius_max <= 0:
        return None

    # Se ITEM_SIZE_CV = 0 -> tutti uguali (ricade nel caso precedente)
    if ITEM_SIZE_CV > 0:
        # lognormal per garantire positività; sigma da CV
        sigma = np.sqrt(np.log(1 + ITEM_SIZE_CV**2))
        # mean dell'underlying normal: uso log(base_radius) (poi area budget corregge eventuale drift)
        r_raw = _RNG.lognormal(mean=np.log(max(base_radius, 1e-6)), sigma=sigma, size=N_target)
        r_raw = np.clip(r_raw, ITEM_RADIUS_RANGE[0], ITEM_RADIUS_RANGE[1])
    else:
        r_raw = np.full(N_target, base_radius, dtype=float)

    # Applica area budget (scala tutti i raggi di un fattore comune per colpire la cumArea target)
    if AREA_BUDGET:
        area_now = np.pi * np.sum(r_raw**2)
        area_tgt = np.random.uniform(*CUMAREA_TARGET)
        s = np.sqrt(max(area_tgt, 1e-6) / max(area_now, 1e-6))
        radii = np.clip(r_raw * s, ITEM_RADIUS_RANGE[0], ITEM_RADIUS_RANGE[1])
    else:
        radii = r_raw

    # posizionamento non-overlap
    centers = []
    attempts = 0
    while len(centers) < N_target and attempts < MAX_PLACEMENT_ATTEMPTS:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, placement_radius_max ** 2))
        cx = int(round(cx_circ + r * np.cos(angle)))
        cy = int(round(cy_circ + r * np.sin(angle)))

        if (cx < 0 or cx >= disp_size or cy < 0 or cy >= disp_size):
            attempts += 1
            continue

        ri = radii[len(centers)]
        ok = True
        for (xj, yj, rj) in centers:
            if np.hypot(cx - xj, cy - yj) < (ri + rj + 1.0):
                ok = False
                break

        # bordi
        if ok and (cx - ri < 0 or cx + ri >= disp_size or cy - ri < 0 or cy + ri >= disp_size):
            ok = False

        if ok:
            centers.append((cx, cy, ri))
        attempts += 1

    if len(centers) < N_target:
        return None

    # disegna
    for cx, cy, ri in centers:
        int_radius = max(1, int(round(ri)))
        rr, cc = disk((cy, cx), int_radius, shape=img.shape)
        img[rr, cc] = 1.0

    img_resized = resize(img, (disp_size, disp_size),
                         anti_aliasing=True, preserve_range=True)

    cum_area = np.sum(img_resized > 0.5)
    try:
        ch = convex_hull_image(img_resized > 0.5).sum()
    except Exception:
        ch = cum_area
    density = cum_area / ch if ch > 0 else 0.0

    mean_r = float(np.mean(radii))
    std_r  = float(np.std(radii))

    img_resized = (img_resized * 255).astype(np.uint8)
    return img_resized, N_target, cum_area, density, mean_r, std_r

# ==========================================================
# GENERAZIONE PER UN LIVELLO (N) — QUOTA SEMPRE RAGGIUNTA
# ==========================================================
def generate_level(args):
    """
    Raccoglie esattamente `samples_per_level` immagini:
    - Applica i vincoli iniziali (cumArea/CH norm).
    - Ogni volta che non si fanno progressi sufficienti, allarga i vincoli.
    - Se dopo alcuni cicli ancora pochi esempi, entra in 'aggressive mode' e disattiva i vincoli di filtro.
    """
    N_level, samples_per_level = args
    images, data = [], []

    # vincoli iniziali
    cumarea_low, cumarea_high = CUMAREA_TARGET
    ch_low, ch_high = CH_TARGET

    relax_step_area = 100
    relax_step_ch = 0.05

    no_progress_iters = 0
    aggressive_mode = False

    while len(images) < samples_per_level:
        result = generate_stimulus(N_level)
        if result is None:
            no_progress_iters += 1
        else:
            img, N, ca, dens, mean_r, std_r = result
            ch_val = ca / dens if dens > 0 else ca
            ch_norm = ch_val / float(GLOBAL_IMG_SIZE**2)

            accept = True
            if not aggressive_mode:
                # con AREA_BUDGET attivo, cumArea è spesso nel range; manteniamo il check
                if not (cumarea_low <= ca <= cumarea_high):
                    accept = False
                if not (ch_low <= ch_norm <= ch_high):
                    accept = False

            if accept:
                images.append(img.flatten())
                data.append([N, ca, ch_val, dens, mean_r, std_r])
                no_progress_iters = 0
            else:
                no_progress_iters += 1

        # RELAX: se non si progredisce da molto, allarghiamo i vincoli
        if len(images) < samples_per_level:
            if no_progress_iters > 5000:
                cumarea_low = max(20, cumarea_low - relax_step_area)
                cumarea_high += relax_step_area
                ch_low = max(0.0, ch_low - relax_step_ch)
                ch_high = min(1.0, ch_high + relax_step_ch)
                no_progress_iters = 0

            # Aggressive mode: se dopo vari relax ancora non basta, disattiva i vincoli
            if (cumarea_low <= 50 and cumarea_high >= 2000) or (ch_low <= 0.05 and ch_high >= 0.95):
                aggressive_mode = True

    return N_level, data, images

# ==========================================================
# GENERAZIONE DATASET COMPLETO + LOG PROGRESSIVO
# ==========================================================
def _save_level_chunk(output_dir, N, data_lvl, imgs_lvl, compressed=False, compress_level=1):
    """
    Salva subito il livello N in stimuli_dataset_adaptive_new/levels/level_XX.npz
    - compressed=False -> np.savez (più veloce)
    - compressed=True  -> np.savez_compressed(..., compress_level=1..9) se disponibile
    """
    import numpy as _np
    lvl_dir = os.path.join(output_dir, "levels")
    os.makedirs(lvl_dir, exist_ok=True)

    data = _np.asarray(data_lvl, dtype=_np.float32)   # [n, 6]
    imgs = _np.asarray(imgs_lvl, dtype=_np.uint8)     # [n, 10000]

    fn = os.path.join(lvl_dir, f"level_{int(N):02d}.npz")
    if compressed:
        try:
            _np.savez_compressed(fn,
                                 D=imgs,
                                 N_list=data[:, 0],
                                 cumArea_list=data[:, 1],
                                 CH_list=data[:, 2],
                                 density=data[:, 3],
                                 mean_item_size=data[:, 4],
                                 std_item_size=data[:, 5],
                                 compress_level=compress_level)
        except TypeError:
            _np.savez_compressed(fn,
                                 D=imgs,
                                 N_list=data[:, 0],
                                 cumArea_list=data[:, 1],
                                 CH_list=data[:, 2],
                                 density=data[:, 3],
                                 mean_item_size=data[:, 4],
                                 std_item_size=data[:, 5])
    else:
        _np.savez(fn,
                  D=imgs,
                  N_list=data[:, 0],
                  cumArea_list=data[:, 1],
                  CH_list=data[:, 2],
                  density=data[:, 3],
                  mean_item_size=data[:, 4],
                  std_item_size=data[:, 5])
    return imgs.shape[0]


def _merge_levels(output_dir, final_name="stimuli_dataset.npz", compressed=True, compress_level=1):
    """
    Merge RAM-safe: carica i chunk per livello e concatena.
    Per velocità: compressed=True con compress_level basso, oppure compressed=False.
    """
    import numpy as _np
    lvl_dir = os.path.join(output_dir, "levels")
    parts = sorted([os.path.join(lvl_dir, f) for f in os.listdir(lvl_dir) if f.endswith(".npz")])
    if not parts:
        raise RuntimeError("Nessun chunk trovato in 'levels/'. Hanno girato i livelli?")

    D_list, N_list, CA_list, CH_list, DEN_list, MU_list, SD_list = [], [], [], [], [], [], []
    for p in parts:
        z = _np.load(p)
        D_list.append(z["D"])
        N_list.append(z["N_list"])
        CA_list.append(z["cumArea_list"])
        CH_list.append(z["CH_list"])
        DEN_list.append(z["density"])
        MU_list.append(z["mean_item_size"])
        SD_list.append(z["std_item_size"])

    D  = _np.concatenate(D_list, axis=0)
    N  = _np.concatenate(N_list, axis=0)
    CA = _np.concatenate(CA_list, axis=0)
    CH = _np.concatenate(CH_list, axis=0)
    DE = _np.concatenate(DEN_list, axis=0)
    MU = _np.concatenate(MU_list, axis=0)
    SD = _np.concatenate(SD_list, axis=0)

    fn = os.path.join(output_dir, final_name)
    if compressed:
        try:
            _np.savez_compressed(fn, D=D, N_list=N, cumArea_list=CA, CH_list=CH,
                                 density=DE, mean_item_size=MU, std_item_size=SD,
                                 compress_level=compress_level)
        except TypeError:
            _np.savez_compressed(fn, D=D, N_list=N, cumArea_list=CA, CH_list=CH,
                                 density=DE, mean_item_size=MU, std_item_size=SD)
    else:
        _np.savez(fn, D=D, N_list=N, cumArea_list=CA, CH_list=CH,
                  density=DE, mean_item_size=MU, std_item_size=SD)
    return D.shape[0]


def generate_dataset(output_dir="stimuli_dataset_adaptive_new", samples_per_level=2000):
    """
    Versione robusta:
    - salva SUBITO ogni livello (checkpoint)
    - rilascia RAM tra i livelli
    - merge finale con compressione bassa/none
    """
    import gc

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    tasks = [(int(N), samples_per_level) for N in GLOBAL_NUM_LEVELS]
    # CSV opzionale
    csv_path = os.path.join(output_dir, "progress.csv")
    if LOG_CSV:
        with open(csv_path, "w") as f:
            f.write("level,nsamples,mean_cumArea,std_cumArea,mean_CH,std_CH,mean_density,std_density,mean_item,std_item\n")

    # 1) genera per livello e SALVA SUBITO il chunk
    with Pool(processes=min(cpu_count(), 12)) as pool:
        for N, data, images in tqdm(pool.imap_unordered(generate_level, tasks), total=len(tasks)):
            lvl = int(N)
            # report per livello
            if len(data) > 0:
                arr = np.asarray(data, dtype=np.float32)
                ca = arr[:, 1]; ch = arr[:, 2]; den = arr[:, 3]; mu = arr[:, 4]; sd = arr[:, 5]
                print(f"\n[Level N={lvl:2d}] "
                      f"samples={len(data)} | "
                      f"cumArea μ={ca.mean():.1f} σ={ca.std():.1f} | "
                      f"CH μ={ch.mean():.1f} σ={ch.std():.1f} | "
                      f"density μ={den.mean():.3f} σ={den.std():.3f} | "
                      f"item μ={mu.mean():.2f} σ={sd.mean():.2f}")
                if LOG_CSV:
                    with open(csv_path, "a") as f:
                        f.write(f"{lvl},{len(data)},{ca.mean():.4f},{ca.std():.4f},{ch.mean():.4f},{ch.std():.4f},"
                                f"{den.mean():.6f},{den.std():.6f},{mu.mean():.6f},{sd.mean():.6f}\n")

            # checkpoint del livello (no/low compression per velocità)
            _ = _save_level_chunk(output_dir, lvl, data, images,
                                  compressed=False,  # più veloce e non “si pianta”
                                  compress_level=1)

            # libera RAM di questo livello
            del data, images
            gc.collect()

    # 2) MERGE finale (puoi usare compressione leggera per avere un unico file)
    n_total = _merge_levels(output_dir, final_name="stimuli_dataset.npz",
                            compressed=True, compress_level=1)
    print(f"\n[Merged] campioni totali: {n_total}")

    # 3) report correlazioni finali (caricando dal file unico)
    z = np.load(os.path.join(output_dir, "stimuli_dataset.npz"))
    mask = z["N_list"] > 5
    N_arr = z["N_list"][mask]
    print("\nCorrelazioni FINALi con N (N>5):")
    names = ["cumArea_list", "CH_list", "density", "mean_item_size", "std_item_size"]
    for name in names:
        try:
            r = scipy.stats.pearsonr(N_arr, z[name][mask])[0]
        except Exception:
            r = np.nan
        print(f"  {name:14s}: {r:.3f}")

    return os.path.join(output_dir, "stimuli_dataset.npz")


if __name__ == "__main__":
    # fn = generate_dataset(output_dir='stimuli_dataset_kuinan_test', samples_per_level=100)

    # ATTENZIONE: assicurati che la cartella qui sotto esista davvero, altrimenti togli questa riga extra di merge.
    n_total = _merge_levels("stimuli_dataset_kuinan_test", final_name="stimuli_dataset.npz",
                             compressed=True, compress_level=1)
    print(f"\n[Merged] campioni totali: {n_total}")

    # 3) report correlazioni finali (caricando dal file unico)

    # z = np.load(os.path.join("stimuli_dataset_adaptive_new", "stimuli_dataset.npz"))
    # mask = z["N_list"] > 5
    # N_arr = z["N_list"][mask]
    # print("\nCorrelazioni FINALi con N (N>5):")
    # names = ["cumArea_list", "CH_list", "density", "mean_item_size", "std_item_size"]
    # for name in names:
    #     try:
    #         r = scipy.stats.pearsonr(N_arr, z[name][mask])[0]
    #     except Exception:
    #         r = np.nan
    #     print(f"  {name:14s}: {r:.3f}")

    # print("Dataset file:")
