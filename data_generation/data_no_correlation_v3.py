import os
import numpy as np
from skimage.draw import disk
from skimage.morphology import convex_hull_image
from skimage.transform import resize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
import scipy.stats
import math
import random
from collections import deque

# ==========================================================
# CONFIG
# ==========================================================
GLOBAL_IMG_SIZE = 100
GLOBAL_NUM_LEVELS = np.arange(1, 33)  # N = 1..32

ITEM_RADIUS_RANGE = (3, 10)           # r per item (px)
FIELD_RADIUS_RANGE = (15, 60)         # r del "campo" (px)

# target di cumArea (in pixel ON) — il budget d’area viene centrato qui
CUMAREA_TARGET = (200, 1200)
# target CH normalizzato (CH / img_area) — usato come filtro ex-post
CH_TARGET = (0.2, 0.3)

MAX_PLACEMENT_ATTEMPTS = 5000
MAX_TOTAL_ATTEMPTS_FACTOR = 10000

# ==========================================================
# UTILS
# ==========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def zscore_N(N_level, N_min, N_max):
    # normalizza N su [-1, 1] circa
    if N_max == N_min:
        return 0.0
    return 2.0 * ( (N_level - N_min) / (N_max - N_min) ) - 1.0

def softclip(x, lo, hi, margin=0.05):
    # clamp morbido (evita stickiness ai bordi durante l'update)
    return clamp(x, lo + margin*(hi-lo), hi - margin*(hi-lo))

# ==========================================================
# CONTROLLER ADATTIVO
# ==========================================================
class CorrTuner:
    """
    Controller che aggiorna automaticamente:
      - p_tie_base  : probabilità media che due item condividano r (raggi "uguali")
      - beta_tie    : dipendenza di p_tie da N (lineare in z-score)
      - radius_var  : varianza intra-immagine dei raggi
      - spread_base : quanto si espandono i centri (raggio effettivo di posizionamento)
      - beta_spread : dipendenza dello spread da N
      - min_gap_base, beta_gap : gap anti-overlap e sua dipendenza da N
    L'update minimizza |corr| per cumArea, CH, density (target = 0).
    """
    def __init__(self,
                 p_tie_base=0.5, beta_tie=0.0,
                 radius_var=0.35,
                 spread_base=0.85, beta_spread=0.0,
                 min_gap_base=1.0, beta_gap=0.0,
                 lr=0.15,                       # learning rate controller
                 eval_window=4000,              # calcolo correlazioni ogni X campioni accettati
                 corr_smooth=0.9,               # smoothing esponenziale
                 seed=42):
        self.p_tie_base = p_tie_base
        self.beta_tie = beta_tie
        self.radius_var = radius_var
        self.spread_base = spread_base
        self.beta_spread = beta_spread
        self.min_gap_base = min_gap_base
        self.beta_gap = beta_gap

        self.lr = lr
        self.eval_window = eval_window
        self.corr_smooth = corr_smooth

        random.seed(seed)
        np.random.seed(seed)

        # buffer per valutare correlazioni
        self.hist_N = deque(maxlen=eval_window)
        self.hist_CA = deque(maxlen=eval_window)
        self.hist_CH = deque(maxlen=eval_window)
        self.hist_DEN = deque(maxlen=eval_window)

        # EMA delle correlazioni
        self.ema_corr = {"cumArea": 0.0, "CH": 0.0, "density": 0.0}

        # limiti parametri
        self.bounds = {
            "p_tie_base": (0.0, 1.0),
            "beta_tie":   (-0.8, 0.8),
            "radius_var": (0.05, 0.65),
            "spread_base":(0.4, 1.0),
            "beta_spread":(-0.6, 0.6),
            "min_gap_base":(0.5, 4.0),
            "beta_gap":   (-0.6, 0.6),
        }

    def current_level_params(self, N_level, Nmin, Nmax):
        z = zscore_N(N_level, Nmin, Nmax)
        # p_tie dipende da N: più alto => più item condividono lo stesso raggio
        p_tie = clamp(self.p_tie_base + self.beta_tie * z, 0.0, 1.0)
        spread = clamp(self.spread_base + self.beta_spread * z, 0.3, 1.0)
        min_gap = clamp(self.min_gap_base + self.beta_gap * z, 0.3, 6.0)
        return {
            "p_tie": p_tie,
            "radius_var": self.radius_var,
            "spread": spread,
            "min_gap": min_gap
        }

    def push_sample(self, N, cumArea, CH, density):
        self.hist_N.append(float(N))
        self.hist_CA.append(float(cumArea))
        self.hist_CH.append(float(CH))
        self.hist_DEN.append(float(density))

    def _corr(self, x, y):
        if len(x) < 10:
            return 0.0
        try:
            r = scipy.stats.pearsonr(np.asarray(x), np.asarray(y))[0]
            if np.isnan(r):
                return 0.0
            return float(r)
        except Exception:
            return 0.0

    def maybe_update(self):
        """
        Se abbiamo abbastanza campioni nel buffer, calcola corr(N, ·) e aggiorna i parametri.
        Ritorna un dict con le correnti corr per logging.
        """
        if len(self.hist_N) < self.hist_N.maxlen:
            return None

        r_ca = self._corr(self.hist_N, self.hist_CA)
        r_ch = self._corr(self.hist_N, self.hist_CH)
        r_den= self._corr(self.hist_N, self.hist_DEN)

        # EMA
        self.ema_corr["cumArea"] = self.corr_smooth*self.ema_corr["cumArea"] + (1-self.corr_smooth)*r_ca
        self.ema_corr["CH"]      = self.corr_smooth*self.ema_corr["CH"]      + (1-self.corr_smooth)*r_ch
        self.ema_corr["density"] = self.corr_smooth*self.ema_corr["density"] + (1-self.corr_smooth)*r_den

        # --- aggiornamento parametri (gradient-free, sign-based) ---
        # 1) cumArea ~ 0: area budget già aiuta. Se persiste correlazione:
        #    - se r_ca > 0: aumentiamo varianza raggi e p_tie verso pattern che rompano la monotonia
        #    - se r_ca < 0: riduciamo un po' la varianza
        delta = self.lr * np.sign(self.ema_corr["cumArea"])
        self.radius_var = softclip(self.radius_var + (-delta*0.10), *self.bounds["radius_var"])
        self.p_tie_base = softclip(self.p_tie_base + (-delta*0.06), *self.bounds["p_tie_base"])
        self.beta_tie   = softclip(self.beta_tie   + (-delta*0.05), *self.bounds["beta_tie"])

        # 2) CH ~ 0: se r_ch > 0 (CH cresce con N), riduciamo spread ai grandi N (beta_spread negativo),
        #             se r_ch < 0, aumentiamo spread con N (beta_spread positivo)
        delta_ch = self.lr * np.sign(self.ema_corr["CH"])
        self.beta_spread = softclip(self.beta_spread + (-delta_ch*0.10), *self.bounds["beta_spread"])
        # e spostiamo leggermente lo spread base verso il centro
        self.spread_base = softclip(self.spread_base + (-delta_ch*0.03), *self.bounds["spread_base"])

        # 3) density ~ 0: agiamo su min_gap e sua dipendenza da N
        delta_den = self.lr * np.sign(self.ema_corr["density"])
        self.beta_gap = softclip(self.beta_gap + (-delta_den*0.10), *self.bounds["beta_gap"])
        self.min_gap_base = softclip(self.min_gap_base + (-delta_den*0.03), *self.bounds["min_gap_base"])

        return {"corr_cumArea": r_ca, "corr_CH": r_ch, "corr_density": r_den,
                "ema_cumArea": self.ema_corr["cumArea"],
                "ema_CH": self.ema_corr["CH"],
                "ema_density": self.ema_corr["density"]}

# ==========================================================
# GENERAZIONE DI UN SINGOLO STIMOLO (mix raggi uguali/diversi + budget d'area)
# ==========================================================
def generate_stimulus(
    N_target,
    disp_size,
    tuner_params,                       # dict da CorrTuner.current_level_params
    area_budget=True,
    area_target_range=CUMAREA_TARGET
):
    """
    - Crea k raggi unici (k deciso da p_tie) e li assegna agli N item (ties possibili).
    - Scala i raggi con un fattore s per centrare la cumArea target (area budget).
    - Posiziona i centri con spread controllato e gap minimo dipendenti da tuner_params.
    Ritorna: (img_uint8, N_target, cum_area, density)
    """
    img = np.zeros((disp_size, disp_size), dtype=np.float32)

    # Campo e centro
    field_radius = np.random.uniform(*FIELD_RADIUS_RANGE)
    center = disp_size // 2

    # --- 1) Scegli quanti raggi unici (k) ---
    p_tie = tuner_params["p_tie"]
    # k va da 1 (tutti uguali) a N (tutti diversi). Più alto p_tie -> meno k.
    # mapping semplice: k = round((1 - p_tie)*N) ma almeno 1
    k_unique = int(clamp(round((1.0 - p_tie) * N_target), 1, N_target))

    # --- 2) Campiona i k raggi unici con varianza controllata ---
    base_r = np.random.uniform(*ITEM_RADIUS_RANGE)
    radius_var = tuner_params["radius_var"]
    # lognormal attorno a base_r
    eps = np.random.normal(loc=0.0, scale=radius_var, size=k_unique)
    unique_r = base_r * np.exp(eps)
    unique_r = np.clip(unique_r, ITEM_RADIUS_RANGE[0], ITEM_RADIUS_RANGE[1])

    # --- 3) Assegna i raggi agli N item (con ties) ---
    r_list = np.random.choice(unique_r, size=N_target, replace=True)

    # --- 4) Area budget: scala i raggi per centrare cumArea target (≈ somma pi r^2) ---
    if area_budget:
        area_min, area_max = area_target_range
        area_now = np.pi * np.sum(r_list ** 2)
        area_tgt = np.random.uniform(area_min, area_max)
        if area_now > 0:
            s = np.sqrt(area_tgt / area_now)
            r_list = r_list * s
        r_list = np.clip(r_list, ITEM_RADIUS_RANGE[0], ITEM_RADIUS_RANGE[1])

    # --- 5) Spread e gap dal tuner ---
    spread = tuner_params["spread"]          # (0.3..1.0) fattore su raggio campo effettivo
    min_gap_px = tuner_params["min_gap"]

    # raggio massimo per posizionamento, tenendo il più grande r_list
    placement_radius_max = min(field_radius, center - (np.max(r_list) + 1))
    placement_radius_max = max(placement_radius_max * spread, 5.0)
    if placement_radius_max <= 0:
        return None

    # --- 6) Posizionamento non-overlap con gap dinamico ---
    centers = []
    attempts = 0
    max_attempts = MAX_PLACEMENT_ATTEMPTS * 2

    while len(centers) < N_target and attempts < max_attempts:
        angle = np.random.uniform(0, 2 * np.pi)
        rr = np.sqrt(np.random.uniform(0, placement_radius_max ** 2))
        cx = int(round(center + rr * np.cos(angle)))
        cy = int(round(center + rr * np.sin(angle)))

        if (cx < 0 or cx >= disp_size or cy < 0 or cy >= disp_size):
            attempts += 1
            continue

        idx = len(centers)
        ri = r_list[idx]
        ok = True
        for (xj, yj, rj) in centers:
            if np.hypot(cx - xj, cy - yj) < (ri + rj + min_gap_px):
                ok = False
                break
        if ok:
            centers.append((cx, cy, ri))
        attempts += 1

    if len(centers) < N_target:
        return None

    # --- 7) Disegna dischi ---
    for cx, cy, ri in centers:
        int_radius = max(1, int(round(ri)))
        rr, cc = disk((cy, cx), int_radius, shape=img.shape)
        img[rr, cc] = 1.0

    # --- 8) Features ---
    img_resized = resize(img, (disp_size, disp_size), anti_aliasing=False, preserve_range=True)
    cum_area = np.sum(img_resized > 0.5)
    try:
        ch = convex_hull_image(img_resized > 0.5).sum()
    except Exception:
        ch = cum_area
    density = cum_area / ch if ch > 0 else 0.0
    img_resized = (img_resized * 255).astype(np.uint8)

    return img_resized, N_target, cum_area, density

# ==========================================================
# GENERAZIONE ADATTIVA PER UN LIVELLO (usa CorrTuner)
# ==========================================================
def generate_level(args):
    N_level, samples_per_level, tuner, disp_size, Nmin, Nmax = args
    images, data = [], []

    # vincoli iniziali per filtro CH e cumArea
    cumarea_low, cumarea_high = CUMAREA_TARGET
    ch_low, ch_high = CH_TARGET

    tolerance_step_area = 100
    tolerance_step_ch = 0.05

    attempts = 0
    max_attempts = samples_per_level * MAX_TOTAL_ATTEMPTS_FACTOR

    while len(images) < samples_per_level and attempts < max_attempts:
        params = tuner.current_level_params(N_level, Nmin, Nmax)
        result = generate_stimulus(
            N_target=N_level,
            disp_size=disp_size,
            tuner_params=params,
            area_budget=True,
            area_target_range=CUMAREA_TARGET
        )
        attempts += 1
        if result is None:
            continue

        img, N, ca, dens = result
        ch_val = ca / dens if dens > 0 else ca

        # filtro CH normalizzato
        if cumarea_low <= ca <= cumarea_high and ch_low <= (ch_val / (disp_size**2)) <= ch_high:
            images.append(img.flatten())
            data.append([N, ca, ch_val, dens])

            # invia al tuner per la stima corr (aggiornamento globale nel chiamante)
            tuner.push_sample(N, ca, ch_val, dens)

        # rilassa vincoli se fatica a trovare esempi
        if attempts % (MAX_PLACEMENT_ATTEMPTS * 5) == 0 and len(images) < samples_per_level:
            cumarea_low = max(50, cumarea_low - tolerance_step_area)
            cumarea_high = cumarea_high + tolerance_step_area
            ch_low = max(0.0, ch_low - tolerance_step_ch)
            ch_high = min(1.0, ch_high + tolerance_step_ch)

    return N_level, data, images

# ==========================================================
# GENERAZIONE DATASET COMPLETO CON ADATTAMENTO DEL TUNER
# ==========================================================
def generate_dataset(output_dir="stimuli_dataset_adaptive_auto", samples_per_level=1000, eval_window=4000):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # inizializza tuner (eval_window uguale per tutti i processi)
    tuner = CorrTuner(eval_window=eval_window)

    # Distribuiamo il tuner in subprocess? Qui lo passiamo per riferimento;
    # ogni processo aggiornerà il proprio tuner locale ma noi vogliamo
    # aggiornamenti globali: soluzione semplice -> generiamo per blocchi/loop
    # senza parallelo stretto, così possiamo aggiornare il tuner periodicamente.
    # Per mantenere semplicità e coerenza del controller, qui useremo un loop sequenziale
    # sui livelli (è più deterministico per l’adattamento).
    # Se vuoi assolutamente la parallelizzazione, si può fare per batch e
    # sincronizzare i parametri tra batch.

    all_data, all_images = [], []
    Nmin, Nmax = int(GLOBAL_NUM_LEVELS.min()), int(GLOBAL_NUM_LEVELS.max())

    for N in tqdm(GLOBAL_NUM_LEVELS, desc="Levels (adaptive)"):
        N = int(N)
        # genera per livello (sequenziale per permettere l'adattamento globale)
        lvl_args = (N, samples_per_level, tuner, GLOBAL_IMG_SIZE, Nmin, Nmax)
        _, data, images = generate_level(lvl_args)
        all_data.extend(data)
        all_images.extend(images)

        # ogni fine livello prova un update del tuner (se finestra piena)
        stats = tuner.maybe_update()
        if stats is not None:
            print(f"\n[TUNER] after level N={N}: "
                  f"corr(N,cumArea)={stats['corr_cumArea']:.3f}, "
                  f"corr(N,CH)={stats['corr_CH']:.3f}, "
                  f"corr(N,density)={stats['corr_density']:.3f}  "
                  f"| EMA: {tuner.ema_corr}\n"
                  f"params: p_tie_base={tuner.p_tie_base:.3f}, beta_tie={tuner.beta_tie:.3f}, "
                  f"radius_var={tuner.radius_var:.3f}, spread={tuner.spread_base:.3f}, "
                  f"beta_spread={tuner.beta_spread:.3f}, min_gap={tuner.min_gap_base:.3f}, "
                  f"beta_gap={tuner.beta_gap:.3f}")

    all_data = np.array(all_data, dtype=np.float32)
    all_images = np.array(all_images, dtype=np.uint8)

    np.savez_compressed(os.path.join(output_dir, "stimuli_dataset.npz"),
                        D=all_images,
                        N_list=all_data[:, 0],
                        cumArea_list=all_data[:, 1],
                        CH_list=all_data[:, 2],
                        density=all_data[:, 3])

    # Correlazioni globali (N>5, come facevi tu)
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
    # Esempio: 1500 per livello, finestra di valutazione 4000 campioni
    fn = generate_dataset(samples_per_level=1500, eval_window=4000)
    print("Dataset file:", fn)
