import numpy as np
import os
import random
from skimage.draw import disk
from skimage.morphology import convex_hull_image
from skimage.transform import resize
from tqdm import tqdm

# Parameters
fname = 'NumStim_1to32'
disp_x, disp_y = 200, 200
downsample = 0.5
min_spacing = 4
center_displays = 0
repetitions = 13
max_tries = 20000

minNumerosity = 1
maxNumerosity = 40
minSize = 2e5
maxSize = 8e5
minSpacing = 7.5e6
maxSpacing = 30e6

Num_range = np.linspace(minNumerosity, maxNumerosity, maxNumerosity, dtype=int)
Siz_range = np.logspace(np.log10(minSize), np.log10(maxSize), 13)
Spa_range = np.logspace(np.log10(minSpacing), np.log10(maxSpacing), 13)

# Max possible patterns
tot_patterns = len(Num_range) * len(Siz_range) * len(Spa_range) * repetitions

# Preallocate max space (will be trimmed to actual size later)
D_raw = np.zeros((disp_x * disp_y, tot_patterns), dtype=np.float32)
N_list = np.zeros(tot_patterns, dtype=np.int32)
TSA_list = np.zeros(tot_patterns, dtype=np.float32)
cumArea_list = np.zeros(tot_patterns, dtype=np.float32)
FA_list = np.zeros(tot_patterns, dtype=np.float32)
CH_list = np.zeros(tot_patterns, dtype=np.float32)

index = 0

with tqdm(total=tot_patterns, desc="Generating stimuli") as pbar:
    for n in Num_range:
        for Sz in Siz_range:
            for Sp in Spa_range:
                TSA = np.sqrt(Sz * n)
                ISA = np.sqrt(Sz / n)
                FA = np.sqrt(Sp * n)

                r = int(np.round(np.sqrt(ISA / np.pi)))
                r_FA = int(np.round(np.sqrt(FA / np.pi)))

                for _ in range(repetitions):
                    currDisp = np.zeros((disp_x, disp_y), dtype=np.uint8)
                    count = 0
                    tries = 0
                    success = True

                    rowGrid, colGrid = np.meshgrid(np.arange(disp_y), np.arange(disp_x))

                    if center_displays:
                        FA_center_x, FA_center_y = disp_x // 2, disp_y // 2
                    else:
                        if disp_x - r_FA <= r_FA or disp_y - r_FA <= r_FA:
                            pbar.update(1)
                            continue
                        FA_center_x = random.randint(r_FA, disp_x - r_FA)
                        FA_center_y = random.randint(r_FA, disp_y - r_FA)

                    FA_circle = (rowGrid - FA_center_y)**2 + (colGrid - FA_center_x)**2 <= r_FA**2

                    while count < n:
                        if tries > max_tries:
                            success = False
                            break
                        tries += 1

                        coords = np.argwhere(FA_circle)
                        pos_y, pos_x = coords[random.randint(0, len(coords) - 1)]
                        pos_x = np.clip(pos_x, r + min_spacing, disp_x - r - min_spacing)
                        pos_y = np.clip(pos_y, r + min_spacing, disp_y - r - min_spacing)

                        rr, cc = disk((pos_y, pos_x), r, shape=currDisp.shape)
                        if currDisp[rr, cc].sum() == 0:
                            currDisp[rr, cc] = 1
                            count += 1

                    if success:
                        D_raw[:, index] = currDisp.flatten()
                        N_list[index] = n
                        TSA_list[index] = TSA
                        cumArea_list[index] = currDisp.sum()
                        FA_list[index] = FA
                        CH_list[index] = convex_hull_image(currDisp).sum()
                        index += 1

                    pbar.update(1)

# Trim arrays to actual number of successful patterns
D_raw = D_raw[:, :index]
N_list = N_list[:index]
TSA_list = TSA_list[:index]
cumArea_list = cumArea_list[:index]
FA_list = FA_list[:index]
CH_list = CH_list[:index]

# Downsampling
if downsample != 1:
    new_x = int(disp_x * downsample)
    new_y = int(disp_y * downsample)
    D_scaled = np.zeros((new_x * new_y, index), dtype=np.float32)
    for p in range(index):
        img = D_raw[:, p].reshape(disp_x, disp_y)
        img_resized = resize(img, (new_x, new_y), anti_aliasing=True)
        D_scaled[:, p] = img_resized.flatten()
    D = D_scaled
else:
    D = D_raw

# Save
save_dir = '/home/student/Desktop/Groundeep/training_tensors/uniform'
os.makedirs(save_dir, exist_ok=True)

out_name = f'{fname}_{int(disp_x * downsample)}x{int(disp_y * downsample)}.npz'
np.savez_compressed(
    os.path.join(save_dir, out_name),
    D=D,
    N_list=N_list,
    TSA_list=TSA_list,
    cumArea_list=cumArea_list,
    FA_list=FA_list,
    CH_list=CH_list
)

print(f"Saved dataset to {os.path.join(save_dir, out_name)}")
print(f"Total samples saved: {index}")
