import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import pickle
from skimage.morphology import convex_hull_image
import cv2
from tqdm.auto import tqdm
import os
import hashlib

np.random.seed(13)

# Code to generate numerosity data stimuli as in DeWind et al. (2015)
type_ = '_TR'  # Use '_TR' for training and '_TE' for testing
disp_x, disp_y = 200, 200  # Display dimensions

downsample = 1  # If different than 1, save also scaled images
min_spacing = 4  # Minimum gap from borders and between items
center_displays = False  # Center displays on screen?
max_min_check = False # Check if if extreme values would fit the display; used for debugging when changing parameters

distribution = 'uniform' # choose between zipfian or uniform

if distribution == 'zipf':
    ### zipfian distribution parameter
    def shifted_zipf_pmf(k, a, s):
        return (k + s)**(-a) / sum((np.arange(1, max(k) + 1) + s)**(-a))
    # MSCOCO Fitted Parameters
    a = 112.27
    s = 714.33

    TOTAL_Number_Stimuli = 500 # Total number of stimuli for each Spacing and Size combination
    # Range of k values
    k_values = np.arange(1, 41)  # Example range from 1 to 40
    # Calculate probabilities and abs frequencies
    probabilities = shifted_zipf_pmf(k_values, a, s)
    n_images = TOTAL_Number_Stimuli*probabilities
    n_images_dict = {
        num: freq for num, freq in enumerate(n_images, start=1)}

if distribution == 'uniform':
    n_images_dict = {
        num: 14 for num in range(1,41)}
    
max_tries = 20000  # How many trials before trying a different placement [placing object without overlap can be hard]
max_dot_tries = 10000
# our parameters (DeWind had displays of 400x400 pixels, so his parameters were a bit different)
# note that these parameters vary depending on minNumerosity and maxNumerosity
minNumerosity = 1
maxNumerosity = 40
minSize = 1.6e5   #[5 to 25: 1; 1 to 32: 2; 7 to 28: 2.6]???
maxSize = 6.4e5  #[7 to 28: 10.4]???
minSpacing = 6e6    #[5 to 25: 6.5; 1 to 32: 9; 7 to 28: 8]???
maxSpacing = 24e6  #[7 to 28: 32]???

fname = f'NumStim_{minNumerosity}to{maxNumerosity}'
save_dir = '/home/student/Desktop/Groundeep/training_tensors/uniform'
os.makedirs(save_dir, exist_ok=True)
# Generate parameter ranges

# Use linspace to cover all numerosities:
Num_range = np.linspace(minNumerosity, maxNumerosity, 40)
# Use logspace to reproduce DeWind methodology:
#Num_range = np.logspace(np.log10(minNumerosity),np.log10(maxNumerosity),13)
Siz_range = np.logspace(np.log10(minSize), np.log10(maxSize), 13)
Spa_range = np.logspace(np.log10(minSpacing), np.log10(maxSpacing), 13)
#print(Siz_range, Spa_range)
if max_min_check == True:
    Num_range = [min(Num_range), max(Num_range)]
    Siz_range = [min(Siz_range), max(Siz_range)]
    Spa_range = [min(Spa_range), max(Spa_range)]
    fname = f'CHECK_NumStim_{minNumerosity}to{maxNumerosity}'

tot_patterns = sum(int(repetitions) + 1 for repetitions in n_images_dict.values()) * len(Siz_range) * len(Spa_range)
D = np.zeros((disp_x * disp_y, tot_patterns), dtype=np.uint8)
D_list = []
N_list, TSA_list, cumArea_list, FA_list, CH_list = [], [], [], [], []

colGrid, rowGrid = np.meshgrid(np.arange(disp_x), np.arange(disp_y))
# Loop through all combinations
with tqdm(total=tot_patterns, desc="Generating stimuli") as pbar:
    index = 0
    for n in np.flip(Num_range):
        n = round(n)
        target_reps = n_images_dict[n] + 1
        print(f'Generating for Num:{n}')
        for Sz in Siz_range:
            for Sp in Spa_range:
                # Calculate parameters
                TSA = np.sqrt(Sz * n)
                ISA = np.sqrt(Sz / n)
                FA = np.sqrt(Sp * n)
                r = round(np.sqrt(ISA / np.pi))
                r_FA = round(np.sqrt(FA / np.pi))
                
                # Track unique patterns for this combination
                combo_hashes = set()
                generated = 0
                attempts = 0
                
                while generated < target_reps and attempts < max_tries:
                    # Create unique seed for this attempt
                    seed_str = f"{n}_{Sz}_{Sp}_{generated}_{attempts}"
                    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
                    np.random.seed(seed)
                    
                    # Generate display
                    currDisp = np.zeros((disp_x, disp_y), dtype=np.uint8)
                    count = 0
                    item_positions = []
                    
                    # Generate FA circle
                    if center_displays:
                        FA_circle = (rowGrid - disp_x//2)**2 + (colGrid - disp_y//2)**2 <= r_FA**2
                    else:
                        try:
                            FA_center_x = np.random.randint(r_FA, disp_x - r_FA)
                            FA_center_y = np.random.randint(r_FA, disp_y - r_FA)
                            FA_circle = (rowGrid - FA_center_x)**2 + (colGrid - FA_center_y)**2 <= r_FA**2
                        except:
                            break

                    # Place items
                    dot_tries = 0
                    while count < n and dot_tries < max_dot_tries:
                        # Get available positions
                        y_idx, x_idx = np.where(FA_circle)
                        if len(x_idx) == 0: break
                        
                        # Select position
                        choice = np.random.choice(len(x_idx))
                        pos_x = x_idx[choice]
                        pos_y = y_idx[choice]
                        
                        # Check spacing
                        if item_positions:
                            distances = np.sqrt((np.array(item_positions) - [pos_x, pos_y])**2 @ [1,1])
                            if np.any(distances < (2*r + min_spacing)):
                                dot_tries += 1
                                if dot_tries > 0 and dot_tries % 1000 ==0:
                                    print(f'Throwing dots failed in FA for Num: {n} for {dot_tries} times')
                                    if dot_tries == max_dot_tries:
                                        print(f'Failed for Num{n} with one FA setting, moving to next one')
                                continue
                        
                        # Place item
                        cv2.circle(currDisp, (pos_x, pos_y), r, (255,), thickness=-1, lineType=cv2.LINE_AA)
                        item_positions.append([pos_x, pos_y])
                        count += 1
                    
                    # Check success and uniqueness
                    if count == n:
                        curr_hash = hashlib.md5(currDisp.tobytes()).hexdigest()
                        if curr_hash not in combo_hashes:
                            combo_hashes.add(curr_hash)

                            # Store data
                            D_list.append(currDisp.flatten(order='F'))
                            N_list.append(n)
                            TSA_list.append(TSA)
                            cumArea_list.append(np.sum(currDisp))
                            FA_list.append(FA)
                            CH_list.append(np.sum(convex_hull_image(currDisp)))
                            generated += 1
                            index += 1
                            
                    attempts += 1
                    if attempts%1000 == 0 and attempts > 0:
                        print(f"Retrying patterns for n={n}, Sz={Sz:.1e}, Sp={Sp:.1e} {attempts}/{max_tries}")
                    if attempts == max_tries:
                        print(f"Failed to generate {target_reps} patterns for n={n}, Sz={Sz:.1e}, Sp={Sp:.1e}")
                    pbar.update(1)

                # Report generation success
                if generated < target_reps:
                    print(f"\nShortfall: {target_reps-generated}/{target_reps} patterns for n={n}, Sz={Sz:.1e}, Sp={Sp:.1e}")

# Convert to final arrays
D = np.column_stack(D_list) if D_list else np.empty((disp_x*disp_y, 0), dtype=np.uint8)
N_list = np.array(N_list)
TSA_list = np.array(TSA_list)
cumArea_list = np.array(cumArea_list)
FA_list = np.array(FA_list)
CH_list = np.array(CH_list)

# Save dataset
# with open(f'{fname}_{disp_x}x{disp_y}{type_}.pkl', 'wb') as f:
#     pickle.dump({'D': D, 'N_list': N_list, 'TSA_list': TSA_list,
#                  'cumArea_list': cumArea_list, 'FA_list': FA_list, 'CH_list': CH_list}, f)
    
# Downsample and save
to_save = 1

if downsample != 1:
    D_scaled = np.array([scipy.ndimage.zoom(D[:, i].reshape(disp_x, disp_y), downsample).flatten(order='F')
                        for i in range(D.shape[1])]).T
else:
    D_scaled = D
    if to_save:
        with open(os.path.join(save_dir,f'{fname}_{int(disp_x * downsample)}x{int(disp_y * downsample)}{type_}_{distribution}.pkl'), 'wb') as f:
            pickle.dump({'D': D_scaled, 'N_list': N_list, 'TSA_list': TSA_list,
                        'cumArea_list': cumArea_list, 'FA_list': FA_list, 'CH_list': CH_list}, f)

# Print final shapes
print("Dataset saved with the following shapes:")
print(f"D: {D_scaled.shape}")
print(f"N_list: {N_list.shape}")
print(f"TSA_list: {TSA_list.shape}")
print(f"cumArea_list: {cumArea_list.shape}")
print(f"FA_list: {FA_list.shape}")
print(f"CH_list: {CH_list.shape}")

# Show scatters as in DeWind
ISA_list = np.array(TSA_list) / np.array(N_list)
Size = np.array(TSA_list) * ISA_list
sparsity_FA = np.array(FA_list) / np.array(N_list)
Spacing = sparsity_FA * np.array(FA_list)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(ISA_list, TSA_list, s=10)
plt.subplot(2, 2, 2)
plt.scatter(np.log2(ISA_list), np.log2(TSA_list), s=10)
plt.subplot(2, 2, 3)
plt.scatter(sparsity_FA, FA_list, s=10)
plt.subplot(2, 2, 4)
plt.scatter(np.log2(sparsity_FA), np.log2(FA_list), s=10)
plt.show()

# Display random image sample
tot_patterns = D_scaled.shape[1]
disp_size = int(np.sqrt(D_scaled.shape[0]))
r, c = 5, 5
fig, axes = plt.subplots(r, c, figsize=(10, 10))
for i in range(r):
    for j in range(c):
        n = np.random.randint(tot_patterns)
        axes[i, j].imshow(D[:, n].reshape(disp_x, disp_y), cmap='gray')
        axes[i, j].axis('off')
plt.show()
