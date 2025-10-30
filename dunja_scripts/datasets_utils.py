# ADJUSTED FOR ISA2, change back before use!!!!



import numpy as np
import pandas as pd
from scipy import io
import scipy
from scipy.stats import norm
import torch
import h5py
try:
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tf = None
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def train_dataset_detailed(train_dataset, train_file):
    train_contents = scipy.io.loadmat(train_file)

    N_list = train_contents['N_list']
    TSA_list = train_contents['TSA_list']
    FA_list = train_contents['FA_list']

    numLeft = []
    numRight = []
    isaLeft = []
    isaRight = []
    faLeft = []
    faRight = []

    N_list = np.squeeze(N_list)
    TSA_list = np.squeeze(TSA_list)
    FA_list = np.squeeze(FA_list)

    idxs_flat = train_dataset['idxs'].view(-1, 2)
    for idx_pair in idxs_flat:
        idx_left, idx_right = int(idx_pair[0])-1, int(idx_pair[1])-1

        numLeft.append(N_list[idx_left])
        numRight.append(N_list[idx_right])
        isaLeft.append(TSA_list[idx_left] / N_list[idx_left])
        isaRight.append(TSA_list[idx_right] / N_list[idx_right])
        faLeft.append(FA_list[idx_left])
        faRight.append(FA_list[idx_right])

    numLeft, numRight, isaLeft, isaRight, faLeft, faRight = np.array(numLeft),np.array(numRight), np.array(isaLeft), np.array(isaRight), np.array(faLeft), np.array(faRight)

    tsaLeft = isaLeft * numLeft
    sizeLeft = isaLeft * tsaLeft
    sparLeft = faLeft / numLeft
    spaceLeft = sparLeft * faLeft

    tsaRight = isaRight * numRight
    sizeRight = isaRight * tsaRight
    sparRight = faRight / numRight
    spaceRight = sparRight * faRight

    numRatio = (numRight / numLeft)
    sizeRatio = (sizeRight / sizeLeft)
    spaceRatio = (spaceRight / spaceLeft)

    numRatio_batched = numRatio.reshape(152, 100,1)
    sizeRatio_batched = sizeRatio.reshape(152, 100,1)
    spaceRatio_batched = spaceRatio.reshape(152, 100,1)
    data_batched = train_dataset['data'].reshape(152, 100,20000)
    labels_batched = train_dataset['labels'].reshape(152, 100,2)
    idxs_batched = train_dataset['idxs'].reshape(152, 100,2)

    train_dataset_extra = {
        'data': data_batched,
        'labels': labels_batched,
        'idxs': idxs_batched,
        'numRatio': numRatio_batched,
        'sizeRatio': sizeRatio_batched,
        'spaceRatio': spaceRatio_batched
    }

    print('\nData shapes:')
    for key, value in train_dataset_extra.items():
        print(f"Key: {key}, Shape: {value.shape}")

    return train_dataset_extra

def stratified_dataset_size_reduction(train_dataset, train_file,reduction_ratio,batch_size=152):
    train_dataset_extra = train_dataset_detailed(train_dataset, train_file)

    if reduction_ratio !=1:
        # Flatten the dataset
        data_flat = train_dataset_extra["data"].reshape(-1, train_dataset_extra["data"].shape[-1])
        labels_flat = train_dataset_extra["labels"].reshape(-1, train_dataset_extra["labels"].shape[-1])
        idxs_flat = train_dataset_extra["idxs"].reshape(-1, train_dataset_extra["idxs"].shape[-1])
        numRatio_flat = train_dataset_extra["numRatio"].reshape(-1)
        sizeRatio_flat = train_dataset_extra["sizeRatio"].reshape(-1)
        spaceRatio_flat = train_dataset_extra["spaceRatio"].reshape(-1)

        num_bins = np.digitize(numRatio_flat, bins=np.linspace(numRatio_flat.min(), numRatio_flat.max(), 13))
        size_bins = np.digitize(sizeRatio_flat, bins=np.linspace(sizeRatio_flat.min(), sizeRatio_flat.max(), 13))
        space_bins = np.digitize(spaceRatio_flat, bins=np.linspace(spaceRatio_flat.min(), spaceRatio_flat.max(), 13))

        stratification_keys = [(n, s, sp) for n, s, sp in zip(num_bins, size_bins, space_bins)]
        unique_keys = np.unique(stratification_keys, axis=0)
        print('number of stratification keys:', len(stratification_keys))
        print('number of unique stratification keys:', len(unique_keys))

        # Proportional sampling from each group
        reduced_indices = []
        #indices_left = []
        forced_single_samples = []
        for key in unique_keys:
            key_indices = np.where(
                (num_bins == key[0]) &
                (size_bins == key[1]) &
                (space_bins == key[2])
            )[0]
            n_samples = max(1, int(len(key_indices) * reduction_ratio + 0.5))

            if n_samples == 1:
                forced_single_samples.extend(key_indices)
            else:
                sampled_indices = np.random.choice(key_indices, int(n_samples), replace=False)
                #nonsampled_indices = np.setdiff1d(key_indices, sampled_indices)
                reduced_indices.extend(sampled_indices)
                #indices_left.extend(nonsampled_indices)


        total_samples_needed = int(len(numRatio_flat) * reduction_ratio)
        current_samples = len(reduced_indices)
        missing_samples = total_samples_needed - current_samples

        if missing_samples > 0 and len(forced_single_samples) >= missing_samples:
            sampled_from_forced = np.random.choice(forced_single_samples, missing_samples, replace=True)
            reduced_indices.extend(sampled_from_forced)
        # elif missing_samples > 0 and len(forced_single_samples) < missing_samples:
        #     reduced_indices.extend(forced_single_samples)
        #     extra_missing_samples = missing_samples - len(forced_single_samples)
        #     extra_sampled_indices = np.random.choice(indices_left, extra_missing_samples, replace=False)
        #     reduced_indices.extend(extra_sampled_indices)

        reduced_indices = np.array(reduced_indices)
        shuffled_indices = np.random.permutation(reduced_indices) # ???
        reduced_data = data_flat[shuffled_indices]
        reduced_labels = labels_flat[shuffled_indices]
        reduced_idxs = idxs_flat[shuffled_indices]

        batch_size_1 = batch_size
        while total_samples_needed % batch_size_1 > 0:
            batch_size_1 += 1

        batch_size_2 = batch_size
        while total_samples_needed % batch_size_2 > 0:
            batch_size_2 -= 1

        if (batch_size_1 - batch_size) >= (batch_size - batch_size_2):
            batch_size = batch_size_2
        else:
            batch_size = batch_size_1

        # Reshape to new batch dimensions
        reduced_data = reduced_data.reshape(-1, batch_size, train_dataset_extra["data"].shape[-1])
        reduced_labels = reduced_labels.reshape(-1, batch_size, train_dataset_extra["labels"].shape[-1])
        reduced_idxs = reduced_idxs.reshape(-1, batch_size, train_dataset_extra["idxs"].shape[-1])

        print(f"Reduced data shape: {reduced_data.shape}")
        print(f"Reduced labels shape: {reduced_labels.shape}")
        print(f"Reduced idxs shape: {reduced_idxs.shape}")

        # Return reduced dataset
        train_dataset_reduced = {
            'data': torch.tensor(reduced_data).clone().detach(),
            'labels': torch.tensor(reduced_labels).clone().detach(),
            'idxs': torch.tensor(reduced_idxs).clone().detach(),
        }

    else:
        train_dataset_reduced = {
            'data': train_dataset_extra["data"],
            'labels': train_dataset_extra["labels"],
            'idxs': train_dataset_extra["idxs"]
        }

    return train_dataset_reduced

def single_stimuli_dataset(file, ref_num = 14.5, num_samples=10000, batch_size = 100):
    contents = scipy.io.loadmat(file)
    N_list = contents['N_list'].flatten()
    TSA_list = contents['TSA_list'].flatten()
    FA_list = contents['FA_list'].flatten()
    D = contents['D']

    data_list = []
    labels_list = []
    idxs_list = []

    # Unique ratios as keys for stratification
    unique_N = np.unique(N_list)
    unique_TSA = np.unique(TSA_list)
    unique_FA = np.unique(FA_list)

    print(f"Unique n: {len(unique_N)}")
    print(f"Unique TSA: {len(unique_TSA)}")
    print(f"Unique FA: {len(unique_FA)}")

    stratification_keys = [(n, tsa, fa) for n in unique_N for tsa in unique_TSA for fa in unique_FA]
    print('number of stratification keys:', len(stratification_keys))

    reduced_indices = []
    forced_single_samples = []
    for key in stratification_keys:
        key_indices = np.where(
            (N_list == key[0]) &
            (TSA_list == key[1]) &
            (FA_list == key[2])
        )[0]

        n_samples = max(1,num_samples//len(stratification_keys))
        if len(key_indices) < n_samples:
            sampled_indices = np.random.choice(key_indices, len(key_indices), replace=False)
        else:
            sampled_indices = np.random.choice(key_indices, n_samples, replace=False)


        if n_samples == 1:
            forced_single_samples.extend(key_indices)
        else:
            sampled_indices = np.random.choice(key_indices, n_samples, replace=False)
            reduced_indices.extend(sampled_indices)


    total_samples_needed = num_samples
    current_samples = len(reduced_indices)
    missing_samples = total_samples_needed - current_samples

    if missing_samples > 0 and len(forced_single_samples) >= missing_samples:
        sampled_from_forced = np.random.choice(forced_single_samples, missing_samples, replace=False)
        reduced_indices.extend(sampled_from_forced)


    reduced_indices = np.array(reduced_indices)
    reduced_indices = np.random.shuffle(reduced_indices)
    #reduced_indices = np.random.permutation(len(reduced_indices))

    # N_reduced = N_list[reduced_indices]
    # ref_num = N_reduced.mean()
    # print(f"Reference number: {ref_num}")

    for idx in reduced_indices:
            # Get the image data
            image_data = D[:, idx]  # Shape: (10000,)
            data_list.append(image_data)

            N_value = N_list[idx]
            label = 0 if N_value < ref_num else 1
            labels_list.append(label)

            idxs_list.append(idx)

    data_tensor = torch.tensor(np.array(data_list)).float()  # Shape: (num_samples, 10000)
    labels_tensor = torch.tensor(np.array(labels_list)).view(-1, 1)  # Shape: (num_samples, 1)
    idxs_tensor = torch.tensor(np.array(idxs_list)).view(-1, 1)  # Shape: (num_samples, 1)

    batch_size_1 = batch_size
    while len(idxs_list)%batch_size_1>0:
        batch_size_1 +=1

    batch_size_2 = batch_size
    while len(idxs_list)%batch_size_2>0:
        batch_size_2 -=1

    if (batch_size_1-batch_size)>= (batch_size-batch_size_2):
        batch_size = batch_size_2
    else:
        batch_size = batch_size_1

    num_batches = num_samples//batch_size

    data_tensor = data_tensor.view(num_batches, batch_size, -1)  # Shape: (num_batches, 100, 10000)
    labels_tensor = labels_tensor.view(num_batches, batch_size, 1)  # Shape: (num_batches, 100, 1)
    idxs_tensor = idxs_tensor.view(num_batches, batch_size, 1)  # Shape: (num_batches, 100, 1)

    dataset = {
        'data': data_tensor,
        #'labels': labels_tensor,
        'idxs': idxs_tensor
    }

    return dataset

def single_stimuli_dataset_modified(file, ref_num=14.5, num_samples=10000, batch_size=100, num_percentage_dict=None, binarize = False):

    try:
        contents = scipy.io.loadmat(file)
        N_list = contents['N_list'].flatten()
        TSA_list = contents['cumArea_list'].flatten()
        FA_list = contents['CH_list'].flatten()
        D = contents['D']

    except NotImplementedError:
        with h5py.File(file, 'r') as mat_file:
            N_list = mat_file['N_list'][()].flatten()
            TSA_list = mat_file['TSA_list'][()].flatten()
            FA_list = mat_file['FA_list'][()].flatten()
            D = mat_file['D'][()]
            D = D.T

    print(f'D shape: {D.shape}')

    data_list = []
    labels_list = []
    idxs_list = []

    unique_N = np.unique(np.array(N_list))
##############################################################################################################

    max_percentage_n = max(num_percentage_dict, key=num_percentage_dict.get)
    max_percentage_value = num_percentage_dict[max_percentage_n]
    max_n_indices = np.where(N_list == max_percentage_n)[0]
    max_n_count = len(max_n_indices)

    reduced_n_indices = []
    for n in unique_N:
        n_indices = np.where(N_list == n)[0]

        if n == max_percentage_n:
            reduced_n_indices.extend(n_indices)
        else:
            scaling_factor = num_percentage_dict[n] / max_percentage_value
            num_samples_to_keep = int((max_n_count * scaling_factor)+0.5)
            num_samples_to_keep = min(len(n_indices), num_samples_to_keep)
            sampled_indices = np.random.choice(n_indices, num_samples_to_keep, replace=False)
            reduced_n_indices.extend(sampled_indices)

    reduced_n_indices = np.array(reduced_n_indices)

    N_list_reduced = N_list[reduced_n_indices]
    TSA_list_reduced = TSA_list[reduced_n_indices]
    FA_list_reduced = FA_list[reduced_n_indices]
    print(N_list_reduced.shape)

    print(f"max number of samples: {len(reduced_n_indices)}")

    if num_samples > len(reduced_n_indices):
        num_samples = len(reduced_n_indices)

    print(f"number of samples: {num_samples}")

    # chosen_indices  = []
    # left_indices = []
    # num_to_keep_sum = 0

    # print(f"num dict keys sum: {sum(num_percentage_dict.values())}")

    # for N in unique_N:
    #     N_indices = np.where(N_list_reduced == N)[0]
    #     num_samples_N = len(N_indices)
    #     print(f'num samples N{num_samples_N}')
    #     print(f'numebr: {N}, perc: {num_percentage_dict[N]}')
    #     print(f'num samples toatal: {num_samples}')
    #     num_samples_N_to_keep = int(num_samples*(num_percentage_dict[N]/100))
    #     num_samples_N_to_keep = min(num_samples_N, num_samples_N_to_keep)
    #     print(f'num samples N to keep: {num_samples_N_to_keep}')
    #     sampled_indices = np.random.choice(N_indices, num_samples_N_to_keep, replace=False)
    #     print(f'num samples N to kept: {len(sampled_indices)}')
    #     left_indices.extend(np.setdiff1d(N_indices, sampled_indices))
    #     chosen_indices.extend(sampled_indices)
    #     num_to_keep_sum += num_samples_N_to_keep

    # print(f'num to keep sum: {num_to_keep_sum}')

    # print(f'initial chosen indices: {len(chosen_indices)}')

    # missing_samples = num_samples - len(chosen_indices)
    # if missing_samples > 0:
    #     sampled_from_left = np.random.choice(left_indices, missing_samples, replace=False)
    #     chosen_indices.extend(sampled_from_left)


    # print(f"number of chosen indices: {len(chosen_indices)}")
    # reduced_indices = chosen_indices
##############################################################################################################

    # Unique ratios as keys for stratification
    # unique_N = np.unique(np.array(N_list))
    # unique_TSA = np.unique(np.array(TSA_list))
    # unique_FA = np.unique(np.array(FA_list))
    # TSA_bins = np.digitize(TSA_list, bins=np.linspace(TSA_list.min(), TSA_list.max(), 13))
    # FA_bins = np.digitize(FA_list, bins=np.linspace(FA_list.min(), FA_list.max(), 13))


    # stratification_keys = [(n, tsa, fa) for n in unique_N for tsa in TSA_bins for fa in FA_bins]
    # print('number of stratification keys:', len(stratification_keys))

    # reduced_indices = []
    # forced_single_samples = []
    # for key in stratification_keys:
    #     key_indices = np.where(
    #         (N_list == key[0]) &
    #         (TSA_list == key[1]) &
    #         (FA_list == key[2])
    #     )[0]

    #     n_samples = max(1,num_samples//len(stratification_keys))
    #     if len(key_indices) < n_samples:
    #         sampled_indices = np.random.choice(key_indices, len(key_indices), replace=False)
    #     else:
    #         sampled_indices = np.random.choice(key_indices, n_samples, replace=False)


    #     if n_samples == 1:
    #         forced_single_samples.extend(key_indices)
    #     else:
    #         sampled_indices = np.random.choice(key_indices, n_samples, replace=False)
    #         reduced_indices.extend(sampled_indices)


    # total_samples_needed = num_samples
    # current_samples = len(reduced_indices)
    # missing_samples = total_samples_needed - current_samples

    # if missing_samples > 0 and len(forced_single_samples) >= missing_samples:
    #     sampled_from_forced = np.random.choice(forced_single_samples, missing_samples, replace=False)
    #     reduced_indices.extend(sampled_from_forced)

##############################################################################################################
    reduced_n_indices = [idx for idx in reduced_n_indices if N_list[idx] != ref_num] # Exclude the reference value!


    reduced_indices = np.random.choice(reduced_n_indices, num_samples, replace=False)


##############################################################################################################

    N_unique = np.unique(N_list)
    print(f"Unique numerosities: {N_unique}")

    # N_reduced = N_list[reduced_indices]
    # ref_num = N_reduced.mean()
    # print(f"Reference number: {ref_num}")
    num_list = []
    print(f'D shape: {D.shape}')

    if binarize == True:
        # Normalize each image to [0,1]
        D_min = D.min(axis=0, keepdims=True)
        D_max = D.max(axis=0, keepdims=True)
        D_norm = (D - D_min) / (D_max - D_min + 1e-8)  # avoid division by zero

        # Threshold at 0.5
        D_binary = (D_norm >= 0.5).astype(int)
        D = D_binary


    for idx in reduced_indices:
            # Get the image data
            image_data = D[:, idx]  # Shape: (10000,)
            data_list.append(image_data)

            N_value = N_list[idx]
            if N_value > ref_num:
                label = 1
            elif N_value < ref_num:
                label = 0
            else:
                raise ValueError('You kept the reference value!')

            labels_list.append(label)

            idxs_list.append(idx)
            num_list.append(N_value)


    print(f"Unique numerosities: {np.unique(num_list)}")

    data_tensor = torch.tensor(np.array(data_list)).float()  # Shape: (num_samples, 10000)
    labels_tensor = torch.tensor(np.array(labels_list)).view(-1, 1)  # Shape: (num_samples, 1)
    idxs_tensor = torch.tensor(np.array(idxs_list)).view(-1, 1)  # Shape: (num_samples, 1)

    num_batches = len(idxs_list) // batch_size

    data_tensor = data_tensor[: num_batches * batch_size].view(num_batches, batch_size, -1)  # Shape: (num_batches, batch_size, 10000)
    labels_tensor = labels_tensor[: num_batches * batch_size].view(num_batches, batch_size, 1)  # Shape: (num_batches, batch_size, 1)
    idxs_tensor = idxs_tensor[: num_batches * batch_size].view(num_batches, batch_size, 1)  # Shape: (num_batches, batch_size, 1)

    dataset = {
        'data': data_tensor,
        'labels': labels_tensor,
        'idxs': idxs_tensor
    }

    return dataset

def is_dot_within_center_circle(image_flat, image_size=100, radius=40, threshold=0.1, plot=False):
    """
    Checks if the dot (or dot blob) in the image is within a central circle of given radius.
    
    Parameters:
    - image_flat: Flattened image array
    - image_size: Width and height of the (square) image
    - radius: Radius of the central circle
    - threshold: Pixel intensity threshold to detect dot
    - plot: If True, plot the image with center circle and dot centroid
    """
    image = image_flat.reshape(image_size, image_size)
    dot_positions = np.argwhere(image >= threshold)

    if len(dot_positions) == 0:
        return False

    # Compute centroid of the dot blob
    y, x = dot_positions.mean(axis=0)

    center = image_size / 2
    distance = np.sqrt((x - center)**2 + (y - center)**2)

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        
        # Draw central circle
        circle = plt.Circle((center, center), radius, color='blue', fill=False, linestyle='--', label='Center circle')
        ax.add_patch(circle)
        ax.set_title(f'Distance from center: {distance:.2f} px')
        plt.axis('off')
        plt.show()

    return distance <= radius

def single_stimuli_dataset_naming(file, num_samples=10000, batch_size=100, num_percentage_dict=None, label_type = 'int', add_zero_images = False, limit_FA = True, limit_radius = 40):

    try:
        contents = scipy.io.loadmat(file)
        N_list = contents['N_list'].flatten()
        TSA_list = contents['cumArea_list'].flatten()
        FA_list = contents['CH_list'].flatten()
        D = contents['D']

    except NotImplementedError:
        with h5py.File(file, 'r') as mat_file:
            N_list = mat_file['N_list'][()].flatten()
            TSA_list = mat_file['TSA_list'][()].flatten()
            FA_list = mat_file['FA_list'][()].flatten()
            D = mat_file['D'][()]
            D = D.T

    print(f'D shape: {D.shape}')

    data_list = []
    labels_list = []
    idxs_list = []

    unique_N = np.unique(np.array(N_list))

##############################################################################################################
    # if limit_FA:
    #     print("Filtering numerosity-1 images outside center circle...")
    #     valid_one_dot_indices = []
    #     plotted = False 

    #     # Get indices where numerosity == 1
    #     n1_indices = np.where(N_list == 1)[0]

    #     for idx in n1_indices:
    #         image_data = D[:, idx]
    #         if is_dot_within_center_circle(image_data, plot=not plotted):
    #             valid_one_dot_indices.append(idx)
    #         if not plotted:
    #             plotted = True


    #     valid_one_dot_indices = np.array(valid_one_dot_indices)
    #     print(f"Kept {len(valid_one_dot_indices)} valid numerosity-1 images (inside circle)")
    ##############################################################################################################

    if limit_FA: 
        print("Filtering all images to keep only those with dots inside center circle...")
        valid_indices_by_n = {}
        plotted = False

        for n in unique_N:
            n_indices = np.where(N_list == n)[0]
            valid_indices = []

            for idx in n_indices:
                image_data = D[:, idx]
                if is_dot_within_center_circle(image_data, plot=not plotted, radius=limit_radius):
                    valid_indices.append(idx)
                if not plotted:
                    plotted = True  # Plot only once, for the first image that passes

            valid_indices_by_n[n] = np.array(valid_indices)
            print(f"Kept {len(valid_indices)} valid images for numerosity {n}")

##############################################################################################################
    # if limit_FA:
    #     max_percentage_n = 1
    #     max_percentage_value = num_percentage_dict[max_percentage_n]
    #     max_n_count = len(valid_one_dot_indices)
    # else:
    #     max_percentage_n = max(num_percentage_dict, key=num_percentage_dict.get)
    #     max_percentage_value = num_percentage_dict[max_percentage_n]
    #     max_n_indices = np.where(N_list == max_percentage_n)[0]
    #     max_n_count = len(max_n_indices)

    
    # reduced_n_indices = []
    # for n in unique_N:
    #     if limit_FA:
    #         if n == 1:
    #             n_indices = valid_one_dot_indices
    #         else:
    #             n_indices = np.where(N_list == n)[0]
    #     else:
    #         n_indices = np.where(N_list == n)[0]

    #     if n == max_percentage_n:
    #         reduced_n_indices.extend(n_indices)
    #     else:
    #         scaling_factor = num_percentage_dict[n] / max_percentage_value
    #         num_samples_to_keep = int((max_n_count * scaling_factor)+0.5)
    #         num_samples_to_keep = min(len(n_indices), num_samples_to_keep)
    #         sampled_indices = np.random.choice(n_indices, num_samples_to_keep, replace=False)
    #         reduced_n_indices.extend(sampled_indices)

    # reduced_n_indices = np.array(reduced_n_indices)
    ##############################################################################################################

    if limit_FA:
        max_percentage_n = max(num_percentage_dict, key=num_percentage_dict.get)
        max_percentage_value = num_percentage_dict[max_percentage_n]
        max_n_indices = valid_indices_by_n[max_percentage_n]
        max_n_count = len(max_n_indices)
    else:
        max_percentage_n = max(num_percentage_dict, key=num_percentage_dict.get)
        max_percentage_value = num_percentage_dict[max_percentage_n]
        max_n_indices = np.where(N_list == max_percentage_n)[0]
        max_n_count = len(max_n_indices)

    reduced_n_indices = []

    for n in unique_N:
        if limit_FA:
            n_indices = valid_indices_by_n[n]
        else:
            n_indices = np.where(N_list == n)[0]

        if n == max_percentage_n:
            reduced_n_indices.extend(n_indices)
        else:
            scaling_factor = num_percentage_dict[n] / max_percentage_value
            num_samples_to_keep = int((max_n_count * scaling_factor) + 0.5)
            num_samples_to_keep = min(len(n_indices), num_samples_to_keep)
            sampled_indices = np.random.choice(n_indices, num_samples_to_keep, replace=False)
            reduced_n_indices.extend(sampled_indices)

    reduced_n_indices = np.array(reduced_n_indices)

    N_list_reduced = N_list[reduced_n_indices]
    TSA_list_reduced = TSA_list[reduced_n_indices]
    FA_list_reduced = FA_list[reduced_n_indices]
    print(N_list_reduced.shape)

    print(f"max number of samples: {len(reduced_n_indices)}")

    if num_samples > len(reduced_n_indices):
        num_samples = len(reduced_n_indices)

    print(f"number of samples: {num_samples}")

##############################################################################################################

    reduced_indices = np.random.choice(reduced_n_indices, num_samples, replace=False)

    unique_N_new = np.unique(N_list_reduced)
    num_samples_per_num = num_samples//len(unique_N_new)

    reduced_indices = []
    for n in unique_N_new:
        # Get indices where numerosity is equal to n
        indices_n = [idx for idx in reduced_n_indices if N_list[idx] == n]

        # Sample from these
        if len(indices_n) < num_samples_per_num:
            raise ValueError(f"Not enough samples for numerosity {n}: needed {num_samples_per_num}, found {len(indices_n)}")

        sampled_indices = np.random.choice(indices_n, num_samples_per_num, replace=False)
        reduced_indices.extend(sampled_indices)

    reduced_indices = np.array(reduced_indices)
    reduced_indices = np.random.permutation(reduced_indices)


##############################################################################################################

    N_unique = np.unique(N_list)
    print(f"Unique numerosities: {N_unique}")

    # N_reduced = N_list[reduced_indices]
    # ref_num = N_reduced.mean()
    # print(f"Reference number: {ref_num}")
    num_list = []
    print(f'D shape: {D.shape}')

    nonlog_label_list = []

    for idx in reduced_indices:
            # Get the image data
            image_data = D[:, idx]  # Shape: (10000,)
            data_list.append(image_data)

            N_value = np.uint(N_list[idx])

            label = N_value

            if label_type == 'int':
                label = N_value
            # elif label_type == 'log':
            #     # nonlog_label_list.append(N_value)
            #     # label = np.log(N_value)
            else:
                print('Invalid label type (int or log)')


            labels_list.append(label)

            idxs_list.append(idx)
            num_list.append(N_value)


    print(f"Unique numerosities: {np.unique(num_list)}")

    data_array = np.array(data_list)  # Shape: (N, 10000)
    labels_array = np.array(labels_list).reshape(-1, 1)  # Shape: (N, 1)
    idxs_array = np.array(idxs_list).reshape(-1, 1)  # Shape: (N, 1)

    if add_zero_images:
        num_blank_samples = num_samples//len(unique_N)  # Adding as many blank images as real ones

        # Create blank images (same shape as real images)
        blank_images = np.zeros((num_blank_samples, D.shape[0]))  # Shape: (num_blank_samples, 10000)
        blank_labels = np.ones((num_blank_samples, 1))  # Labels = 0
        blank_idxs = np.full((num_blank_samples, 1), -1)  # Assign -1 for blank indices (or another marker)

       # Stack real and blank samples together and shuffle
        combined_data = np.vstack((data_array, blank_images))
        combined_labels = np.vstack((labels_array, blank_labels))
        combined_idxs = np.vstack((idxs_array, blank_idxs))

        # Shuffle the dataset (preserving alignment between data, labels, and indices)
        shuffle_indices = np.random.permutation(len(combined_data))
        data_shuffled = combined_data[shuffle_indices]
        labels_shuffled = combined_labels[shuffle_indices]
        idxs_shuffled = combined_idxs[shuffle_indices]

        data_tensor = torch.tensor(data_shuffled).float()
        labels_tensor = torch.tensor(labels_shuffled).view(-1, 1)
        idxs_tensor = torch.tensor(idxs_shuffled).view(-1, 1)

        num_total_samples = data_tensor.shape[0]

    else:
        data_tensor = torch.tensor(np.array(data_list)).float()  # Shape: (num_samples, 10000)
        labels_tensor = torch.tensor(np.array(labels_list)).view(-1, 1)  # Shape: (num_samples, 1)
        idxs_tensor = torch.tensor(np.array(idxs_list)).view(-1, 1)  # Shape: (num_samples, 1)

        num_total_samples = data_tensor.shape[0]

    num_batches = num_total_samples // batch_size  # Compute total batches

    # Reshape to maintain consistency
    data_tensor = data_tensor[: num_batches * batch_size].view(num_batches, batch_size, -1)  # (num_batches, 100, 10000)
    labels_tensor = labels_tensor[: num_batches * batch_size].view(num_batches, batch_size, 1)  # (num_batches, 100, 1)
    idxs_tensor = idxs_tensor[: num_batches * batch_size].view(num_batches, batch_size, 1)  # (num_batches, 100, 1)

    dataset = {
        'data': data_tensor,
        'labels': labels_tensor,
        'idxs': idxs_tensor
    }

    return dataset

def create_congruency_dataset(train_dataset, train_file, congruent=True, congruency_feature='CH', batch_size=100):
    data_flat = train_dataset["data"].reshape(-1, train_dataset["data"].shape[-1])
    labels_flat = train_dataset["labels"].reshape(-1, train_dataset["labels"].shape[-1])
    idxs_flat = train_dataset["idxs"].reshape(-1, train_dataset["idxs"].shape[-1])

    train_contents = scipy.io.loadmat(train_file)
    N_list = train_contents['N_list']
    TSA_list = train_contents['TSA_list']
    FA_list = train_contents['FA_list']
    CH_list = train_contents['CH_list']

    numLeft = []
    numRight = []
    isaLeft = []
    isaRight = []
    faLeft = []
    faRight = []
    chRight = []
    chLeft = []

    N_list = np.squeeze(N_list)
    TSA_list = np.squeeze(TSA_list)
    FA_list = np.squeeze(FA_list)
    CH_list = np.squeeze(CH_list)

    idxs_flat = train_dataset['idxs'].view(-1, 2)
    for idx_pair in idxs_flat:
        idx_left, idx_right = int(idx_pair[0])-1, int(idx_pair[1])-1
        numLeft.append(N_list[idx_left])
        numRight.append(N_list[idx_right])
        isaLeft.append(TSA_list[idx_left] / N_list[idx_left])
        isaRight.append(TSA_list[idx_right] / N_list[idx_right])
        faLeft.append(FA_list[idx_left])
        faRight.append(FA_list[idx_right])
        chLeft.append(CH_list[idx_left])
        chRight.append(CH_list[idx_right])
    print(f"ex. idx_left: {idx_left}, idx_right: {idx_right}")

    numLeft, numRight, isaLeft, isaRight, faLeft, faRight, chLeft, chRight = np.array(numLeft), np.array(numRight), np.array(isaLeft), np.array(isaRight), np.array(faLeft), np.array(faRight), np.array(chLeft), np.array(chRight)

    tsaLeft = isaLeft * numLeft
    sizeLeft = isaLeft * tsaLeft
    sparLeft = faLeft / numLeft
    spaceLeft = sparLeft * faLeft
    tpLeft = 2*np.sqrt(np.pi) + sizeLeft**(1/4) + numLeft**(3/4)

    tsaRight = isaRight * numRight
    sizeRight = isaRight * tsaRight
    sparRight = faRight / numRight
    spaceRight = sparRight * faRight
    tpRight = 2*np.sqrt(np.pi) + sizeRight**(1/4) + numRight**(3/4)

    numRatio = (numRight / numLeft)
    sizeRatio = (sizeRight / sizeLeft)
    spaceRatio = (spaceRight / spaceLeft)
    tsaRatio = (tsaRight/tsaLeft)
    tpRatio = (tpRight/tpLeft)

    FARatio = (faRight / faLeft)
    print(f"FA ratio range: {np.min(FARatio)} - {np.max(FARatio)}")

    CHRatio = (chRight / chLeft)
    print(f"CH ratio range: {np.min(CHRatio)} - {np.max(CHRatio)}")
    print(f"TSA ratio range: {np.min(tsaRatio)} - {np.max(tsaRatio)}")
    print(f"Size ratio range: {np.min(sizeRatio)} - {np.max(sizeRatio)}")
    print(f"Spacing ratio range: {np.min(spaceRatio)} - {np.max(spaceRatio)}")


    # Choose congruency feature
    feature_map = {'CH': CHRatio, 'FA': FARatio, 'TSA': tsaRatio, 'size': sizeRatio, 'space': spaceRatio, 'TP': tpRatio}
    featureRatio = feature_map.get(congruency_feature)
    if featureRatio is None:
        raise ValueError(f"Invalid congruency feature: {congruency_feature}\n Possible features: {list(feature_map.keys())}")

    # congruency conditions
    if congruent:
        condition_1_mask = (numRatio > 1) & (featureRatio > 1)
        condition_2_mask = (numRatio < 1) & (featureRatio < 1)
        combined_mask = condition_1_mask | condition_2_mask
    else:
        condition_1_mask = (numRatio > 1) & (featureRatio < 1)
        condition_2_mask = (numRatio < 1) & (featureRatio > 1)
        combined_mask = condition_1_mask | condition_2_mask


    selected_indices = np.where(combined_mask)[0]

    print(f"Number of selected indices: {len(selected_indices)}")

    selected_indices = np.array(selected_indices)

    reduced_labels = labels_flat[selected_indices]
    reduced_data = data_flat[selected_indices]
    reduced_idxs = idxs_flat[selected_indices]
    print(reduced_labels.shape)

    # Ensure equal number of 2 classes
    label_0_indices = np.where(reduced_labels[:,0] == 0)[0]
    label_1_indices = np.where(reduced_labels[:,0] == 1)[0]

    print(f"zeros: {len(label_0_indices)}\nones: {len(label_1_indices)}")
#######################################################################################################
# Ensure both classes are balanced
    # min_class_count = min(len(label_0_indices), len(label_1_indices))
    # if min_class_count == 0:
    #     raise ValueError("One of the classes has no samples. Balancing is not possible.")

    # # Adjust min_class_count to be a multiple of 50, but ensure it's not zero
    # min_class_count = max(50, (min_class_count // 50) * 50)

    # balanced_indices = np.concatenate([
    #     np.random.choice(label_0_indices, min_class_count, replace=False),
    #     np.random.choice(label_1_indices, min_class_count, replace=False)
    # ])

    # np.random.shuffle(balanced_indices)

  

    # # Recompute indices for each class to verify balance
    # label_0_indices = np.where(reduced_labels[:, 0] == 0)[0]
    # label_1_indices = np.where(reduced_labels[:, 0] == 1)[0]

    # print(f"zeros after balancing: {len(label_0_indices)}\nones after balancing: {len(label_1_indices)}")
################################################################################################
    num_samples = len(selected_indices)
    # Compute number of samples needed to fill last batch
    padding_needed = (batch_size - (num_samples % batch_size)) % batch_size

    if padding_needed > 0:
        pad_indices = np.random.choice(selected_indices, padding_needed, replace=False)
        balanced_indices = np.concatenate([selected_indices, pad_indices])

    # Shuffle after padding
    np.random.shuffle(balanced_indices)
    reduced_data = data_flat[balanced_indices]
    reduced_labels = labels_flat[balanced_indices]
    reduced_idxs = idxs_flat[balanced_indices]

    reduced_data = reduced_data.reshape(-1, batch_size, train_dataset['data'].shape[-1])
    reduced_labels = reduced_labels.reshape(-1, batch_size, train_dataset['labels'].shape[-1])
    reduced_idxs = reduced_idxs.reshape(-1, batch_size, train_dataset['idxs'].shape[-1])

    print(f"Reduced data shape: {reduced_data.shape}")
    print(f"Reduced labels shape: {reduced_labels.shape}")
    print(f"Reduced idxs shape: {reduced_idxs.shape}")

    congruency_dataset = {
        'data': torch.tensor(reduced_data).clone().detach(),
        'labels': torch.tensor(reduced_labels).clone().detach(),
        'idxs': torch.tensor(reduced_idxs).clone().detach(),
    }

    return congruency_dataset
