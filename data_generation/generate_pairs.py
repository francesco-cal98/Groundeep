import numpy as np
import pickle
from numpy.random import default_rng

# Initialize RNG
rng = default_rng()

# Load dataset
data = np.load('/home/student/Desktop/Groundeep/circle_dataset_100x100/circle_dataset_100x100_v2.npz')
D = data['D']  # Image data, shape: (num_samples, 10000)
N_list = data['N_list'].flatten()  # e.g., [7, 9, 14, ...]

# Pair definitions: (smaller N, larger N)
small_list = np.array([7, 7, 8, 9, 9, 10, 11, 14, 7, 8, 9, 10, 11, 12, 14, 16, 7, 8, 9, 10, 12, 14, 16, 8, 9, 10, 11, 16, 18, 20, 22, 7, 10, 12, 14, 16, 22, 25])
large_list = np.array([14, 12, 16, 18, 16, 20, 22, 28, 11, 14, 14, 16, 16, 18, 20, 25, 10, 11, 12, 14, 16, 18, 22, 10, 11, 12, 14, 20, 22, 25, 28, 8, 11, 14, 16, 18, 25, 28])

n_repetitions = 200
disp_size = int(np.sqrt(D.shape[1]))  # e.g., 100

# Step 1: Generate all pairs of indices (from original dataset)
all_idxs_N1 = []
all_idxs_N2 = []

for N1, N2 in zip(small_list, large_list):
    idxs_smaller = np.where(N_list == N1)[0]
    idxs_larger = np.where(N_list == N2)[0]

    if len(idxs_smaller) < n_repetitions or len(idxs_larger) < n_repetitions:
        raise ValueError(f"Not enough samples for N1={N1} or N2={N2}")

    idxs_N1 = rng.choice(idxs_smaller, n_repetitions, replace=False)
    idxs_N2 = rng.choice(idxs_larger, n_repetitions, replace=False)

    all_idxs_N1.append(idxs_N1)
    all_idxs_N2.append(idxs_N2)

all_idxs_N1 = np.concatenate(all_idxs_N1)
all_idxs_N2 = np.concatenate(all_idxs_N2)

# Create all pairs (left-small/right-large)
all_pairs = np.vstack([all_idxs_N1, all_idxs_N2]).T  # shape (num_pairs, 2)

# Step 2: Shuffle and split the dataset indices (train/test)
all_data_indices = np.arange(len(N_list))
perm = rng.permutation(all_data_indices)
split_ratio = 0.75
split_idx = int(len(all_data_indices) * split_ratio)
train_indices = np.sort(perm[:split_idx])  # sorted for indexing ease
test_indices = np.sort(perm[split_idx:])

# Step 3: Filter pairs so both indices belong to train or test
train_set = set(train_indices)
test_set = set(test_indices)

train_pairs = [pair for pair in all_pairs if pair[0] in train_set and pair[1] in train_set]
test_pairs = [pair for pair in all_pairs if pair[0] in test_set and pair[1] in test_set]

train_pairs = np.array(train_pairs)
test_pairs = np.array(test_pairs)

# Step 4: Remap pair indices to their positions in train/test subsets
train_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(train_indices)}
test_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(test_indices)}

train_pairs_remapped = np.array([[train_map[pair[0]], train_map[pair[1]]] for pair in train_pairs])
test_pairs_remapped = np.array([[test_map[pair[0]], test_map[pair[1]]] for pair in test_pairs])

# Step 5: Extract data subsets
train_data = D[train_indices]
test_data = D[test_indices]
train_N_list = N_list[train_indices]
test_N_list = N_list[test_indices]

# Step 6: Prepare inputs and labels for train pairs
IMGs_train_left = train_data[train_pairs_remapped[:, 0]]
IMGs_train_right = train_data[train_pairs_remapped[:, 1]]
input_train_small_left = np.hstack([IMGs_train_left, IMGs_train_right])
labels_train_small_left = np.tile([0, 1], (input_train_small_left.shape[0], 1))

input_train_large_left = np.hstack([IMGs_train_right, IMGs_train_left])
labels_train_large_left = np.tile([1, 0], (input_train_large_left.shape[0], 1))

train_inputs = np.vstack([input_train_small_left, input_train_large_left])
train_labels = np.vstack([labels_train_small_left, labels_train_large_left])
train_pair_indices = np.vstack([train_pairs_remapped, train_pairs_remapped[:, ::-1]])

# Step 7: Prepare inputs and labels for test pairs
IMGs_test_left = test_data[test_pairs_remapped[:, 0]]
IMGs_test_right = test_data[test_pairs_remapped[:, 1]]
input_test_small_left = np.hstack([IMGs_test_left, IMGs_test_right])
labels_test_small_left = np.tile([0, 1], (input_test_small_left.shape[0], 1))

input_test_large_left = np.hstack([IMGs_test_right, IMGs_test_left])
labels_test_large_left = np.tile([1, 0], (input_test_large_left.shape[0], 1))

test_inputs = np.vstack([input_test_small_left, input_test_large_left])
test_labels = np.vstack([labels_test_small_left, labels_test_large_left])
test_pair_indices = np.vstack([test_pairs_remapped, test_pairs_remapped[:, ::-1]])

# Step 8: Save to pkl files (flat versions)
with open('/home/student/Desktop/Groundeep/pairs_from_mat_train_v2.pkl', 'wb') as f:
    pickle.dump({'data': train_inputs, 'labels': train_labels, 'idxs': train_pair_indices}, f)

with open('/home/student/Desktop/Groundeep/pairs_from_mat_test_v2.pkl', 'wb') as f:
    pickle.dump({'data': test_inputs, 'labels': test_labels, 'idxs': test_pair_indices}, f)

# Step 9: Optional batching function
def make_batches(x, batch_size):
    n = x.shape[0]
    num_batches = n // batch_size
    x = x[:num_batches * batch_size]
    return x.reshape(num_batches, batch_size, -1)

batch_size = 100

train_batched = make_batches(train_inputs, batch_size)
test_batched = make_batches(test_inputs, batch_size)
train_labels_batched = make_batches(train_labels, batch_size)
test_labels_batched = make_batches(test_labels, batch_size)
train_idxs_batched = make_batches(train_pair_indices, batch_size)
test_idxs_batched = make_batches(test_pair_indices, batch_size)

# Step 10: Save batched versions
with open('/home/student/Desktop/Groundeep/pairs_from_mat_train_v2_batched.pkl', 'wb') as f:
    pickle.dump({
        'data': train_batched,
        'labels': train_labels_batched,
        'idxs': train_idxs_batched
    }, f)

with open('/home/student/Desktop/Groundeep/pairs_from_mat_test_v2_batched.pkl', 'wb') as f:
    pickle.dump({
        'data': test_batched,
        'labels': test_labels_batched,
        'idxs': test_idxs_batched
    }, f)
