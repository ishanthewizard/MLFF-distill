import os
import time
import torch

set_num = 3
parent_path = '/data/christine/esen_final_node_embedding'
paths = [
    '/data/ishan-amin/OMOL/TOY/ligand_pocket_300/train', # plp300 train
    '/data/ishan-amin/OMOL/TOY/ligand_pocket_300/val', # plp300 val
    '/data/ishan-amin/OMOL/4M/subsets/OMol_subset/protein_ligand_pockets_train_4M', # plp 20k train
    '/data/ishan-amin/OMOL/4M/subsets/OMol_subset/protein_ligand_pockets_val' # plp 20k val
]

info_paths = [
    'plp300_train_unshuffled',
    'plp300_eval_unshuffled',
    'plp20k_train_unshuffled',
    'plp20k_eval_unshuffled'
]

path = paths[set_num]
info_path = info_paths[set_num]

print(f'path: {path}')
print(f'info_path: {info_path}')
print(f'starting at {time.strftime("%Y-%m-%d %H:%M:%S")}')

time_start = time.time()
whole_path = f'{parent_path}/{info_path}'
num_batches = len(os.listdir(whole_path)) // 2  # since this contains x_message and batch
x_all = []
for b in range(num_batches):
    x_message = torch.load(f"{whole_path}/x_message_{b}.pt", map_location="cpu")
    batch = torch.load(f"{whole_path}/batch_{b}.pt", map_location="cpu")
    x_parts = [x_message[batch == i] for i in range(max(batch) + 1)]
    x_all.extend(x_parts)

print(f'num structures: {len(x_all)}')
print(f"Time taken to load x_all: {time.time() - time_start} seconds")

# this is the maximum number of atoms in a structure in the dataset divided by 10
# i would do 10% of the atoms for every structure but then the resulting tensor is not the same size for every structure
# which causes a problem when i try to save the tensor at the end
max_num_points = 35

time_start = time.time()
diverse_idxs = []
for x in x_all:
    # Step 2: Compute pairwise L2 distance matrix [N, N]
    # (Optional: use other distances like cosine if preferred)
    diffs = x[:, None, :] - x[None, :, :]  # shape (N, N, 128)
    dists = torch.norm(diffs, dim=2)       # shape (N, N)
    num_points = x.shape[0] // 10

    # Step 3: Greedy Max-Min Diversity selection
    selected = [torch.randint(0, len(x), (1,)).item()]  # Start from a random point
    remaining = set(range(len(x))) - set(selected)

    for _ in range(num_points - 1):
        # For each remaining candidate, get the distance to the closest already selected point
        candidate_to_min_dist = [
            (i, torch.min(dists[i, selected]).item()) for i in remaining
        ]
        # Pick the candidate with the **maximum** of these minimum distances
        next_idx = max(candidate_to_min_dist, key=lambda t: t[1])[0]
        selected.append(next_idx)
        remaining.remove(next_idx)
    
    # pad with -1s to make the tensor the same size for every structure
    selected.extend([-1] * (max_num_points - num_points))

    diverse_idxs.append(selected)

print(f'num diverse idxs: {len(diverse_idxs)}')
print(f"Time taken to get diverse idxs: {(time.time() - time_start) / 60} minutes")

div_idxs_tensor = torch.tensor(diverse_idxs)
torch.save(div_idxs_tensor, f"{parent_path}/diverse_idxs_{info_path.rsplit('_', 1)[0]}.pt")

print('done')
