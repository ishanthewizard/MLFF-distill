from fairchem.core.common.registry import registry
from fairchem.src.fairchem.core.common.data_parallel import BalancedBatchSampler, OCPCollater
from torch.utils.data import DataLoader
from tqdm import tqdm
# dataset_path = '/data/shared/MLFF/MD22/95_lmdb/Ac-Ala3-NHMe/train/'
dataset_path = '/data/ishan-amin/post_data/md17/ethanol/50k/train'
print(registry)
config = {"src": dataset_path}
dataset = registry.get_dataset_class("lmdb")(config)

# Initialize sums and counts
y_sum = 0.0
forces_sum = 0.0
y_count = 0
forces_count = 0

# Iterate over all batches
for sample in tqdm(dataset):
    y_sum += sample.y.sum().item()
    forces_sum += sample['force'].sum().item()
    y_count += sample.y.numel()
    forces_count += sample['force'].numel()

# Calculate means
y_mean = y_sum / y_count
forces_mean = forces_sum / forces_count

print(f"Mean of dataset.y: {y_mean}")
print(f"Mean of dataset['forces']: {forces_mean}")
