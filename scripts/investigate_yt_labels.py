# Paths
import os
from ffairchem.core.common.registry import registry
from src.distill_datasets import SimpleDataset


dataset_path = '/data/ishan-amin/MPtraj/mptraj_seperated/Yttrium/val'
labels_path = output_label_path = 'labels/mace_mp_all_splits_Yttrium'




# dataset_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Yttrium/'
# labels_folder = 'labels/mace_mp_large_Yttrium'
# output_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Yttrium/'
# output_label_path = 'labels/mace_mp_all_splits_Yttrium'

val_dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
teacher_force_val_dataset = SimpleDataset(os.path.join(labels_path, 'val_forces'))
# Load datasets




train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(main_path, 'train')})
force_jac_dataset = SimpleDataset(os.path.join(labels_folder, 'force_jacobians'))
teacher_force_train_dataset = SimpleDataset(os.path.join(labels_folder, 'train_forces'))

# Ensure dataset lengths match