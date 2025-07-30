import os
import lmdb
from tqdm import tqdm
from src_v2.distill_datasets import LmdbDataset, LmdbHessianIndexDataset
import numpy as np

def record_and_save(dataset, file_path, fn, is_idxs=False):
    # Assuming train_loader is your DataLoader
    map_size = 1099511627776 * 2

    env = lmdb.open(file_path, map_size=map_size)
    env_info = env.info()
    
    with env.begin(write=True) as txn:
        i = 0
        print(len(dataset))
        # if is_idxs:
        #     breakpoint()
        for j, sample in tqdm(enumerate(dataset)):
            # if np.sqrt(len(sample) / 9 ) < 3 and fjacs:
            #     continue
            # if len(sample) / 3 < 3:
            #     continue
            # if torch.isnan(sample).any():
            #     raise Exception("yup, has nans")
            sample_id = str(i) if not is_idxs else str(i) + "_idxs"

            sample_output = fn(sample) # this function needs to output an array where each element correponds to the label for an entire molecule
            # Convert tensor to bytes and write to LMDB
            txn.put(sample_id.encode(), sample_output.tobytes())
            i += 1
    env.close()
    print('SAVED:', i)
    print(f"All tensors saved to LMDB:{file_path}")

def convert_labels(labels_folder, new_labels_folder, model="medium"):
    os.makedirs(os.path.join(new_labels_folder, 'force_jacobians'))
    os.makedirs(os.path.join(new_labels_folder, 'train_forces'))
    os.makedirs(os.path.join(new_labels_folder, 'val_forces'))
    force_jac_dataset = LmdbDataset(os.path.join(labels_folder,  'force_jacobians'), div_2=True)
    teacher_force_train_dataset = LmdbDataset(os.path.join(labels_folder,  'train_forces' ))
    teacher_force_val_dataset = LmdbDataset(os.path.join(labels_folder,  'val_forces'  ))
    force_jac_idxs_dataset = LmdbHessianIndexDataset(os.path.join(labels_folder,  'force_jacobians'), dtype=np.float32, div_2=True)
    identity_idx = lambda x: x.long().numpy()
    flip = lambda x: -x.numpy()
    multiply_std = lambda x: (1.423  * x).numpy()
    # record_and_save(teacher_force_train_dataset, os.path.join(new_labels_folder, 'train_forces', 'train_forces.lmdb'), multiply_std)
    record_and_save(force_jac_idxs_dataset, os.path.join(new_labels_folder, 'force_jacobians', 'force_jacobians.lmdb'), identity_idx, is_idxs=True)
    
    record_and_save(force_jac_dataset, os.path.join(new_labels_folder, 'force_jacobians', 'force_jacobians.lmdb'), multiply_std)
    record_and_save(teacher_force_train_dataset, os.path.join(new_labels_folder, 'train_forces', 'train_forces.lmdb'), multiply_std)
    record_and_save(teacher_force_val_dataset, os.path.join(new_labels_folder, 'val_forces', 'val_forces.lmdb'), multiply_std)
    
    

if __name__ == "__main__":
    # labels_folder = 'labels/mace_off_large_Perovskites'
    labels_folder = '/data/ishan-amin/OMOL/electrolytes_application/labels/NAPF6/s1p1_NORMED_OLD'
    # labels_folder = '/data/ericqu/EScAIP_labels/iodine'
    new_labels_folder = '/data/ishan-amin/OMOL/electrolytes_application/labels/NAPF6/s1p1'

    convert_labels(labels_folder, new_labels_folder)
