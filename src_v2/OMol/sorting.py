import torch
import sys
sys.path.append("/home/yuejian/project/MLFF-distill")
# sys.path.append("../..")
from src_v2.distill_datasets import LmdbDataset
from fairchem.core.datasets import AseDBDataset
from torch.utils.data import Subset
from src_v2.dataset_utils import get_inverse_indices
import lmdb,os
from tqdm import tqdm
import argparse



if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Sort and save hessian results.")
    argparser.add_argument(
        "--subset_root",
        type=str,
        required=True,
        default="/home/yuejian/project/MLFF-distill/OMol_Whole/toy/label_collate",
        help="Directory containing the subset root.",
    )
    args = argparser.parse_args()

    
    subset_root = args.subset_root
    for dir in os.listdir(subset_root):
        subset_dir = os.path.join(subset_root, dir)
        
        merged_label_path = os.path.join(subset_dir, "merged_labels")
        final_label_path = os.path.join(subset_dir, "final_labels")
        if not os.path.exists(final_label_path):
            os.makedirs(final_label_path)

        # label folders
        hessian_folder = os.path.join(merged_label_path, "force_jacobians")
        train_force_folder = os.path.join(merged_label_path, "train_forces")
        val_force_folder = os.path.join(merged_label_path, "val_forces")

        # indices
        indices_folder = os.path.join(merged_label_path, "indices")


        # load lmdb dataset
        hessian_dataset = LmdbDataset(
            folder_path= hessian_folder,
        )
        # The code snippet you provided is creating a dataset object named `hessian_dataset` using the `SimpleDataset` class. It is loading data from the folder path "/data/yuejian/OMol_subset/toy_train/labels/force_jacobians".
        train_force_dataset = LmdbDataset(
            folder_path= train_force_folder,
        )

        val_force_dataset = LmdbDataset(
            folder_path= val_force_folder,
        )

        train_indices = torch.load(os.path.join(indices_folder, "train_indices.pt"))
        val_indices = torch.load(os.path.join(indices_folder, "val_indices.pt"))

        inverse_train_indices = get_inverse_indices(train_indices)
        inverse_val_indices = get_inverse_indices(val_indices)

        sorted_hessian = Subset(
            dataset=hessian_dataset,
            indices=inverse_train_indices
        )
        sorted_train_force = Subset(
            dataset=train_force_dataset,
            indices=inverse_train_indices
        )

        sorted_val_force = Subset(
            dataset=val_force_dataset,
            indices=inverse_val_indices
        )

        # dump to new lmdb dataset
        file_path = os.path.join(final_label_path, "force_jacobians", "data0000.lmdb")
        os.makedirs(file_path, exist_ok=True)

        map_size= 1099511627776 * 2
        env = lmdb.open(file_path, map_size=map_size)


        with env.begin(write=True) as txn:
            for idx, data in enumerate(sorted_hessian):
                txn.put(str(int(idx)).encode(), data.numpy().tobytes())
                
        env.close()

        # dump to new lmdb dataset
        file_path = os.path.join(final_label_path, "train_forces", "data0000.lmdb")
        os.makedirs(file_path, exist_ok=True)
        map_size= 1099511627776 * 2
        env = lmdb.open(file_path, map_size=map_size)


        with env.begin(write=True) as txn:
            for idx, data in enumerate(sorted_train_force):
                txn.put(str(int(idx)).encode(), data.numpy().tobytes())
                
        env.close()

        # dump to new lmdb dataset
        file_path = os.path.join(final_label_path, "val_forces", "data0000.lmdb")
        os.makedirs(file_path, exist_ok=True)
        map_size= 1099511627776 * 2
        env = lmdb.open(file_path, map_size=map_size)


        with env.begin(write=True) as txn:
            for idx, data in enumerate(sorted_val_force):
                txn.put(str(int(idx)).encode(), data.numpy().tobytes())
                
        env.close()