import sys
sys.path.append("/home/yuejian/project/MLFF-distill")
from src_v2.dataset_utils import merge_lmdb_shards, merge_indices_pt_shards
import os
import argparse

if __name__ == "__main__":

    # make subset_dir into a argument through argparse with a default value of "/data/ericqu/OMol_subset/protein_core_train_4M"
    argparser = argparse.ArgumentParser(description="Merge LMDB shards in a directory.")
    argparser.add_argument(
        "--labels_dir",
        type=str,
        default="/data/ishan-amin/OMOL/TOY/labels/ligand_pocket_300/",
        help="Directory containing the LMDB shards to merge.",
    )
    args = argparser.parse_args()

    raw_labels_dir = os.path.join(args.labels_dir, 'RAW')
    merged_labels_dir = os.path.join(args.labels_dir, 'JOINED')
    if not os.path.exists(merged_labels_dir):
        os.makedirs(merged_labels_dir)
    for labels_type in os.listdir(raw_labels_dir):
        if labels_type == 'indices':
            continue
        curr_labels_folder = os.path.join(raw_labels_dir, labels_type)
        os.makedirs(os.path.join(merged_labels_dir, labels_type))
        output_file = os.path.join(merged_labels_dir, labels_type,"data.0000.lmdb")
        if not any(file.endswith('.lmdb') for file in os.listdir(curr_labels_folder)):
            print(f"No LMDB files found in {curr_labels_folder}, skipping...")
            continue
        
        print(f"Merging {curr_labels_folder} into {output_file}...")
        try:
            merge_lmdb_shards(
                input_dir=curr_labels_folder,
                output_path=output_file
            )
        except:
            print(f"Failed to merge {curr_labels_folder}, skipping...")
            continue
    

    # merge all shards in indices subdir
    indices_dir = os.path.join(raw_labels_dir, "indices")
    indices_output_path = os.path.join(merged_labels_dir, "indices")
    os.makedirs(indices_output_path, exist_ok=True)
    print(f"Merging indices from {indices_dir}")
    merge_indices_pt_shards(indices_dir, output_path=indices_output_path)