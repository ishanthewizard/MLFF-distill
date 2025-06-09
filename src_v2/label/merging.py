import sys
sys.path.append("/home/yuejian/project/MLFF-distill")
from src_v2.distill_datasets import merge_lmdb_shards, merge_indices_pt_shards
import os
import argparse




if __name__ == "__main__":

    # make subset_dir into a argument through argparse with a default value of "/data/ericqu/OMol_subset/protein_core_train_4M"
    argparser = argparse.ArgumentParser(description="Merge LMDB shards in a directory.")
    argparser.add_argument(
        "--subset_root",
        type=str,
        required=True,
        default="/data/ericqu/OMol_subset/test",
        help="Directory containing the LMDB shards to merge.",
    )
    args = argparser.parse_args()


    for subset in os.listdir(args.subset_root):
        subset_dir = os.path.join(args.subset_root, subset)
        raw_labels_dir = os.path.join(subset_dir, "labels")
        merged_labels_dir = os.path.join(subset_dir, "merged_labels")
        if not os.path.exists(merged_labels_dir):
            os.makedirs(merged_labels_dir)
        for subdir in os.listdir(raw_labels_dir):
            os.makedirs(os.path.join(merged_labels_dir, subdir), exist_ok=True)
            
            
        # merge all sharded lmdb files in each subdir
        for subdir in os.listdir(raw_labels_dir):
            # print(f"Checking {subdir}...")
            if subdir == "indices":
                continue
            # print(f"Getting path: {subdir}")
            shard_dir = os.path.join(raw_labels_dir, subdir)
            output_file = os.path.join(merged_labels_dir,subdir,"data.0000.lmdb")
            
            # print(f"looking at:{shard_dir}")
            # if no files end with .lmdb, skip
            if not any(file.endswith('.lmdb') for file in os.listdir(shard_dir)):
                print(f"No LMDB files found in {shard_dir}, skipping...")
                continue
            
            print(f"Merging {shard_dir} into {output_file}...")
            try:
                merge_lmdb_shards(
                    input_dir=shard_dir,
                    output_path=output_file
                )
            except:
                print(f"Failed to merge {shard_dir}, skipping...")
                continue

        # merge all shards in indices subdir
        indices_dir = os.path.join(raw_labels_dir, "indices")
        output_path = os.path.join(merged_labels_dir, "indices")
        print(f"Merging indices from {indices_dir}")
        merge_indices_pt_shards(indices_dir, output_path=output_path)