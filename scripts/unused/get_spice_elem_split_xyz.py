import os
import argparse
from pathlib import Path
import ase.io
from tqdm import tqdm


def group_xyz_by_dataset_name(xyz_path, output_dir, dataset_type):
    # Read all atoms from the original XYZ file
    atoms_list = ase.io.read(xyz_path, ":")  # Load all configurations
    categorized_samples = {"Iodine": []}

    # Group atoms by 'dataset_name' (assuming it's in the 'info' dictionary of each atom)
    for atoms in tqdm(atoms_list, desc="Grouping by dataset_name"):
        dataset_name = atoms.info.get("config_type", "unknown")
        if 53 in atoms.get_atomic_numbers():
            categorized_samples["Iodine"].append(atoms)

    # Write separate XYZ files for each group
    for dataset_name, group_atoms in categorized_samples.items():
        dataset_dir = os.path.join(output_dir, dataset_name, dataset_type)
        os.makedirs(dataset_dir, exist_ok=True)
        output_file = os.path.join(dataset_dir, f"data.xyz")

        # Write grouped atoms to a new XYZ file
        ase.io.write(output_file, group_atoms)
        print(f"Saved {len(group_atoms)} configurations to {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Group XYZ files by dataset_name and save into separate files.")
    # parser.add_argument("--xyz_path", type=str, required=True, help="Path to the input XYZ file.")
    # parser.add_argument("--output_dir", type=str, required=True, help="Directory where grouped XYZ files will be saved.")
    # args = parser.parse_args()
    # xyz_path = args.xyz_path
    # output_dir = args.output_dir
    xyz_path = "/data/ishan-amin/preprocessed_stuff/train_large_neut_no_bad_clean.xyz"
    # xyz_path = '/data/ishan-amin/preprocessed_stuff/test_large_neut_all.xyz'
    output_dir = "/data/ishan-amin/spice_seperated_xyz"
    # Run the grouping function
    dataset_type = "train"
    group_xyz_by_dataset_name(xyz_path, output_dir, dataset_type=dataset_type)
