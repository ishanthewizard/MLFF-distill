import os
import pickle
from fairchem.core.common.registry import registry
from mp_api.client import MPRester
from ase.io import read
from src.distill_datasets import SimpleDataset
from tqdm import tqdm
# with MPRester("PZ6Gx8dJTmeySErT5xuuaGhypYyg86p4") as mpr:
#     docs = mpr.summary.search(material_ids= ['mp-689577'])
#     mp_number = 'mp-1120767'
#     folder_path = '/data/shared/MPTrj/original'
#     file_name = f"{mp_number}.extxyz"  # Construct the file name
#     file_path = os.path.join(folder_path, file_name)
#     if os.path.exists(file_path):
#             all_atoms = []
#             for atoms in read(file_path, index=":"):
#                 all_atoms.append(atoms)
#             atom = all_atoms[0]
#     breakpoint()

# main_path = '/data/shared/MPTrj/lmdb/'
# train = os.path.join(main_path, 'train')
# val = os.path.join(main_path, 'val')
# # test = os.path.join(main_path, 'test')

# train_dataset = registry.get_dataset_class("lmdb")({"src": train})
# val_dataset = registry.get_dataset_class("lmdb")({"src": val})
# mace_train_dataset = registry.get_dataset_class("lmdb")({"src": '/data/ishan-amin/MPtraj/mace_mp_split/train'})
# # test_dataset = registry.get_dataset_class("lmdb")({"src": test})
# total =  len(train_dataset) + len(val_dataset)
# print("ERIC TOTAL:", len(train_dataset) + len(val_dataset))
# print("MACE TOTAL:", len(mace_train_dataset) )
# print("DIFFERENCE:",total - len(mace_train_dataset))


main_path = '/data/shared/MLFF/OC20/s2ef/2M/'
train = os.path.join(main_path, 'train')
# test = os.path.join(main_path, 'test')
pickle_file = 'labels/oc20_data_mapping.pkl'

with open(pickle_file, 'rb') as f:
    data_dict = pickle.load(f)

train_dataset = registry.get_dataset_class("lmdb")({"src": train})

# Initialize dictionaries for element counts in each class
element_dict = {1: 0, 7: 0, 8: 0, 12: 0}  # Element atomic numbers for hydrogen, nitrogen, oxygen, magnesium
class_dict = {
    0: element_dict.copy(),  # intermetallics
    1: element_dict.copy(),  # metalloids
    2: element_dict.copy(),  # non-metals
    3: element_dict.copy(),  # halides
}

# Iterate through the dataset and count occurrences of each element in the adsorbates based on their class
for sample in tqdm(train_dataset):
    key = "random" + str(sample.sid)
    elements = [1, 7, 8, 12]  # H, N, O, Mg

    for atomic_num in elements:
        # Check if the element is present in adsorbates (assuming sample.tags == 2 denotes adsorbates)
        if (sample.atomic_numbers[sample.tags == 2] == atomic_num).any():
            class_dict[data_dict[key]['class']][atomic_num] += 1

# Define group and element names for better output
groups = ['intermetallics', 'metalloids', 'non-metals', 'halides']
element_names = {1: 'hydrogen', 7: 'nitrogen', 8: 'oxygen', 12: 'magnesium'}

# Print the results nicely
for class_id, group_name in enumerate(groups):
    print(f"\nIn {group_name.capitalize()} category:")
    for atomic_num, element_count in class_dict[class_id].items():
        element_name = element_names.get(atomic_num, f"Element {atomic_num}")
        print(f"  {element_name.capitalize()}: {element_count}")


# breakpoint()

# force_jac_dataset = SimpleDataset(os.path.join(labels_folder,  'force_jacobians'))
# teacher_force_train_dataset = SimpleDataset(os.path.join(labels_folder,  'train_forces' ))

# print(len(force_jac_dataset))
# breakpoint()
# print(len(teacher_force_train_dataset))
# # teacher_force_val_dataset = SimpleDataset(os.path.join(labels_folder,  'val_forces'  ))




# with MPRester("PZ6Gx8dJTmeySErT5xuuaGhypYyg86p4") as mpr:
    
#     # Query for all materials containing halogens (F, Cl, Br, I, At)
#     halogen_elements = ["F", "Cl", "Br", "I", "At"]
    
#     halide_elements = ["F", "Cl", "Br", "I", "At"]
#     results = []
#     for elem in halide_elements:
#         results.extend(mpr.summary.search(elements=[elem], fields=["material_id"]))
#     print(len(results))
#     breakpoint
#     # # Print results
#     # for material in halides:
#     #     print(f"Material ID: {material['material_id']}, Formula: {material['pretty_formula']}, Elements: {material['elements']}, Spacegroup: {material['spacegroup.symbol']}")



    