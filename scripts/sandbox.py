import os
import pickle
from fairchem.core.common.registry import registry
# from mp_api.client import MPRester
# from ase.io import read
from src.distill_datasets import SimpleDataset
from tqdm import tqdm
import torch
import numpy as np
import lmdb
from torch.utils.data import Subset
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


# main_path = '/data/shared/MLFF/OC20/s2ef/2M/'
# train = os.path.join(main_path, 'train')
# # test = os.path.join(main_path, 'test')
# pickle_file = 'labels/oc20_data_mapping.pkl'

# with open(pickle_file, 'rb') as f:
#     data_dict = pickle.load(f)

# train_dataset = registry.get_dataset_class("lmdb")({"src": train})

# # Initialize dictionaries for element counts in each class
# element_dict = {1: 0, 7: 0, 8: 0, 12: 0}  # Element atomic numbers for hydrogen, nitrogen, oxygen, magnesium
# class_dict = {
#     0: element_dict.copy(),  # intermetallics
#     1: element_dict.copy(),  # metalloids
#     2: element_dict.copy(),  # non-metals
#     3: element_dict.copy(),  # halides
# }

# # Iterate through the dataset and count occurrences of each element in the adsorbates based on their class
# for sample in tqdm(train_dataset):
#     key = "random" + str(sample.sid)
#     elements = [1, 7, 8, 12]  # H, N, O, Mg

#     for atomic_num in elements:
#         # Check if the element is present in adsorbates (assuming sample.tags == 2 denotes adsorbates)
#         if (sample.atomic_numbers[sample.tags == 2] == atomic_num).any():
#             class_dict[data_dict[key]['class']][atomic_num] += 1

# # Define group and element names for better output
# groups = ['intermetallics', 'metalloids', 'non-metals', 'halides']
# element_names = {1: 'hydrogen', 7: 'nitrogen', 8: 'oxygen', 12: 'magnesium'}

# # Print the results nicely
# for class_id, group_name in enumerate(groups):
#     print(f"\nIn {group_name.capitalize()} category:")
#     for atomic_num, element_count in class_dict[class_id].items():
#         element_name = element_names.get(atomic_num, f"Element {atomic_num}")
#         print(f"  {element_name.capitalize()}: {element_count}")


# breakpoint()

# main_path = '/data/ishan-amin/MPtraj/mptraj_seperated_noIons/Perovskites/'
# main_path = '/data/ishan-amin/SPICE/spice_separated/Iodine'
# main_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Bandgap_greater_than_5'
# main_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Perovskites_noIons'
# main_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Yttrium'
main_path = '/data/shared/MPTrj/lmdb/'
train = os.path.join(main_path, 'train')
# val = os.path.join(main_path, 'val')
# test = os.path.join(main_path, 'test')

train_dataset = registry.get_dataset_class("lmdb")({"src": train})
breakpoint()
# val_dataset = registry.get_dataset_class("lmdb")({"src": val})
# test_dataset = registry.get_dataset_class("lmdb")({"src": test})
# total =  len(train_dataset) + len(val_dataset) + len(test_dataset)
# print(len(train_dataset))
# print(len(val_dataset))
# print(len(test_dataset))
# print(len(test_dataset) / total)
# print(val_dataset[0])
# print(test_dataset[0])
for sample in train_dataset:
    print(sample.force)
    breakpoint()
# force_jac_dataset = SimpleDataset(os.path.join(labels_folder,  'force_jacobians'))
# teacher_force_train_dataset = SimpleDataset(os.path.join(labels_folder,  'train_forces' ))

# print(len(force_jac_dataset))
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
#     # # Print results
#     # for material in halides:
#     #     print(f"Material ID: {material['material_id']}, Formula: {material['pretty_formula']}, Elements: {material['elements']}, Spacegroup: {material['spacegroup.symbol']}")




# Paths
# main_path = '/pscratch/sd/i/ishan_a/OC20_seperated/Nonmetals_N_in_adsorbate'
main_path = '/pscratch/sd/i/ishan_a/OC20_seperated/Halides'
# val = os.path.join(main_path, 'val')
# # test = os.path.join(main_path, 'test')

train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(main_path, 'train')})
val_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(main_path, 'val')})
normal_label_path = '/pscratch/sd/i/ishan_a/OC20_labels/Nonmetals_N_in_adsorbate_gemnet_oc_large_s2ef_all_md_final'
# normal_label_path = '/pscratch/sd/i/ishan_a/OC20_labels/Nonmetals_N_in_adsorbate_gemnet_oc_large_s2ef_all_md'
# normal_label_path = '/pscratch/sd/i/ishan_a/OC20_labels/nonmetals_oc_fjac_combined'
# augmented_label_path = '/pscratch/sd/i/ishan_a/OC20_labels/Nonmetals_N_in_adsorbate_gemnet_oc_large_s2ef_all_md_LAST_HESSIANS'

# normal_label_path = '/pscratch/sd/i/ishan_a/OC20_labels/Nonmetals_N_in_adsorbate_gemnet_oc_large_s2ef_all_md_FINAL_FRRRRR'

# normal_label_path = 'pscratch/sd/i/ishan_a/labels/Halides_eq2_31M_ec4_allmd_LAST_OF_THE_MOHESSIANS'

force_jac_dataset = SimpleDataset(os.path.join(normal_label_path, 'force_jacobians'))
# augemented_jac_dataset = SimpleDataset(os.path.join(augmented_label_path, 'force_jacobians'))

teacher_val_forces = SimpleDataset(os.path.join(normal_label_path, 'val_forces'))
teacher_train_forces = SimpleDataset(os.path.join(normal_label_path, 'train_forces'))

print(len(train_dataset))
# print(len(force_jac_dataset))
# print(len(teacher_train_forces))
print(" ")
# print(len(teacher_val_forces))
print(len(val_dataset))
breakpoint()
# force_mae = 0
# indx = np.random.default_rng(seed=123).choice(
#                     len(train_dataset), 
#                     10000, 
#                     replace=False
#                 )
# train_dataset = Subset(train_dataset, torch.tensor(indx))
# teacher_train_forces = Subset(teacher_train_forces, torch.tensor(indx))
# for i, sample in tqdm(enumerate(train_dataset)):
#     teach_forces = teacher_train_forces[i]
#     true_forces = sample.force / 2.887317180633545 
#     force_mae += torch.abs(teach_forces.reshape(true_forces.shape[0], 3) - true_forces).mean().item()
# force_mae /= len(teacher_train_forces)    
# print("TEACHER FORCE MAE:", force_mae)

# force_mae = 0
# for i, sample in tqdm(enumerate(val_dataset)):
#     teach_forces = teacher_val_forces[i]
#     true_forces = sample.force / 2.887317180633545 
#     force_mae += torch.abs(teach_forces.reshape(true_forces.shape[0], 3) - true_forces).mean().item()
# force_mae /= len(teacher_train_forces)    
# print("TEACHER FORCE MAE:", force_mae)

for i, sample in tqdm(enumerate(train_dataset)):
    force_jac = force_jac_dataset[i]
    num_free_atoms = sum(sample.fixed == 0).item()
    # # if we get an exception here: save the idx to an np array. after the array is done, save the array to an np file Nonmetals_missing called
    if len(sample.pos) * num_free_atoms * 9 != len(force_jac):
        print("UH OH!!")
        
breakpoint()
# Array to store the indexes where exceptions occur
missing_indexes = []

# Iterate over the force_jac_dataset
# force_jac_dataset.merge_lmdb_files('/pscratch/sd/i/ishan_a/OC20_labels/nonmetals_oc_fjac_combined')
samples = []

with augemented_jac_dataset.envs[0].begin() as txn:
    cursor = txn.cursor()
    for key, _ in cursor:
        # Decode the key from bytes to string and print it
        readable_key = key.decode('utf-8')  # Adjust encoding if needed
        print(readable_key)


new_db_path = '/pscratch/sd/i/ishan_a/OC20_labels/nonmetals_oc_merged_last_hessians'
augemented_jac_dataset.merge_lmdb_files(new_db_path)
augemented_jac_dataset2 = SimpleDataset(new_db_path)
for i in tqdm(range(len(train_dataset))):
    try:
        sample = force_jac_dataset[i]
        # sample = train_dataset[i]
        # num_free_atoms = sum(sample.fixed == 0).item()
        # # if we get an exception here: save the idx to an np array. after the array is done, save the array to an np file Nonmetals_missing called
        # if len(sample.pos) * num_free_atoms * 9 != len(force_jac):
        #     print("UH OH!!")

    except Exception as e:
        # If an exception occurs, append the index to missing_indexes
        sample = augemented_jac_dataset2[i]
        missing_indexes.append(i)
    samples.append(sample)
    
db_path = '/pscratch/sd/i/ishan_a/OC20_labels/Nonmetals_N_in_adsorbate_gemnet_oc_large_s2ef_all_md_ALL_HESSIANS'

env = lmdb.open(os.path.join(db_path, 'data.lmdb'), map_size=int(1099511627776 * 2)) 
with env.begin(write=True) as txn:
    for i, sample in tqdm(enumerate(samples)):
        sample_id = str(i)
        sample_output = sample.numpy()  # this function needs to output an array where each element correponds to the label for an entire molecule
            # Convert tensor to bytes and write to LMDB
        txn.put(sample_id.encode(), sample_output.tobytes())

# Close the new LMDB environment
env.close()
print("SAVED:", len(samples))
print(len(missing_indexes))
# breakpoint()
# # After the loop, save the missing indexes array to a np file
# np.save('Nonmetals_missing.npy', np.array(missing_indexes))

# for i, true_force_jac in tqdm(enumerate(force_jac_dataset)):
#     sample = train_dataset[i]
#     # if we get an exception here: save the idx to an np array. after the array is done, save the array to an np file Nonmetals_missing called
#     num_free_atoms = sum(sample.fixed == 0).item()
#     assert len(sample.pos) * num_free_atoms * 9 == len(true_force_jac)

# loss = 0






    