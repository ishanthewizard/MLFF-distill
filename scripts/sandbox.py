import os
from fairchem.core.common.registry import registry
from mp_api.client import MPRester
from ase.io import read
from src.distill_datasets import SimpleDataset
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




# main_path = '/data/ishan-amin/MPtraj/mptraj_seperated_noIons/Perovskites/'
# main_path = '/data/ishan-amin/SPICE/spice_separated/Iodine'
# main_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Bandgap_greater_than_5'
# main_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Perovskites_noIons'
main_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Yttrium'
train = os.path.join(main_path, 'train')
val = os.path.join(main_path, 'val')
test = os.path.join(main_path, 'test')

train_dataset = registry.get_dataset_class("lmdb")({"src": train})
val_dataset = registry.get_dataset_class("lmdb")({"src": val})
test_dataset = registry.get_dataset_class("lmdb")({"src": test})
total =  len(train_dataset) + len(val_dataset) + len(test_dataset)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))
print(len(test_dataset) / total)
print(val_dataset[0])
print(test_dataset[0])
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



    