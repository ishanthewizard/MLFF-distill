import os
from mp_api.client import MPRester
from ase.io import read
with MPRester("PZ6Gx8dJTmeySErT5xuuaGhypYyg86p4") as mpr:
    docs = mpr.summary.search(material_ids= ['mp-689577'])
    mp_number = 'mp-1120767'
    folder_path = '/data/shared/MPTrj/original'
    file_name = f"{mp_number}.extxyz"  # Construct the file name
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
            all_atoms = []
            for atoms in read(file_path, index=":"):
                all_atoms.append(atoms)
            atom = all_atoms[0]
    breakpoint()


    