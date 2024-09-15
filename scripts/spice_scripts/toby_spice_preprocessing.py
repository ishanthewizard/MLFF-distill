import os
import argparse
from pathlib import Path
import pdb
import pickle
import lmdb
import numpy as np
from tqdm import tqdm
from urllib import request as request
from sklearn.model_selection import train_test_split
from arrays_to_graphs import AtomsToGraphs
#from mdsim.common.utils import EV_TO_KCAL_MOL
import torch
import ase.io
import pickle
an_2_atomic_reference_energy = {35: -70045.28385080204, 6: -1030.5671648271828, 17: -12522.649269035726, 9: -2715.318528602957, 1: -13.571964772646918, 53: -8102.524593409054, 7: -1486.3750255780376, 8: -2043.933693071156, 15: -9287.407133426237, 16: -10834.4844708122}
def write_to_lmdb(xyz_path, db_path, split_name='train'):
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=False,
        device='cpu'
    )
    print(f'Loading xyz file {xyz_path}....')
    atoms = ase.io.read(xyz_path, ':')
    #import ipdb; ipdb.set_trace()
    R = [a.positions for a in atoms]
    forces = [a.get_forces() for a in atoms]
    energies = [a.get_potential_energy() for a in atoms]
    #forces = [a.arrays['MACE_forces'] for a in atoms]
    #energies = [a.info['MACE_energy'] for a in atoms]
    z = [a.get_atomic_numbers() for a in atoms]
    smiles = [a.info.get('smiles', '') for a in atoms]
    origin_datset_name = [a.info.get('config_type', '') for a in atoms]
    atomic_reference_energies = [sum([an_2_atomic_reference_energy[an] for an in an_list]) for an_list in z]
    atomic_reference_energies = np.array(atomic_reference_energies)
    reference_energies = np.array(energies) - atomic_reference_energies
    n_points = len(R)
    atomic_numbers = z
    #atomic_numbers = atomic_numbers.astype(np.int64) 
    positions = R
    
    force = forces
    energy = np.array(energies)
    energy = energy.reshape(-1, 1)  # Reshape energy into 2D array ADDED TODO
    
    lengths = np.ones(3)[None, :] * 30.
    angles = np.ones(3)[None, :] * 90.
    
    norm_stats = {
        'e_mean': energy.mean(),
        'e_std': energy.std(),
        'f_mean': np.concatenate([f.flatten() for f in force]).mean(),
        'f_std': np.concatenate([f.flatten() for f in force]).std(),
        'r_e_mean': reference_energies.mean(),
        'r_e_std': reference_energies.std(),
    }
    save_path = Path(db_path)
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / 'metadata', norm_stats)
        
    print(f'processing split {split_name}.')
    save_path = Path(db_path) / split_name
    save_path.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(
        str(save_path / 'data.lmdb'),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    for idx in tqdm(range(len(positions))):
        #natoms = np.array([positions.shape[1]] * 1, dtype=np.int64)
        natoms = np.array([len(positions[idx])] * 1, dtype=np.int64)
        data = a2g.convert(natoms, positions[idx], atomic_numbers[idx], 
                        lengths, angles, energy[idx], force[idx])
        data.sid = 0
        data.fid = idx
        data.smiles = smiles[idx]
        data.dataset_name = origin_datset_name[idx]
        data.reference_energy = torch.tensor(reference_energies[idx]).unsqueeze(0)
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()
    db.sync()
    db.close()
write_to_lmdb(db_path='/data/shared/spice/maceoff_split', xyz_path='/data/shared/spice/train_large_neut_no_bad_clean.xyz', split_name='train')
write_to_lmdb(db_path='/data/shared/spice/maceoff_split', xyz_path='/data/shared/spice/test_large_neut_all.xyz', split_name='test')