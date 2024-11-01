"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# pylint: disable=stop-iteration-return

import h5py
import numpy as np
from ase import Atoms
import math
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}


def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def generator(formula, rxn, grp, trajectory_subset = None, frac_neb_iterations = 1):
    """Iterates through a h5 group"""

    
    fid = 0
    positions = grp["positions"]
    if trajectory_subset is not None:
        start, end = trajectory_subset[0], trajectory_subset[1] + 1
    else:
        start, end = 0, 8

    all_neb_iterations = int((positions.shape[0] - 10) / 8)
    
    num_neb_iterations = int(frac_neb_iterations * all_neb_iterations)

    assert num_neb_iterations > 0, "Number of NEB iterations must be greater than 0"
    
    
    # remove first 10 frames and then reshape into 8 frames/images per NEB iteration
    positions = positions[10:].reshape(-1,8, positions.shape[-2], positions.shape[-1])
    energies = grp["wB97x_6-31G(d).energy"][10:].reshape(-1,8)
    forces = grp["wB97x_6-31G(d).forces"][10:].reshape(-1,8, positions.shape[-2], positions.shape[-1])

    # take the relevant frames
    
    assert positions.shape[0] >= num_neb_iterations
    
    positions = positions[-num_neb_iterations:, start:end]
    energies = energies[-num_neb_iterations:, start:end]
    forces = forces[-num_neb_iterations:, start:end]
    

    # reshape into original shape
    positions = positions.reshape(-1, positions.shape[-2], positions.shape[-1])
qq    
    energies = energies.reshape(-1)
    forces = forces.reshape(-1, forces.shape[-2], forces.shape[-1])
    atomic_numbers = list(grp["atomic_numbers"])
    molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)
    
    assert len(energies) == len(forces) == len(positions), "Length of energies, forces and positions must be the same"
    for energy, force, positions in zip(energies, forces, positions):
        # get ase atoms object
        atoms = Atoms(atomic_numbers, positions=positions)
        sp_calc = SPCalc(atoms=atoms, energy=energy, forces=force.tolist())
        sp_calc.implemented_properties = ["energy", "forces"]
        atoms.set_calculator(sp_calc)
        atoms.set_tags(2 * np.ones(len(atomic_numbers)))
        id = (f"{formula}_{rxn}", fid)
        fid += 1

        """d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": positions,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }"""

        yield id, atoms


class Dataloader:
    """
    Can iterate through h5 data set for paper ####

    hdf5_file: path to data
    only_final: if True, the iterator will only loop through reactant, product and transition
    state instead of all configurations for each reaction and return them in dictionaries.
    """

    def __init__(self, hdf5_file, datasplit="data", subgroup = None, trajectory_subset = None, frac_neb_iterations = 1):
        self.hdf5_file = hdf5_file
        if subgroup:
            assert subgroup in [
                "reactant",
                "product",
                "transition_state",
            ], "subgroup must be one of 'reactant', 'product' or 'transition_state'"
        self.subgroup = subgroup
        self.trajectory_subset = trajectory_subset
        self.frac_neb_iterations = frac_neb_iterations
        if trajectory_subset is not None:
            assert not subgroup, "subgroup cannot be specified if trajectory_subset is specified"

        self.datasplit = datasplit
        if datasplit:
            assert datasplit in [
                "data",
                "train",
                "val",
                "test",
            ], "datasplit must be one of 'all', 'train', 'val' or 'test'"

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]
            for formula, grp in split.items():
                for rxn, subgrp in grp.items():
                    # reactant = next(generator(formula, rxn, subgrp["reactant"]))
                    # product = next(generator(formula, rxn, subgrp["product"]))

                    """if self.only_final:
                        transition_state = next(
                            generator(formula, rxn, subgrp["transition_state"])
                        )
                        yield {
                            "rxn": rxn,
                            "reactant": reactant,
                            "product": product,
                            "transition_state": transition_state,
                        }"""
                    # yield (reactant, "reactant")
                    # yield (product, "product")        
                    subgrp = subgrp[self.subgroup] if self.subgroup else subgrp # filter by subgroup (reactant, product, transition_state, etc.)
                    rxn_atoms_list = []
                    id_list = []
                    sm_sys = None
                    for id, molecule in generator(formula, rxn, subgrp, self.trajectory_subset, self.frac_neb_iterations):
                        rxn_atoms_list.append(molecule)
                        id_list.append(id)
                    
                    if self.trajectory_subset is not None:
                        shape = int(self.frac_neb_iterations * ((subgrp["positions"].shape[0] - 10)/8)) * (1 + self.trajectory_subset[1] - self.trajectory_subset[0])
                    else:
                        shape = int(self.frac_neb_iterations * (subgrp["positions"].shape[0] - 10))
                    
                    if len(rxn_atoms_list) != shape:
                        import pdb; pdb.set_trace()
                    assert len(rxn_atoms_list) == shape, f"Length of rxn_atoms_list ({len(rxn_atoms_list)}) does not match shape ({shape})"
                    # marking systems that have less than 4 atoms
                    if subgrp["atomic_numbers"].shape[0] < 4:
                        sm_sys = (f"{formula}_{rxn}", subgrp["atomic_numbers"].shape[0])
                    yield id_list, rxn_atoms_list, sm_sys
