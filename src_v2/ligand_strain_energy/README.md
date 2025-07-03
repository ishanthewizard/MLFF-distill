# Ligand Strain Energy

This directory contains tools and utilities for calculating ligand strain energy.

## TODO

- [ ] Currently charge and spin are set to 0 and 1 in rdkit to ase function, need to change this in the future 
  - **Location:** `strain_relief/src/strain_relief/io/utils_mol_format.py` lines 24-25
  - **Context:** TODO comment about temporary fix for spin=1, charge=0 assumption
- [ ] also you need to sanity check that all mols in the test set are neutral and their spin are 1
- [ ] double check about the units for each model output