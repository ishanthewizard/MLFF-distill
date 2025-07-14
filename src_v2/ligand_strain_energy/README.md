# Ligand Strain Energy

This directory contains tools and utilities for calculating ligand strain energy.

## TODO

- [ ] Currently charge and spin are set to 0 and 1 in rdkit to ase function, need to change this in the future 
  - **Location:** `strain_relief/src/strain_relief/io/utils_mol_format.py` lines 24-25
  - **Context:** TODO comment about temporary fix for spin=1, charge=0 assumption
- [ ] also you need to sanity check that all mols in the test set are neutral and their spin are 1
- [ ] double check about the units for each model output

## Environment Setup

To run the strain energy scripts, you need to set up your Python environment with the required dependencies from both the `strain_relief` and `fairchem_v2` packages. Follow these steps:

1. Clone or have access to both the `strain_relief` and `fairchem_v2` repositories.
2. Install both packages in development mode:
   ```bash
   pip install -e /path/to/strain_relief
   pip install -e /path/to/fairchem_v2
   ```
3. Downgrade numpy to version 1.26.4 (required for compatibility):
   ```bash
   pip install numpy==1.26.4
   ```

Make sure to activate your virtual environment before installing these packages.

## How to Run `strain_energy_calculator.py`

### Running from the ligand_strain_energy Directory

If you are already in the `ligand_strain_energy` directory, you can run the script with:

```bash
python strain_energy_calculator.py
```

This will use the default configuration and paths as described above. Make sure your environment is set up as described in the Environment Setup section.

## Configuration

The behavior of `strain_energy_calculator.py` is controlled by a YAML configuration file, managed by Hydra. By default, the script loads `configs/strain_energy/default.yaml`.

### How to Specify a Config

- To use the default config, simply run:
  ```bash
  python strain_energy_calculator.py
  ```
- To use a different config file in the same directory, run:
  ```bash
  python strain_energy_calculator.py --config-name <your_config_name>
  ```
  Replace `<your_config_name>` with the name of your YAML file (without the `.yaml` extension).

### What the Config Controls
- Input and output file paths
- Methods and parameters for local/global minimization and energy evaluation
- Model paths (if using machine learning models)
- Thresholds and conformer generation settings

### Example Config Snippet
```yaml
io:
  input:
    path: /path/to/input.parquet
    mol_col_name: mol
    id_col_name: ligand_id
  output:
    path: /path/to/output.parquet
local_min:
  method: MMFF
  max_iters: 200
conformers:
  num_confs: 20
global_min:
  method: MMFF
  max_iters: 500
energy_eval:
  method: MMFF
threshold: 10.0
```

Edit the config YAML file to customize your run as needed. For more advanced options, see the configs in `configs/strain_energy/`.

## Jupyter Notebook Example

The notebook `example_for_read_in_parquet.ipynb` demonstrates how to:
- Read molecular data from a parquet file
- Extract molecule bytes and convert them into RDKit Mol objects
- Access atomic numbers, coordinates, charge, and spin multiplicity

This can be helpful for inspecting your data or preparing it for further processing with RDKit or ASE.
