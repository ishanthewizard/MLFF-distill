# DFT Calculation Data Pipeline

This directory contains scripts for preparing and processing data for DFT (Density Functional Theory) calculations using ORCA.

## Scripts Overview

### 1. `prepare_orca_input.py`

A Python script that processes SDF (Structure Data File) files and generates corresponding ORCA input files for DFT calculations.

### 2. `split_sdf_to_ligands.py`

A script that splits a large SDF file containing multiple molecules into individual SDF files, one per ligand.

### 3. `test_split_sdf_to_ligands.py`

A test script that validates the splitting process by comparing the original molecules with the split SDF files.

## Usage

### `prepare_orca_input.py`

**Purpose**: Converts SDF files to ORCA input files for DFT calculations.

**Basic Usage**:
```bash
python prepare_orca_input.py <base_directory>
```

**Full Usage**:
```bash
python prepare_orca_input.py <base_directory> [--nprocs N] [--max-iter N] [--guessmix N]
```

#### Parameters

- `base_directory` (required): Path to the directory containing subdirectories with SDF files
- `--nprocs N` (optional): Number of processors for ORCA calculations (default: 8)
- `--max-iter N` (optional): Maximum number of SCF iterations (default: 100)
- `--guessmix N` (optional): Number for guessmix parameter (default: 50)

#### Examples

```bash
# Basic usage with default parameters
python prepare_orca_input.py data/ligandboundconf3.0/crest_global_min

# Specify number of processors
python prepare_orca_input.py data/ligandboundconf3.0/crest_global_min --nprocs 16

# Specify both nprocs and max iterations
python prepare_orca_input.py data/ligandboundconf3.0/crest_global_min --nprocs 12 --max-iter 200

# Specify all parameters
python prepare_orca_input.py data/ligandboundconf3.0/crest_global_min --nprocs 16 --max-iter 200 --guessmix 50

# High-performance settings
python prepare_orca_input.py data/ligandboundconf3.0/crest_global_min --nprocs 32 --max-iter 300 --guessmix 100
```

#### Input Structure

The script expects a directory structure like:
```
base_directory/
├── ligand1/
│   ├── ligand1.sdf
│   └── ...
├── ligand2/
│   ├── ligand2.sdf
│   └── ...
└── ...
```

#### Output Structure

For each SDF file, the script creates:
```
base_directory/
├── ligand1/
│   ├── ligand1.sdf
│   ├── DFT/
│   │   └── input/
│   │       └── ligand1.inp
│   └── ...
├── ligand2/
│   ├── ligand2.sdf
│   ├── DFT/
│   │   └── input/
│   │       └── ligand2.inp
│   └── ...
└── ...
```

#### ORCA Input File Format

The generated ORCA input files use the following configuration:

```
! UKS wB97M-V DEF2-TZVPD TightSCF DEFGRID3 RIJCOSX
%scf
  MaxIter <max_iter>
  Thresh 1e-12        # Integral threshold
  TCut   1e-13        # Primitive batch threshold
  guessmix <guessmix>
end

%pal
  nprocs <nprocs>     # Number of processors
end

* xyz <charge> 1
<atomic_coordinates>
*
```

#### Features

- **Automatic SDF Parsing**: Extracts atomic coordinates and charge information
- **Configurable Parameters**: Control number of processors and SCF iterations
- **Directory Structure**: Creates organized `DFT/input/` directories
- **File Overwriting**: Re-runs update existing files with new parameters
- **Error Handling**: Reports processing errors while continuing with other files
- **Progress Tracking**: Shows processing status and final statistics

#### Requirements

- Python 3.6+
- Standard library modules: `os`, `sys`, `argparse`, `pathlib`

**Additional Dependencies** (for other scripts):
- RDKit (for molecular operations)
- NumPy (for numerical operations)
- tqdm (for progress bars)
- pytest (for testing)

#### Example Output

```
Processing directory: data/ligandboundconf3.0/crest_global_min
Number of processors: 16
Maximum SCF iterations: 200
ORCA input files will be saved in DFT/input/ subdirectories within each ligand directory
Existing .inp files will be overwritten
--------------------------------------------------
Processing: data/ligandboundconf3.0/crest_global_min/0A1_3QTC_A_811/0A1_3QTC_A_811.sdf
  Generated: data/ligandboundconf3.0/crest_global_min/0A1_3QTC_A_811/DFT/input/0A1_3QTC_A_811.inp
...

Processing complete!
Successfully processed: 3137 files
Errors: 0 files
```

## Troubleshooting

### Common Issues

1. **Directory not found**: Ensure the base directory path is correct
2. **Permission errors**: Check file/directory permissions
3. **SDF parsing errors**: Verify SDF files are in correct format
4. **Memory issues**: For large datasets, consider processing in batches

### Error Messages

- `Error: Directory {base_dir} does not exist.`: Check the directory path
- `Could not find atom block in SDF file`: SDF file format issue
- `Error processing {file}: {error}`: Individual file processing error

## Notes

- The script processes all `.sdf` files found in subdirectories
- Existing `.inp` files are overwritten when re-running the script
- The script creates necessary directories automatically
- Processing continues even if individual files fail

## Future Enhancements

Potential improvements for future versions:
- Support for different ORCA calculation types
- Batch processing options
- Configuration file support
- Parallel processing for large datasets
- Additional SCF convergence parameters

### `split_sdf_to_ligands.py`

**Purpose**: Splits a large SDF file containing multiple molecules into individual SDF files, one per ligand.

**Usage**:
```bash
python split_sdf_to_ligands.py
```

**Configuration**:
The script uses hardcoded paths that need to be modified:
- `input_sdf`: Path to the input SDF file containing multiple molecules
- `output_base`: Base directory where individual SDF files will be saved

**Input**: A single SDF file containing multiple molecules
**Output**: Individual SDF files organized in subdirectories by ligand ID

**Features**:
- Extracts ligand IDs from molecule properties (`_Name` or `ligand_id`)
- Creates organized directory structure
- Handles missing or invalid ligand IDs gracefully

### `test_split_sdf_to_ligands.py`

**Purpose**: Validates the splitting process by comparing original molecules with split SDF files.

**Usage**:
```bash
python test_split_sdf_to_ligands.py
```

**Configuration**:
The script uses hardcoded paths that need to be modified:
- `input_sdf`: Path to the original SDF file
- `output_base`: Base directory containing the split SDF files

**Features**:
- Compares atom types between original and split molecules
- Validates atomic positions (with tolerance for numerical differences)
- Provides detailed error reporting for mismatches
- Uses progress bars for large datasets

**Requirements**:
- RDKit
- NumPy
- tqdm (for progress bars)
- pytest (for testing framework)

## Workflow

The typical workflow for processing molecular data for DFT calculations is:

1. **Split large SDF file** (if needed):
   ```bash
   python split_sdf_to_ligands.py
   ```

2. **Validate splitting** (optional):
   ```bash
   python test_split_sdf_to_ligands.py
   ```

3. **Generate ORCA input files**:
   ```bash
   python prepare_orca_input.py data/ligandboundconf3.0/crest_global_min --nprocs 16 --max-iter 200 --guessmix 50
   ``` 