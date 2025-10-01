# DFT Single-Point Energy Calculation Script

## Overview

This script automates DFT (Density Functional Theory) single-point energy calculations using ORCA quantum chemistry software. It processes multiple ligands in a directory structure, manages calculation status, and provides progress tracking and error handling.

## Features

- **Batch Processing**: Automatically processes all ligands in a directory structure
- **Status Management**: Tracks calculation success/failure with status files
- **Resume Capability**: Skips completed calculations and retries failed ones
- **Progress Tracking**: Shows real-time progress and summary statistics
- **Error Handling**: Comprehensive error handling with timeout protection
- **Flexible Configuration**: Configurable ORCA path and timeout settings

## Prerequisites

### Software Requirements
- **ORCA**: Quantum chemistry software (tested with ORCA 6.0.0)
- **Python 3.6+**: For running the script
- **OpenMPI**: Required for ORCA parallel calculations

### System Requirements
- Linux/Unix environment
- Sufficient disk space for calculation outputs
- Adequate memory for DFT calculations

## Installation

1. **Install ORCA** (if not already installed):
   ```bash
   # Download ORCA from the official website
   # Extract to your preferred location
   # Example: /home/yuejian/project/MLFF-distill/yuejian/orca/
   ```

2. **Verify ORCA installation**:
   ```bash
   /path/to/orca/orca --version
   ```

## Directory Structure

The script expects the following directory structure:

```
root_directory/
├── ligand_1/
│   └── DFT/
│       ├── input/
│       │   └── ligand_1.inp          # ORCA input file
│       ├── success.txt               # Created on successful completion
│       └── failure.txt               # Created on failure
├── ligand_2/
│   └── DFT/
│       ├── input/
│       │   └── ligand_2.inp
│       └── ...
└── ...
```

### Directory Structure Requirements

- **Root Directory**: Contains subdirectories for each ligand
- **Ligand Directories**: Named directories for each molecule/ligand
- **DFT Directory**: Must contain an `input` subdirectory
- **Input Directory**: Must contain `.inp` files (ORCA input files)
- **Status Files**: Automatically created by the script
  - `success.txt`: Indicates successful calculation completion
  - `failure.txt`: Indicates calculation failure with error details

## Usage

### Basic Usage

```bash
python calculate_single_point_energy.py /path/to/ligands/directory
```

### Advanced Usage

```bash
python calculate_single_point_energy.py /path/to/ligands/directory --orca-path /custom/path/to/orca
```

### Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `root_dir` | string | Yes | - | Root directory containing ligand subdirectories |
| `--orca-path` | string | No | `/home/yuejian/project/MLFF-distill/yuejian/orca/orca_6_0_0_shared_openmpi416/orca` | Path to ORCA executable |

### Examples

1. **Process ligands in current directory**:
   ```bash
   python calculate_single_point_energy.py .
   ```

2. **Process ligands with custom ORCA path**:
   ```bash
   python calculate_single_point_energy.py /data/ligands --orca-path /usr/local/bin/orca
   ```

3. **Process specific ligand directory**:
   ```bash
   python calculate_single_point_energy.py /path/to/specific/ligand
   ```

## ORCA Input File Format

The script expects ORCA input files (`.inp`) in the `input` directory. Example ORCA input file:

```
! RKS B3LYP def2-SVP
! Grid4 GridX4
! RIJCOSX
! def2/J
! D3BJ
! TightSCF
! CPKS

%pal
nprocs 8
end

%maxcore 8000

* xyz 0 1
C 0.0 0.0 0.0
H 1.0 0.0 0.0
H -0.5 0.866 0.0
H -0.5 -0.866 0.0
*
```

## Script Behavior

### Calculation Process

1. **Directory Scanning**: Scans root directory for ligand subdirectories
2. **Status Check**: Checks for existing `success.txt` or `failure.txt` files
3. **Input Validation**: Verifies presence of `.inp` files in input directory
4. **Calculation Execution**: Runs ORCA with timeout protection (24 hours)
5. **Status Recording**: Creates success/failure status files
6. **Progress Reporting**: Displays real-time progress and results

### Status Management

- **Completed Calculations**: Skipped if `success.txt` exists
- **Failed Calculations**: Retried if `failure.txt` exists
- **New Calculations**: Processed normally

### Error Handling

- **Timeout Protection**: 24-hour timeout for each calculation
- **ORCA Errors**: Captures and reports ORCA-specific error messages
- **File System Errors**: Handles missing directories and files
- **Process Errors**: Manages subprocess execution failures

## Output and Logging

### Console Output

The script provides detailed console output including:

- Progress indicators: `[1/10] Processing: ligand_name`
- Status messages: `✓ Success` or `✗ Failure`
- Summary statistics at completion
- Error messages for failed calculations

### Status Files

#### Success File (`success.txt`)
```
Calculation completed successfully at 2024-01-15 14:30:25
```

#### Failure File (`failure.txt`)
```
Calculation failed at 2024-01-15 14:30:25
Error: ORCA did not terminate normally
```

### ORCA Output Files

- **`.out` files**: ORCA calculation outputs (same name as input file)
- **`.gbw` files**: ORCA binary files (if generated)
- **Other ORCA files**: Depending on calculation type

## Configuration

### Timeout Settings

The script uses a 24-hour timeout for each calculation. To modify this, edit the `timeout_seconds` variable in the `run_orca_calculation` function:

```python
timeout_seconds = 24 * 3600  # 24 hours
```

### ORCA Path

Set the default ORCA path in the argument parser or use the `--orca-path` argument:

```python
parser.add_argument(
    "--orca-path",
    type=str,
    default="/home/yuejian/project/MLFF-distill/yuejian/orca/orca_6_0_0_shared_openmpi416/orca",
    help="Path to ORCA executable"
)
```

## Troubleshooting

### Common Issues

1. **ORCA Not Found**:
   ```
   Error: ORCA executable not found at /path/to/orca
   ```
   **Solution**: Verify ORCA installation and update the `--orca-path` argument

2. **Missing Input Files**:
   ```
   No .inp files found in /path/to/input/directory
   ```
   **Solution**: Ensure `.inp` files exist in the `input` directory

3. **Permission Errors**:
   ```
   Permission denied
   ```
   **Solution**: Check file permissions and ensure write access to output directories

4. **Calculation Timeout**:
   ```
   Calculation timed out
   ```
   **Solution**: Increase timeout value or optimize calculation parameters

### Debugging

1. **Check ORCA Installation**:
   ```bash
   /path/to/orca --version
   ```

2. **Verify Directory Structure**:
   ```bash
   ls -la /path/to/ligands/
   find /path/to/ligands/ -name "*.inp"
   ```

3. **Test Single Calculation**:
   ```bash
   # Run ORCA manually on one input file
   /path/to/orca /path/to/ligand/DFT/input/ligand.inp
   ```

## Performance Considerations

- **Parallel Processing**: ORCA can use multiple cores (configure in `.inp` file)
- **Memory Usage**: Adjust `%maxcore` in ORCA input files based on available RAM
- **Disk Space**: Monitor disk usage as calculations generate large output files
- **Network**: For cluster environments, ensure stable network connections

## Best Practices

1. **Backup Data**: Always backup important data before running calculations
2. **Test Small Set**: Test with a few ligands before running large batches
3. **Monitor Resources**: Monitor CPU, memory, and disk usage during calculations
4. **Regular Checkpoints**: Use status files to resume interrupted calculations
5. **Documentation**: Keep records of calculation parameters and results

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify ORCA installation and configuration
3. Review error messages in `failure.txt` files
4. Check system resources and permissions

## License

This script is part of the MLFF-distill project. Please refer to the project's license terms. 