# Multi-Node MD Simulation Setup

This directory contains the merged multi-node version of the molecular dynamics simulation scripts.

## Files

- `multinode_12gpu_merged.sh` - Main SLURM script for 3-node, 12-GPU execution
- `README.md` - This documentation file

## Configuration Details

### Hardware Configuration
- **Nodes**: 3 compute nodes
- **GPUs per node**: 4 GPUs 
- **Total GPUs**: 12 GPUs
- **CPUs per GPU**: 32 cores
- **Memory per node**: 200GB

### Simulation Configuration
The script merges all trajectory-model pairs from `g1.sh`, `g2.sh`, and `g3.sh`:

#### Trajectory Sources:
- **g1.sh**: 4 trajectories (GPUs 0-3)
- **g2.sh**: 4 trajectories (GPUs 4-7) 
- **g3.sh**: 4 trajectories (GPUs 8-11)

#### Total: 12 trajectory-model pairs

### Model Distribution:
- **Trajectories 1-4**: `trained_on_all_systems` model
- **Trajectories 5-8**: Mixed (1 `trained_on_all_systems` + 3 `micro_student`)
- **Trajectories 9-12**: `micro_student` model

### Parameter Variations:
- **Temperatures**: 298.0K (most), 273.2K (one trajectory)
- **Timesteps**: 0.7fs (g1 trajectories), 1.0fs (g2/g3 trajectories)
- **Target steps**: 20,000,000 steps for all

## Usage

### Submit the Job
```bash
cd /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS_config/multinode_version
sbatch multinode_12gpu_merged.sh
```

### Monitor Job Status
```bash
# Check job status
squeue -u $USER

# View output logs
tail -f /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/log/multinode_md_multinode_12gpu_*
```

### Cancel Job (if needed)
```bash
scancel <job_id>
```

## Key Features

1. **Automatic Mode Detection**: The Python script will detect multi-node execution automatically via environment variables set by `torchrun`

2. **Distributed Synchronization**: All 12 processes synchronize at startup and cleanup

3. **Robust Error Handling**: Comprehensive validation of configuration arrays

4. **Detailed Logging**: Extensive logging of job parameters, node assignments, and progress

5. **Scalable Design**: Can be easily adapted for different node/GPU configurations

## Comparison with Single-Node Scripts

| Feature | Single-Node (g1/g2/g3) | Multi-Node (this script) |
|---------|------------------------|---------------------------|
| Nodes | 1 | 3 |
| Total GPUs | 4 | 12 |
| Trajectories | 4 per script | 12 total |
| Process Management | `mp.spawn` | `torchrun` + SLURM |
| Scalability | Limited to 1 node | Scales across nodes |
| Execution | Direct Python call | `srun torchrun python` |

## Troubleshooting

### Common Issues:
1. **Path Access**: Ensure all trajectory paths are accessible from all nodes
2. **Network**: Verify inter-node communication is working
3. **Resource Limits**: Check available GPU/memory resources
4. **File Permissions**: Ensure output directories are writable

### Debugging:
- Check SLURM output/error files for detailed logs
- Verify node connectivity with `srun hostname`
- Test distributed setup with a simple torch.distributed example

## Customization

To modify for different configurations:
1. Update `#SBATCH --nodes=N` for different node counts
2. Adjust trajectory/model arrays for your data
3. Modify `TARGET_STEPS`, `INTERVAL` as needed
4. Change `MASTER_PORT` if port conflicts occur