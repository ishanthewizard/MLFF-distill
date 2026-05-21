# Continuous Multi-Node MD Simulation System

## Overview

This system allows you to run MD simulations continuously beyond the 4-hour interactive session limit by automatically launching successive sessions.

## Files

- `continuous_simulation_wrapper.sh` - Main wrapper that handles the continuous execution
- `simulation_control.sh` - Easy-to-use control interface  
- `launch_interactive.sh` - Single 4-hour session launcher (existing)
- `interactive_12gpu_srun.sh` - Simulation execution script (existing)

## Quick Start

### 1. Start Continuous Simulation

```bash
# Run for 10 cycles (40 hours total)
bash simulation_control.sh start 10

# Run for 5 cycles (20 hours total) 
bash simulation_control.sh start 5
```

### 2. Check Status

```bash
bash simulation_control.sh status
```

### 3. View Logs

```bash
# Show last 50 lines
bash simulation_control.sh logs

# Show last 100 lines  
bash simulation_control.sh logs 100
```

### 4. Stop Gracefully

```bash
bash simulation_control.sh stop
```

## How It Works

### Cycle Management
- Each cycle runs for 4 hours (SLURM interactive limit)
- Simulations automatically resume from checkpoint files
- 60-second pause between cycles to avoid queue issues
- Automatic detection of cycle completion

### Monitoring & Logging
- All activities logged with timestamps
- Separate log files for each cycle
- Real-time status monitoring
- Queue wait time tracking

### Safety Features
- Graceful stop mechanism (completes current cycle)
- Emergency stop via `STOP_CONTINUOUS` file
- Process monitoring and cleanup
- Timeout handling for queue waits

## Advanced Usage

### Manual Control

```bash
# Direct wrapper usage
bash continuous_simulation_wrapper.sh 8  # 8 cycles = 32 hours

# Emergency stop (creates stop file)
touch STOP_CONTINUOUS

# Check running processes
pgrep -f continuous_simulation_wrapper.sh
pgrep -f "salloc.*interactive"
```

### Log Management

```bash
# View all log files
ls -la continuous_logs/

# Clean old logs (>7 days)
bash simulation_control.sh clean 7

# Monitor live progress
tail -f continuous_logs/continuous_*.log
```

## Simulation Resume Behavior

The MD simulation script automatically resumes from existing checkpoints:
- Reads existing `.traj` files to determine current step
- Continues from last saved step
- No data loss between cycles
- Seamless continuation across sessions

## Expected Output Structure

```
continuous_logs/
├── continuous_20260419_143022.log          # Main run log
├── continuous_20260419_143022_cycle_1.log  # Cycle 1 details  
├── continuous_20260419_143022_cycle_2.log  # Cycle 2 details
└── nohup_20260419_143022.log              # Background execution log
```

## Monitoring Examples

### Real-time Status
```bash
# Check every 30 seconds
watch -n 30 "bash simulation_control.sh status"

# Follow main log  
tail -f continuous_logs/continuous_*.log
```

### Progress Tracking
```bash
# Count completed cycles
grep "Completed.*successfully" continuous_logs/continuous_*.log

# Check simulation progress in trajectory directories
ls -la /path/to/trajectory/dirs/*/md_*.log
```

## Troubleshooting

### Common Issues

1. **Queue Wait Timeout**
   - Increase timeout in wrapper script
   - Check SLURM queue status: `squeue -u $USER`

2. **Session Not Starting**
   - Verify account access: `sacctmgr show assoc user=$USER`
   - Check resource availability: `sinfo -p gpu`

3. **Simulation Errors**
   - Check cycle-specific logs in `continuous_logs/`
   - Verify trajectory and checkpoint file integrity

### Emergency Recovery

```bash
# Kill all related processes
pkill -f continuous_simulation_wrapper.sh
pkill -f "salloc.*interactive.*m5250_g"

# Remove stop file if stuck
rm -f STOP_CONTINUOUS

# Clean up and restart
bash simulation_control.sh clean 0
bash simulation_control.sh start N
```

## Performance Tips

1. **Queue Management**
   - Run during off-peak hours for faster allocation
   - Monitor queue status before starting long runs

2. **Resource Optimization**  
   - Verify checkpoint file sizes periodically
   - Monitor disk usage in trajectory directories

3. **Reliability**
   - Test with 1-2 cycles before long runs
   - Keep backup copies of important trajectory data

## Integration with Existing Workflow

This system builds on your existing scripts:
- Uses the same `launch_interactive.sh` and `interactive_12gpu_srun.sh`
- Maintains all current configurations and parameters
- Compatible with existing trajectory and checkpoint structures
- No changes needed to the MD simulation code

The continuous system simply orchestrates multiple 4-hour sessions to achieve longer total runtime.