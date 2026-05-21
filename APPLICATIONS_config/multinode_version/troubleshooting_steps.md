# Multi-Node Troubleshooting Guide

## Issues Identified from Error Logs

### 1. **Primary Python Script Failure**
- Processes are exiting with code 1 before distributed setup completes
- This suggests an import error, path issue, or other Python-level problem
- The error occurs on rank 4 (local_rank 0 on the third node)

### 2. **Potential Causes**

#### A. **Path/Import Issues**
- The `get_calc` import might fail on some nodes
- Project root path calculation might be different across nodes
- Shared filesystem access issues

#### B. **Memory/Resource Issues** 
- 128GB memory per node might not be sufficient for large trajectory files
- GPU memory allocation conflicts

#### C. **File Access Issues**
- Trajectory files might not be accessible from all nodes
- Model checkpoint files might have permission issues

### 3. **Debugging Steps**

Run these in order:

#### Step 1: Test Basic Multi-Node Setup
```bash
cd /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS_config/multinode_version
bash debug_multinode.sh
```

#### Step 2: Test Simple 3-GPU Setup (if Step 1 works)
```bash
bash simple_test.sh
```

#### Step 3: Check File Accessibility
```bash
salloc --nodes 3 --qos interactive --time 01:00:00 --constraint "gpu" --gpus 3 --mem 128g --ntasks-per-node 1 --cpus-per-task 32 --account m5250_g

# Once allocated, test file access from all nodes:
srun ls -la /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/smaller_dt_and_all_c_all_sys_copy/100ps_double_data_breaking_init_boxes/20ns_solvent_0_1M/md_omol_naotf_diglyme_1m_s1p1/
```

### 4. **Workarounds**

#### A. **Reduce Resource Requirements**
- Start with 3 nodes, 1 GPU each (instead of 4 GPUs per node)
- Use smaller test trajectories first

#### B. **Fix Python Environment**
- Ensure all Python dependencies are available on all nodes
- Check that the conda/virtual environment is properly activated

#### C. **Use Alternative Approach**
- Fall back to single-node multi-GPU for now
- Use job arrays instead of true multi-node

### 5. **Alternative Launch Commands**

If the full 12-GPU version fails, try:

```bash
# 3 nodes, 1 GPU each (total 3 GPUs)
salloc --nodes 3 --qos interactive --time 04:00:00 --constraint "gpu" --gpus 3 --mem 128g --ntasks-per-node 1 --cpus-per-task 32 --account m5250_g

# 2 nodes, 2 GPUs each (total 4 GPUs) 
salloc --nodes 2 --qos interactive --time 04:00:00 --constraint "gpu" --gpus 4 --mem 128g --ntasks-per-node 2 --cpus-per-task 32 --account m5250_g
```

### 6. **Error Patterns to Watch For**

- **Import Errors**: Module not found errors
- **File Access**: Permission denied or file not found
- **Memory**: OOM errors or allocation failures  
- **Network**: Connection timeouts or broken pipes
- **GPU**: CUDA errors or device allocation failures

### 7. **Success Indicators**

Look for these messages in the output:
- ✅ "Detected execution mode: multi_node"
- ✅ "Global rank X (local rank Y): Initialized multi-node distributed setup"  
- ✅ "Global rank X: Assigned trajectory: ..."
- ✅ "Starting simulation for ..."

### 8. **Next Steps After Debugging**

1. If debugging works → Use the simple test with 3 GPUs
2. If simple test works → Gradually increase to more GPUs/trajectories
3. If still failing → Switch to single-node multi-GPU approach
4. If single-node works → Submit multiple single-node jobs instead