# MLFF-distill: Force-Jacobian Model Distillation for Machine Learning Force Fields
Based off the fairchem repo with environments and distributed training instructions taken from EGAP (https://github.com/EricZQu/EGAP)

This repository contains the WIP code for MLFF-distill.

## Install 
(taken from EGAP)

Step 1: Install mamba solver for conda (optional)

```bash
conda install mamba -n base -c conda-forge
```

Step 2: Check the CUDA is in `PATH` and `LD_LIBRARY_PATH`

```bash
$ echo $PATH | tr ':' '\n' | grep cuda
/usr/local/cuda/bin

$ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/usr/local/cuda/lib64
```

If not, add something like following (depends on the location) to your `.bashrc` or `.zshrc`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

Step 3: Install the dependencies

```bash
mamba env create -f env.yml
conda activate egap
```
## Start Training Runs
Training is identical to training with the fairchem repo, just with different configs. 
For example, to perform a normal (undistilled) training run of Gemnet-dT on Ac-Ala3, run

```
python main.py --mode train --config-yml configs/md22/Ac-Ala3/gemnet-dT.yml
```
For a distilled version of the above training run, go into the distill folder that is always in the same directory as the original config. The distill 
config specifies attributes unique to distillation training. It also contains a link with all the attributes of the original config. 

```
python main.py --mode train configs/md22/Ac-Ala3/distill/gemnet-dT.yml
```

## Distributed Trainig
For distributed training on NERSC, please see the [Nersc Distributed Training README](NERSC_dist_train.md) (adapted from the EGAP repository)
