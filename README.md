# MLFF-Distill: Model Distillation for Machine Learning Force Fields via Energy Hessians

This repository contains the code for the paper:  
**Amin, I., Raja, S., Krishnapriyan, A.S. (2024). Towards Fast, Specialized Machine Learning Force Fields: Distilling Foundation Models via Energy Hessians.**  
*Accepted to ICLR 2025.* [arXiv:2501.09009](https://arxiv.org/abs/2501.09009).

We built our implementation of Hessian distillation on top of the [Fairchem repository](https://github.com/FAIR-Chem/fairchem).  
The environment and NERSC training instructions were adapted from the [EScAIP repository](https://github.com/ASK-Berkeley/EScAIP/tree/main).

---


## Installing the conda environment 
(environment is from the [EScAIP repository](https://github.com/ASK-Berkeley/EScAIP/tree/main))

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

If not, add something like the following (depends on location) to your `.bashrc` or `.zshrc`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

Step 3: Install the dependencies

```bash
mamba env create -f env.yml
conda activate escaip
```
## Downloading the Data
Below we provide the links to repositories where the foundation model weights were obtained, as well as the training data:

- [Mace-OFF repository](https://github.com/ACEsuit/mace-off)
- [Mace-MP repository](https://github.com/ACEsuit/mace-mp)
- [JMP repository](https://github.com/facebookresearch/JMP)
- [Spice Dataset](https://www.repository.cam.ac.uk/items/d50227cd-194f-4ba4-aeb7-2643a69f025f)
- [MPtrj Dataset](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842)

Scripts to process the data can be found in the [scripts](scripts/) folder. We may add postprocessed labels and data to this repo at some point in the future.


## Start Training Runs

To perform a standard (non-distilled) training run of Gemnet-dT on the Solvated Amino Acids subset of SPICE, execute:

```bash
python main.py --mode train --config-yml configs/SPICE/solvated_amino_acids/gemnet-dT-small.yml
```
For more info and options relating to training command inputs, please see the [Fairchem repository](https://github.com/FAIR-Chem/fairchem).

For a Hessian distillation version of the above training run, navigate to the hessians folder, located in the same directory as the original configuration. The distillation configuration specifies attributes unique to distillation training and includes a link to all the attributes of the original (non-distilled) configuration:

```bash
python main.py --mode train --config-yml configs/SPICE/solvated_amino_acids/hessian/gemnet-dT-small.yml
```

Similiarly, you can run some of the baselines we ran in our paper by selecting a config from the baselines folder, which is located in the same directory as the undistilled configuration:

```bash
python main.py --mode train --config-yml configs/SPICE/solvated_amino_acids/baselines/gemnet-dT-small-n2n.yml
```

## Distributed Training
For distributed training on NERSC, please see the [Nersc Distributed Training README](NERSC_dist_train.md) taken from the  [EScAIP repository](https://github.com/ASK-Berkeley/EScAIP/tree/main)

