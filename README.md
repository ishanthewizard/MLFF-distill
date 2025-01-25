# MLFF-Distill: Model Distillation for Machine Learning Force Fields via Energy Hessians

This repository contains the code for the paper:  
**Amin, I., Raja, S., Krishnapriyan, A.S. (2024). Towards Fast, Specialized Machine Learning Force Fields: Distilling Foundation Models via Energy Hessians.**  
*Accepted to ICLR 2025.* [arXiv:2501.09009](https://arxiv.org/abs/2501.09009).

We built our implementation of Hessian distillation on top of the [Fairchem repository](https://github.com/FAIR-Chem/fairchem).  
The environment and NERSC training instructions were adapted from the [EScAIP repository](https://github.com/ASK-Berkeley/EScAIP/tree/main).

If you have any questions about the repo feel free to email ishanthewizard@berkeley.edu.

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

## Generating Hessian Labels

### Mace-OFF Labels
To generate the labels for **Mace-OFF**, use the script [`scripts/spice_scripts/get_maceOFF_labels.py`](scripts/spice_scripts/get_maceOFF_labels.py).  
Set the `dataset_path` and `labels_folder` variables in the `main` function to the correct source and destination.

### Mace-MP0 Labels
To generate the labels for **Mace-MP0**, use the script [`scripts/mptraj_scripts/get_maceMP0_labels.py`](scripts/mptraj_scripts/get_maceMP0_labels.py).  
Similarly, set the `dataset_path` and `labels_folder` variables in the `main` function appropriately.


### Generating Hessian Labels with a Teacher Checkpoint
If you have a teacher checkpoint that is runnable in the **Fairchem** repository, you can generate Hessian labels using the following steps:

1. Go to a Hessian configuration file.
2. Set the `trainer` to `src.distill_trainer.LabelsTrainer`.
3. Ensure your Hessian configuration includes the following structure:

```yaml
dataset:
  train:
    teacher_checkpoint_path: data/teacher_checkpoints/nanotube_jmp-l.ckpt 
    teacher_labels_folder: data/labels/md22_labels/jmp-large_double-walled_nanotube/
    label_force_batch_size: 32
    label_jac_batch_size: 64
    vectorize_teach_jacs: False
```
- **`teacher_checkpoint_path`**: Path to your teacher checkpoint.  
- **`teacher_labels_folder`**: Destination path for the generated labels.  
- **`label_force_batch_size`**: Batch size for generating force labels.  
- **`label_jac_batch_size`**: Batch size for generating Hessians (note: setting this too high may cause memory overflow).  
- **`vectorize_teach_jacs`**: If set to `True`, Hessian generation speed will increase using `vmap`, but there is a risk of memory overflow.  

Please see [this file](configs/SPICE/solvated_amino_acids/hessian/gemnet-dT-small.yml) for an example of label generation, the label generation settings are the ones commented out.

- **Important**: Ensure the dataset specified in your base configuration matches the dataset you plan to distill with (i.e., the dataset you want to generate labels for). Also ensure that the linked config with the model attributes is the teacher's config, not the student's.

We created the JMP labels from the [JMP repository](https://github.com/facebookresearch/JMP), essentially by just copying over our src/labels_trainer.py 

## Distributed Training
For distributed training on NERSC, please see the [Nersc Distributed Training README](NERSC_dist_train.md) taken from the  [EScAIP repository](https://github.com/ASK-Berkeley/EScAIP/tree/main)

## Citation
If you find this work useful, please consider citing the following:

```bibtex
@article{amin2025distilling,
      title={Towards Fast, Specialized Machine Learning Force Fields: Distilling Foundation Models via Energy Hessians},
      author={Ishan Amin, Sanjeev Raja, and Krishnapriyan, A.S.},
      journal={International Conference on Learning Representations 2025},
      year={2025},
      archivePrefix={arXiv},
      eprint={2501.09009},
}
```