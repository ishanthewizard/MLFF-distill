#!/bin/bash


source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairchem_V2

############# label generation

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/OMol_fairchem_v2/training_4M/eSEN_md.yaml


############# label merging

# python draft/collate_hessian_results/merging.py \
# --subset_root /data/ericqu/OMol_subset/test

############# label sorting

# python draft/collate_hessian_results/sorting.py \
# --subset_root /data/ericqu/OMol_subset/test


############# model training

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/germain/protein_ligand_pocket/eSEN_md_train.yaml







############### model distillation

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/germain/protein_ligand_pocket/distill.yaml


python main_v2.py \
-c /home/yuejian/project/MLFF-distill/configs/germain/protein_ligand_fragments/distill.yaml



############# debugging
############# debug ckpt loading

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/OMol_fairchem_v2/debug_ckpt/eSEN_md.yaml