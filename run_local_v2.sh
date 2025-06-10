#!/bin/bash


source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairchem_V2

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export MASTER_PORT=13379

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

######## training with no distillation

python main_v2.py \
-c /home/yuejian/project/MLFF-distill/configs/escher/protein_ligand_pocket_autograd/eSEN_md_train.yaml



############### model distillation


######## distillation with autograd

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/escher/protein_ligand_pocket_autograd/distill.yaml

######## distillation with finite difference

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/escher/protein_ligand_pocket_finite_diff/distill.yaml



############# debugging
############# debug ckpt loading

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/OMol_fairchem_v2/debug_ckpt/eSEN_md.yaml