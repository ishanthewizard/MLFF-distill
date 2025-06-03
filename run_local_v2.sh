#!/bin/bash


# label generation

python main_v2.py \
-c /private/home/ericqu/MLFF-distill/configs/OMol_fairchem_v2/training_4M/eSEN_md.yaml

# model training, distillation

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/OMol_fairchem_v2/distill/eSEN_md.yaml

# debug ckpt loading

# python main_v2.py \
# -c /home/yuejian/project/MLFF-distill/configs/OMol_fairchem_v2/debug_ckpt/eSEN_md.yaml