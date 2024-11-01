# List of commands to make new splits for the transition1x dataset (just find/replace the trajectory_subset values with the ones you want)
cd /data/shared/ishan_stuff/transition1x
mkdir -p traj_4_4_0.1
mkdir -p traj_4_4_0.1/train
mkdir -p traj_4_4_0.1/val
mkdir -p traj_4_4_0.1/test
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_traj --transition1x_h5 transition1x-release.h5 --split train --traj_dir traj_4_4_0.1/train --num_workers 32 --trajectory_subset 4 4 --frac_neb_iterations 0.1
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_traj --transition1x_h5 transition1x-release.h5 --split val --traj_dir traj_4_4_0.1/val --num_workers 32 --trajectory_subset 4 4 --frac_neb_iterations 0.1
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_traj --transition1x_h5 transition1x-release.h5 --split test --traj_dir traj_4_4_0.1/test --num_workers 32 --trajectory_subset 4 4 --frac_neb_iterations 0.1
mkdir -p lmdb_4_4_0.1
mkdir -p lmdb_4_4_0.1/train
mkdir -p lmdb_4_4_0.1/val
mkdir -p lmdb_4_4_0.1/test
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_lmdbs --data_path traj_4_4_0.1/train --out_path lmdb_4_4_0.1/train --split train --num_workers 32
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_lmdbs --data_path traj_4_4_0.1/val --out_path lmdb_4_4_0.1/val --split val --num_workers 32
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_lmdbs --data_path traj_4_4_0.1/test --out_path lmdb_4_4_0.1/test --split test --num_workers 32
cd ~/MLFF-distill/JMP/scripts
python generate_metadata.py --src /data/shared/ishan_stuff/transition1x/lmdb_4_4_0.1/train --type pretrain

cd /data/shared/ishan_stuff/transition1x
mkdir -p traj_3_5_0.1
mkdir -p traj_3_5_0.1/train
mkdir -p traj_3_5_0.1/val
mkdir -p traj_3_5_0.1/test
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_traj --transition1x_h5 transition1x-release.h5 --split train --traj_dir traj_3_5_0.1/train --num_workers 32 --trajectory_subset 3 5 --frac_neb_iterations 0.1
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_traj --transition1x_h5 transition1x-release.h5 --split val --traj_dir traj_3_5_0.1/val --num_workers 32 --trajectory_subset 3 5 --frac_neb_iterations 0.1
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_traj --transition1x_h5 transition1x-release.h5 --split test --traj_dir traj_3_5_0.1/test --num_workers 32 --trajectory_subset 3 5 --frac_neb_iterations 0.1
mkdir -p lmdb_3_5_0.1
mkdir -p lmdb_3_5_0.1/train
mkdir -p lmdb_3_5_0.1/val
mkdir -p lmdb_3_5_0.1/test
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_lmdbs --data_path traj_3_5_0.1/train --out_path lmdb_3_5_0.1/train --split train --num_workers 32
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_lmdbs --data_path traj_3_5_0.1/val --out_path lmdb_3_5_0.1/val --split val --num_workers 32
python -m jmp.datasets.scripts.transition1x_preprocess.trans1x_write_lmdbs --data_path traj_3_5_0.1/test --out_path lmdb_3_5_0.1/test --split test --num_workers 32
cd ~/MLFF-distill/JMP/scripts
python generate_metadata.py --src /data/shared/ishan_stuff/transition1x/lmdb_3_5_0.1/train --type pretrain