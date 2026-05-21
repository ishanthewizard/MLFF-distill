from ase.io import read
from get_calc import get_uma_calc
import torch
import numpy as np
traj = read('/global/homes/y/yuejian/project/MLFF-distill/m4558/Sep_18/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj', index=':')
print(traj[0])


calc = get_uma_calc(
    uma_path="/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt",
    small_model=False
)

atom = traj[0]
ref_forces = atom.get_forces()

atom.calc = calc
forces = atom.get_forces()

# force mae torch tensor
mae = np.mean(np.abs(forces - ref_forces))
print(mae)