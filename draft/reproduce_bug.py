import torch


grid_cell_atom_count = torch.load("/global/homes/y/yuejian/project/MLFF-distill/yuejian/test/grid_cell_atom_count.pt")
source_atom_grid_id = torch.load("/global/homes/y/yuejian/project/MLFF-distill/yuejian/test/source_atom_grid_id.pt")


grid_cell_atom_count.index_add_(
    0, source_atom_grid_id, torch.ones_like(source_atom_grid_id)
) 