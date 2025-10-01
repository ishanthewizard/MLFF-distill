import os, sys
from fairchem.core.common.registry import registry
from fairchem.core import OCPCalculator
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs
import numpy as np
import lmdb
from ase.io import read
from ase import Atoms
# from mace.calculators import mace_off, mace_mp
from tqdm import tqdm
from torch.utils.data import Subset
from torch.autograd.functional import jacobian
from torch.autograd import grad
import torch
from torch import vmap
from torch.amp import autocast
def record_and_save(dataset, file_path, fn):
    # Assuming train_loader is your DataLoader
    avg_num_atoms = dataset[0].natoms.item()
    map_size = 1099511627776 * 2
    print("file_path",file_path)
    # sys.exit()
    env = lmdb.open(file_path, map_size=map_size)
    env_info = env.info()
    with env.begin(write=True) as txn:
        for sample in tqdm(dataset):
            sample_id = str(int(sample.id))
            print("sample",sample)
            if sample['atomic_numbers'].shape[0] >30:
                continue

                # sys.exit()
            # sys.exit()
            sample_output = fn(sample)  # this function needs to output an array where each element correponds to the label for an entire molecule
            continue
            # Convert tensor to bytes and write to LMDB
            txn.put(sample_id.encode(), sample_output.tobytes())
    env.close()
    print(f"All tensors saved to LMDB:{file_path}")

def record_labels(labels_folder, dataset_path, model="large"):
    os.makedirs(labels_folder, exist_ok=True)
    # Load the dataset
    train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'train')})
    
    val_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'val')})
    print("train_dataset", train_dataset)
    print("val_dataset", val_dataset)
    calc = OCPCalculator(checkpoint_path="/home/yuejian/project/MLFF-distill/collection/eSEN/esen_30m_mptrj.pt")
    # sys.exit(0)
    # Load the model
    # calc = mace_mp(model=model, dispersion=False, default_dtype="float32", device='cuda')

    def get_forces(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(), cell=sample.cell.numpy()[0], pbc=True)
        atoms.calc = calc
        return atoms.get_forces()
    def get_hessians(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        natoms = sample.natoms
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(), cell=sample.cell.numpy()[0], pbc=True)
        
        # transform the atoms object to a graph object
        a2g = AtomsToGraphs(
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_pbc=True,
            r_edges=not True # config["model"]["otf_graph"],
        )

        data_object = a2g.convert(atoms)
        batch = data_list_collater([data_object], otf_graph=True)
        
        
        model = calc.trainer.model.to('cuda')
        device = next(model.parameters()).device
        print("Model is on:", device)
        batch = batch.to('cuda')
        batch.pos = batch.pos.detach().requires_grad_()
        # result = model(batch)
        # check if forces are detached
        # forces = result['forces']
        # print("forces.requires_grad:", forces.requires_grad)
        

        # def calc_func(x):
        #     batch.pos = x
        #     result = model(batch)
        #     return result['forces']
        
        # # directly calculate Jacobian
        # J = jacobian(calc_func, batch.pos)
        
        # # calculate through VJP+vmap
        # pos = batch.pos.detach().requires_grad_()  # ensure it's a leaf with grad tracking
        # batch.pos = pos  # update the batch to use the new `pos` with grad
        # result = model(batch)
        # forces = result['forces']         # shape: (86, 3)
        # forces_flat = forces.view(-1)     # shape: (258,)

        # def VJP_one_row(v):
        #     VJP = grad(outputs = forces_flat, # 
        #                     inputs = pos,  # 
        #                     grad_outputs = v, # 
        #                     create_graph=False)[0].view(-1) # (258)
        #     return VJP
        # # print("jacobian", jacobian)
        # # Step 3: Create identity matrix as batch of one-hot vectors (each is grad_outputs)
        # eye = torch.eye(forces_flat.numel(), device=forces_flat.device)  # [258, 258]
        
        # # Step 4: Vectorize over one-hot vectors to get full Jacobian
        # with autocast(device_type="cuda"):
        #     jacobian = vmap(VJP_one_row)(eye)  # shape: [258, 258]
        # print("jacobian", jacobian,jacobian.shape)
        
        # calculate through VJP and for loop
        # Ensure pos is a leaf with grad enabled
        pos = batch.pos.detach().requires_grad_()
        batch.pos = pos  # update the batch

        # Run model once
        result = model(batch)
        forces = result['forces']        # shape: [86, 3]
        forces_flat = forces.view(-1)    # shape: [258]

        # Identity matrix of grad_outputs
        eye = torch.eye(forces_flat.numel(), device=forces_flat.device)

        # Initialize Jacobian tensor
        jacobian_rows = []

        # Loop over each output dimension
        for i in range(forces_flat.numel()):
            v = eye[i]  # one-hot grad_outputs vector: shape [258]
            J_row = grad(
                outputs=forces_flat,
                inputs=pos,
                grad_outputs=v,
                retain_graph=True,  # important for multiple grads
                create_graph=False  # only True if you need second derivatives later
            )[0].view(-1)  # shape: [258]
            jacobian_rows.append(J_row)

        # Stack rows into full Jacobian
        jacobian = torch.stack(jacobian_rows, dim=0)  # shape: [258, 258]
        print("jacobian", jacobian,jacobian.shape)
        
        sys.exit(0)
        
        
        
        # print("show",J.shape)
        # print("sum", J.sum())
        
        # print("results", calc.results)
        # sys.exit()
        # hessian = calc.get_hessian(atoms=atoms)
        return - 1 * J # this is SUPER IMPORTANT!!! multiply by -1
    
    def get_final_node_features(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        natoms = sample.natoms
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(), cell=sample.cell.numpy()[0], pbc=True)
        atoms.calc = calc
        return calc.get_final_node_features(atoms=atoms)


    # def get_atom_embeddings():
    #     return calc.get_atom_embeddings()
    print("calc", calc)
    # sys.exit()
    # atom_embeddings = get_atom_embeddings()
    # print("atom_embeddings", atom_embeddings.shape)
    # sys.exit()
    record_and_save(train_dataset, os.path.join(labels_folder, 'force_jacobians', 'force_jacobians.lmdb'), get_hessians)
    
    sys.exit()
    record_and_save(train_dataset, os.path.join(labels_folder, 'train_forces', 'train_forces.lmdb'), get_forces)
    record_and_save(val_dataset, os.path.join(labels_folder, 'val_forces', 'val_forces.lmdb'), get_forces)
    sys.exit(0)
    np.save(os.path.join(labels_folder, 'atom_embeddings.npy'), get_atom_embeddings())
    # record_and_save(train_dataset, os.path.join(labels_folder, 'final_node_features.lmdb'), get_final_node_features)
    # record_and_save(val_dataset, os.path.join(labels_folder, 'final_node_features.lmdb'), get_final_node_features)
    # record_and_save(train_dataset, os.path.join(labels_folder, 'train_forces.lmdb'), get_forces)
    # record_and_save(train_dataset, os.path.join(labels_folder, 'force_jacobians.lmdb'), get_hessians)
    # record_and_save(val_dataset, os.path.join(labels_folder, 'val_forces.lmdb'), get_forces)

if __name__ == "__main__":
    labels_folder = '/data/yuejian/MLFF_DIST/data/labels/MPtrj_labels/eSEN_mp_all_splits_Bandgap_greater_5'
    dataset_path = '/data/yuejian/MLFF_DIST/data/MPtrj/MPtrj_separated_all_splits/Bandgap_greater_than_5'

    record_labels(labels_folder, dataset_path)

    # labels_folder = '/data/shared/ishan_stuff/labels/mace_off_large_SpiceIodine'
    # dataset_path = '/data/ishan-amin/SPICE/spice_separated/Iodine'

    # record_labels(labels_folder, dataset_path)


    # labels_folder = '/data/shared/ishan_stuff/labels/mace_off_large_SpiceAminos'
    # dataset_path = '/data/ishan-amin/SPICE/spice_separated/Solvated_Amino_Acids'

    # record_labels(labels_folder, dataset_path)


