from __future__ import annotations
import pickle
from typing import TYPE_CHECKING
import torch
from fairchem.core import __version__
from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer
import math
import lmdb 

@registry.register_trainer("worstRow")
class WorstRowTrainer(OCPTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.worst_force_row_dict = {}
        lmdb_path = self.config["optim"].get("worst_rows_lmdb_path")
        self.lmdb_env = lmdb.open(lmdb_path, map_size=int(1e9)) 
        self.true_epoch = 0
        
    def  _compute_loss(self, out, batch):
        loss = super()._compute_loss(out, batch)
        with torch.no_grad():
            if self.config["optim"].get("active_learning", False):
                if (
                    self.true_epoch
                    % self.config["optim"].get("worst_force_update_freq", 5)
                    == 0
                ):
                    self._update_worst_force_loss_dict(out, batch)
        return loss
        
    def validate(self, **kwargs):
        metrics = super().validate(**kwargs)    
        self.true_epoch += 1
        return metrics
        
    def _update_worst_force_loss_dict(self, out, batch):
        # recompute worst k rows of force loss
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0

        loss = []
        for loss_fn in self.loss_functions:
            target_name, loss_info = loss_fn
            if target_name == "forces":

                target = batch[target_name]
                pred = out[target_name]

                natoms = batch.natoms
                natoms = torch.repeat_interleave(natoms, natoms)

                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["train_on_free_atoms"]
                ):
                    target = target[mask]
                    pred = pred[mask]
                    natoms = natoms[mask]

                # to keep the loss coefficient weights balanced we remove linear references
                # subtract element references from target data
                if target_name in self.elementrefs:
                    target = self.elementrefs[target_name].dereference(target, batch)
                # normalize the targets data
                if target_name in self.normalizers:
                    target = self.normalizers[target_name].norm(target)

                force_loss_per_atom = torch.linalg.vector_norm(
                    pred - target, ord=2, dim=-1
                )

        batch_size = batch.batch.max().item() + 1
        top_k_arr = torch.ceil(self.config["optim"].get("frac_worst_rows", 0.5) * batch.natoms)
        grouped_force_loss_per_atom = [
            force_loss_per_atom[batch.batch == i] for i in range(batch_size)
        ]

        # Step 3: Apply topk to each batch's values
        topk_indices = []
        topk_values = []

        for i in range(batch_size):
            if len(grouped_force_loss_per_atom[i]) > 0:
                top_vals, top_idx = torch.topk(
                    grouped_force_loss_per_atom[i],
                    int(top_k_arr[i]),
                )
                topk_values.append(top_vals)
                topk_indices.append(top_idx)
            else:
                topk_values.append(torch.tensor([]))  # No elements for this batch
                topk_indices.append(torch.tensor([], dtype=torch.long))

        # # update worst rows dict
        # for i in range(batch_size):
        #     self.worst_force_row_dict[batch.fid[i].item()] = final_topk_indices[i]
        
        # Store in LMDB
        with self.lmdb_env.begin(write=True) as txn:
            for i in range(batch_size):
                key = f"{batch.fid[i].item()}_{self.true_epoch}".encode()
                value = pickle.dumps(topk_indices[i])
                txn.put(key, value)