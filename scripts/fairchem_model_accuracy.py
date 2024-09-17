
import argparse
import copy
import logging
from typing import TYPE_CHECKING

from fairchem.core.common.flags import flags
from fairchem.core.common.utils import build_config
from fairchem.core.common.utils import new_trainer_context
from tqdm import tqdm
import time
from fairchem.core.modules.loss import L2MAELoss
import numpy as np
import torch
from src.distill_utils import print_cuda_memory_usage
def evaluate_model(trainer, eval_loss):
    loss_fn = L2MAELoss()
    losses = []
    trainer.model.eval()
    start = None
    start_record_idx = 15
    batch_size = trainer.config["optim"].get("eval_batch_size", trainer.config["optim"]["batch_size"])
    print(f"BATCH SIZE: {batch_size}")
    print(f"TRAINER EPOCH: {trainer.epoch}")
    print(f"Trainer best val metric{trainer.best_val_metric}")
    # Use torch.no_grad to disable gradient tracking
    with torch.no_grad():
        for i, batch in enumerate(tqdm(trainer.test_loader)):
            if i >= start_record_idx and start is None:
                start = time.time()

            forces = trainer._forward(batch)["forces"]
            if eval_loss:
                losses.append(torch.abs(forces - batch["forces"]).mean().item())
            # print_cuda_memory_usage()

    iter_per_s = (len(trainer.test_loader) - start_record_idx) / (time.time() -  start)
    print("LEN DATASET:", len(trainer.test_dataset))
    print(f"INTERATIONS PER SECOND: {iter_per_s}")
    print(f"THROUGHPUT PER SECOND: {iter_per_s * batch_size}")
    if eval_loss:
        print(f"FORCE MAE: {np.mean(np.array(losses)) * 1000} meV")
        # print(f"FORCE MAE MEDIAN: {np.median(np.array(losses)) * 1000} meV")
        # print(f"FORCE MAE MAX: {np.max(np.array(losses)) * 1000} meV")
if __name__ == "__main__":
    # config_path = 'configs/SPICE/solvated_amino_acids/painn/painn-small.yml' # you need to supply this at the command line
    # checkpoint_path = 'checkpoints/2024-09-08-10-29-36-solvated-PaiNN-DIST/best_checkpoint.pt'
    # test_dataset  = '/data/ishan-amin/spice_separated/Iodine/test'
    batch_size = 32
    parser: argparse.ArgumentParser = flags.get_parser()
    parser.add_argument("--nersc", action="store_true", help="Run with NERSC")
    parser.add_argument("--eval_loss", action="store_true", default=False)
    args: argparse.Namespace
    override_args: list[str]
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    # config['dataset']['test'] = {'src' : test_dataset}
    config['is_debug'] = True
    config['optim']['eval_batch_size'] = batch_size
    config['dataset']['test'] = {'src': config['dataset']['val']['src'][:-3] + 'train'}
    print(config['dataset']['test'])
    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    with new_trainer_context(config=config, distributed=False) as ctx:
        trainer = ctx.trainer
        trainer.load_checkpoint(args.checkpoint)
        evaluate_model(trainer, args.eval_loss)