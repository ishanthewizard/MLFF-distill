
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
from src.distill_utils import print_cuda_memory_usage
def evaluate_model(trainer):
    loss_fn = L2MAELoss()
    losses = []
    trainer.model.eval()
    start = None
    start_record_idx = 15
    batch_size = trainer.config["optim"].get("eval_batch_size", trainer.config["optim"]["batch_size"])
    print(f"BATCH SIZE: {batch_size}")
    for i, batch in enumerate(tqdm(trainer.test_loader)):
        if i >= start_record_idx and start == None:
            start = time.time()
        forces = trainer._forward(batch)["forces"]
        # losses.append(loss_fn(forces, batch["forces"]))

    iter_per_s = (len(trainer.test_loader) - start_record_idx) / (time.time() -  start)

    print(f"INTERATIONS PER SECOND: {iter_per_s}")
    print(f"THROUGHPUT PER SECOND: {iter_per_s * batch_size}")
    # print(f"FORCE MAE: {np.mean(np.array(losses)) * 100} meV")
if __name__ == "__main__":
    # config_path = 'configs/SPICE/monomers/distill/gemnet-dT-small.yml' # you need to supply this at the command line
    checkpoint_path = 'checkpoints/2024-09-05-13-45-52-solvated-gemSmall-DIST-4s/checkpoint.pt'
    test_dataset  = '/data/ishan-amin/spice_separated/Solvated_Amino_Acids/train'
    parser: argparse.ArgumentParser = flags.get_parser()
    parser.add_argument("--nersc", action="store_true", help="Run with NERSC")
    args: argparse.Namespace
    override_args: list[str]
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    config['dataset']['test'] = {'src' : test_dataset}
    config['is_debug'] = True
    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    with new_trainer_context(config=config, distributed=False) as ctx:
        trainer = ctx.trainer
        trainer.load_checkpoint(checkpoint_path)
        evaluate_model(trainer)