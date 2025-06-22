"""
Adopted from fairchem.core._cli.py
Add more Slurm configs
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid
import tempfile
import argparse
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from submitit import AutoExecutor
from fairchem.core._cli import (
    SchedulerType,
    DeviceType,
    RunType,
    DistributedInitMethod,
    Metadata,
    Submitit,
    get_timestamp_uid,
    get_commit_hash,
    get_cluster_name,
    LOG_DIR_NAME,
    CHECKPOINT_DIR_NAME,
    RESULTS_DIR,
    CONFIG_FILE_NAME,
    PREEMPTION_STATE_DIR_NAME,
    get_canonical_config,
    _runner_wrapper,
)
from omegaconf import OmegaConf
from fairchem.core.common import distutils
import multiprocessing as mp


@dataclass
class SlurmConfig:
    mem_gb: int = 80
    timeout_hr: int = 168
    cpus_per_task: int = 8
    partition: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    qos: Optional[str] = None  # omegaconf in python 3.9 does not backport annotations
    account: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    
    # added for germain
    partition: Optional[str] = (
        None 
    )
    # timeout_hr: Optional[str] = (
    #     None  # omegaconf in python 3.9 does not backport annotations
    # )
    nodelist: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    constraint: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    exclude: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )


@dataclass
class SchedulerConfig:
    mode: SchedulerType = SchedulerType.LOCAL
    distributed_init_method: DistributedInitMethod = DistributedInitMethod.TCP
    ranks_per_node: int = 1
    num_nodes: int = 1
    num_array_jobs: int = 1
    slurm: SlurmConfig = field(default_factory=lambda: SlurmConfig)


@dataclass
class JobConfig:
    run_name: str = field(
        default_factory=lambda: get_timestamp_uid() + uuid.uuid4().hex.upper()[0:4]
    )
    timestamp_id: str = field(default_factory=lambda: get_timestamp_uid())
    run_dir: str = field(default_factory=lambda: tempfile.TemporaryDirectory().name)
    device_type: DeviceType = DeviceType.CUDA
    debug: bool = False
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig)
    logger: Optional[dict] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    seed: int = 0
    deterministic: bool = False
    runner_state_path: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    # read-only metadata about the job, not user inputs
    metadata: Optional[Metadata] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    graph_parallel_group_size: Optional[int] = None

    def __post_init__(self) -> None:
        self.run_dir = os.path.abspath(self.run_dir)
        self.metadata = Metadata(
            commit=get_commit_hash(),
            log_dir=os.path.join(self.run_dir, self.timestamp_id, LOG_DIR_NAME),
            checkpoint_dir=os.path.join(
                self.run_dir, self.timestamp_id, CHECKPOINT_DIR_NAME
            ),
            results_dir=os.path.join(self.run_dir, self.timestamp_id, RESULTS_DIR),
            config_path=os.path.join(self.run_dir, self.timestamp_id, CONFIG_FILE_NAME),
            preemption_checkpoint_dir=os.path.join(
                self.run_dir,
                self.timestamp_id,
                CHECKPOINT_DIR_NAME,
                PREEMPTION_STATE_DIR_NAME,
            ),
            cluster_name=get_cluster_name(),
        )


def get_hydra_config_from_yaml(
    config_yml: str, overrides_args: list[str]
) -> DictConfig:
    # Load the configuration from the file
    os.environ["HYDRA_FULL_ERROR"] = "1"
    config_directory = os.path.dirname(os.path.abspath(config_yml))
    config_name = os.path.basename(config_yml)
    hydra.initialize_config_dir(config_directory, version_base="1.1")
    cfg = hydra.compose(config_name=config_name, overrides=overrides_args)
    # merge default structured config with initialized job object
    cfg = OmegaConf.merge({"job": OmegaConf.structured(JobConfig)}, cfg)
    # canonicalize config (remove top level keys that just used replacing variables)
    return get_canonical_config(cfg)


def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str, required=True)
        args, override_args = parser.parse_known_args()

    cfg = get_hydra_config_from_yaml(args.config, override_args)
    log_dir = cfg.job.metadata.log_dir
    os.makedirs(cfg.job.run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    OmegaConf.save(cfg, cfg.job.metadata.config_path)
    logging.info(f"saved canonical config to {cfg.job.metadata.config_path}")

    scheduler_cfg = cfg.job.scheduler
    logging.info(f"Running fairchemv2 cli with {cfg}")
    if scheduler_cfg.mode == SchedulerType.SLURM:  # Run on cluster
        assert (
            os.getenv("SLURM_SUBMIT_HOST") is None
        ), "SLURM DID NOT SUBMIT JOB!! Please do not submit jobs from an active slurm job (srun or otherwise)"
        executor = AutoExecutor(folder=log_dir, slurm_max_num_timeout=3)
        executor.update_parameters(
            name=cfg.job.run_name,
            mem_gb=scheduler_cfg.slurm.mem_gb,
            timeout_min=scheduler_cfg.slurm.timeout_hr * 60,
            slurm_partition=scheduler_cfg.slurm.partition,
            gpus_per_node=scheduler_cfg.ranks_per_node,
            cpus_per_task=scheduler_cfg.slurm.cpus_per_task,
            tasks_per_node=scheduler_cfg.ranks_per_node,
            nodes=scheduler_cfg.num_nodes,
            slurm_qos=scheduler_cfg.slurm.qos,
            slurm_account=scheduler_cfg.slurm.account,
            slurm_nodelist=scheduler_cfg.slurm.nodelist,
            slurm_constraint=scheduler_cfg.slurm.constraint,
            slurm_exclude=scheduler_cfg.slurm.exclude,
        )
        if scheduler_cfg.num_array_jobs == 1:
            job = executor.submit(Submitit(), cfg)
            logging.info(
                f"Submitted job id: {cfg.job.timestamp_id}, slurm id: {job.job_id}, logs: {cfg.job.metadata.log_dir}"
            )
            jobs = [job]
        elif scheduler_cfg.num_array_jobs > 1:
            executor.update_parameters(
                slurm_array_parallelism=scheduler_cfg.num_array_jobs,
            )

            jobs = []
            with executor.batch():
                for job_number in range(scheduler_cfg.num_array_jobs):
                    _cfg = cfg.copy()
                    _cfg.job.metadata.array_job_num = job_number
                    job = executor.submit(Submitit(), _cfg)
                    jobs.append(job)
            logging.info(f"Submitted {len(jobs)} jobs: {jobs[0].job_id.split('_')[0]}")

        if "reducer" in cfg:
            job_id = jobs[0].job_id.split("_")[0]
            executor.update_parameters(
                name=f"{cfg.job.run_name}_reduce",
                # set a single node, or do we want the same config as the Runner or a separate JobConfig
                nodes=1,
                slurm_dependency=f"afterok:{job_id}",
                slurm_additional_parameters={
                    "kill-on-invalid-dep": "yes"
                },  # kill the reducer if run fails
            )
            executor.submit(Submitit(), cfg, RunType.REDUCE)
    else:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        assert (
            (scheduler_cfg.num_nodes) <= 1
        ), f"You cannot use more than one node (scheduler_cfg.num_nodes={scheduler_cfg.num_nodes}) in LOCAL mode"
        if scheduler_cfg.ranks_per_node > 1:
            logging.info(
                f"Running in local mode with {scheduler_cfg.ranks_per_node} ranks using device_type:{cfg.job.device_type}"
            )
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=scheduler_cfg.ranks_per_node,
                rdzv_backend="c10d",
                max_restarts=0,
            )
            elastic_launch(launch_config, _runner_wrapper)(cfg)
            if "reducer" in cfg:
                elastic_launch(launch_config, _runner_wrapper)(cfg, RunType.REDUCE)
        else:
            logging.info("Running in local mode without elastic launch")
            distutils.setup_env_local()
            Submitit()(cfg)
            if "reducer" in cfg:
                Submitit()(cfg, RunType.REDUCE)


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    os.environ["WANDB_MODE"] = "disabled"
    OmegaConf.register_new_resolver("merge", lambda x, y : x + y)
    main()