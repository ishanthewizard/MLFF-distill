# Guide For Distributed Training on NERSC

In this guide, we will show how to run distributed training on NERSC with the `submitit` package from FAIR.

Note: the `submitit` package auto checkpoint before 2 minutes of the job ending. It should auto requque the job after that, but this is not supported on NERSC. Therefore, now we have to manually resume the job after the checkpointing.

Step 1: Modify `configs/slurm/base.yml` to include the account and constraints for your job. (Optional: change the email for job notifications). Update the `configs/slurm/regular.yml` or `configs/slurm/debug.yml` to include the `qos` and `time` for your job.

Step 2: Include this config in the `base.yml` of the dataset like so:
```yaml
includes:
  - configs/slurm/regular.yml # or configs/slurm/debug.yml
```

Step 3: Run the following command to submit the job:
```bash
python main.py --distributed --num-gpus 4 --num-nodes N --submit --nersc --mode train --config-yml EXP_CONFIG_PATH --identifier NAME_OF_JOB --timestamp-id XXXX --checkpoint checkpoints/XXXX/checkpoint.pt
```

Explanation of the arguments:
- `--num-gpus 4`: Number of GPUs per node. In Perlmutter, each node has 4 GPUs.
- `--num-nodes N`: Number of nodes to use for distributed training. The total number of GPUs used will be `num-gpus * num-nodes`.
- `--nersc`: Flag to indicate that the job is being run on NERSC.
- `--identifier`: Name of the job in the Slurm queue.
- `--timestamp-id XXXX`: Recommended to set one for resuming training later.
- `--checkpoint checkpoints/XXXX/checkpoint.pt`: Path to the checkpoint to resume training from. If you are starting a new job, you can omit this argument.
