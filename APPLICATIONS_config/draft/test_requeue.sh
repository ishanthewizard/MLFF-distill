#!/bin/bash

# === SLURM Job Parameters ===
#SBATCH --account=m4558
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --job-name=fl2
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --qos=preempt                  # preemptible queue
#SBATCH --time=00:03:00                # short for testing
#SBATCH --signal=B:USR1@60             # send USR1 to batch script 60s before timeout
#SBATCH --requeue                      # auto requeue on PREEMPT
#SBATCH --open-mode=append
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/test_log/md_flex_%x_%j.out
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/test_log/md_flex_%x_%j.err

# === Timeout handler: called when USR1 is delivered ===
timeout_handler() {
    echo "[$(date)] ENTERED timeout_handler: approaching time limit"
    echo "[$(date)] SLURM_JOB_ID=$SLURM_JOB_ID, SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"

    # 1) Requeue ASAP for timeout case
    echo "[$(date)] Calling: scontrol requeue $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID" || echo "[$(date)] WARNING: scontrol requeue FAILED"

    # 2) Try to stop Python cleanly
    if [[ -n "$PY_PID" ]]; then
        echo "[$(date)] Sending SIGTERM to python process PID $PY_PID"
        kill -TERM "$PY_PID"
        echo "[$(date)] Sleeping 20s to give Python time to clean up..."
        sleep 20
    else
        echo "[$(date)] WARNING: PY_PID is empty; nothing to kill"
    fi

    echo "[$(date)] timeout_handler exiting so requeued job can start later"
    exit 0
}

# === Preempt handler: called when SIGTERM is delivered (preemption) ===
preempt_handler() {
    echo "[$(date)] ENTERED preempt_handler: received SIGTERM (preemption)"
    if [[ -n "$PY_PID" ]]; then
        echo "[$(date)] Forwarding SIGTERM to python PID $PY_PID"
        kill -TERM "$PY_PID"
    else
        echo "[$(date)] WARNING: PY_PID empty in preempt_handler"
    fi
    # With --requeue, Slurm will requeue the job after this handler returns.
}

# === Install traps as early as possible ===
echo "[$(date)] Shell PID = $$"
echo "[$(date)] Installing signal traps..."

trap 'echo "[$(date)] DEBUG: SIGUSR1 received by bash"; timeout_handler' USR1
trap 'preempt_handler' SIGTERM

echo "[$(date)] Traps for USR1 and SIGTERM are active."

# === Environment Setup and Logging ===
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Requeue Count: ${SLURM_RESTART_COUNT:-0}"
echo "=========================================="

# === Change to Working Directory ===
cd /global/homes/y/yuejian/project/MLFF-distill

# === Run your MD script ===
python APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py \
    /global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_speed/requeue/md_omol_naotf_diglyme_1m_s1p1 \
    --models /global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_speed/student_ckpt/inference_ckpt.pt \
    --steps 10000000 \
    --interval 10 &

PY_PID=$!
echo "[$(date)] Launched python, PID=$PY_PID"

wait $PY_PID
EXIT_CODE=$?

echo "[$(date)] Python finished with exit code $EXIT_CODE"
exit $EXIT_CODE