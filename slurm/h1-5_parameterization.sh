#!/usr/bin/env bash

#SBATCH --job-name=h1-5
#SBATCH --chdir=/home/lucentlab/wmahler/Stochastic-Tussock-Model
#SBATCH --output=h1-5.%j.batch.out
#SBATCH --error=h1-5.%j.batch.err
#SBATCH --time=1000:00:00
#SBATCH --partition=lgmem
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --nodelist=lgmem-01,lgmem-02

set -euo pipefail

CONFIG_DIR="${1:-parameterization_configs/h1-5_configs}"
PARAM_SCRIPT="${2:-scripts/parameterization.py}"

export CONFIG_DIR
export PARAM_SCRIPT

echo "Batch script running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Allocated nodes: ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "CONFIG_DIR: ${CONFIG_DIR}"
echo "PARAM_SCRIPT: ${PARAM_SCRIPT}"
echo "========================================"

# Make sure the config files exist before launching workers.
TOTAL_CONFIGS=$(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.ini" | wc -l)

echo "Total configs found by batch script: ${TOTAL_CONFIGS}"

if [ "$TOTAL_CONFIGS" -eq 0 ]; then
    echo "ERROR: No .ini config files found in ${CONFIG_DIR}"
    exit 1
fi

# Remove old worker logs from previous runs of this same job name if desired.
# Comment these out if you want to keep all previous worker logs.
rm -f "worker-${SLURM_JOB_ID}"-*.out "worker-${SLURM_JOB_ID}"-*.err

srun \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --cpus-per-task=80 \
    --cpu-bind=none \
    --kill-on-bad-exit=0 \
    --label \
    --output="worker-${SLURM_JOB_ID}-%N-%t.out" \
    --error="worker-${SLURM_JOB_ID}-%N-%t.err" \
    bash -lc '
set -euxo pipefail

echo "========================================"
echo "Worker starting"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-missing}"
echo "SLURM_PROCID=${SLURM_PROCID:-missing}"
echo "SLURM_LOCALID=${SLURM_LOCALID:-missing}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-missing}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-missing}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "PARAM_SCRIPT=${PARAM_SCRIPT}"
echo "========================================"

echo "Checking paths on $(hostname)"
ls -ld "$CONFIG_DIR"
ls -l "$PARAM_SCRIPT"
ls -l /home/lucentlab/wmahler/miniconda3/etc/profile.d/conda.sh

source /home/lucentlab/wmahler/miniconda3/etc/profile.d/conda.sh
conda activate tussock

echo "Python path on $(hostname): $(which python)"
python --version

mapfile -t ALL_CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.ini" | sort)

TOTAL=${#ALL_CONFIGS[@]}
NTASKS=${SLURM_NTASKS}
RANK=${SLURM_PROCID}

START=$(( RANK * TOTAL / NTASKS ))
END=$(( (RANK + 1) * TOTAL / NTASKS ))

echo "Total configs visible on $(hostname): $TOTAL"
echo "Worker rank: $RANK"
echo "Worker host: $(hostname)"
echo "This worker handles indices: $START to $((END - 1))"

if [ "$START" -lt "$END" ]; then
    echo "First assigned config: ${ALL_CONFIGS[$START]}"
else
    echo "No configs assigned to this worker"
fi

echo "========================================"

for ((i=START; i<END; i++)); do
    cfg="${ALL_CONFIGS[$i]}"
    echo "[rank $RANK / host $(hostname)] Starting config index $i: $cfg"
    python -u "$PARAM_SCRIPT" --config "$cfg" --site "TL"
    echo "[rank $RANK / host $(hostname)] Finished config index $i: $cfg"
done

echo "[rank $RANK / host $(hostname)] Finished all assigned configs"
'

STATUS=$?

echo "========================================"
echo "srun status: ${STATUS}"

echo "Worker logs produced:"
ls -lh "worker-${SLURM_JOB_ID}"-*.out "worker-${SLURM_JOB_ID}"-*.err || true

if [ "$STATUS" -ne 0 ]; then
    echo "One or more workers failed."
    exit 1
fi

echo "All parameterization jobs completed successfully."