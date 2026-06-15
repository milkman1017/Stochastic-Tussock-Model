#!/usr/bin/env bash

#SBATCH --job-name=h3
#SBATCH --chdir=/home/lucentlab/wmahler/Stochastic-Tussock-Model
#SBATCH --output=logs/h3_%A_%a.out
#SBATCH --error=logs/h3%A_%a.err
#SBATCH --time=1000:00:00
#SBATCH --partition=lgmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=lgmem-01
#SBATCH --mail-user=wes.mahler1017@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

CONFIG_LIST="$1"
PARAM_SCRIPT="${2:-scripts/parameterization.py}"

source /home/lucentlab/wmahler/miniconda3/etc/profile.d/conda.sh
conda activate tussock

TASK_ID="${SLURM_ARRAY_TASK_ID}"

CONFIG=$(sed -n "$((TASK_ID + 1))p" "$CONFIG_LIST")

if [ -z "$CONFIG" ]; then
    echo "ERROR: No config found for SLURM_ARRAY_TASK_ID=$TASK_ID"
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-none}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Config list: $CONFIG_LIST"
echo "Config: $CONFIG"
echo "Parameter script: $PARAM_SCRIPT"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "========================================"

python -u "$PARAM_SCRIPT" --config "$CONFIG" --sites "TL"

echo "Finished config: $CONFIG"
echo "Finished on host: $(hostname)"