#!/usr/bin/env bash

#SBATCH --job-name=h0
#SBATCH --chdir=/home/lucentlab/wmahler/Stochastic-Tussock-Model
#SBATCH --output=h0.out
#SBATCH --error=h0.err
#SBATCH --time=1000:00:00
#SBATCH --nodelist=lgmem-01
#SBATCH --exclusive

set -euo pipefail

source /home/lucentlab/wmahler/miniconda3/etc/profile.d/conda.sh
conda activate tussock

CONFIG_DIR="${1:-parameterization_configs/h0}"
PARAM_SCRIPT="${2:-scripts/parameterization.py}"

for cfg in "$CONFIG_DIR"/*.ini; do
    [ -e "$cfg" ] || continue
    echo "========================================"
    echo "Running parameterization for: $cfg"
    echo "========================================"
    python "$PARAM_SCRIPT" --config "$cfg" --site "TL"
done