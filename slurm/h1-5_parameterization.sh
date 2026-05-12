#!/usr/bin/env bash

#SBATCH --job-name=h1-5
#SBATCH --chdir=/home/lucentlab/wmahler/Stochastic-Tussock-Model
#SBATCH --output=h1-5.out
#SBATCH --error=h1-5.err
#SBATCH --time=1000:00:00
#SBATCH --nodelist=lgmem-01,lgmem-02
#SBATCH --exclusive

set -euo pipefail

source /home/lucentlab/wmahler/miniconda3/etc/profile.d/conda.sh
conda activate tussock

CONFIG_DIR="${1:-parameterization_configs/h1-5_configs}"
PARAM_SCRIPT="${2:-scripts/parameterization.py}"

# Get all config files and split into two arrays for parallel processing on both nodes
declare -a ALL_CONFIGS=()
for cfg in "$CONFIG_DIR"/*.ini; do
    [ -e "$cfg" ] || continue
    ALL_CONFIGS+=("$cfg")
done

TOTAL=${#ALL_CONFIGS[@]}
HALF=$((TOTAL / 2))

echo "Total configs: $TOTAL"
echo "Splitting across 2 nodes: lgmem-01 (configs 0-$((HALF-1))), lgmem-02 (configs $HALF-$((TOTAL-1)))"
echo "========================================"

# Function to run configs on a specific node
run_configs_on_node() {
    local node=$1
    local start=$2
    local end=$3
    
    for ((i=start; i<end; i++)); do
        cfg="${ALL_CONFIGS[$i]}"
        echo "[Node: $node] Running parameterization for: $cfg"
        python "$PARAM_SCRIPT" --config "$cfg" --site "TL"
    done
}

# Run first half on lgmem-01 and second half on lgmem-02 in parallel
run_configs_on_node "lgmem-01" 0 $HALF &
PID1=$!

run_configs_on_node "lgmem-02" $HALF $TOTAL &
PID2=$!

# Wait for both to complete
wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

echo "========================================"
echo "lgmem-01 job status: $STATUS1"
echo "lgmem-02 job status: $STATUS2"

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "One or more jobs failed!"
    exit 1
fi

echo "All parameterization jobs completed successfully!"