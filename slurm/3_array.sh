#!/usr/bin/env bash

set -euo pipefail

CONFIG_DIR="${1:-parameterization_configs/h3_configs}"
PARAM_SCRIPT="${2:-scripts/parameterization.py}"
MAX_CONCURRENT="${3:-16}"

mkdir -p logs

CONFIG_LIST="configs_h3_$(date +%Y%m%d_%H%M%S).txt"

find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.ini" | sort > "$CONFIG_LIST"

TOTAL=$(wc -l < "$CONFIG_LIST")

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No .ini files found in $CONFIG_DIR"
    exit 1
fi

LAST_INDEX=$((TOTAL - 1))

echo "Found $TOTAL configs"
echo "Config list: $CONFIG_LIST"
echo "Submitting array: 0-${LAST_INDEX}%${MAX_CONCURRENT}"
echo "Parameter script: $PARAM_SCRIPT"
echo "========================================"

sbatch \
    --array=0-"$LAST_INDEX"%"$MAX_CONCURRENT" \
    slurm/run_one_array.sh "$CONFIG_LIST" "$PARAM_SCRIPT"

    