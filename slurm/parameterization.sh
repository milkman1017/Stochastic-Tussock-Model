#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-parameterization_configs/h1_configs}"
PARAM_SCRIPT="${2:-scripts/parameterization.py}"

for cfg in "$CONFIG_DIR"/*.ini; do
    [ -e "$cfg" ] || continue
    echo "========================================"
    echo "Running parameterization for: $cfg"
    echo "========================================"
    python "$PARAM_SCRIPT" --config "$cfg"
done