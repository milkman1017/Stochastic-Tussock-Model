#!/bin/bash
#SBATCH --job-name=ecotype_param
#SBATCH --chdir=/home/lucentlab/wmahler/Stochastic-Tussock-Model
#SBATCH --output=eco_param.out
#SBATCH --error=eco_param.err
#SBATCH --time=500:00:00
#SBATCH --nodelist=lgmem-02

set -euo pipefail

source /home/lucentlab/wmahler/miniconda3/etc/profile.d/conda.sh
conda activate tussock

PY="$CONDA_PREFIX/bin/python"

echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "PYTHON (env): $PY"
echo "PYTHON (which): $(which python || true)"
"$PY" -V
"$PY" -c "import sys; print('executable:', sys.executable); import pandas as pd; print('pandas:', pd.__version__)"

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "START: $(date)"

# ---- build model (force rebuild) ----
cd model
echo "=== cleaning + building ==="
make clean
make

echo "=== built binary ==="
ls -lh ./tussock_model
sha256sum ./tussock_model
echo "MTIME: $(stat -c %y ./tussock_model)"
cd ..

OUTBASE="parameterization_outputs"
if [ -d "$OUTBASE" ]; then
  echo "=== removing old outputs: $OUTBASE ==="
  rm -rf "$OUTBASE"
fi

echo "=== running parameterization ==="
"$PY" scripts/parameterization.py --sites Toolik_7

echo "END: $(date)"
