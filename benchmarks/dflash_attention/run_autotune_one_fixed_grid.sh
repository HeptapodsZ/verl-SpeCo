#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

source ~/.bashrc
source ~/venvs/cuda-triton/bin/activate
cd "${REPO_ROOT}"

# Keep an external production tuning profile from changing the teaching run.
unset VERL_SPECO_DFLASH_TUNING_PROFILE

python benchmarks/dflash_attention/autotune_one_fixed_grid.py "$@"
