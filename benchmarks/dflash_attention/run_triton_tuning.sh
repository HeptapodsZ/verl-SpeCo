#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODE="${DFLASH_TUNING_MODE:-smoke}"

source ~/.bashrc
source ~/venvs/cuda-triton/bin/activate
cd "${REPO_ROOT}"
unset VERL_SPECO_DFLASH_TUNING_PROFILE

case "${MODE}" in
  smoke)
    matrix_args=(
      --batch-sizes 1
      --context-lens 512
      --head-dims 64
      --warmup 3
      --rounds 3
      --min-iterations 10
      --min-seconds 0.05
      --output benchmark_results/dflash_triton_tuning_smoke.json
    )
    ;;
  standard)
    matrix_args=(
      --batch-sizes 1
      --context-lens 512,2048,8192
      --head-dims 64,128
      --warmup 5
      --rounds 5
      --min-iterations 20
      --min-seconds 0.2
      --output benchmark_results/dflash_triton_tuning_standard.json
    )
    ;;
  full)
    matrix_args=(
      --batch-sizes 1,2
      --context-lens 512,2048,8192,16384,32768,65536
      --head-dims 64,128
      --warmup 5
      --rounds 5
      --min-iterations 20
      --min-seconds 0.2
      --full-search
      --output benchmark_results/dflash_triton_tuning_full.json
    )
    ;;
  *)
    echo "Unknown DFLASH_TUNING_MODE=${MODE}; use smoke, standard, or full" >&2
    exit 2
    ;;
esac

python benchmarks/dflash_attention/tune_triton.py "${matrix_args[@]}" "$@"
