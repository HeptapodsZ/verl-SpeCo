#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
source ~/venvs/cuda-triton/bin/activate

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${1:-${REPO_ROOT}/benchmark_results/nsight}"
BACKEND="${BACKEND:-triton}"
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

cases=(
  "1 512 16"
  "1 8192 16"
  "1 65536 16"
)

for item in "${cases[@]}"; do
  read -r batch context block <<<"${item}"
  name="${BACKEND}_b${batch}_c${context}_bs${block}"
  python tests/special_standalone/dflash_attention_memory_preflight.py \
    --batch-size "${batch}" --context-len "${context}" --block-size "${block}" \
    --backend "${BACKEND}" --minimum-headroom 0.20 || continue
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output "${OUTPUT_ROOT}/${name}" \
    python benchmarks/dflash_attention/profile_case.py \
      --backend "${BACKEND}" --batch-size "${batch}" \
      --context-len "${context}" --block-size "${block}" --iterations 20
  ncu \
    --target-processes all \
    --profile-from-start off \
    --set full \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --force-overwrite \
    --export "${OUTPUT_ROOT}/${name}" \
    python benchmarks/dflash_attention/profile_case.py \
      --backend "${BACKEND}" --batch-size "${batch}" \
      --context-len "${context}" --block-size "${block}" --iterations 1
done
