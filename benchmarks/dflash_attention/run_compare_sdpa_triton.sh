source ~/.bashrc
source ~/venvs/cuda-triton/bin/activate
cd /mnt/d/Code_projects/VeRL/verl_speco/verl-SpeCo


for fixed_grid_size in 40 80; do
  python benchmarks/dflash_attention/compare_sdpa_triton.py \
    --batch-sizes 1 \
    --context-lens 512,2048,8192,16384,32768,65536 \
    --block-sizes 16 \
    --num-anchors 64 \
    --query-heads 32 \
    --kv-heads 8 \
    --head-dim 128 \
    --fixed-grid-size "${fixed_grid_size}" \
    --backends sdpa,triton_one_fixed_grid \
    --dtype bfloat16 \
    --anchor-distribution uniform \
    --seed 0 \
    --warmup 10 \
    --rounds 5 \
    --min-iterations 20 \
    --min-seconds 2 \
    --output "benchmark_results/dflash_attention_fixed_grid_${fixed_grid_size}_compare_rtx5080.json"
done
