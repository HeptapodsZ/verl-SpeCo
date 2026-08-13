# DFlash SDPA vs Triton benchmark

## Purpose

`compare_sdpa_triton.py` is a focused attention-core benchmark for the two
DFlash backends. It measures only:

- forward latency, useful throughput, and peak CUDA memory;
- forward-plus-backward latency, useful throughput, and peak CUDA memory.

SDPA and Triton run in separate Python processes. This prevents one backend's
CUDA caching allocator and compiled-kernel state from contaminating the other
backend's peak-memory measurement.

## Run

All project Python commands must run in the configured WSL environment:

```bash
source ~/.bashrc
source ~/venvs/cuda-triton/bin/activate
cd /mnt/d/Code_projects/VeRL/verl_speco/verl-SpeCo
```

Run the default six-context comparison:

```bash
python benchmarks/dflash_attention/compare_sdpa_triton.py \
  --batch-sizes 1 \
  --context-lens 512,2048,8192,16384,32768,65536 \
  --block-sizes 16 \
  --num-anchors 64 \
  --query-heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype bfloat16 \
  --anchor-distribution uniform \
  --seed 0 \
  --warmup 10 \
  --rounds 5 \
  --min-iterations 20 \
  --min-seconds 2 \
  --output benchmark_results/dflash_attention_sdpa_vs_triton_rtx5080.json
```

The command writes both JSON and CSV:

- `benchmark_results/dflash_attention_sdpa_vs_triton_rtx5080.json` contains
  environment metadata, raw round samples, p50/p90/p99, throughput, memory,
  and direct comparison fields;
- `benchmark_results/dflash_attention_sdpa_vs_triton_rtx5080.csv` contains one
  side-by-side row per shape and phase.

A short integration smoke command is:

```bash
python benchmarks/dflash_attention/compare_sdpa_triton.py \
  --context-lens 512 \
  --warmup 2 --rounds 2 --min-iterations 2 --min-seconds 0.1 \
  --output /tmp/dflash_sdpa_vs_triton_smoke.json
```

Run the correctness contract separately before interpreting performance:

```bash
pytest -q tests/unit/test_dflash_attention_kernels.py -s
pytest -q tests/unit/test_dflash_attention_compare_benchmark.py
```

## Measurement semantics

The default shape is `B=1`, `A=64`, `block_size=16`, `Lq=1024`,
`Hq/Hkv=32/8`, `D=128`, bf16. Anchors are uniformly distributed from zero to
the context length and all blocks are kept.

- Latency is GPU elapsed time measured with CUDA events. Each reported round
  is the mean of multiple iterations; p50 is the median of five round means.
  Warmup and first compilation are excluded.
- Forward executes under `torch.no_grad()`.
- Forward-plus-backward creates a new graph and computes `dQ`, `dK`, and `dV`
  with `torch.autograd.grad` in every iteration.
- Query throughput is `B * Lq / p50_latency`. It is useful draft query tokens
  per second, not head-expanded tokens.
- Effective QK throughput counts only pairs allowed by the DFlash mask. For
  forward-plus-backward it represents useful pairs per completed training
  step, not a claim about total backward FLOPs.
- Before a one-iteration memory measurement, the worker synchronizes, runs
  Python GC, calls `torch.cuda.empty_cache()`, and resets peak statistics.
  Inputs and the persistent dense mask or structural metadata remain resident.
- `peak_allocated_mib` is the main comparable peak. The JSON/CSV also records
  the baseline, peak increment, and caching-allocator reserved memory.
- Dense-mask construction happens before latency timing. SDPA still reads the
  mask during attention, and its GQA `repeat_interleave` remains inside the
  measured call. The resident dense mask is included in absolute peak memory.

## RTX 5080 results

Measured on 2026-08-12 with:

- NVIDIA GeForce RTX 5080, compute capability 12.0, approximately 16 GiB;
- Python 3.12.3;
- PyTorch 2.12.1+cu130 and CUDA 13.0;
- Triton 3.7.1;
- repository commit `7f1e1f7270236ae8699239b6c125211c87df6a42`, plus the
  uncommitted benchmark changes described in this document.

### Latency and throughput

Latency is p50. `Kq/s` is thousands of useful draft query tokens per second;
`Gpair/s` is billions of effective visible QK pairs per second.

| C | Phase | SDPA ms | Triton ms | Triton speedup | SDPA Kq/s | Triton Kq/s | SDPA Gpair/s | Triton Gpair/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | Forward | 0.909 | 0.109 | 8.34x | 1126.0 | 9390.7 | 9.80 | 81.74 |
| 512 | F+B | 3.560 | 0.531 | 6.70x | 287.7 | 1928.1 | 2.50 | 16.78 |
| 2,048 | Forward | 1.840 | 0.272 | 6.78x | 556.5 | 3770.9 | 18.52 | 125.49 |
| 2,048 | F+B | 6.977 | 1.249 | 5.59x | 146.8 | 819.8 | 4.88 | 27.28 |
| 8,192 | Forward | 5.517 | 1.060 | 5.20x | 185.6 | 965.9 | 24.42 | 127.10 |
| 8,192 | F+B | 20.640 | 4.153 | 4.97x | 49.6 | 246.6 | 6.53 | 32.44 |
| 16,384 | Forward | 10.625 | 1.940 | 5.48x | 96.4 | 527.8 | 25.31 | 138.62 |
| 16,384 | F+B | 38.922 | 8.045 | 4.84x | 26.3 | 127.3 | 6.91 | 33.43 |
| 32,768 | Forward | 20.758 | 3.885 | 5.34x | 49.3 | 263.6 | 25.89 | 138.34 |
| 32,768 | F+B | 75.314 | 15.608 | 4.83x | 13.6 | 65.6 | 7.14 | 34.43 |
| 65,536 | Forward | 40.420 | 7.740 | 5.22x | 25.3 | 132.3 | 26.58 | 138.79 |
| 65,536 | F+B | 150.486 | 30.937 | 4.86x | 6.8 | 33.1 | 7.14 | 34.72 |

Observed on this matrix, Triton is `5.20-8.34x` faster for forward and
`4.83-6.70x` faster for forward-plus-backward. Throughput speedup is the same
ratio because both backends complete the same useful DFlash workload.

### Peak allocated memory

`Peak` includes resident inputs and mask/metadata. `Delta` is additional
allocated memory above that backend's pre-phase baseline. All values are MiB.

| C | Phase | SDPA peak | Triton peak | Peak reduction | SDPA delta | Triton delta |
|---:|---|---:|---:|---:|---:|---:|
| 512 | Forward | 58.5 | 30.1 | 48.5% | 35.0 | 8.1 |
| 512 | F+B | 126.8 | 45.3 | 64.3% | 103.3 | 23.3 |
| 2,048 | Forward | 94.0 | 36.1 | 61.6% | 62.0 | 8.1 |
| 2,048 | F+B | 186.3 | 56.3 | 69.8% | 154.3 | 28.3 |
| 8,192 | Forward | 232.0 | 60.1 | 74.1% | 171.0 | 8.1 |
| 8,192 | F+B | 420.3 | 104.3 | 75.2% | 359.3 | 52.3 |
| 16,384 | Forward | 416.0 | 92.1 | 77.9% | 314.0 | 8.1 |
| 16,384 | F+B | 732.3 | 168.3 | 77.0% | 630.3 | 84.3 |
| 32,768 | Forward | 784.0 | 156.1 | 80.1% | 602.0 | 8.1 |
| 32,768 | F+B | 1356.3 | 296.3 | 78.2% | 1174.3 | 148.3 |
| 65,536 | Forward | 1520.0 | 284.1 | 81.3% | 1178.0 | 8.1 |
| 65,536 | F+B | 2604.3 | 552.3 | 78.8% | 2262.3 | 276.3 |

The Triton forward delta remains about 8.1 MiB because it mainly allocates the
8 MiB output and 0.125 MiB FP32 LSE. SDPA's delta grows with context length due
to dense-mask handling, GQA K/V expansion, and general attention workspace.

## Validation performed

- Script compilation and focused aggregation tests: `2 passed`.
- Existing DFlash forward/backward correctness contracts: `7 passed`.
- Short isolated-worker benchmark smoke: passed for both backends and phases.
- Full six-context comparison: all 12 backend/case runs completed successfully.

Warnings about unavailable `flash_attn`, deprecated TorchScript APIs, and an
initial CUDA context were non-fatal and do not select either measured backend.

## Interpretation limits

These are local attention-core measurements, not complete `DFlashAttention`,
drafter-training, or end-to-end RLHF throughput. They do not include projection,
RMSNorm, RoPE, optimizer, rollout, feature transfer, or distributed waits.
Do not extrapolate the result to H100, B greater than one, other head layouts,
other block sizes, other anchor distributions, or another PyTorch SDPA release
without rerunning the same controlled comparison.
