# `_forward_grid_stride_kernel` / `one_fixed_grid` autotune experiment

## 1. Implementation summary

This experiment is deliberately limited to the DFlash forward
`_forward_grid_stride_kernel` with `ANCHOR_MAJOR=True`, which is the
`triton_one_fixed_grid` work ordering. It does not modify the production kernel,
dispatcher, model, backward kernels, or production tuning profiles.

### Changed files

| File | Purpose |
|---|---|
| `benchmarks/dflash_attention/autotune_one_fixed_grid.py` | Imports the real `_forward_grid_stride_kernel`, wraps it with `triton.autotune`, builds representative contiguous DFlash inputs, launches only the `one_fixed_grid` mapping, checks the selected output against SDPA, measures production-default baseline versus the autotuned winner, and compares the winner with the SDPA backend for latency and peak allocated memory. |
| `benchmarks/dflash_attention/run_autotune_one_fixed_grid.sh` | Enters the repository, activates the required WSL CUDA/Triton environment, removes any external production tuning-profile override, and runs both comparisons. Extra CLI arguments are forwarded to the Python script. |
| `benchmarks/dflash_attention/README_autotune_one_fixed_grid.md` | Records the implementation, methodology, measured result, limitations, and exact reproduction command. |

### Autotune space

The default search contains 24 configurations:

```text
BLOCK_M          = 16              # fixed: exact DFlash draft block
BLOCK_N          = {32, 64, 128}
num_warps        = {4, 8}
num_stages       = {1, 2}
PIPELINE_STAGES  = num_stages
NUM_PROGRAMS     = {40, 80}
ANCHOR_MAJOR     = True
```

`NUM_PROGRAMS` controls both `gridDim.x` and the grid-stride step, so every
candidate remains a valid fixed-worker-pool implementation. The autotune key
includes the complete Q/K/V workload identity needed by this standalone run.

The baseline is the current production default for this variant:

```text
BLOCK_M=16, BLOCK_N=64, num_warps=4, num_stages=2, NUM_PROGRAMS=40
```

### Correctness contract

The script constructs the same DFlash mask used by the repository:

- an active block sees context `[0, anchor)` and its complete local draft block;
- the first block is a dummy and each query sees only its matching local key;
- dropout is zero and the scale is `1/sqrt(head_dim)`;
- GQA K/V heads are repeated for the SDPA reference.

After `triton.autotune` selects the lowest-latency candidate, the selected output
is compared with PyTorch SDPA using `atol=rtol=2e-2`. The script reports
`allclose`, maximum absolute error, relative L2 error, and cosine similarity; it
raises `AssertionError` if `allclose` fails.

Important: standard `triton.autotune` selects by performance and the SDPA gate is
then applied to the winner. It does not correctness-gate every candidate and it
does not tune the shared backward kernels. The production offline tuner remains
the appropriate tool for correctness-gated forward/backward profile generation.

## 2. Performance results

### Workload and method

Measured on 2026-08-17:

```text
GPU:            NVIDIA GeForce RTX 5080
Batch:          1
Context length: 512
Anchors:        64
Block size:     16
Query length:   1024
Hq/Hkv:         32/8
Head dimension: 128
Dtype:          bfloat16
Anchor layout:  uniformly spaced sorted anchors; first block is dummy
Warmup:         10 launches per latency path
Measurement:    5 CUDA Event rounds, 100 launches per round
Kernel timing:  O and LSE preallocated outside measured regions
Backend calls:  output allocation performed inside each call
```

Compilation and autotune search time are excluded from the reported steady-state
kernel latency. Baseline and tuned use identical tensors, output buffers, timing
code, and `ANCHOR_MAJOR=True` mapping.

The best-config-versus-SDPA comparison invokes complete attention backend calls.
The tuned call allocates O and LSE. The SDPA dense boolean mask is constructed
once and remains resident, while GQA K/V `repeat_interleave` and SDPA's output and
workspace allocations remain inside the call. Latency is CUDA-event device time,
so Python dispatch and host-side allocator overhead are not included. Peak memory is
`torch.cuda.max_memory_allocated()` for one forward after synchronization, Python
GC, `torch.cuda.empty_cache()`, and reset of peak statistics. Resident Q/K/V and
backend structural data are included; the per-call delta is also reported. The
tuned peak is measured before the SDPA-only dense mask is materialized.

### Baseline versus tuned

| Variant | `BLOCK_N` | Warps | Stages | Programs | Forward p50 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Production-default baseline | 64 | 4 | 2 | 40 | 0.218270 ms | 1.0000x |
| `triton.autotune` winner | 64 | 4 | 2 | 80 | 0.127229 ms | **1.7156x** |

Raw per-round means in milliseconds:

```text
baseline:
  0.2182697678, 0.2203727913, 0.2168611145, 0.2204118347, 0.2136617661

tuned:
  0.1171001625, 0.1272288036, 0.1284825611, 0.1283513641, 0.1260803223
```

### Autotuned best config versus SDPA

This backend-call comparison is intentionally separate from the preallocated
kernel-only table above. Allocations made by each call are captured by the memory
metric; latency remains CUDA-event device time.

| Backend | Forward p50 | Relative latency | Peak allocated | Per-call peak delta |
|---|---:|---:|---:|---:|
| Autotuned `one_fixed_grid` best | 0.139421 ms | 1.0000x | 22.126 MiB | 8.125 MiB |
| PyTorch SDPA | 0.979469 ms | 7.0252x slower | 50.501 MiB | 35.000 MiB |

For this workload, the tuned backend is **7.0252x faster** than SDPA and lowers
peak allocated memory by **28.375 MiB (56.19%)**. Raw per-round latency means:

```text
tuned best:
  0.1401276779, 0.1483676815, 0.1394214439, 0.1373043156, 0.1256886387

SDPA:
  0.9250201416, 0.9794694519, 0.9981839752, 0.9987609863, 0.9367225647
```

The selected kernel passed the SDPA output gate:

```text
allclose:    True
max_abs:     0.00390625
relative_l2: 0.0007560361
cosine:      0.9999997616
```

The measured winner changes only the fixed worker-pool size from 40 to 80; its
tile, warp count, and pipeline depth match the production default. This is an
attention-forward microbenchmark on one GPU. It is not evidence of the same
speedup for backward, the complete DFlash module, drafter training, multi-GPU
training, or end-to-end RLHF throughput.

## 3. Reproduction

Run from Windows through WSL, or from an existing WSL shell. The executable
wrapper `benchmarks/dflash_attention/run_autotune_one_fixed_grid.sh` locates the
repository relative to itself, activates the workspace-required virtual
environment, and reports both baseline-vs-tuned and tuned-best-vs-SDPA results:

```bash
cd /mnt/d/Code_projects/VeRL/verl_speco/verl-SpeCo
bash benchmarks/dflash_attention/run_autotune_one_fixed_grid.sh
```

The exact default experiment is equivalent to:

```bash
source ~/.bashrc
source ~/venvs/cuda-triton/bin/activate
cd /mnt/d/Code_projects/VeRL/verl_speco/verl-SpeCo
unset VERL_SPECO_DFLASH_TUNING_PROFILE

python benchmarks/dflash_attention/autotune_one_fixed_grid.py \
  --batch-size 1 \
  --context-len 512 \
  --num-anchors 64 \
  --query-heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype bfloat16 \
  --block-ns 32,64,128 \
  --warp-counts 4,8 \
  --stage-counts 1,2 \
  --grid-sizes 40,80 \
  --baseline-grid-size 40 \
  --warmup 10 \
  --iterations 100 \
  --rounds 5 \
  --seed 0
```

For a quick integration smoke test rather than a performance result:

```bash
bash benchmarks/dflash_attention/run_autotune_one_fixed_grid.sh \
  --context-len 17 \
  --num-anchors 3 \
  --query-heads 4 \
  --kv-heads 2 \
  --head-dim 64 \
  --block-ns 32 \
  --warp-counts 4 \
  --stage-counts 1 \
  --grid-sizes 2 \
  --baseline-grid-size 2 \
  --warmup 2 \
  --iterations 10 \
  --rounds 3
```

Results can vary with GPU architecture, clocks, temperature, driver, PyTorch,
Triton version, background load, and tensor shape. Rerun tuning independently
for each target workload and GPU class.
