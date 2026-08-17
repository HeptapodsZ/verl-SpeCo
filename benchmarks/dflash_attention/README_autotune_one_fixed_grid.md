# `_forward_grid_stride_kernel` / `one_fixed_grid` autotune experiment

## 1. Implementation summary

This experiment is deliberately limited to the DFlash forward
`_forward_grid_stride_kernel` with `ANCHOR_MAJOR=True`, which is the
`triton_one_fixed_grid` work ordering. It does not modify the production kernel,
dispatcher, model, backward kernels, or production tuning profiles.

### Changed files

| File | Purpose |
|---|---|
| `benchmarks/dflash_attention/autotune_one_fixed_grid.py` | Imports the real `_forward_grid_stride_kernel`, wraps it with `triton.autotune`, builds representative contiguous DFlash inputs, launches only the `one_fixed_grid` mapping, checks the selected output against SDPA, and measures production-default baseline versus the autotuned winner. |
| `benchmarks/dflash_attention/run_autotune_one_fixed_grid.sh` | Enters the repository, activates the required WSL CUDA/Triton environment, removes any external production tuning-profile override, and runs the experiment. Extra CLI arguments are forwarded to the Python script. |
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
Warmup:         10 launches
Measurement:    5 CUDA Event rounds, 100 launches per round
Allocation:     O and LSE preallocated outside measured regions
```

Compilation and autotune search time are excluded from the reported steady-state
kernel latency. Baseline and tuned use identical tensors, output buffers, timing
code, and `ANCHOR_MAJOR=True` mapping.

### Baseline versus tuned

| Variant | `BLOCK_N` | Warps | Stages | Programs | Forward p50 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Production-default baseline | 64 | 4 | 2 | 40 | 0.217265 ms | 1.0000x |
| `triton.autotune` winner | 64 | 4 | 2 | 80 | 0.122127 ms | **1.7790x** |

Raw per-round means in milliseconds:

```text
baseline:
  0.2172649574, 0.2106851196, 0.2182959938, 0.2164678383, 0.2189423943

tuned:
  0.1217712021, 0.1221273613, 0.1267084789, 0.1224342442, 0.1163779163
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

Run from Windows through WSL, or from an existing WSL shell. The wrapper locates
the repository relative to itself and activates the workspace-required virtual
environment:

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
