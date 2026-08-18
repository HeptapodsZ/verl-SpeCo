# DFlash triton_npu_v1: Triton-Ascend Forward Kernel for Ascend NPU

Date: 2026-08-17 · Branch: feature/dflash_kernel_npu · Status: forward-only v1, correctness-validated on CUDA side, pending NPU hardware validation.

## 1. Implementation summary

New backend `triton_npu_v1` implements the DFlash block-sparse attention forward pass
with a Triton-Ascend kernel and is registered in the existing dispatch chain.

| File | Change |
| --- | --- |
| `verl_speco/models/dflash/kernels/triton_npu_attention.py` | **New.** `_forward_npu_kernel` (FlashAttention-v2-style persistent forward kernel), the raw launcher `triton_npu_dflash_attention_forward` (returns output + natural-log LSE), `default_npu_program_count()` (env `VERL_SPECO_DFLASH_NPU_PROGRAMS` → torch-npu core count → 20), and the `_TritonNpuDFlashAttention` autograd wrapper `triton_npu_dflash_attention` (NPU-gated, forward-only, backward raises `NotImplementedError`). |
| `verl_speco/models/dflash/kernels/dispatch.py` | Adds `triton_npu_v1` to `DFLASH_ATTENTION_BACKENDS` and `_CUSTOM_BACKENDS`. `_validate_custom_inputs` now accepts `cuda` or `npu` device types. Routing: `triton_npu_v1` is dispatched before the `startswith("triton")` branch, converts anchors/keep to int32, requires `query.device.type == "npu"` (fail closed), and calls `triton_npu_dflash_attention`. |
| `verl_speco/models/dflash/modeling_dflash.py` | Adds `triton_npu_v1` to the custom-backend branch, so `DFlashAttention.forward` passes anchors/keep/block_size as structural metadata instead of a materialized mask. |
| `verl_speco/backends/dflash_trainer_backend.py` | Adds `triton_npu_v1` to the `custom_attention_backend` tuple so training-side validation accepts it. |
| `verl_speco/config/speco_base.yaml` | Documents the new opt-in value next to `dflash_attention_backend`. |
| `benchmarks/dflash_attention/run_triton_npu_v1.py` | **New.** Standalone runner (section 3): NPU correctness vs the FP32 dense reference + LSE + latency; training-scale timing mode; `--smoke-cuda` compile/numerics smoke; `--direct` tuning mode for `--block-n` / `--num-programs`. |
| `docs/exp/dflash_triton_npu_v1_kernel.md` | This document. |

Behavior contract preserved: `auto` still routes to Flex/SDPA, all CUDA backends are untouched,
and the new backend is explicit opt-in. Non-NPU tensors are rejected with a clear error at the
dispatch boundary and inside the autograd wrapper.

## 2. Kernel design

### 2.1 DFlash attention recap (inputs, sparsity pattern)

- Q: `[B, HQ, draft_len, D]`, where `draft_len = num_anchors × block_size` and `block_size == 16`. Q rows come from draft hidden states only.
- K/V: `[B, HK, ctx_len + draft_len, D]` — context keys (target prefix) followed by draft keys; GQA with `groups = HQ / HK`.
- `anchor_positions`: `[B, num_anchors]`, ascending per batch, each anchor in `[0, ctx_len]`.
- `block_keep_mask`: `[B, num_anchors]`; `keep == 0` marks dummy blocks excluded from the loss.
- Sparse pattern (exact definition in `kernels/reference.py::build_dflash_dense_attention_mask`):
  - context region: query block `b` attends keys `[0, anchor_b)` — a variable-length prefix, so the block mask is a staircase when anchors are sorted;
  - draft region: query block `b` attends exactly its own 16 draft keys `[ctx_len + 16b, ctx_len + 16b + 16)` — block-diagonal, fully connected (non-causal), no cross-block leakage;
  - dummy rows attend only their own key (`ctx_len + q_index`), which keeps softmax finite and reproduces the SDPA contract.

### 2.2 Work decomposition

One program processes one logical work item `(batch, kv_head, GQA head chunk, anchor pair)`:

- `ANCHORS_PER_PROGRAM = 2` consecutive anchors per program. The two 16-token draft blocks form **one contiguous 32-row K/V tile** in memory, which removes the M-dim starvation problem of a bare 16-row block.
- `HEADS_PER_PROGRAM = min(2, groups)` query heads of one KV head share every K/V tile load (GQA without `repeat_kv` materialization), giving `QUERY_ROWS = 2 × 16 × 2 = 64` rows — a real Cube M extent.
- Grid: **persistent grid-stride** (`grid = NUM_PROGRAMS`, `for work_id in range(pid, TOTAL_WORK, NUM_PROGRAMS)`), exactly the pattern of the Triton-Ascend tutorial `04_fused_attention`. `NUM_PROGRAMS` defaults to the torch-npu core count (fallback 20), so the whole launch is one wave per NPU core instead of ~`B×HK×num_anchors` tiny programs.

### 2.3 Two phases with online softmax

Phase 1 — context prefix (staircase generalization of the tutorial's STAGE1/2 causal tiling):

```text
max_anchor = max(anchor_row)          # per-program loop bound
for start_n in range(0, min(max_anchor, ctx_len), BLOCK_N):
    k, v = load K/V tile [BLOCK_N, D]            # masked at the tail
    scores = dot(q, trans(k)) * SCALE
    allowed = query_valid & (offs_n < anchor_row) & keep_row     # per-row staircase
    scores = where(allowed, scores, -inf)
    # online softmax: m_ij = max(m_i, max(scores)); alpha = exp2(m_i - m_ij);
    # l_i = l_i*alpha + sum(p); acc = acc*alpha + dot(p, v)  (in-place dot accumulate)
```

Phase 2 — draft blocks: one contiguous `2×16` K/V tile, block-diagonal mask
(`same_anchor`), with dummy rows collapsed to their diagonal element. Every row has at
least one finite score here, so the softmax update is unconditional.

Numerics: fp32 online softmax with `exp2`/`log2e` rescaling (same scheme as the CUDA
Triton kernels); output = `acc / l`; LSE = `m + log(l)` stored in fp32 for the future
backward. Dummy rows keep the "empty identity" state (`has_context_tile` guard) through
the context phase so `(-inf) - (-inf)` never produces NaN — the correctness trap
identified in the design session.

### 2.4 NPU-fit evidence

Directly mirroring the Ascend tutorial and the prior design-session findings:

1. **Persistent grid** (`range(pid, TOTAL_WORK, NUM_PROGRAMS)`): identical to the tutorial's `_attn_fwd` scheduling loop; avoids multi-wave launch overhead on the NPU scheduler. `NUM_PROGRAMS` is core-count based (torch-npu property or the tutorial's 20-core constant).
2. **M-dim packing**: the tutorial's Cube unit needs `BLOCK_M ≥ 16`, and a single DFlash block is exactly 16 rows — with head/anchor packing the dot runs at 64 rows, while BLOCK_N ∈ {16,32,64} and HEAD_DIM ∈ {64,128} satisfy Cube 16×16×16 minimums in every supported shape.
3. **Contiguous head dim**: all 2-D tile loads (K/V context, local draft tile, Q gather, O store) keep `head_dim` as the innermost contiguous axis, which is the layout the tutorial selects with `order=(1, 0)` block pointers. v1 uses pointer arithmetic with masked loads instead of `tl.make_block_ptr` because masked block-pointer loads are rejected by the CUDA-side Triton used for the numerics smoke, and block-pointer `boundary_check` support on Triton-Ascend could not be verified without hardware; the accesses are the same contiguous pattern. Switching to block pointers / `order=(1,0)` is a measured-future refinement, not a requirement.
4. **Cube/Vector split**: `tl.dot` (Cube) for QK^T and P@V, Vector unit for max/sum/exp2 — the CV fusion pattern the tutorial relies on. `P@V` uses the in-place `tl.dot(p, v, acc)` accumulate form shown in the tutorial to cut one extra UB round trip.
5. **fp32 accumulation**: online softmax state and the P@V accumulator are fp32, matching the tutorial.
6. **Staircase mask**: `offs_n < anchor_row` is a tensor-vs-tensor int32 comparison evaluated per tile; per the design notes, comparisons that could scalarize on NPU are kept inside `tl.where`/mask selects on full tiles rather than scalar control flow. The only runtime scalar is the context loop bound.
7. **Addressing**: batch/KV-head base offsets are widened to int64 (tutorial pattern) so large KV buffers never overflow int32.
8. **UB budget**: worst registered tile (64 rows, D=128, BLOCK_N=64, fp16/fp32 mix: q 16 KB + k 16 KB + v 16 KB + scores 32 KB + acc 32 KB ≈ 112 KB) fits the 192 KB UB of Ascend 910-series parts; BLOCK_N is capped at 64 until NPU measurements confirm otherwise.

Conclusion: the design matches the NPU kernel style of the reference tutorial point by point
(persistent scheduling, packed Cube tiles, online softmax, contiguous-D tile layout, CV
fusion). What is **not** yet evidence: actual NPU codegen quality, UB/CV alignment of the
256-byte fp16/bf16 head-dim rows, and measured throughput — those require the Triton-Ascend
toolchain and an Ascend part.

### 2.5 Known limitations and next steps

- **Forward-only**: backward raises `NotImplementedError`. DFlash is a training-time draft model, so `triton_npu_v1` cannot run a training step yet; it is usable for eval/benchmarking of the forward path. Next milestone is the dQ/dK/dV backward (pull schedule, reuse of the CUDA backward structure with LSE saved from forward).
- Default `BLOCK_N=64`, `NUM_PROGRAMS` auto. Both are tunable (`--direct` mode / env var); on-hardware tuning should sweep them plus block-pointer loads and `num_stages` pipelining.
- `head_dim=128` fp16/bf16 rows are 256 B — below the 512 B CV alignment sweet spot noted in the design session; needs on-NPU profiling before drawing conclusions.

## 3. Running the kernel standalone

On an Ascend NPU node with `torch_npu` and Triton-Ascend (triton-ascend-3.2.2):

```bash
# Correctness vs the FP32 dense reference + LSE + latency (small shapes)
python benchmarks/dflash_attention/run_triton_npu_v1.py

# Training-scale forward timing (ctx=8192, anchors=512, 32/8 heads, D=128)
python benchmarks/dflash_attention/run_triton_npu_v1.py --large \
    --ctx-len 8192 --num-anchors 512 --query-heads 32 --kv-heads 8 --iters 50

# Tuning knobs (raw launcher; program count and K-tile width)
python benchmarks/dflash_attention/run_triton_npu_v1.py --direct --block-n 32 --num-programs 40

# Compile + numerics smoke of the same kernel text on a CUDA dev box
# (validates the Triton frontend and the algorithm, NOT NPU codegen/performance)
python benchmarks/dflash_attention/run_triton_npu_v1.py --smoke-cuda

# Program count override without touching the launch config
VERL_SPECO_DFLASH_NPU_PROGRAMS=40 python benchmarks/dflash_attention/run_triton_npu_v1.py --large ...
```

Through the public API:

```python
from verl_speco.models.dflash.kernels import dflash_sparse_attention

output = dflash_sparse_attention(
    q, k, v, anchor_positions, block_keep_mask,
    ctx_len=ctx_len, block_size=16, backend="triton_npu_v1",
)
```

In training, opt in with `drafter.dflash_attention_backend: triton_npu_v1` (forward-only;
backward will raise).

## 4. Validation status

What was run (WSL, Python 3.12.3, torch 2.12.1+cu130, Triton 3.7.1, RTX 5080) and passed:

- **Numerics smoke of the exact kernel text on CUDA**, 12 cases vs the FP32 dense reference
  (`dense_dflash_attention_reference`), gated at `atol/rtol 2e-2`, `rel_l2 5e-3`, cosine
  `0.9999`, LSE `max_abs < 2e-2`, zero NaN/Inf: default bf16 GQA, fp16 D=64, MHA (groups=1),
  ctx=0, odd anchor counts (5, 1), non-aligned ctx (100), all-dummy, no-dummy, BLOCK_N
  ∈ {16, 32, 64}, B=2 ctx=2048 HQ/HK=32/8. Worst observed: `max_abs 7.6e-3`, `rel_l2 2.1e-3`,
  cosine ≥ `0.999998`; LSE matched at ≤ `1.2e-6`. The all-dummy case reproduced the
  `safe_self` identity exactly (`max_abs = 0`).
- **Integration**: backend registered in `DFLASH_ATTENTION_BACKENDS`; dispatch rejects
  `triton_npu_v1` on CUDA and CPU with clear errors; autograd wrapper fails closed on CUDA;
  `DFlashTrainingModel(attention_backend="triton_npu_v1")` passes constructor validation.
- **No regression**: `tests/unit` + `tests/config` → 44 passed, 3 skipped
  (`test_draft_training_loop` could not be collected in this venv because the `verl`
  dependency is absent — pre-existing environment limitation, unrelated to this change).

What was **not** run (no NPU available locally):

- NPU compilation and execution (requires the Triton-Ascend toolchain and an Ascend part);
- NPU performance/UB/alignment measurements (BLOCK_N, program count, block-pointer variants);
- backward path (not implemented by design in v1).

The CUDA-side results are evidence of algorithm correctness and Triton frontend validity
only; they say nothing about NPU codegen or throughput, and toy results must not be
extrapolated to production-scale multi-NPU RLHF throughput.
