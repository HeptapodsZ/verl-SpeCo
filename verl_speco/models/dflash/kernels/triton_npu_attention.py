# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton-Ascend forward kernel for DFlash block-sparse attention (NPU).

Backend ``triton_npu_v1``. This is a forward-only FlashAttention-v2-style
kernel written for Ascend NPUs through the Triton-Ascend compiler:

- one persistent, grid-strided program grid over logical work items
  (batch, KV head, GQA head chunk, anchor pair), avoiding the giant
  per-anchor grids the CUDA baseline launches;
- K/V tiles are streamed with masked pointer-arithmetic loads that keep the
  head dim as the contiguous, CV-friendly axis (block-pointer ``order=(1, 0)``
  streaming is a measured-future refinement, see the design doc);
- the anchor prefix is applied as a per-row staircase mask inside each tile,
  and dummy blocks keep the empty-identity online-softmax state until their
  finite self-only local tile is visited;
- the online softmax runs in fp32 with ``exp2`` rescaling and accumulates
  ``P @ V`` directly into the fp32 accumulator through the Cube unit.

The autograd wrapper is forward-only and fails closed on backward because
draft-model training needs a matching backward kernel (planned follow-up).
"""

from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - exercised only in minimal CPU installs
    raise ImportError("The DFlash triton_npu_v1 backend requires the triton package") from exc


_LOG2E = tl.constexpr(1.4426950408889634)
_NPU_DEFAULT_CORE_COUNT = 20
_NPU_PROGRAMS_ENV = "VERL_SPECO_DFLASH_NPU_PROGRAMS"


@triton.jit
def _forward_npu_kernel(
    Q,
    K,
    V,
    ANCHORS,
    KEEP,
    O,
    LSE,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    CTX_LEN: tl.constexpr,
    NUM_ANCHORS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    HEAD_CHUNKS: tl.constexpr,
    ANCHORS_PER_PROGRAM: tl.constexpr,
    ACTIVE_ROWS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    NUM_ANCHOR_GROUPS: tl.constexpr,
    TOTAL_WORK: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_per_anchor: tl.constexpr = HEADS_PER_PROGRAM * BLOCK_SIZE
    local_n: tl.constexpr = ANCHORS_PER_PROGRAM * BLOCK_SIZE

    # Persistent grid-stride scheduling: each NPU core walks a strided slice
    # of the logical work queue instead of the compiler launching one tiny
    # program per (anchor, head chunk) pair.
    for work_id in range(pid, TOTAL_WORK, NUM_PROGRAMS):
        anchor_group = work_id % NUM_ANCHOR_GROUPS
        batch_kv_chunk = work_id // NUM_ANCHOR_GROUPS
        batch_id = batch_kv_chunk // (HK * HEAD_CHUNKS)
        kv_chunk = batch_kv_chunk - batch_id * HK * HEAD_CHUNKS
        kv_head = kv_chunk // HEAD_CHUNKS
        head_chunk = kv_chunk - kv_head * HEAD_CHUNKS

        # Query tile rows pack (anchor lane, head lane, token) so that a
        # 16-token block is never the only M extent of a Cube operation.
        offs_m = tl.arange(0, QUERY_ROWS)
        anchor_lane = offs_m // rows_per_anchor
        head_token_offset = offs_m - anchor_lane * rows_per_anchor
        head_lane = head_token_offset // BLOCK_SIZE
        token_offset = head_token_offset - head_lane * BLOCK_SIZE
        anchor_id = anchor_group * ANCHORS_PER_PROGRAM + anchor_lane
        anchor_valid = anchor_id < NUM_ANCHORS
        query_head = kv_head * GROUPS + head_chunk * HEADS_PER_PROGRAM + head_lane
        query_index = anchor_id * BLOCK_SIZE + token_offset
        query_valid = (
            (offs_m < ACTIVE_ROWS)
            & anchor_valid
            & (query_head < (kv_head + 1) * GROUPS)
        )
        offs_d = tl.arange(0, HEAD_DIM)

        q_base = Q + batch_id.to(tl.int64) * (HQ * Q_LEN * HEAD_DIM)
        q_ptrs = (
            q_base
            + (query_head[:, None] * Q_LEN + query_index[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=query_valid[:, None], other=0.0)
        anchor_row = tl.load(
            ANCHORS + batch_id * NUM_ANCHORS + anchor_id, mask=anchor_valid, other=0
        )
        keep_row = (
            tl.load(
                KEEP + batch_id * NUM_ANCHORS + anchor_id,
                mask=anchor_valid,
                other=0,
            )
            != 0
        )

        row_max = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
        row_sum = tl.zeros((QUERY_ROWS,), tl.float32)
        acc = tl.zeros((QUERY_ROWS, HEAD_DIM), tl.float32)

        kv_base = (
            K
            + batch_id.to(tl.int64) * (HK * KV_LEN * HEAD_DIM)
            + kv_head.to(tl.int64) * (KV_LEN * HEAD_DIM)
        )
        vv_base = (
            V
            + batch_id.to(tl.int64) * (HK * KV_LEN * HEAD_DIM)
            + kv_head.to(tl.int64) * (KV_LEN * HEAD_DIM)
        )

        # Context phase: scan [0, max anchor) once per program and mask each
        # row against its own anchor, generalizing the tutorial's causal
        # staircase from token positions to anchor positions. K/V tiles use
        # pointer arithmetic so masked tail loads compile on both CUDA Triton
        # and Triton-Ascend; the head dim stays the contiguous axis.
        if CTX_LEN > 0:
            max_anchor = tl.max(anchor_row, axis=0)
            loop_hi = tl.minimum(max_anchor, CTX_LEN)
            for start_n in range(0, loop_hi, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                key_valid = offs_n < loop_hi
                k_ptrs = kv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
                v_ptrs = vv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
                k = tl.load(k_ptrs, mask=key_valid[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)
                scores = tl.dot(q, tl.trans(k)) * SCALE
                allowed = (
                    query_valid[:, None]
                    & (offs_n[None, :] < anchor_row[:, None])
                    & keep_row[:, None]
                )
                scores = tl.where(allowed, scores, -float("inf"))
                tile_max = tl.max(scores, axis=1)
                # Dummy blocks have no context.  Keep their online-softmax
                # state at the empty identity until the finite local tile;
                # otherwise (-inf) - (-inf) produces NaNs that survive the
                # select.
                has_context_tile = query_valid & keep_row & (start_n < anchor_row)
                new_max = tl.where(
                    has_context_tile, tl.maximum(row_max, tile_max), row_max
                )
                alpha = tl.where(
                    has_context_tile,
                    tl.exp2((row_max - new_max) * _LOG2E),
                    1.0,
                )
                probs = tl.where(
                    allowed,
                    tl.exp2((scores - new_max[:, None]) * _LOG2E),
                    0.0,
                )
                acc = acc * alpha[:, None]
                acc = tl.dot(probs.to(q.dtype), v, acc)
                row_sum = row_sum * alpha + tl.sum(probs, axis=1)
                row_max = new_max

        # Draft phase: the two consecutive anchor blocks form one contiguous
        # 32-row K/V tile.  Each query row attends its own block (all 16
        # tokens, non-causal); dummy rows collapse to their diagonal key.
        local_offs = tl.arange(0, local_n)
        local_lane = local_offs // BLOCK_SIZE
        local_token = local_offs - local_lane * BLOCK_SIZE
        local_anchor_id = anchor_group * ANCHORS_PER_PROGRAM + local_lane
        local_valid = local_anchor_id < NUM_ANCHORS
        local_base = CTX_LEN + anchor_group * local_n
        k_ptrs = (
            kv_base + (local_base + local_offs[:, None]) * HEAD_DIM + offs_d[None, :]
        )
        v_ptrs = (
            vv_base + (local_base + local_offs[:, None]) * HEAD_DIM + offs_d[None, :]
        )
        k = tl.load(k_ptrs, mask=local_valid[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=local_valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        same_anchor = anchor_id[:, None] == local_anchor_id[None, :]
        valid_rows_cols = query_valid[:, None] & local_valid[None, :] & same_anchor
        local_allowed = valid_rows_cols & (
            keep_row[:, None] | (token_offset[:, None] == local_token[None, :])
        )
        scores = tl.where(local_allowed, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.exp2((row_max - new_max) * _LOG2E)
        probs = tl.exp2((scores - new_max[:, None]) * _LOG2E)
        acc = acc * alpha[:, None]
        acc = tl.dot(probs.to(q.dtype), v, acc)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_max

        output = acc / row_sum[:, None]
        o_base = O + batch_id.to(tl.int64) * (HQ * Q_LEN * HEAD_DIM)
        o_ptrs = (
            o_base
            + (query_head[:, None] * Q_LEN + query_index[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        lse_base = LSE + batch_id.to(tl.int64) * (HQ * Q_LEN)
        lse_ptrs = lse_base + query_head * Q_LEN + query_index
        tl.store(o_ptrs, output.to(Q.dtype.element_ty), mask=query_valid[:, None])
        tl.store(lse_ptrs, row_max + tl.log(row_sum), mask=query_valid)


def default_npu_program_count() -> int:
    """Number of programs for the persistent grid on this NPU.

    Prefers ``VERL_SPECO_DFLASH_NPU_PROGRAMS``, then the core count exposed by
    torch-npu, and falls back to the 20-core tutorial constant.
    """
    env_value = os.environ.get(_NPU_PROGRAMS_ENV)
    if env_value:
        try:
            count = int(env_value)
        except ValueError as exc:
            raise RuntimeError(
                f"{_NPU_PROGRAMS_ENV} must be a positive integer, got {env_value!r}"
            ) from exc
        if count > 0:
            return count
    try:
        import torch_npu  # noqa: F401

        device_index = torch_npu.npu.current_device()
        props = torch_npu.npu.get_device_properties(device_index)
        for attr in ("core_num", "multi_processor_count"):
            value = getattr(props, attr, None)
            if value is not None and int(value) > 0:
                return int(value)
    except Exception:
        pass
    return _NPU_DEFAULT_CORE_COUNT


def triton_npu_dflash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    ctx_len: int,
    block_size: int,
    num_programs: int | None = None,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Triton-Ascend forward kernel and return output plus LSE.

    The raw launcher is device-agnostic so the same kernel text can be
    compile-checked on CUDA; the NPU device gate lives at the public API
    boundary (``triton_npu_dflash_attention`` / dispatch).
    """
    if int(block_n) not in (16, 32, 64):
        raise ValueError(
            "DFlash triton_npu_v1 block_n must be one of (16, 32, 64); "
            f"larger tiles exceed the validated register/UB budget, got {block_n}"
        )
    bsz, num_query_heads, query_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    num_anchors = anchor_positions.shape[1]
    groups = num_query_heads // num_kv_heads
    heads_per_program = min(2, groups)
    head_chunks = triton.cdiv(groups, heads_per_program)
    anchors_per_program = 2
    num_anchor_groups = triton.cdiv(num_anchors, anchors_per_program)
    active_rows = int(block_size) * heads_per_program * anchors_per_program
    query_rows = max(16, triton.next_power_of_2(active_rows))
    total_work = bsz * num_kv_heads * head_chunks * num_anchor_groups
    if num_programs is None or int(num_programs) <= 0:
        num_programs = default_npu_program_count()
    num_programs = max(1, min(int(num_programs), total_work))

    output = torch.empty_like(query)
    lse = torch.empty(
        (bsz, num_query_heads, query_len), device=query.device, dtype=torch.float32
    )
    _forward_npu_kernel[(num_programs,)](
        query,
        key,
        value,
        anchor_positions,
        block_keep_mask,
        output,
        lse,
        HQ=num_query_heads,
        HK=num_kv_heads,
        Q_LEN=query_len,
        KV_LEN=key.shape[2],
        CTX_LEN=int(ctx_len),
        NUM_ANCHORS=num_anchors,
        SCALE=1.0 / math.sqrt(head_dim),
        BLOCK_SIZE=int(block_size),
        HEAD_DIM=head_dim,
        BLOCK_N=int(block_n),
        GROUPS=groups,
        HEADS_PER_PROGRAM=heads_per_program,
        HEAD_CHUNKS=head_chunks,
        ANCHORS_PER_PROGRAM=anchors_per_program,
        ACTIVE_ROWS=active_rows,
        QUERY_ROWS=query_rows,
        NUM_ANCHOR_GROUPS=num_anchor_groups,
        TOTAL_WORK=total_work,
        NUM_PROGRAMS=num_programs,
    )
    return output, lse


class _TritonNpuDFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        ctx_len: int,
        block_size: int,
        num_programs: int | None,
        block_n: int,
    ) -> torch.Tensor:
        if query.device.type != "npu":
            raise RuntimeError(
                "DFlash triton_npu_v1 attention requires NPU (Ascend) tensors, "
                f"got device type {query.device.type!r}"
            )
        output, _ = triton_npu_dflash_attention_forward(
            query,
            key,
            value,
            anchor_positions,
            block_keep_mask,
            ctx_len=int(ctx_len),
            block_size=int(block_size),
            num_programs=num_programs,
            block_n=int(block_n),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError(
            "DFlash triton_npu_v1 is forward-only in this version. Draft-model "
            "training needs the matching backward kernel, which is not "
            "implemented yet; use a CUDA backend (triton, tilelang, flex, or "
            "sdpa) for training runs."
        )


def triton_npu_dflash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    ctx_len: int,
    block_size: int,
    num_programs: int | None = None,
    block_n: int = 64,
) -> torch.Tensor:
    """Forward-only DFlash attention on Ascend NPU via Triton-Ascend."""
    return _TritonNpuDFlashAttention.apply(
        query,
        key,
        value,
        anchor_positions,
        block_keep_mask,
        int(ctx_len),
        int(block_size),
        num_programs,
        int(block_n),
    )
