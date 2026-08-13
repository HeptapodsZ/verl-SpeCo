# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Triton forward and backward kernels for DFlash block-sparse attention."""

from __future__ import annotations

import math

import torch

from .tuning import get_triton_tuning

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - exercised only in minimal CPU installs
    raise ImportError("The DFlash Triton backend requires the triton package") from exc


_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _forward_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    HEAD_CHUNKS: tl.constexpr,
    ACTIVE_ROWS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
):
    anchor_id = tl.program_id(0)
    batch_kv_chunk = tl.program_id(1)
    batch_id = batch_kv_chunk // (HK * HEAD_CHUNKS)
    kv_chunk = batch_kv_chunk - batch_id * HK * HEAD_CHUNKS
    kv_head = kv_chunk // HEAD_CHUNKS
    head_chunk = kv_chunk - kv_head * HEAD_CHUNKS

    offs_m = tl.arange(0, QUERY_ROWS)
    head_lane = offs_m // BLOCK_SIZE
    token_offset = offs_m - head_lane * BLOCK_SIZE
    query_head = kv_head * GROUPS + head_chunk * HEADS_PER_PROGRAM + head_lane
    offs_n_local = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    query_index = anchor_id * BLOCK_SIZE + token_offset
    query_valid = (offs_m < ACTIVE_ROWS) & (query_head < (kv_head + 1) * GROUPS)
    q_ptrs = (
        Q + ((batch_id * HQ + query_head[:, None]) * Q_LEN + query_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=query_valid[:, None], other=0.0)
    anchor = tl.load(ANCHORS + batch_id * NUM_ANCHORS + anchor_id)
    keep = tl.load(KEEP + batch_id * NUM_ANCHORS + anchor_id) != 0

    row_max = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    row_sum = tl.zeros((QUERY_ROWS,), tl.float32)
    acc = tl.zeros((QUERY_ROWS, HEAD_DIM), tl.float32)

    # The loop bound is a runtime anchor, so 65K contexts do not get unrolled.
    for start_n in tl.range(0, anchor, BLOCK_N, num_stages=2):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        key_valid = offs_n < anchor
        k_ptrs = (
            K
            + ((batch_id * HK + kv_head) * KV_LEN + offs_n[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        v_ptrs = (
            V
            + ((batch_id * HK + kv_head) * KV_LEN + offs_n[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        k = tl.load(k_ptrs, mask=key_valid[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        allowed = query_valid[:, None] & key_valid[None, :] & keep
        scores = tl.where(allowed, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        # A dummy block cannot see context.  Keep its online-softmax state at
        # the empty identity until the finite self-only local tile is visited;
        # otherwise (-inf) - (-inf) produces NaNs that survive tl.where.
        has_context = query_valid & keep
        new_max = tl.where(has_context, tl.maximum(row_max, tile_max), row_max)
        alpha = tl.where(
            has_context,
            tl.exp2((row_max - new_max) * _LOG2E),
            1.0,
        )
        probs = tl.where(
            allowed,
            tl.exp2((scores - new_max[:, None]) * _LOG2E),
            0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(probs.to(Q.dtype.element_ty), v)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_max

    local_index = CTX_LEN + anchor_id * BLOCK_SIZE + offs_n_local
    local_valid = offs_n_local < BLOCK_SIZE
    k_ptrs = (
        K
        + ((batch_id * HK + kv_head) * KV_LEN + local_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + ((batch_id * HK + kv_head) * KV_LEN + local_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    k = tl.load(k_ptrs, mask=local_valid[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=local_valid[:, None], other=0.0)
    scores = tl.dot(q, tl.trans(k)) * SCALE
    valid_rows_cols = query_valid[:, None] & local_valid[None, :]
    local_allowed = tl.where(
        keep,
        valid_rows_cols,
        valid_rows_cols & (token_offset[:, None] == offs_n_local[None, :]),
    )
    scores = tl.where(local_allowed, scores, -float("inf"))
    tile_max = tl.max(scores, axis=1)
    new_max = tl.maximum(row_max, tile_max)
    alpha = tl.exp2((row_max - new_max) * _LOG2E)
    probs = tl.exp2((scores - new_max[:, None]) * _LOG2E)
    acc = acc * alpha[:, None] + tl.dot(probs.to(Q.dtype.element_ty), v)
    row_sum = row_sum * alpha + tl.sum(probs, axis=1)
    row_max = new_max

    output = acc / row_sum[:, None]
    o_ptrs = (
        O + ((batch_id * HQ + query_head[:, None]) * Q_LEN + query_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    lse_ptrs = LSE + (batch_id * HQ + query_head) * Q_LEN + query_index
    tl.store(o_ptrs, output, mask=query_valid[:, None])
    tl.store(lse_ptrs, row_max + tl.log(row_sum), mask=query_valid)


@triton.jit
def _delta_kernel(
    O,
    DO,
    DELTA,
    HQ: tl.constexpr,
    Q_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    anchor_id = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_id = batch_head // HQ
    query_head = batch_head - batch_id * HQ
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    query_index = anchor_id * BLOCK_SIZE + offs_m
    valid = offs_m < BLOCK_SIZE
    ptrs = (
        ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    output = tl.load(O + ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
    grad_output = tl.load(DO + ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(output * grad_output, axis=1)
    tl.store(
        DELTA + (batch_id * HQ + query_head) * Q_LEN + query_index,
        delta,
        mask=valid,
    )


@triton.jit
def _backward_dq_kernel(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    ANCHORS,
    KEEP,
    DQ,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    CTX_LEN: tl.constexpr,
    NUM_ANCHORS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
):
    anchor_id = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_id = batch_head // HQ
    query_head = batch_head - batch_id * HQ
    kv_head = query_head // GROUPS
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    query_index = anchor_id * BLOCK_SIZE + offs_m
    query_valid = offs_m < BLOCK_SIZE
    q_ptrs = (
        Q
        + ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    do_ptrs = (
        DO
        + ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    output_ptrs = (
        O
        + ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=query_valid[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=query_valid[:, None], other=0.0)
    output = tl.load(output_ptrs, mask=query_valid[:, None], other=0.0).to(tl.float32)
    lse = tl.load(
        LSE + (batch_id * HQ + query_head) * Q_LEN + query_index,
        mask=query_valid,
        other=0.0,
    )
    delta = tl.sum(output * do.to(tl.float32), axis=1)
    tl.store(
        DELTA + (batch_id * HQ + query_head) * Q_LEN + query_index,
        delta,
        mask=query_valid,
    )
    anchor = tl.load(ANCHORS + batch_id * NUM_ANCHORS + anchor_id)
    keep = tl.load(KEEP + batch_id * NUM_ANCHORS + anchor_id) != 0
    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for start_n in tl.range(0, anchor, BLOCK_N, num_stages=2):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        key_valid = offs_n < anchor
        k_ptrs = (
            K
            + ((batch_id * HK + kv_head) * KV_LEN + offs_n[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        v_ptrs = (
            V
            + ((batch_id * HK + kv_head) * KV_LEN + offs_n[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        k = tl.load(k_ptrs, mask=key_valid[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        allowed = query_valid[:, None] & key_valid[None, :] & keep
        probs = tl.where(
            allowed,
            tl.exp2((scores - lse[:, None]) * _LOG2E),
            0.0,
        )
        dp = tl.dot(do, tl.trans(v))
        ds = probs * (dp - delta[:, None]) * SCALE
        dq += tl.dot(ds.to(Q.dtype.element_ty), k)

    offs_n = tl.arange(0, BLOCK_M)
    local_index = CTX_LEN + anchor_id * BLOCK_SIZE + offs_n
    local_valid = offs_n < BLOCK_SIZE
    k_ptrs = (
        K
        + ((batch_id * HK + kv_head) * KV_LEN + local_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + ((batch_id * HK + kv_head) * KV_LEN + local_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    k = tl.load(k_ptrs, mask=local_valid[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=local_valid[:, None], other=0.0)
    scores = tl.dot(q, tl.trans(k)) * SCALE
    valid_rows_cols = query_valid[:, None] & local_valid[None, :]
    allowed = tl.where(
        keep,
        valid_rows_cols,
        valid_rows_cols & (offs_m[:, None] == offs_n[None, :]),
    )
    probs = tl.where(
        allowed,
        tl.exp2((scores - lse[:, None]) * _LOG2E),
        0.0,
    )
    dp = tl.dot(do, tl.trans(v))
    # Dummy blocks are exactly one-element softmaxes.  Their Q/K gradient is
    # mathematically zero; enforcing that identity avoids reconstruction noise.
    probs = tl.where(keep, probs, allowed.to(tl.float32))
    ds = tl.where(keep, probs * (dp - delta[:, None]) * SCALE, 0.0)
    dq += tl.dot(ds.to(Q.dtype.element_ty), k)
    dq_ptrs = (
        DQ
        + ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    tl.store(dq_ptrs, dq, mask=query_valid[:, None])


@triton.jit
def _backward_dkv_context_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    ANCHORS,
    KEEP,
    DK,
    DV,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    CTX_LEN: tl.constexpr,
    NUM_ANCHORS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
):
    key_tile = tl.program_id(0)
    batch_kv_head = tl.program_id(1)
    batch_id = batch_kv_head // HK
    kv_head = batch_kv_head - batch_id * HK
    offs_n = key_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    key_valid = offs_n < CTX_LEN
    kv_ptrs = (
        ((batch_id * HK + kv_head) * KV_LEN + offs_n[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    k = tl.load(K + kv_ptrs, mask=key_valid[:, None], other=0.0)
    v = tl.load(V + kv_ptrs, mask=key_valid[:, None], other=0.0)
    dk = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    offs_m = tl.arange(0, BLOCK_M)
    query_valid = offs_m < BLOCK_SIZE

    for anchor_id in tl.range(0, NUM_ANCHORS, loop_unroll_factor=1):
        anchor = tl.load(ANCHORS + batch_id * NUM_ANCHORS + anchor_id)
        keep = tl.load(KEEP + batch_id * NUM_ANCHORS + anchor_id) != 0
        if keep & (key_tile * BLOCK_N < anchor):
            query_index = anchor_id * BLOCK_SIZE + offs_m
            for group_id in tl.static_range(0, GROUPS):
                query_head = kv_head * GROUPS + group_id
                q_ptrs = (
                    Q
                    + ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None])
                    * HEAD_DIM
                    + offs_d[None, :]
                )
                do_ptrs = (
                    DO
                    + ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None])
                    * HEAD_DIM
                    + offs_d[None, :]
                )
                q = tl.load(q_ptrs, mask=query_valid[:, None], other=0.0)
                do = tl.load(do_ptrs, mask=query_valid[:, None], other=0.0)
                lse = tl.load(
                    LSE + (batch_id * HQ + query_head) * Q_LEN + query_index,
                    mask=query_valid,
                    other=0.0,
                )
                delta = tl.load(
                    DELTA + (batch_id * HQ + query_head) * Q_LEN + query_index,
                    mask=query_valid,
                    other=0.0,
                )
                scores = tl.dot(q, tl.trans(k)) * SCALE
                allowed = query_valid[:, None] & key_valid[None, :] & (offs_n[None, :] < anchor)
                probs = tl.where(
                    allowed,
                    tl.exp2((scores - lse[:, None]) * _LOG2E),
                    0.0,
                )
                dv += tl.dot(tl.trans(probs.to(Q.dtype.element_ty)), do)
                dp = tl.dot(do, tl.trans(v))
                ds = probs * (dp - delta[:, None]) * SCALE
                dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)

    tl.store(DK + kv_ptrs, dk, mask=key_valid[:, None])
    tl.store(DV + kv_ptrs, dv, mask=key_valid[:, None])


@triton.jit
def _backward_dkv_draft_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    KEEP,
    DK,
    DV,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    CTX_LEN: tl.constexpr,
    NUM_ANCHORS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS: tl.constexpr,
):
    anchor_id = tl.program_id(0)
    batch_kv_head = tl.program_id(1)
    batch_id = batch_kv_head // HK
    kv_head = batch_kv_head - batch_id * HK
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    query_valid = offs_m < BLOCK_SIZE
    key_valid = offs_n < BLOCK_SIZE
    query_index = anchor_id * BLOCK_SIZE + offs_m
    local_index = CTX_LEN + anchor_id * BLOCK_SIZE + offs_n
    kv_ptrs = (
        ((batch_id * HK + kv_head) * KV_LEN + local_index[:, None]) * HEAD_DIM
        + offs_d[None, :]
    )
    k = tl.load(K + kv_ptrs, mask=key_valid[:, None], other=0.0)
    v = tl.load(V + kv_ptrs, mask=key_valid[:, None], other=0.0)
    keep = tl.load(KEEP + batch_id * NUM_ANCHORS + anchor_id) != 0
    dk = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for group_id in tl.static_range(0, GROUPS):
        query_head = kv_head * GROUPS + group_id
        q_ptrs = (
            Q
            + ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        do_ptrs = (
            DO
            + ((batch_id * HQ + query_head) * Q_LEN + query_index[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=query_valid[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=query_valid[:, None], other=0.0)
        lse = tl.load(
            LSE + (batch_id * HQ + query_head) * Q_LEN + query_index,
            mask=query_valid,
            other=0.0,
        )
        delta = tl.load(
            DELTA + (batch_id * HQ + query_head) * Q_LEN + query_index,
            mask=query_valid,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * SCALE
        valid_rows_cols = query_valid[:, None] & key_valid[None, :]
        allowed = tl.where(
            keep,
            valid_rows_cols,
            valid_rows_cols & (offs_m[:, None] == offs_n[None, :]),
        )
        probs = tl.where(
            allowed,
            tl.exp2((scores - lse[:, None]) * _LOG2E),
            0.0,
        )
        probs = tl.where(keep, probs, allowed.to(tl.float32))
        dv += tl.dot(tl.trans(probs.to(Q.dtype.element_ty)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = tl.where(keep, probs * (dp - delta[:, None]) * SCALE, 0.0)
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)

    tl.store(DK + kv_ptrs, dk, mask=key_valid[:, None])
    tl.store(DV + kv_ptrs, dv, mask=key_valid[:, None])


def _launch_config(
    block_size: int, ctx_len: int, device: torch.device
) -> tuple[int, int, int, int]:
    config = get_triton_tuning(
        block_size=block_size, ctx_len=ctx_len, device=device
    )
    return config.block_m, config.block_n, config.num_warps, config.num_stages


def triton_dflash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    ctx_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Triton forward kernel and return output plus natural-log LSE."""
    bsz, num_query_heads, query_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    num_anchors = anchor_positions.shape[1]
    block_m, block_n, num_warps, num_stages = _launch_config(
        block_size, ctx_len, query.device
    )
    groups = num_query_heads // num_kv_heads
    heads_per_program = min(2, groups)
    head_chunks = triton.cdiv(groups, heads_per_program)
    active_rows = int(block_size) * heads_per_program
    query_rows = max(16, triton.next_power_of_2(active_rows))
    output = torch.empty_like(query)
    lse = torch.empty((bsz, num_query_heads, query_len), device=query.device, dtype=torch.float32)
    _forward_kernel[(num_anchors, bsz * num_kv_heads * head_chunks)](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=groups,
        HEADS_PER_PROGRAM=heads_per_program,
        HEAD_CHUNKS=head_chunks,
        ACTIVE_ROWS=active_rows,
        QUERY_ROWS=query_rows,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, lse


def _triton_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    ctx_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    bsz, num_query_heads, query_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    num_anchors = anchor_positions.shape[1]
    groups = num_query_heads // num_kv_heads
    block_m, block_n, num_warps, num_stages = _launch_config(
        block_size, ctx_len, query.device
    )
    delta = torch.empty_like(lse)
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    grid_qa = (num_anchors, bsz * num_query_heads)
    _backward_dq_kernel[grid_qa](
        query,
        key,
        value,
        output,
        grad_output,
        lse,
        delta,
        anchor_positions,
        block_keep_mask,
        grad_query,
        HQ=num_query_heads,
        HK=num_kv_heads,
        Q_LEN=query_len,
        KV_LEN=key.shape[2],
        CTX_LEN=int(ctx_len),
        NUM_ANCHORS=num_anchors,
        SCALE=1.0 / math.sqrt(head_dim),
        BLOCK_SIZE=int(block_size),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=groups,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if ctx_len > 0:
        _backward_dkv_context_kernel[(triton.cdiv(ctx_len, block_n), bsz * num_kv_heads)](
            query,
            key,
            value,
            grad_output,
            lse,
            delta,
            anchor_positions,
            block_keep_mask,
            grad_key,
            grad_value,
            HQ=num_query_heads,
            HK=num_kv_heads,
            Q_LEN=query_len,
            KV_LEN=key.shape[2],
            CTX_LEN=int(ctx_len),
            NUM_ANCHORS=num_anchors,
            SCALE=1.0 / math.sqrt(head_dim),
            BLOCK_SIZE=int(block_size),
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            GROUPS=groups,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    _backward_dkv_draft_kernel[(num_anchors, bsz * num_kv_heads)](
        query,
        key,
        value,
        grad_output,
        lse,
        delta,
        block_keep_mask,
        grad_key,
        grad_value,
        HQ=num_query_heads,
        HK=num_kv_heads,
        Q_LEN=query_len,
        KV_LEN=key.shape[2],
        CTX_LEN=int(ctx_len),
        NUM_ANCHORS=num_anchors,
        SCALE=1.0 / math.sqrt(head_dim),
        BLOCK_SIZE=int(block_size),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        GROUPS=groups,
        num_warps=num_warps,
    )
    return grad_query, grad_key, grad_value


class _TritonDFlashAttention(torch.autograd.Function):
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
    ) -> torch.Tensor:
        output, lse = triton_dflash_attention_forward(
            query,
            key,
            value,
            anchor_positions,
            block_keep_mask,
            ctx_len=int(ctx_len),
            block_size=int(block_size),
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            output,
            lse,
            anchor_positions,
            block_keep_mask,
        )
        ctx.ctx_len = int(ctx_len)
        ctx.block_size = int(block_size)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, output, lse, anchor_positions, block_keep_mask = (
            ctx.saved_tensors
        )
        grad_query, grad_key, grad_value = _triton_backward(
            grad_output,
            query,
            key,
            value,
            output,
            lse,
            anchor_positions,
            block_keep_mask,
            ctx_len=ctx.ctx_len,
            block_size=ctx.block_size,
        )
        return grad_query, grad_key, grad_value, None, None, None, None


def triton_dflash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    ctx_len: int,
    block_size: int,
) -> torch.Tensor:
    return _TritonDFlashAttention.apply(
        query,
        key,
        value,
        anchor_positions,
        block_keep_mask,
        int(ctx_len),
        int(block_size),
    )
