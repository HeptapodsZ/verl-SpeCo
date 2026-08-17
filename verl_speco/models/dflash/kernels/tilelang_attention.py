# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""TileLang forward and backward kernels for DFlash block-sparse attention."""

import functools
import math

import torch

try:
    import tilelang
    import tilelang.language as T
except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
    raise ImportError("The DFlash TileLang backend requires the tilelang package") from exc


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    # Pull kernels assign one unique key tile per program. TileLang's symbolic
    # checker cannot prove uniqueness through key_start + j, but no two
    # programs or loop points write the same GradK/GradV element.
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}


@tilelang.jit(out_idx=[5, 6], pass_configs=_PASS_CONFIGS)
def _forward_factory(
    batch,
    query_heads,
    kv_heads,
    query_len,
    kv_len,
    ctx_len,
    num_anchors,
    block_size,
    head_dim,
    dtype,
    block_m=16,
    block_n=64,
    threads=128,
    num_stages=2,
):
    scale = (1.0 / head_dim) ** 0.5
    log2e = 1.4426950408889634
    groups = query_heads // kv_heads
    heads_per_program = (
        min(2, groups)
        if ctx_len <= 512
        else (groups if block_m == 16 else min(2, groups))
    )
    head_chunks = (groups + heads_per_program - 1) // heads_per_program
    active_rows = block_size * heads_per_program
    rows = max(16, active_rows)
    q_shape = (batch, query_heads, query_len, head_dim)
    kv_shape = (batch, kv_heads, kv_len, head_dim)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Anchors: T.Tensor((batch, num_anchors), T.int32),
        Keep: T.Tensor((batch, num_anchors), T.int32),
        Output: T.Tensor(q_shape, dtype),
        LSE: T.Tensor((batch, query_heads, query_len), T.float32),
    ):
        with T.Kernel(num_anchors, kv_heads * head_chunks, batch, threads=threads) as (
            ba,
            hk_chunk,
            bz,
        ):
            Q_shared = T.alloc_shared((rows, head_dim), dtype)
            K_shared = T.alloc_shared((block_n, head_dim), dtype)
            V_shared = T.alloc_shared((block_n, head_dim), dtype)
            K_local = T.alloc_shared((block_m, head_dim), dtype)
            V_local = T.alloc_shared((block_m, head_dim), dtype)
            P_cast = T.alloc_fragment((rows, block_n), dtype)
            Scores = T.alloc_fragment((rows, block_n), T.float32)
            P_local = T.alloc_fragment((rows, block_m), dtype)
            Scores_local = T.alloc_fragment((rows, block_m), T.float32)
            Acc = T.alloc_fragment((rows, head_dim), T.float32)
            RowMax = T.alloc_fragment((rows,), T.float32)
            RowMaxPrev = T.alloc_fragment((rows,), T.float32)
            RowScale = T.alloc_fragment((rows,), T.float32)
            RowSum = T.alloc_fragment((rows,), T.float32)
            TileSum = T.alloc_fragment((rows,), T.float32)
            kv_head = hk_chunk // head_chunks
            query_head_start = kv_head * groups + (hk_chunk % head_chunks) * heads_per_program
            anchor = Anchors[bz, ba]
            keep = Keep[bz, ba] != 0

            for i, d in T.Parallel(rows, head_dim):
                if i < active_rows:
                    Q_shared[i, d] = Q[
                        bz,
                        query_head_start + i // block_size,
                        ba * block_size + i % block_size,
                        d,
                    ]
                else:
                    Q_shared[i, d] = 0
            T.fill(Acc, 0)
            T.fill(RowSum, 0)
            T.fill(RowMax, -T.infinity(T.float32))

            for kt in T.Pipelined(T.ceildiv(anchor, block_n), num_stages=num_stages):
                for j, d in T.Parallel(block_n, head_dim):
                    key_index = kt * block_n + j
                    if key_index < ctx_len:
                        K_shared[j, d] = K[bz, kv_head, key_index, d]
                        V_shared[j, d] = V[bz, kv_head, key_index, d]
                    else:
                        K_shared[j, d] = 0
                        V_shared[j, d] = 0
                for i, j in T.Parallel(rows, block_n):
                    Scores[i, j] = T.if_then_else(
                        (i < active_rows)
                        and (kt * block_n + j < anchor)
                        and keep,
                        0,
                        -T.infinity(T.float32),
                    )
                T.gemm(
                    Q_shared,
                    K_shared,
                    Scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(RowMax, RowMaxPrev)
                T.fill(RowMax, -T.infinity(T.float32))
                T.reduce_max(Scores, RowMax, dim=1, clear=False)
                for i in T.Parallel(rows):
                    RowMax[i] = T.if_then_else(
                        keep and (kt * block_n < anchor),
                        T.max(RowMax[i] * scale, RowMaxPrev[i]),
                        RowMaxPrev[i],
                    )
                    RowScale[i] = T.if_then_else(
                        keep and (kt * block_n < anchor),
                        T.exp2((RowMaxPrev[i] - RowMax[i]) * log2e),
                        1,
                    )
                for i, j in T.Parallel(rows, block_n):
                    allowed = (
                        (i < active_rows)
                        and (kt * block_n + j < anchor)
                        and keep
                    )
                    Scores[i, j] = T.if_then_else(
                        allowed,
                        T.exp2((Scores[i, j] * scale - RowMax[i]) * log2e),
                        0,
                    )
                T.reduce_sum(Scores, TileSum, dim=1)
                for i in T.Parallel(rows):
                    RowSum[i] = RowSum[i] * RowScale[i] + TileSum[i]
                for i, d in T.Parallel(rows, head_dim):
                    Acc[i, d] *= RowScale[i]
                T.copy(Scores, P_cast)
                T.gemm(P_cast, V_shared, Acc, policy=T.GemmWarpPolicy.FullRow)

            # The local draft block is one final, exact-width softmax tile.
            for j, d in T.Parallel(block_m, head_dim):
                if j < block_size:
                    local_index = ctx_len + ba * block_size + j
                    K_local[j, d] = K[bz, kv_head, local_index, d]
                    V_local[j, d] = V[bz, kv_head, local_index, d]
                else:
                    K_local[j, d] = 0
                    V_local[j, d] = 0
            for i, j in T.Parallel(rows, block_m):
                local_allowed = (i < active_rows) and (j < block_size) and (
                    keep or (i % block_size == j)
                )
                Scores_local[i, j] = T.if_then_else(
                    local_allowed, 0, -T.infinity(T.float32)
                )
            T.gemm(
                Q_shared,
                K_local,
                Scores_local,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(RowMax, RowMaxPrev)
            T.fill(RowMax, -T.infinity(T.float32))
            T.reduce_max(Scores_local, RowMax, dim=1, clear=False)
            for i in T.Parallel(rows):
                RowMax[i] = T.max(RowMax[i] * scale, RowMaxPrev[i])
                RowScale[i] = T.exp2((RowMaxPrev[i] - RowMax[i]) * log2e)
            for i, j in T.Parallel(rows, block_m):
                local_allowed = (i < active_rows) and (j < block_size) and (
                    keep or (i % block_size == j)
                )
                Scores_local[i, j] = T.if_then_else(
                    local_allowed,
                    T.exp2((Scores_local[i, j] * scale - RowMax[i]) * log2e),
                    0,
                )
            T.reduce_sum(Scores_local, TileSum, dim=1)
            for i in T.Parallel(rows):
                RowSum[i] = RowSum[i] * RowScale[i] + TileSum[i]
            for i, d in T.Parallel(rows, head_dim):
                Acc[i, d] *= RowScale[i]
            T.copy(Scores_local, P_local)
            T.gemm(P_local, V_local, Acc, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(rows, head_dim):
                if i < active_rows:
                    Output[
                        bz,
                        query_head_start + i // block_size,
                        ba * block_size + i % block_size,
                        d,
                    ] = Acc[i, d] / RowSum[i]
            for i in T.Parallel(rows):
                if i < active_rows:
                    LSE[
                        bz,
                        query_head_start + i // block_size,
                        ba * block_size + i % block_size,
                    ] = RowMax[i] + T.log(RowSum[i])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS_CONFIGS)
def _delta_factory(
    batch,
    query_heads,
    query_len,
    num_anchors,
    block_size,
    head_dim,
    dtype,
    block_m=16,
    threads=32,
):
    shape = (batch, query_heads, query_len, head_dim)

    @T.prim_func
    def main(
        Output: T.Tensor(shape, dtype),
        GradOutput: T.Tensor(shape, dtype),
        Delta: T.Tensor((batch, query_heads, query_len), T.float32),
    ):
        with T.Kernel(num_anchors, query_heads, batch, threads=threads) as (
            ba,
            hq,
            bz,
        ):
            Product = T.alloc_fragment((block_m, head_dim), T.float32)
            RowSum = T.alloc_fragment((block_m,), T.float32)
            for i, d in T.Parallel(block_m, head_dim):
                Product[i, d] = T.if_then_else(
                    i < block_size,
                    Output[bz, hq, ba * block_size + i, d]
                    * GradOutput[bz, hq, ba * block_size + i, d],
                    0,
                )
            T.reduce_sum(Product, RowSum, dim=1)
            for i in T.Parallel(block_m):
                if i < block_size:
                    Delta[bz, hq, ba * block_size + i] = RowSum[i]

    return main


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _backward_atomic_factory(
    batch,
    query_heads,
    kv_heads,
    query_len,
    kv_len,
    ctx_len,
    num_anchors,
    block_size,
    head_dim,
    dtype,
    block_m=16,
    block_n=16,
    threads=32,
    num_stages=2,
    write_dkv=True,
):
    scale = (1.0 / head_dim) ** 0.5
    log2e = 1.4426950408889634
    groups = query_heads // kv_heads
    heads_per_program = groups if block_m == 16 else min(2, groups)
    head_chunks = (groups + heads_per_program - 1) // heads_per_program
    active_rows = block_size * heads_per_program
    rows = max(16, active_rows)
    q_shape = (batch, query_heads, query_len, head_dim)
    kv_shape = (batch, kv_heads, kv_len, head_dim)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        GradOutput: T.Tensor(q_shape, dtype),
        LSE: T.Tensor((batch, query_heads, query_len), T.float32),
        Delta: T.Tensor((batch, query_heads, query_len), T.float32),
        Anchors: T.Tensor((batch, num_anchors), T.int32),
        Keep: T.Tensor((batch, num_anchors), T.int32),
        GradQ: T.Tensor(q_shape, T.float32),
        GradK: T.Tensor(kv_shape, T.float32),
        GradV: T.Tensor(kv_shape, T.float32),
    ):
        with T.Kernel(num_anchors, kv_heads * head_chunks, batch, threads=threads) as (
            ba,
            hk_chunk,
            bz,
        ):
            Q_shared = T.alloc_shared((rows, head_dim), dtype)
            DO_shared = T.alloc_shared((rows, head_dim), dtype)
            K_shared = T.alloc_shared((block_n, head_dim), dtype)
            V_shared = T.alloc_shared((block_n, head_dim), dtype)
            K_local = T.alloc_shared((block_n, head_dim), dtype)
            V_local = T.alloc_shared((block_n, head_dim), dtype)
            P_shared = T.alloc_shared((rows, block_n), dtype)
            DS_shared = T.alloc_shared((rows, block_n), dtype)
            Scores = T.alloc_fragment((rows, block_n), T.float32)
            DP = T.alloc_fragment((rows, block_n), T.float32)
            DQ = T.alloc_fragment((rows, head_dim), T.float32)
            DK = T.alloc_fragment((block_n, head_dim), T.float32)
            DV = T.alloc_fragment((block_n, head_dim), T.float32)
            kv_head = hk_chunk // head_chunks
            query_head_start = (
                kv_head * groups + (hk_chunk % head_chunks) * heads_per_program
            )
            anchor = Anchors[bz, ba]
            keep = Keep[bz, ba] != 0

            for i, d in T.Parallel(rows, head_dim):
                if i < active_rows:
                    hq = query_head_start + i // block_size
                    Q_shared[i, d] = Q[
                        bz, hq, ba * block_size + i % block_size, d
                    ]
                    DO_shared[i, d] = GradOutput[
                        bz, hq, ba * block_size + i % block_size, d
                    ]
                else:
                    Q_shared[i, d] = 0
                    DO_shared[i, d] = 0
            T.clear(DQ)

            for kt in T.Pipelined(T.ceildiv(anchor, block_n), num_stages=num_stages):
                for j, d in T.Parallel(block_n, head_dim):
                    key_index = kt * block_n + j
                    if key_index < ctx_len:
                        K_shared[j, d] = K[bz, kv_head, key_index, d]
                        V_shared[j, d] = V[bz, kv_head, key_index, d]
                    else:
                        K_shared[j, d] = 0
                        V_shared[j, d] = 0
                for i, j in T.Parallel(rows, block_n):
                    Scores[i, j] = 0
                T.gemm(
                    Q_shared,
                    K_shared,
                    Scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(rows, block_n):
                    hq = query_head_start + i // block_size
                    allowed = (
                        (i < active_rows)
                        and (kt * block_n + j < anchor)
                        and keep
                    )
                    Scores[i, j] = T.if_then_else(
                        allowed,
                        T.exp2(
                            (
                                Scores[i, j] * scale
                                - LSE[bz, hq, ba * block_size + i % block_size]
                            )
                            * log2e
                        ),
                        0,
                    )
                if write_dkv:
                    T.copy(Scores, P_shared)
                T.clear(DP)
                T.gemm(
                    DO_shared,
                    V_shared,
                    DP,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(rows, block_n):
                    hq = query_head_start + i // block_size
                    DS_shared[i, j] = Scores[i, j] * (
                        DP[i, j]
                        - Delta[bz, hq, ba * block_size + i % block_size]
                    ) * scale
                T.gemm(DS_shared, K_shared, DQ, policy=T.GemmWarpPolicy.FullRow)
                if write_dkv:
                    T.clear(DK)
                    T.clear(DV)
                    T.gemm(DS_shared, Q_shared, DK, transpose_A=True)
                    T.gemm(P_shared, DO_shared, DV, transpose_A=True)
                    for j, d in T.Parallel(block_n, head_dim):
                        key_index = kt * block_n + j
                        if key_index < anchor and keep:
                            T.atomic_add(GradK[bz, kv_head, key_index, d], DK[j, d])
                            T.atomic_add(GradV[bz, kv_head, key_index, d], DV[j, d])

            for j, d in T.Parallel(block_n, head_dim):
                if j < block_size:
                    local_index = ctx_len + ba * block_size + j
                    K_local[j, d] = K[bz, kv_head, local_index, d]
                    V_local[j, d] = V[bz, kv_head, local_index, d]
                else:
                    K_local[j, d] = 0
                    V_local[j, d] = 0
            T.clear(Scores)
            T.gemm(
                Q_shared,
                K_local,
                Scores,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(rows, block_n):
                hq = query_head_start + i // block_size
                allowed = (i < active_rows) and (j < block_size) and (
                    keep or (i % block_size == j)
                )
                Scores[i, j] = T.if_then_else(
                    allowed,
                    T.if_then_else(
                        keep,
                        T.exp2(
                            (
                                Scores[i, j] * scale
                                - LSE[bz, hq, ba * block_size + i % block_size]
                            )
                            * log2e
                        ),
                        1,
                    ),
                    0,
                )
            if write_dkv:
                T.copy(Scores, P_shared)
            T.clear(DP)
            T.gemm(
                DO_shared,
                V_local,
                DP,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(rows, block_n):
                hq = query_head_start + i // block_size
                DS_shared[i, j] = T.if_then_else(
                    keep,
                    Scores[i, j]
                    * (
                        DP[i, j]
                        - Delta[bz, hq, ba * block_size + i % block_size]
                    )
                    * scale,
                    0,
                )
            T.gemm(DS_shared, K_local, DQ, policy=T.GemmWarpPolicy.FullRow)
            if write_dkv:
                T.clear(DK)
                T.clear(DV)
                T.gemm(DS_shared, Q_shared, DK, transpose_A=True)
                T.gemm(P_shared, DO_shared, DV, transpose_A=True)
                for j, d in T.Parallel(block_n, head_dim):
                    if j < block_size:
                        local_index = ctx_len + ba * block_size + j
                        T.atomic_add(GradK[bz, kv_head, local_index, d], DK[j, d])
                        T.atomic_add(GradV[bz, kv_head, local_index, d], DV[j, d])
            for i, d in T.Parallel(rows, head_dim):
                if i < active_rows:
                    hq = query_head_start + i // block_size
                    GradQ[bz, hq, ba * block_size + i % block_size, d] = DQ[i, d]

    return main


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _backward_dkv_context_factory(
    batch,
    query_heads,
    kv_heads,
    query_len,
    kv_len,
    ctx_len,
    num_anchors,
    block_size,
    head_dim,
    dtype,
    block_m=16,
    block_n=32,
    threads=128,
):
    scale = (1.0 / head_dim) ** 0.5
    log2e = 1.4426950408889634
    groups = query_heads // kv_heads
    q_shape = (batch, query_heads, query_len, head_dim)
    kv_shape = (batch, kv_heads, kv_len, head_dim)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        GradOutput: T.Tensor(q_shape, dtype),
        LSE: T.Tensor((batch, query_heads, query_len), T.float32),
        Delta: T.Tensor((batch, query_heads, query_len), T.float32),
        Anchors: T.Tensor((batch, num_anchors), T.int32),
        Keep: T.Tensor((batch, num_anchors), T.int32),
        GradK: T.Tensor(kv_shape, T.float32),
        GradV: T.Tensor(kv_shape, T.float32),
    ):
        with T.Kernel(T.ceildiv(ctx_len, block_n), kv_heads, batch, threads=threads) as (
            kt,
            hk,
            bz,
        ):
            Q_shared = T.alloc_shared((block_m, head_dim), dtype)
            DO_shared = T.alloc_shared((block_m, head_dim), dtype)
            K_shared = T.alloc_shared((block_n, head_dim), dtype)
            V_shared = T.alloc_shared((block_n, head_dim), dtype)
            P_shared = T.alloc_shared((block_m, block_n), dtype)
            DS_shared = T.alloc_shared((block_m, block_n), dtype)
            Scores = T.alloc_fragment((block_m, block_n), T.float32)
            DP = T.alloc_fragment((block_m, block_n), T.float32)
            DK = T.alloc_fragment((block_n, head_dim), T.float32)
            DV = T.alloc_fragment((block_n, head_dim), T.float32)
            key_start = kt * block_n

            for j, d in T.Parallel(block_n, head_dim):
                key_index = key_start + j
                if key_index < ctx_len:
                    K_shared[j, d] = K[bz, hk, key_index, d]
                    V_shared[j, d] = V[bz, hk, key_index, d]
                else:
                    K_shared[j, d] = 0
                    V_shared[j, d] = 0
            T.clear(DK)
            T.clear(DV)
            for ba in T.serial(num_anchors):
                anchor = Anchors[bz, ba]
                if (Keep[bz, ba] != 0) and (key_start < anchor):
                    for group in T.serial(groups):
                        hq = hk * groups + group
                        for i, d in T.Parallel(block_m, head_dim):
                            if i < block_size:
                                Q_shared[i, d] = Q[bz, hq, ba * block_size + i, d]
                                DO_shared[i, d] = GradOutput[
                                    bz, hq, ba * block_size + i, d
                                ]
                            else:
                                Q_shared[i, d] = 0
                                DO_shared[i, d] = 0
                        T.clear(Scores)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            Scores,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            allowed = (i < block_size) and (key_start + j < anchor)
                            Scores[i, j] = T.if_then_else(
                                allowed,
                                T.exp2(
                                    (
                                        Scores[i, j] * scale
                                        - LSE[bz, hq, ba * block_size + i]
                                    )
                                    * log2e
                                ),
                                0,
                            )
                        T.copy(Scores, P_shared)
                        T.clear(DP)
                        T.gemm(
                            DO_shared,
                            V_shared,
                            DP,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            DS_shared[i, j] = Scores[i, j] * (
                                DP[i, j] - Delta[bz, hq, ba * block_size + i]
                            ) * scale
                        T.gemm(DS_shared, Q_shared, DK, transpose_A=True)
                        T.gemm(P_shared, DO_shared, DV, transpose_A=True)
            for j, d in T.Parallel(block_n, head_dim):
                key_index = key_start + j
                if key_index < ctx_len:
                    GradK[bz, hk, key_index, d] = DK[j, d]
                    GradV[bz, hk, key_index, d] = DV[j, d]

    return main


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _backward_dkv_draft_factory(
    batch,
    query_heads,
    kv_heads,
    query_len,
    kv_len,
    ctx_len,
    num_anchors,
    block_size,
    head_dim,
    dtype,
    block_m=16,
    threads=64,
):
    scale = (1.0 / head_dim) ** 0.5
    log2e = 1.4426950408889634
    groups = query_heads // kv_heads
    q_shape = (batch, query_heads, query_len, head_dim)
    kv_shape = (batch, kv_heads, kv_len, head_dim)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        GradOutput: T.Tensor(q_shape, dtype),
        LSE: T.Tensor((batch, query_heads, query_len), T.float32),
        Delta: T.Tensor((batch, query_heads, query_len), T.float32),
        Keep: T.Tensor((batch, num_anchors), T.int32),
        GradK: T.Tensor(kv_shape, T.float32),
        GradV: T.Tensor(kv_shape, T.float32),
    ):
        with T.Kernel(num_anchors, kv_heads, batch, threads=threads) as (ba, hk, bz):
            Q_shared = T.alloc_shared((block_m, head_dim), dtype)
            DO_shared = T.alloc_shared((block_m, head_dim), dtype)
            K_shared = T.alloc_shared((block_m, head_dim), dtype)
            V_shared = T.alloc_shared((block_m, head_dim), dtype)
            P_shared = T.alloc_shared((block_m, block_m), dtype)
            DS_shared = T.alloc_shared((block_m, block_m), dtype)
            Scores = T.alloc_fragment((block_m, block_m), T.float32)
            DP = T.alloc_fragment((block_m, block_m), T.float32)
            DK = T.alloc_fragment((block_m, head_dim), T.float32)
            DV = T.alloc_fragment((block_m, head_dim), T.float32)
            keep = Keep[bz, ba] != 0

            for j, d in T.Parallel(block_m, head_dim):
                if j < block_size:
                    local_index = ctx_len + ba * block_size + j
                    K_shared[j, d] = K[bz, hk, local_index, d]
                    V_shared[j, d] = V[bz, hk, local_index, d]
                else:
                    K_shared[j, d] = 0
                    V_shared[j, d] = 0
            T.clear(DK)
            T.clear(DV)
            for group in T.serial(groups):
                hq = hk * groups + group
                for i, d in T.Parallel(block_m, head_dim):
                    if i < block_size:
                        Q_shared[i, d] = Q[bz, hq, ba * block_size + i, d]
                        DO_shared[i, d] = GradOutput[bz, hq, ba * block_size + i, d]
                    else:
                        Q_shared[i, d] = 0
                        DO_shared[i, d] = 0
                T.clear(Scores)
                T.gemm(
                    Q_shared,
                    K_shared,
                    Scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(block_m, block_m):
                    allowed = (i < block_size) and (j < block_size) and (
                        keep or (i == j)
                    )
                    Scores[i, j] = T.if_then_else(
                        allowed,
                        T.exp2(
                            (
                                Scores[i, j] * scale
                                - LSE[bz, hq, ba * block_size + i]
                            )
                            * log2e
                        ),
                        0,
                    )
                T.copy(Scores, P_shared)
                T.clear(DP)
                T.gemm(
                    DO_shared,
                    V_shared,
                    DP,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(block_m, block_m):
                    DS_shared[i, j] = Scores[i, j] * (
                        DP[i, j] - Delta[bz, hq, ba * block_size + i]
                    ) * scale
                T.gemm(DS_shared, Q_shared, DK, transpose_A=True)
                T.gemm(P_shared, DO_shared, DV, transpose_A=True)
            for j, d in T.Parallel(block_m, head_dim):
                if j < block_size:
                    local_index = ctx_len + ba * block_size + j
                    GradK[bz, hk, local_index, d] = DK[j, d]
                    GradV[bz, hk, local_index, d] = DV[j, d]

    return main


def _tilelang_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return T.bfloat16
    if dtype == torch.float16:
        return T.float16
    raise TypeError(f"Unsupported TileLang DFlash dtype {dtype}")


@functools.lru_cache(maxsize=128)
def _compiled_forward(
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_len: int,
    kv_len: int,
    ctx_len: int,
    num_anchors: int,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype,
):
    block_m = 16 if block_size <= 16 else 32
    groups = query_heads // kv_heads
    heads_per_program = (
        min(2, groups)
        if ctx_len <= 512
        else (groups if block_m == 16 else min(2, groups))
    )
    rows = max(16, block_size * heads_per_program)
    return _forward_factory(
        batch,
        query_heads,
        kv_heads,
        query_len,
        kv_len,
        ctx_len,
        num_anchors,
        block_size,
        head_dim,
        _tilelang_dtype(dtype),
        block_m=block_m,
        block_n=64,
        threads=128 if rows >= 64 else (64 if rows >= 32 else 32),
        num_stages=2,
    )


@functools.lru_cache(maxsize=128)
def _compiled_delta(
    batch: int,
    query_heads: int,
    query_len: int,
    num_anchors: int,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype,
):
    block_m = 16 if block_size <= 16 else 32
    return _delta_factory(
        batch,
        query_heads,
        query_len,
        num_anchors,
        block_size,
        head_dim,
        _tilelang_dtype(dtype),
        block_m=block_m,
        threads=32 if block_m == 16 else 64,
    )


@functools.lru_cache(maxsize=128)
def _compiled_backward(
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_len: int,
    kv_len: int,
    ctx_len: int,
    num_anchors: int,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype,
    write_dkv: bool,
):
    block_m = 16 if block_size <= 16 else 32
    groups = query_heads // kv_heads
    heads_per_program = groups if block_m == 16 else min(2, groups)
    rows = block_m * heads_per_program
    return _backward_atomic_factory(
        batch,
        query_heads,
        kv_heads,
        query_len,
        kv_len,
        ctx_len,
        num_anchors,
        block_size,
        head_dim,
        _tilelang_dtype(dtype),
        block_m=block_m,
        block_n=block_m,
        threads=128 if rows >= 64 else (64 if rows >= 32 else 32),
        num_stages=2,
        write_dkv=write_dkv,
    )


@functools.lru_cache(maxsize=128)
def _compiled_dkv_context(
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_len: int,
    kv_len: int,
    ctx_len: int,
    num_anchors: int,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype,
):
    block_m = 16 if block_size <= 16 else 32
    return _backward_dkv_context_factory(
        batch,
        query_heads,
        kv_heads,
        query_len,
        kv_len,
        ctx_len,
        num_anchors,
        block_size,
        head_dim,
        _tilelang_dtype(dtype),
        block_m=block_m,
        block_n=32,
        threads=128,
    )


@functools.lru_cache(maxsize=128)
def _compiled_dkv_draft(
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_len: int,
    kv_len: int,
    ctx_len: int,
    num_anchors: int,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype,
):
    block_m = 16 if block_size <= 16 else 32
    return _backward_dkv_draft_factory(
        batch,
        query_heads,
        kv_heads,
        query_len,
        kv_len,
        ctx_len,
        num_anchors,
        block_size,
        head_dim,
        _tilelang_dtype(dtype),
        block_m=block_m,
        threads=64 if block_m == 16 else 128,
    )


def tilelang_dflash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    ctx_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, query_heads, query_len, head_dim = query.shape
    kernel = _compiled_forward(
        bsz,
        query_heads,
        key.shape[1],
        query_len,
        key.shape[2],
        int(ctx_len),
        anchor_positions.shape[1],
        int(block_size),
        head_dim,
        query.dtype,
    )
    return kernel(query, key, value, anchor_positions, block_keep_mask)


def _tilelang_backward(
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
    bsz, query_heads, query_len, head_dim = query.shape
    num_anchors = anchor_positions.shape[1]
    delta_kernel = _compiled_delta(
        bsz,
        query_heads,
        query_len,
        num_anchors,
        int(block_size),
        head_dim,
        query.dtype,
    )
    delta = delta_kernel(output, grad_output)
    grad_query_fp32 = torch.zeros_like(query, dtype=torch.float32)
    grad_key_fp32 = torch.zeros_like(key, dtype=torch.float32)
    grad_value_fp32 = torch.zeros_like(value, dtype=torch.float32)
    use_pull_dkv = int(ctx_len) > 2048
    backward_kernel = _compiled_backward(
        bsz,
        query_heads,
        key.shape[1],
        query_len,
        key.shape[2],
        int(ctx_len),
        num_anchors,
        int(block_size),
        head_dim,
        query.dtype,
        not use_pull_dkv,
    )
    backward_kernel(
        query,
        key,
        value,
        grad_output,
        lse,
        delta,
        anchor_positions,
        block_keep_mask,
        grad_query_fp32,
        grad_key_fp32,
        grad_value_fp32,
    )
    if use_pull_dkv:
        if int(ctx_len) > 0:
            context_kernel = _compiled_dkv_context(
                bsz,
                query_heads,
                key.shape[1],
                query_len,
                key.shape[2],
                int(ctx_len),
                num_anchors,
                int(block_size),
                head_dim,
                query.dtype,
            )
            context_kernel(
                query,
                key,
                value,
                grad_output,
                lse,
                delta,
                anchor_positions,
                block_keep_mask,
                grad_key_fp32,
                grad_value_fp32,
            )
        draft_kernel = _compiled_dkv_draft(
            bsz,
            query_heads,
            key.shape[1],
            query_len,
            key.shape[2],
            int(ctx_len),
            num_anchors,
            int(block_size),
            head_dim,
            query.dtype,
        )
        draft_kernel(
            query,
            key,
            value,
            grad_output,
            lse,
            delta,
            block_keep_mask,
            grad_key_fp32,
            grad_value_fp32,
        )
    return (
        grad_query_fp32.to(query.dtype),
        grad_key_fp32.to(key.dtype),
        grad_value_fp32.to(value.dtype),
    )


class _TileLangDFlashAttention(torch.autograd.Function):
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
        output, lse = tilelang_dflash_attention_forward(
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
        grad_query, grad_key, grad_value = _tilelang_backward(
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


def tilelang_dflash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    ctx_len: int,
    block_size: int,
) -> torch.Tensor:
    return _TileLangDFlashAttention.apply(
        query,
        key,
        value,
        anchor_positions,
        block_keep_mask,
        int(ctx_len),
        int(block_size),
    )
