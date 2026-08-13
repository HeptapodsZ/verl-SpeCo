# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Validation and dispatch for custom DFlash attention backends."""

from __future__ import annotations

import torch

DFLASH_ATTENTION_BACKENDS = frozenset(
    {
        "auto",
        "flex",
        "sdpa",
        "triton",
        "triton_two_anchor",
        "triton_persistent",
        "tilelang",
    }
)
_CUSTOM_BACKENDS = frozenset(
    {"triton", "triton_two_anchor", "triton_persistent", "tilelang"}
)
_SUPPORTED_BLOCK_SIZES = frozenset({16})
_SUPPORTED_HEAD_DIMS = frozenset({64, 128})


def _validate_custom_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
) -> None:
    if not query.is_cuda:
        raise RuntimeError("DFlash Triton/TileLang attention requires CUDA tensors")
    if key.device != query.device or value.device != query.device:
        raise ValueError("DFlash custom attention requires Q/K/V on the same device")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"DFlash custom attention requires fp16/bf16, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("DFlash custom attention requires matching Q/K/V dtypes")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("DFlash custom attention expects rank-4 Q/K/V")
    if key.shape != value.shape:
        raise ValueError("DFlash custom attention requires identical K/V shapes")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("DFlash custom attention batch/head dimensions are inconsistent")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("DFlash query heads must be divisible by KV heads")
    if query.shape[-1] not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"DFlash custom attention head_dim must be one of {sorted(_SUPPORTED_HEAD_DIMS)}, "
            f"got {query.shape[-1]}"
        )
    if int(block_size) not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"DFlash custom attention block_size must be one of {sorted(_SUPPORTED_BLOCK_SIZES)}, "
            f"got {block_size}"
        )
    if anchor_positions.ndim != 2 or block_keep_mask.shape != anchor_positions.shape:
        raise ValueError("DFlash anchors/keep mask must have matching [B, A] shapes")
    if anchor_positions.shape[0] != query.shape[0]:
        raise ValueError("DFlash anchor batch size does not match Q/K/V")
    expected_q = anchor_positions.shape[1] * int(block_size)
    if query.shape[2] != expected_q:
        raise ValueError(f"DFlash query length must equal anchors*block_size={expected_q}")
    if key.shape[2] != int(ctx_len) + expected_q:
        raise ValueError("DFlash KV length must equal context length plus draft length")
    if int(ctx_len) < 0 or int(ctx_len) > 65536:
        raise ValueError(f"DFlash custom attention context length must be in [0, 65536], got {ctx_len}")
    if anchor_positions.device != query.device or block_keep_mask.device != query.device:
        raise ValueError("DFlash anchors/keep mask must be on the Q/K/V device")


def dflash_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    ctx_len: int,
    block_size: int,
    backend: str,
) -> torch.Tensor:
    """Run a custom DFlash attention backend with a shared autograd contract."""
    backend = str(backend).lower()
    if backend not in _CUSTOM_BACKENDS:
        raise ValueError(f"Unknown custom DFlash attention backend {backend!r}")
    _validate_custom_inputs(
        query, key, value, anchor_positions, block_keep_mask, ctx_len, block_size
    )
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    anchors_i32 = anchor_positions.to(dtype=torch.int32).contiguous()
    keep_i32 = block_keep_mask.to(dtype=torch.int32).contiguous()
    if backend.startswith("triton"):
        from .triton_attention import triton_dflash_attention

        forward_variant = {
            "triton": "baseline",
            "triton_two_anchor": "two_anchor",
            "triton_persistent": "persistent",
        }[backend]
        return triton_dflash_attention(
            query,
            key,
            value,
            anchors_i32,
            keep_i32,
            ctx_len=int(ctx_len),
            block_size=int(block_size),
            forward_variant=forward_variant,
        )

    try:
        from .tilelang_attention import tilelang_dflash_attention
    except ImportError as exc:
        raise RuntimeError(
            "DFlash TileLang backend was selected but TileLang is unavailable"
        ) from exc
    return tilelang_dflash_attention(
        query,
        key,
        value,
        anchors_i32,
        keep_i32,
        ctx_len=int(ctx_len),
        block_size=int(block_size),
    )
