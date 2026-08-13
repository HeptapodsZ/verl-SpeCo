# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Backend-independent DFlash attention references and accuracy metrics."""

from __future__ import annotations

import math

import torch


def build_dflash_dense_attention_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
) -> torch.Tensor:
    """Build the exact boolean mask used by DFlash block-denoising training."""
    bsz, num_blocks = anchor_positions.shape
    device = anchor_positions.device
    draft_len = num_blocks * int(block_size)
    query_indices = torch.arange(draft_len, device=device)
    query_block_ids = query_indices // int(block_size)
    query_anchors = anchor_positions.index_select(1, query_block_ids)
    query_valid = block_keep_mask.index_select(1, query_block_ids)

    context_indices = torch.arange(int(ctx_len), device=device)
    context_allowed = context_indices.view(1, 1, -1) < query_anchors.unsqueeze(-1)
    draft_block_ids = torch.arange(draft_len, device=device) // int(block_size)
    draft_allowed = (
        query_block_ids.view(1, draft_len, 1)
        == draft_block_ids.view(1, 1, draft_len)
    ).expand(bsz, -1, -1)
    allowed = torch.cat([context_allowed, draft_allowed], dim=-1)

    # Dummy rows are excluded from the loss. Giving each query its own draft
    # key keeps softmax finite and matches the existing SDPA contract.
    total_len = int(ctx_len) + draft_len
    key_indices = torch.arange(total_len, device=device)
    safe_self = key_indices.view(1, 1, total_len) == (
        int(ctx_len) + query_indices
    ).view(1, draft_len, 1)
    allowed = torch.where(query_valid.unsqueeze(-1), allowed, safe_self)
    return allowed.unsqueeze(1)


def dense_dflash_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a small-shape FP32 dense reference and natural-log LSE.

    This function intentionally does not call FlexAttention or SDPA. It is for
    correctness tests and must not be used for long-context benchmarks.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("DFlash attention expects rank-4 Q/K/V tensors")
    if key.shape != value.shape:
        raise ValueError("DFlash key and value shapes must match")
    if query.shape[0] != key.shape[0]:
        raise ValueError("DFlash Q/K/V batch sizes must match")
    groups = query.shape[1] // key.shape[1]
    if groups * key.shape[1] != query.shape[1]:
        raise ValueError("DFlash query heads must be divisible by KV heads")

    q = query.float()
    k = key.float().repeat_interleave(groups, dim=1)
    v = value.float().repeat_interleave(groups, dim=1)
    mask = build_dflash_dense_attention_mask(
        anchor_positions, block_keep_mask, ctx_len, block_size
    )
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v), lse


def tensor_error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, float | int | bool]:
    """Return stable on-device reductions used by accuracy benchmarks."""
    actual_f = actual.float()
    reference_f = reference.float()
    diff = actual_f - reference_f
    ref_norm = torch.linalg.vector_norm(reference_f)
    actual_norm = torch.linalg.vector_norm(actual_f)
    rel_l2 = torch.linalg.vector_norm(diff) / ref_norm.clamp_min(1e-12)
    flat_actual = actual_f.reshape(-1)
    flat_reference = reference_f.reshape(-1)
    if flat_actual.numel() == 0 or (actual_norm == 0 and ref_norm == 0):
        cosine = torch.ones((), device=actual.device)
    else:
        cosine = torch.nn.functional.cosine_similarity(
            flat_actual, flat_reference, dim=0, eps=1e-12
        )
    metrics: dict[str, float | int | bool] = {
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "rmse": float(diff.square().mean().sqrt().item()) if diff.numel() else 0.0,
        "relative_l2": float(rel_l2.item()),
        "cosine": float(cosine.item()),
        "actual_nan": float(torch.isnan(actual_f).sum().item()),
        "reference_nan": float(torch.isnan(reference_f).sum().item()),
        "actual_inf": float(torch.isinf(actual_f).sum().item()),
        "reference_inf": float(torch.isinf(reference_f).sum().item()),
    }
    if (atol is None) != (rtol is None):
        raise ValueError("atol and rtol must be specified together")
    if atol is not None and rtol is not None:
        close = torch.isclose(actual_f, reference_f, atol=atol, rtol=rtol)
        metrics["allclose"] = bool(close.all().item())
        metrics["mismatch_count"] = int((~close).sum().item())
    return metrics
