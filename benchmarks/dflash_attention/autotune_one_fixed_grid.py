# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Small, educational autotuner for DFlash ``triton_one_fixed_grid`` forward.

This script deliberately focuses on one kernel and one forward backend:

* ``_forward_grid_stride_kernel``
* ``ANCHOR_MAJOR=True`` (the ``one_fixed_grid`` work ordering)

``triton.autotune`` selects the fastest candidate for the requested shape.  The
selected output is then checked against PyTorch SDPA using the exact dense
DFlash mask.  This is a teaching example, not the production offline tuner in
``tune_triton.py``: autotune chooses by latency first, and the SDPA correctness
gate is applied to the winning configuration afterwards.

Run from the repository root under the project WSL environment, for example:

    python benchmarks/dflash_attention/autotune_one_fixed_grid.py

Use smaller values for a quick demonstration:

    python benchmarks/dflash_attention/autotune_one_fixed_grid.py \
        --context-len 128 --num-anchors 8 --query-heads 4 --kv-heads 2 \
        --head-dim 64 --grid-sizes 20,40
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl_speco.models.dflash.kernels.reference import (  # noqa: E402
    build_dflash_dense_attention_mask,
)
from verl_speco.models.dflash.kernels.triton_attention import (  # noqa: E402
    _forward_grid_stride_kernel,
)


BLOCK_SIZE = 16


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated CLI list such as ``32,64,128``."""
    values = [int(item) for item in text.split(",") if item]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def make_autotuned_kernel(
    block_ns: list[int],
    warp_counts: list[int],
    stage_counts: list[int],
    grid_sizes: list[int],
):
    """Attach a transparent ``triton.autotune`` wrapper to the real kernel.

    ``BLOCK_M`` remains 16 because the current DFlash backend supports an exact
    16-token draft block.  The other values express the useful teaching space:

    * BLOCK_N: number of context K/V rows streamed per iteration;
    * num_warps: threads cooperating on one Triton program;
    * num_stages: K/V load-compute software-pipeline depth;
    * NUM_PROGRAMS: size of the fixed program pool.
    """
    configs = [
        triton.Config(
            {
                "BLOCK_M": BLOCK_SIZE,
                "BLOCK_N": block_n,
                "PIPELINE_STAGES": num_stages,
                "NUM_PROGRAMS": num_programs,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_n in block_ns
        for num_warps in warp_counts
        for num_stages in stage_counts
        for num_programs in grid_sizes
    ]
    return triton.autotune(
        configs=configs,
        key=[
            "HQ",
            "HK",
            "Q_LEN",
            "KV_LEN",
            "CTX_LEN",
            "NUM_ANCHORS",
            "HEAD_DIM",
            "TOTAL_WORK",
        ],
    )(_forward_grid_stride_kernel)


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    """Create contiguous Q/K/V plus representative structural metadata."""
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    query_len = args.num_anchors * BLOCK_SIZE
    kv_len = args.context_len + query_len
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    query = torch.randn(
        args.batch_size,
        args.query_heads,
        query_len,
        args.head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    key = torch.randn(
        args.batch_size,
        args.kv_heads,
        kv_len,
        args.head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    value = torch.randn(
        args.batch_size,
        args.kv_heads,
        kv_len,
        args.head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )

    # Sorted anchors model the real DFlash sampler.  Anchor 0 is made a dummy
    # block so the self-only numerical-safety path is covered as well.
    anchors = torch.linspace(
        0, args.context_len, args.num_anchors, device="cuda"
    ).round()
    anchors = anchors.to(torch.int32).unsqueeze(0)
    anchors = anchors.expand(args.batch_size, -1).contiguous()
    keep = torch.ones_like(anchors, dtype=torch.int32)
    anchors[:, 0] = 0
    keep[:, 0] = 0
    return query, key, value, anchors, keep


def launch(
    kernel,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchors: torch.Tensor,
    keep: torch.Tensor,
    *,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch exactly the production ``one_fixed_grid`` mapping."""
    batch_size, query_heads, query_len, head_dim = query.shape
    kv_heads = key.shape[1]
    num_anchors = anchors.shape[1]
    groups = query_heads // kv_heads
    heads_per_program = min(2, groups)
    head_chunks = triton.cdiv(groups, heads_per_program)
    active_rows = BLOCK_SIZE * heads_per_program
    query_rows = max(16, triton.next_power_of_2(active_rows))
    batch_kv_chunks = batch_size * kv_heads * head_chunks
    total_work = num_anchors * batch_kv_chunks

    output = torch.empty_like(query)
    lse = torch.empty(
        batch_size,
        query_heads,
        query_len,
        device=query.device,
        dtype=torch.float32,
    )

    # NUM_PROGRAMS comes from each triton.Config, so the launch grid and the
    # grid-stride step always agree for that candidate.
    grid = lambda meta: (meta["NUM_PROGRAMS"],)
    kernel[grid](
        query,
        key,
        value,
        anchors,
        keep,
        output,
        lse,
        HQ=query_heads,
        HK=kv_heads,
        Q_LEN=query_len,
        KV_LEN=key.shape[2],
        CTX_LEN=context_len,
        NUM_ANCHORS=num_anchors,
        SCALE=1.0 / math.sqrt(head_dim),
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=head_dim,
        GROUPS=groups,
        HEADS_PER_PROGRAM=heads_per_program,
        HEAD_CHUNKS=head_chunks,
        ACTIVE_ROWS=active_rows,
        QUERY_ROWS=query_rows,
        TOTAL_WORK=total_work,
        ANCHOR_MAJOR=True,
    )
    return output, lse


def sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    anchors: torch.Tensor,
    keep: torch.Tensor,
    *,
    context_len: int,
) -> torch.Tensor:
    """Run SDPA with the same prefix/local-block/dummy-self attention mask."""
    groups = query.shape[1] // key.shape[1]
    dense_mask = build_dflash_dense_attention_mask(
        anchors, keep.bool(), context_len, BLOCK_SIZE
    )
    return F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(groups, dim=1),
        value.repeat_interleave(groups, dim=1),
        attn_mask=dense_mask,
        dropout_p=0.0,
        is_causal=False,
    )


def accuracy_metrics(
    actual: torch.Tensor, reference: torch.Tensor, *, atol: float, rtol: float
) -> dict[str, float | bool]:
    actual_f = actual.float()
    reference_f = reference.float()
    diff = actual_f - reference_f
    return {
        "allclose": torch.allclose(actual_f, reference_f, atol=atol, rtol=rtol),
        "max_abs": diff.abs().max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference_f).clamp_min(1e-12)
        ).item(),
        "cosine": F.cosine_similarity(
            actual_f.reshape(1, -1), reference_f.reshape(1, -1)
        ).item(),
    }


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this script requires a CUDA GPU")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query-heads must be divisible by kv-heads")
    if args.head_dim not in (64, 128):
        raise ValueError("head-dim must be 64 or 128")
    if args.context_len < 0 or args.context_len > 65536:
        raise ValueError("context-len must be in [0, 65536]")
    if any(value not in (2, 4, 8) for value in args.warp_counts):
        raise ValueError("warp-counts may contain only 2, 4, or 8")
    if any(value not in (1, 2, 3, 4) for value in args.stage_counts):
        raise ValueError("stage-counts may contain only 1, 2, 3, or 4")
    if any(value < 16 or value & (value - 1) for value in args.block_ns):
        raise ValueError("block-ns must contain powers of two >= 16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-ns", type=parse_int_list, default=parse_int_list("32,64,128"))
    parser.add_argument("--warp-counts", type=parse_int_list, default=parse_int_list("4,8"))
    parser.add_argument("--stage-counts", type=parse_int_list, default=parse_int_list("1,2"))
    parser.add_argument("--grid-sizes", type=parse_int_list, default=parse_int_list("40,80"))
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    query, key, value, anchors, keep = make_inputs(args)
    kernel = make_autotuned_kernel(
        args.block_ns,
        args.warp_counts,
        args.stage_counts,
        args.grid_sizes,
    )

    print(f"GPU: {torch.cuda.get_device_name(query.device)}")
    print(
        "Shape: "
        f"B={args.batch_size}, C={args.context_len}, A={args.num_anchors}, "
        f"Hq/Hkv={args.query_heads}/{args.kv_heads}, D={args.head_dim}, "
        f"dtype={args.dtype}"
    )
    print(f"Candidates: {len(kernel.configs)} (first call runs Triton autotune)")

    with torch.no_grad():
        # The first call benchmarks all candidates.  A second call uses the
        # cached winner and leaves output/LSE populated by that winner.
        launch(kernel, query, key, value, anchors, keep, context_len=args.context_len)
        output, _ = launch(
            kernel, query, key, value, anchors, keep, context_len=args.context_len
        )
        reference = sdpa_reference(
            query, key, value, anchors, keep, context_len=args.context_len
        )
        torch.cuda.synchronize()

        metrics = accuracy_metrics(
            output, reference, atol=args.atol, rtol=args.rtol
        )

    best = kernel.best_config
    print("\nBest triton.Config:")
    print(f"  kwargs={best.kwargs}")
    print(f"  num_warps={best.num_warps}, num_stages={best.num_stages}")
    print("\nAccuracy vs SDPA:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    if not metrics["allclose"]:
        raise AssertionError(
            f"one_fixed_grid does not match SDPA at atol={args.atol}, rtol={args.rtol}"
        )


if __name__ == "__main__":
    main()
