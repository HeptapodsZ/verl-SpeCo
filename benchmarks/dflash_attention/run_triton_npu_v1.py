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
"""Standalone runner for the DFlash triton_npu_v1 forward kernel.

On Ascend NPU this script validates the kernel against the FP32 dense
reference and reports latency. Without an NPU it can still compile-check and
numerically smoke-test the same kernel text on CUDA (``--smoke-cuda``); that
path validates the algorithm and the Triton frontend only, it is NOT evidence
of NPU performance or NPU codegen correctness.

Examples:

    # Correctness + timing on NPU (small shapes so the dense FP32 reference stays cheap)
    python benchmarks/dflash_attention/run_triton_npu_v1.py

    # Training-scale timing on NPU (no dense reference; finiteness checks only)
    python benchmarks/dflash_attention/run_triton_npu_v1.py --large

    # Compile + numerics smoke of the same kernel text on a CUDA dev box
    python benchmarks/dflash_attention/run_triton_npu_v1.py --smoke-cuda

    # Kernel tuning knobs (raw launcher, bypasses the dispatch NPU gate)
    python benchmarks/dflash_attention/run_triton_npu_v1.py --direct --block-n 128 --num-programs 40
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl_speco.models.dflash.kernels import (
    dense_dflash_attention_reference,
    dflash_sparse_attention,
    tensor_error_metrics,
)
from verl_speco.models.dflash.kernels.triton_npu_attention import (
    default_npu_program_count,
    triton_npu_dflash_attention_forward,
)

FORWARD_LIMITS = {
    "atol": 2e-2,
    "rtol": 2e-2,
    "relative_l2": 5e-3,
    "cosine": 0.9999,
}


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def resolve_device() -> tuple[str, torch.device]:
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu", torch.device("npu")
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    return "cpu", torch.device("cpu")


def _synchronize(device_type: str) -> None:
    if device_type == "npu":
        torch.npu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()


def make_inputs(
    batch: int,
    ctx_len: int,
    num_anchors: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    seed: int = 0,
    dummy_every: int = 7,
) -> tuple[torch.Tensor, ...]:
    """Deterministic inputs: sorted anchors, a 1-in-7 dummy block, both fp types."""
    block_size = 16
    draft_len = num_anchors * block_size
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(
        (batch, query_heads, draft_len, head_dim), generator=generator, dtype=dtype
    )
    key = torch.randn(
        (batch, kv_heads, ctx_len + draft_len, head_dim),
        generator=generator,
        dtype=dtype,
    )
    value = torch.randn(
        (batch, kv_heads, ctx_len + draft_len, head_dim),
        generator=generator,
        dtype=dtype,
    )
    if ctx_len > 0:
        anchors = torch.randint(
            0, ctx_len + 1, (batch, num_anchors), generator=generator, dtype=torch.int64
        )
        anchors = anchors.sort(dim=-1).values
    else:
        anchors = torch.zeros((batch, num_anchors), dtype=torch.int64)
    keep = torch.ones((batch, num_anchors), dtype=torch.bool)
    if dummy_every > 0:
        keep[:, dummy_every - 1 :: dummy_every] = 0
    return (
        query.to(device),
        key.to(device),
        value.to(device),
        anchors.to(device),
        keep.to(device),
    )


def _print_metrics(prefix: str, metrics: dict, limits: dict | None = None) -> bool:
    passed = True
    print(f"  {prefix}:")
    for name in ("max_abs", "mean_abs", "rmse", "relative_l2", "cosine"):
        print(f"    {name:<14s} {metrics[name]:.6e}")
    print(
        f"    nan          actual={metrics['actual_nan']} reference={metrics['reference_nan']}"
        f"  inf actual={metrics['actual_inf']} reference={metrics['reference_inf']}"
    )
    if limits is not None:
        for name in ("atol", "rtol"):
            if not metrics["allclose"]:
                passed = False
        for name in ("relative_l2", "cosine"):
            limit = limits[name]
            if name == "cosine":
                ok = metrics[name] >= limit
            else:
                ok = metrics[name] <= limit
            if not ok:
                passed = False
    return passed


def run_correctness_case(
    device_type: str,
    device: torch.device,
    args: argparse.Namespace,
) -> bool:
    dtype = _dtype(args.dtype)
    block_size = 16
    query, key, value, anchors, keep = make_inputs(
        args.batch,
        args.ctx_len,
        args.num_anchors,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
        dtype,
        device,
    )
    print(
        f"[case] B={args.batch} ctx={args.ctx_len} anchors={args.num_anchors} "
        f"HQ/HK={args.query_heads}/{args.kv_heads} D={args.head_dim} "
        f"dtype={args.dtype} device={device_type}"
    )
    reference, reference_lse = dense_dflash_attention_reference(
        query, key, value, anchors, keep, args.ctx_len, block_size
    )

    if args.direct or device_type != "npu":
        # The dispatch NPU gate rejects non-NPU devices, so the CUDA smoke and
        # explicit tuning runs call the raw launcher directly.
        anchors_i32 = anchors.to(torch.int32).contiguous()
        keep_i32 = keep.to(torch.int32).contiguous()
        actual, lse = triton_npu_dflash_attention_forward(
            query,
            key,
            value,
            anchors_i32,
            keep_i32,
            ctx_len=args.ctx_len,
            block_size=block_size,
            num_programs=args.num_programs,
            block_n=args.block_n,
        )
    else:
        actual = dflash_sparse_attention(
            query,
            key,
            value,
            anchors,
            keep,
            ctx_len=args.ctx_len,
            block_size=block_size,
            backend="triton_npu_v1",
        )
        # LSE is only exposed by the raw launcher; the dispatch contract
        # returns the output tensor alone.
        _, lse = triton_npu_dflash_attention_forward(
            query,
            key,
            value,
            anchors.to(torch.int32).contiguous(),
            keep.to(torch.int32).contiguous(),
            ctx_len=args.ctx_len,
            block_size=block_size,
        )

    out_metrics = tensor_error_metrics(actual, reference, atol=2e-2, rtol=2e-2)
    ok = _print_metrics("output vs dense reference", out_metrics, FORWARD_LIMITS)
    lse_metrics = tensor_error_metrics(lse, reference_lse)
    _print_metrics("lse vs dense reference", lse_metrics)
    ok &= lse_metrics["max_abs"] < 2e-2 and lse_metrics["actual_nan"] == 0
    return ok


def run_large_case(device_type: str, device: torch.device, args: argparse.Namespace) -> None:
    dtype = _dtype(args.dtype)
    block_size = 16
    query, key, value, anchors, keep = make_inputs(
        1,
        args.ctx_len,
        args.num_anchors,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
        dtype,
        device,
    )
    anchors_i32 = anchors.to(torch.int32).contiguous()
    keep_i32 = keep.to(torch.int32).contiguous()
    print(
        f"[large] B=1 ctx={args.ctx_len} anchors={args.num_anchors} "
        f"HQ/HK={args.query_heads}/{args.kv_heads} D={args.head_dim} "
        f"dtype={args.dtype} device={device_type}"
    )

    def call() -> torch.Tensor:
        return triton_npu_dflash_attention_forward(
            query,
            key,
            value,
            anchors_i32,
            keep_i32,
            ctx_len=args.ctx_len,
            block_size=block_size,
            num_programs=args.num_programs,
            block_n=args.block_n,
        )[0]

    output, _ = call()  # compile/warmup
    _synchronize(device_type)
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("[large] FAILED: non-finite output")
        sys.exit(1)
    iterations = int(args.iters)
    times_ms = []
    for _ in range(iterations):
        _synchronize(device_type)
        start = time.perf_counter()
        call()
        _synchronize(device_type)
        times_ms.append((time.perf_counter() - start) * 1e3)
    median = statistics.median(times_ms)
    draft_len = args.num_anchors * block_size
    print(f"[large] median {median:.3f} ms over {iterations} iterations")
    print(
        f"[large] {1e3 / median:.2f} calls/s, "
        f"{args.query_heads * draft_len / (median * 1e-3) / 1e6:.2f} M query tokens/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and validate the DFlash triton_npu_v1 forward kernel"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--num-anchors", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128, choices=(64, 128))
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "bfloat16"))
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-programs", type=int, default=None)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="call the raw launcher so --block-n/--num-programs take effect",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="training-scale timing run without a dense reference",
    )
    parser.add_argument(
        "--smoke-cuda",
        action="store_true",
        help="compile-check and numerics smoke of the same kernel text on CUDA",
    )
    args = parser.parse_args()

    if args.smoke_cuda:
        if not torch.cuda.is_available():
            print("CUDA unavailable; nothing to smoke")
            sys.exit(2)
        device_type, device = "cuda", torch.device("cuda")
        print(
            "[warning] CUDA smoke validates the Triton frontend and the kernel "
            "algorithm only; it is not NPU validation."
        )
        ok = run_correctness_case(device_type, device, args)
        print("[smoke-cuda] PASSED" if ok else "[smoke-cuda] FAILED")
        sys.exit(0 if ok else 1)

    device_type, device = resolve_device()
    if device_type != "npu":
        print(
            "This runner requires an Ascend NPU with torch_npu and "
            "Triton-Ascend. On a CUDA dev box use --smoke-cuda to compile-check "
            "the kernel text. Exiting without running the NPU kernel."
        )
        sys.exit(2)

    print(f"[device] npu core program default: {default_npu_program_count()}")
    if args.large:
        run_large_case(device_type, device, args)
        return
    ok = run_correctness_case(device_type, device, args)
    print("[npu] PASSED" if ok else "[npu] FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
