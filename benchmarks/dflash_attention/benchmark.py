# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Accuracy-gated DFlashAttention benchmark for Flex, SDPA, Triton, and TileLang.

Run this file in a fresh process per backend when comparing peak memory. The
default matrix is the six-point B=1, block_size=16 RTX 5080 acceptance matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl_speco.models.dflash.kernels import (
    build_dflash_dense_attention_mask,
    current_device_profile,
    dflash_sparse_attention,
    tensor_error_metrics,
)


BACKENDS = (
    "flex",
    "sdpa",
    "triton",
    "triton_two_anchor",
    "triton_persistent",
    "triton_one_grid",
    "triton_one_fixed_grid",
    "tilelang",
)
FORWARD_LIMITS = {
    "atol": 2e-2,
    "rtol": 2e-2,
    "relative_l2": 5e-3,
    "cosine": 0.9999,
}
BACKWARD_LIMITS = {
    "atol": 3e-2,
    "rtol": 3e-2,
    "relative_l2": 1e-2,
    "cosine": 0.999,
}


@dataclass(frozen=True)
class Case:
    batch_size: int
    context_len: int
    block_size: int
    num_anchors: int = 64
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "bfloat16"
    anchor_distribution: str = "uniform"

    @property
    def query_len(self) -> int:
        return self.num_anchors * self.block_size

    @property
    def case_id(self) -> str:
        return (
            f"b{self.batch_size}_c{self.context_len}_a{self.num_anchors}_"
            f"bs{self.block_size}_h{self.query_heads}_{self.kv_heads}_d{self.head_dim}"
        )


@dataclass
class Inputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    grad_output: torch.Tensor
    anchors: torch.Tensor
    keep: torch.Tensor


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def make_inputs(case: Case, seed: int) -> Inputs:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape_q = (
        case.batch_size,
        case.query_heads,
        case.query_len,
        case.head_dim,
    )
    shape_kv = (
        case.batch_size,
        case.kv_heads,
        case.context_len + case.query_len,
        case.head_dim,
    )
    dtype = _dtype(case.dtype)
    query = torch.randn(shape_q, generator=generator, device="cuda", dtype=dtype)
    key = torch.randn(shape_kv, generator=generator, device="cuda", dtype=dtype)
    value = torch.randn(shape_kv, generator=generator, device="cuda", dtype=dtype)
    grad_output = torch.randn(
        shape_q, generator=generator, device="cuda", dtype=dtype
    )
    if case.anchor_distribution == "early":
        base = torch.linspace(0, max(case.context_len // 4, 1), case.num_anchors)
    elif case.anchor_distribution == "late":
        base = torch.linspace(
            max(0, case.context_len * 3 // 4), case.context_len, case.num_anchors
        )
    elif case.anchor_distribution == "uniform":
        base = torch.linspace(0, case.context_len, case.num_anchors)
    else:
        raise ValueError(f"Unknown anchor distribution {case.anchor_distribution!r}")
    anchors = base.round().to(device="cuda", dtype=torch.int32)
    anchors = anchors.clamp_(0, case.context_len).unsqueeze(0)
    anchors = anchors.expand(case.batch_size, -1).contiguous()
    keep = torch.ones(
        (case.batch_size, case.num_anchors), device="cuda", dtype=torch.bool
    )
    return Inputs(query, key, value, grad_output, anchors, keep)


def make_block_mask(case: Case, inputs: Inputs):
    from torch.nn.attention.flex_attention import create_block_mask

    anchors = inputs.anchors
    keep = inputs.keep
    context_len = case.context_len
    block_size = case.block_size

    def mask_mod(b, h, q_idx, kv_idx):
        del h
        block_id = q_idx // block_size
        normal = ((kv_idx < context_len) & (kv_idx < anchors[b, block_id])) | (
            (kv_idx >= context_len)
            & (((kv_idx - context_len) // block_size) == block_id)
        )
        return torch.where(
            keep[b, block_id], normal, kv_idx == context_len + q_idx
        )

    return create_block_mask(
        mask_mod,
        B=case.batch_size,
        H=None,
        Q_LEN=case.query_len,
        KV_LEN=case.context_len + case.query_len,
        device="cuda",
    )


def backend_callable(
    backend: str,
    case: Case,
    inputs: Inputs,
    *,
    fixed_grid_size: int = 40,
) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], int]:
    if backend == "flex":
        from torch.nn.attention.flex_attention import flex_attention

        block_mask = make_block_mask(case, inputs)
        compiled_flex = torch.compile(flex_attention, fullgraph=True, dynamic=False)

        def call(query, key, value):
            return compiled_flex(
                query, key, value, block_mask=block_mask, enable_gqa=True
            )

        mask_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in block_mask.as_tuple()
            if isinstance(tensor, torch.Tensor)
        )
        return call, mask_bytes

    if backend == "sdpa":
        dense_mask = build_dflash_dense_attention_mask(
            inputs.anchors, inputs.keep, case.context_len, case.block_size
        )
        groups = case.query_heads // case.kv_heads

        def call(query, key, value):
            return F.scaled_dot_product_attention(
                query,
                key.repeat_interleave(groups, dim=1),
                value.repeat_interleave(groups, dim=1),
                attn_mask=dense_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        return call, dense_mask.numel() * dense_mask.element_size()

    if backend not in (
        "triton",
        "triton_two_anchor",
        "triton_persistent",
        "triton_one_grid",
        "triton_one_fixed_grid",
        "tilelang",
    ):
        raise ValueError(f"Unknown backend {backend!r}")

    def call(query, key, value):
        return dflash_sparse_attention(
            query,
            key,
            value,
            inputs.anchors,
            inputs.keep,
            ctx_len=case.context_len,
            block_size=case.block_size,
            backend=backend,
            fixed_grid_size=fixed_grid_size,
        )

    metadata_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (inputs.anchors, inputs.keep)
    )
    return call, metadata_bytes


def _sync_time(call: Callable[[], object]) -> tuple[object, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1e3


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _bootstrap_median_ci(values: list[float], samples: int = 2000) -> list[float]:
    if len(values) == 1:
        return [values[0], values[0]]
    generator = random.Random(0)
    medians = []
    for _ in range(samples):
        draw = [values[generator.randrange(len(values))] for _ in values]
        medians.append(statistics.median(draw))
    return [_percentile(medians, 0.025), _percentile(medians, 0.975)]


def _bootstrap_ratio_ci(
    numerator: list[float], denominator: list[float], samples: int = 2000
) -> list[float]:
    generator = random.Random(0)
    ratios = []
    for _ in range(samples):
        num = statistics.median(
            [numerator[generator.randrange(len(numerator))] for _ in numerator]
        )
        den = statistics.median(
            [denominator[generator.randrange(len(denominator))] for _ in denominator]
        )
        ratios.append(num / den)
    return [_percentile(ratios, 0.025), _percentile(ratios, 0.975)]


def measure_cuda(
    call: Callable[[], object],
    *,
    warmup: int,
    rounds: int,
    min_iterations: int,
    min_seconds: float,
) -> dict[str, float | int | list[float]]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    _, estimate_ms = _sync_time(call)
    iterations = max(
        min_iterations,
        int(math.ceil(min_seconds * 1000.0 / max(estimate_ms * rounds, 1e-6))),
    )
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {
        "rounds": rounds,
        "iterations_per_round": iterations,
        "p50_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "bootstrap_median_ci95_ms": _bootstrap_median_ci(samples),
        "round_ms": samples,
    }


def _valid_metrics(metrics: dict[str, float], limits: dict[str, float]) -> bool:
    finite_counts_match = (
        metrics["actual_nan"] == metrics["reference_nan"] == 0
        and metrics["actual_inf"] == metrics["reference_inf"] == 0
    )
    return bool(
        finite_counts_match
        and metrics.get("allclose", False)
        and metrics["relative_l2"] <= limits["relative_l2"]
        and metrics["cosine"] >= limits["cosine"]
    )


def compute_accuracy(
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: Inputs,
    reference: dict[str, torch.Tensor],
) -> tuple[dict, bool]:
    def run_once() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        query = inputs.query.detach().clone().requires_grad_(True)
        key = inputs.key.detach().clone().requires_grad_(True)
        value = inputs.value.detach().clone().requires_grad_(True)
        output = call(query, key, value)
        grads = torch.autograd.grad(output, (query, key, value), inputs.grad_output)
        return output.detach(), tuple(grad.detach() for grad in grads)

    output, grads = run_once()
    ctx_len = inputs.key.shape[2] - inputs.query.shape[2]
    forward_metrics = lambda actual, expected: tensor_error_metrics(
        actual,
        expected,
        atol=FORWARD_LIMITS["atol"],
        rtol=FORWARD_LIMITS["rtol"],
    )
    backward_metrics = lambda actual, expected: tensor_error_metrics(
        actual,
        expected,
        atol=BACKWARD_LIMITS["atol"],
        rtol=BACKWARD_LIMITS["rtol"],
    )
    metrics = {
        "output": forward_metrics(output, reference["output"]),
        "dQ": backward_metrics(grads[0], reference["dQ"]),
        "dK": backward_metrics(grads[1], reference["dK"]),
        "dK_context": tensor_error_metrics(
            grads[1][:, :, :ctx_len],
            reference["dK"][:, :, :ctx_len],
            atol=BACKWARD_LIMITS["atol"],
            rtol=BACKWARD_LIMITS["rtol"],
        ),
        "dK_draft": tensor_error_metrics(
            grads[1][:, :, ctx_len:],
            reference["dK"][:, :, ctx_len:],
            atol=BACKWARD_LIMITS["atol"],
            rtol=BACKWARD_LIMITS["rtol"],
        ),
        "dV": backward_metrics(grads[2], reference["dV"]),
        "dV_context": tensor_error_metrics(
            grads[2][:, :, :ctx_len],
            reference["dV"][:, :, :ctx_len],
            atol=BACKWARD_LIMITS["atol"],
            rtol=BACKWARD_LIMITS["rtol"],
        ),
        "dV_draft": tensor_error_metrics(
            grads[2][:, :, ctx_len:],
            reference["dV"][:, :, ctx_len:],
            atol=BACKWARD_LIMITS["atol"],
            rtol=BACKWARD_LIMITS["rtol"],
        ),
    }
    repeat_drift = []
    for _ in range(2):
        repeated_output, repeated_grads = run_once()
        repeat_drift.append(
            {
                "output": forward_metrics(repeated_output, output),
                "dQ": backward_metrics(repeated_grads[0], grads[0]),
                "dK": backward_metrics(repeated_grads[1], grads[1]),
                "dV": backward_metrics(repeated_grads[2], grads[2]),
            }
        )
    metrics["repeat_drift"] = repeat_drift
    valid = _valid_metrics(metrics["output"], FORWARD_LIMITS) and all(
        _valid_metrics(metrics[name], BACKWARD_LIMITS)
        for name in (
            "dQ",
            "dK",
            "dK_context",
            "dK_draft",
            "dV",
            "dV_context",
            "dV_draft",
        )
    )
    valid = valid and all(
        _valid_metrics(drift["output"], FORWARD_LIMITS)
        and all(
            _valid_metrics(drift[name], BACKWARD_LIMITS)
            for name in ("dQ", "dK", "dV")
        )
        for drift in repeat_drift
    )
    return metrics, valid


def make_sdpa_reference(case: Case, inputs: Inputs) -> dict[str, torch.Tensor]:
    call, _ = backend_callable("sdpa", case, inputs)
    query = inputs.query.detach().clone().requires_grad_(True)
    key = inputs.key.detach().clone().requires_grad_(True)
    value = inputs.value.detach().clone().requires_grad_(True)
    output = call(query, key, value)
    grads = torch.autograd.grad(output, (query, key, value), inputs.grad_output)
    return {
        "output": output.detach(),
        "dQ": grads[0].detach(),
        "dK": grads[1].detach(),
        "dV": grads[2].detach(),
    }


def visible_pairs(case: Case, inputs: Inputs) -> int:
    normal = inputs.keep.to(torch.int64) * (
        inputs.anchors.to(torch.int64) + case.block_size
    )
    dummy = (~inputs.keep).to(torch.int64)
    per_query_block = normal + dummy
    return int((per_query_block.sum() * case.block_size * case.query_heads).item())


def add_rates(result: dict, case: Case, pairs: int) -> None:
    query_tokens = case.batch_size * case.query_len
    for phase, flop_multiplier in (("forward", 4), ("backward", 8), ("forward_backward", 12)):
        timing = result.get(phase)
        if not timing:
            continue
        seconds = timing["p50_ms"] / 1000.0
        timing["query_tokens_per_second"] = query_tokens / seconds
        timing["qk_pairs_per_second"] = pairs / seconds
        timing["effective_tflops"] = (
            flop_multiplier * case.head_dim * pairs / seconds / 1e12
        )


def add_acceptance(case_record: dict) -> None:
    results = {result["backend"]: result for result in case_record["results"]}
    if "flex" not in results or "sdpa" not in results:
        return
    flex = results["flex"]
    sdpa = results["sdpa"]
    if flex.get("status") != "ok" or sdpa.get("status") != "ok":
        return
    for backend in ("triton", "tilelang"):
        result = results.get(backend)
        if result is None or result.get("status") != "ok":
            continue
        comparisons = {}
        phases_pass = []
        for phase in ("forward", "backward"):
            custom_samples = result[phase]["round_ms"]
            flex_samples = flex[phase]["round_ms"]
            sdpa_samples = sdpa[phase]["round_ms"]
            sdpa_speedup = sdpa[phase]["p50_ms"] / result[phase]["p50_ms"]
            flex_ratio = result[phase]["p50_ms"] / flex[phase]["p50_ms"]
            sdpa_speedup_ci = _bootstrap_ratio_ci(sdpa_samples, custom_samples)
            flex_ratio_ci = _bootstrap_ratio_ci(custom_samples, flex_samples)
            phase_pass = sdpa_speedup_ci[0] > 1.0 and flex_ratio_ci[1] <= 1.10
            comparisons[phase] = {
                "sdpa_speedup": sdpa_speedup,
                "sdpa_speedup_ci95": sdpa_speedup_ci,
                "latency_vs_flex": flex_ratio,
                "latency_vs_flex_ci95": flex_ratio_ci,
                "accepted": phase_pass,
            }
            phases_pass.append(phase_pass)
        memory_pass = (
            result["memory"]["peak_allocated_delta_mib"]
            <= sdpa["memory"]["peak_allocated_delta_mib"]
        )
        comparisons["memory_not_above_sdpa"] = memory_pass
        comparisons["accepted"] = bool(
            result["accuracy_valid"] and all(phases_pass) and memory_pass
        )
        result["acceptance"] = comparisons


def benchmark_backend(
    backend: str,
    case: Case,
    inputs: Inputs,
    reference: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict:
    result: dict = {"backend": backend, "status": "running"}
    call, mask_bytes = backend_callable(
        backend, case, inputs, fixed_grid_size=args.fixed_grid_size
    )
    result["mask_or_metadata_mib"] = mask_bytes / 2**20

    query = inputs.query.detach().clone().requires_grad_(True)
    key = inputs.key.detach().clone().requires_grad_(True)
    value = inputs.value.detach().clone().requires_grad_(True)
    (output, compile_forward_ms) = _sync_time(lambda: call(query, key, value))
    (_, compile_backward_ms) = _sync_time(
        lambda: torch.autograd.grad(
            output,
            (query, key, value),
            inputs.grad_output,
            retain_graph=True,
        )
    )
    result["compile_forward_ms"] = compile_forward_ms
    result["compile_backward_ms"] = compile_backward_ms
    result["accuracy"], result["accuracy_valid"] = compute_accuracy(
        call, inputs, reference
    )
    if not result["accuracy_valid"]:
        result["status"] = "invalid_accuracy"
        return result
    if args.accuracy_only:
        result["status"] = "accuracy_only"
        return result

    with torch.no_grad():
        result["forward"] = measure_cuda(
            lambda: call(inputs.query, inputs.key, inputs.value),
            warmup=args.warmup,
            rounds=args.rounds,
            min_iterations=args.min_iterations,
            min_seconds=args.min_seconds,
        )

    query = inputs.query.detach().clone().requires_grad_(True)
    key = inputs.key.detach().clone().requires_grad_(True)
    value = inputs.value.detach().clone().requires_grad_(True)
    output = call(query, key, value)
    result["backward"] = measure_cuda(
        lambda: torch.autograd.grad(
            output,
            (query, key, value),
            inputs.grad_output,
            retain_graph=True,
        ),
        warmup=args.warmup,
        rounds=args.rounds,
        min_iterations=args.min_iterations,
        min_seconds=args.min_seconds,
    )

    def forward_backward():
        q = inputs.query.detach().requires_grad_(True)
        k = inputs.key.detach().requires_grad_(True)
        v = inputs.value.detach().requires_grad_(True)
        out = call(q, k, v)
        return torch.autograd.grad(out, (q, k, v), inputs.grad_output)

    result["forward_backward"] = measure_cuda(
        forward_backward,
        warmup=args.warmup,
        rounds=args.rounds,
        min_iterations=args.min_iterations,
        min_seconds=args.min_seconds,
    )
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    memory_values = forward_backward()
    torch.cuda.synchronize()
    del memory_values
    result["memory"] = {
        "baseline_allocated_mib": baseline_allocated / 2**20,
        "baseline_reserved_mib": baseline_reserved / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "peak_allocated_delta_mib": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        )
        / 2**20,
        "peak_reserved_delta_mib": (
            torch.cuda.max_memory_reserved() - baseline_reserved
        )
        / 2**20,
    }
    add_rates(result, case, visible_pairs(case, inputs))
    result["status"] = "ok"
    return result


def environment() -> dict:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": current_device_profile(),
        "git_commit": git_commit,
        "command": " ".join(os.sys.argv),
    }


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def iter_cases(args: argparse.Namespace) -> Iterable[Case]:
    for batch_size in parse_ints(args.batch_sizes):
        for context_len in parse_ints(args.context_lens):
            for block_size in parse_ints(args.block_sizes):
                yield Case(
                    batch_size=batch_size,
                    context_len=context_len,
                    block_size=block_size,
                    num_anchors=args.num_anchors,
                    query_heads=args.query_heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    dtype=args.dtype,
                    anchor_distribution=args.anchor_distribution,
                )


def flatten_rows(payload: dict) -> list[dict]:
    rows = []
    for case_result in payload["cases"]:
        case = case_result["case"]
        for result in case_result["results"]:
            row = {**case, "backend": result["backend"], "status": result["status"]}
            row["accuracy_valid"] = result.get("accuracy_valid")
            for phase in ("forward", "backward", "forward_backward"):
                for key, value in result.get(phase, {}).items():
                    if key != "round_ms":
                        row[f"{phase}_{key}"] = value
            for key, value in result.get("memory", {}).items():
                row[key] = value
            for tensor, metrics in result.get("accuracy", {}).items():
                if tensor == "repeat_drift":
                    row["repeat_drift"] = json.dumps(metrics, separators=(",", ":"))
                    continue
                for key, value in metrics.items():
                    row[f"{tensor}_{key}"] = value
            acceptance = result.get("acceptance", {})
            row["accepted"] = acceptance.get("accepted")
            row["memory_not_above_sdpa"] = acceptance.get("memory_not_above_sdpa")
            for phase in ("forward", "backward"):
                for key, value in acceptance.get(phase, {}).items():
                    row[f"acceptance_{phase}_{key}"] = value
            if "error" in result:
                row["error"] = result["error"]
            rows.append(row)
    return rows


def write_outputs(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    rows = flatten_rows(payload)
    csv_path = output.with_suffix(".csv")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--context-lens", default="512,2048,8192,16384,32768,65536")
    parser.add_argument("--block-sizes", default="16")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--fixed-grid-size", type=int, default=40)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--anchor-distribution", choices=("early", "uniform", "late"), default="uniform"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-iterations", type=int, default=100)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/dflash_attention.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The DFlashAttention benchmark requires CUDA")
    if args.fixed_grid_size <= 0:
        raise ValueError("--fixed-grid-size must be positive")
    backends = tuple(item for item in args.backends.split(",") if item)
    unknown = set(backends) - set(BACKENDS)
    if unknown:
        raise ValueError(f"Unknown backends: {sorted(unknown)}")

    payload = {"schema_version": 1, "environment": environment(), "cases": []}
    for case in iter_cases(args):
        case_record = {"case": asdict(case), "results": []}
        print(f"[dflash] {case.case_id}", flush=True)
        try:
            inputs = make_inputs(case, args.seed)
            reference = make_sdpa_reference(case, inputs)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            for backend in backends:
                case_record["results"].append(
                    {
                        "backend": backend,
                        "status": "capacity_limit_reference",
                        "error": str(exc),
                    }
                )
            payload["cases"].append(case_record)
            write_outputs(payload, args.output)
            continue

        for backend in backends:
            try:
                result = benchmark_backend(backend, case, inputs, reference, args)
            except torch.OutOfMemoryError as exc:
                result = {"backend": backend, "status": "capacity_limit", "error": str(exc)}
                torch.cuda.empty_cache()
            except Exception as exc:  # preserve the remaining matrix and the exact failure
                result = {
                    "backend": backend,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            print(f"  {backend}: {result['status']}", flush=True)
            case_record["results"].append(result)
            write_outputs(payload | {"cases": payload["cases"] + [case_record]}, args.output)
        add_acceptance(case_record)
        payload["cases"].append(case_record)
        del reference, inputs
        torch.cuda.empty_cache()
    write_outputs(payload, args.output)


if __name__ == "__main__":
    main()
