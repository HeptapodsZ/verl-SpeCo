# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Focused SDPA-versus-Triton benchmark for DFlash attention.

The public process launches one fresh worker process per backend. This keeps
CUDA allocator state and backend compilation state from contaminating the
peak-memory comparison. Only forward and forward-plus-backward are measured.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.dflash_attention.benchmark import (  # noqa: E402
    Case,
    backend_callable,
    make_inputs,
    visible_pairs,
)


BACKENDS = ("sdpa", "triton")
PHASES = ("forward", "forward_backward")


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _discard(value: object) -> None:
    del value


def measure_cuda_latency(
    call: Callable[[], object],
    *,
    warmup: int,
    rounds: int,
    min_iterations: int,
    min_seconds: float,
) -> dict[str, float | int | list[float]]:
    """Measure steady-state GPU elapsed time with one sample per round."""
    for _ in range(warmup):
        _discard(call())
    torch.cuda.synchronize()

    start_cpu = time.perf_counter()
    estimate_value = call()
    torch.cuda.synchronize()
    estimate_ms = (time.perf_counter() - start_cpu) * 1e3
    del estimate_value
    iterations = max(
        min_iterations,
        int(math.ceil(min_seconds * 1000.0 / max(estimate_ms * rounds, 1e-6))),
    )

    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            value = call()
            del value
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)

    return {
        "rounds": rounds,
        "iterations_per_round": iterations,
        "p50_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "round_ms": samples,
    }


def measure_cuda_memory(call: Callable[[], object]) -> dict[str, float]:
    """Measure one phase after clearing only unoccupied CUDA cache blocks."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    value = call()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del value
    torch.cuda.synchronize()

    mib = float(2**20)
    return {
        "baseline_allocated_mib": baseline_allocated / mib,
        "baseline_reserved_mib": baseline_reserved / mib,
        "peak_allocated_mib": peak_allocated / mib,
        "peak_reserved_mib": peak_reserved / mib,
        "peak_allocated_delta_mib": (peak_allocated - baseline_allocated) / mib,
        "peak_reserved_delta_mib": (peak_reserved - baseline_reserved) / mib,
    }


def _add_throughput(
    phase: dict[str, object], *, query_tokens: int, effective_qk_pairs: int
) -> None:
    seconds = float(phase["p50_ms"]) / 1000.0
    phase["query_tokens_per_second"] = query_tokens / seconds
    phase["effective_qk_pairs_per_second"] = effective_qk_pairs / seconds


def _iter_cases(args: argparse.Namespace) -> Iterable[Case]:
    for batch_size in _parse_ints(args.batch_sizes):
        for context_len in _parse_ints(args.context_lens):
            for block_size in _parse_ints(args.block_sizes):
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


def _environment() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    try:
        import triton

        triton_version = triton.__version__
    except (ImportError, AttributeError):
        triton_version = "unknown"
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            ".".join(str(item) for item in torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
        "git_commit": commit,
    }


def _benchmark_case(
    backend: str, case: Case, args: argparse.Namespace
) -> dict[str, object]:
    inputs = make_inputs(case, args.seed)
    call, structural_bytes = backend_callable(backend, case, inputs)
    effective_pairs = visible_pairs(case, inputs)
    dense_pairs = (
        case.batch_size
        * case.query_heads
        * case.query_len
        * (case.context_len + case.query_len)
    )
    query_tokens = case.batch_size * case.query_len

    def forward() -> torch.Tensor:
        with torch.no_grad():
            return call(inputs.query, inputs.key, inputs.value)

    def forward_backward() -> tuple[torch.Tensor, ...]:
        query = inputs.query.detach().requires_grad_(True)
        key = inputs.key.detach().requires_grad_(True)
        value = inputs.value.detach().requires_grad_(True)
        output = call(query, key, value)
        return torch.autograd.grad(
            output,
            (query, key, value),
            inputs.grad_output,
        )

    phase_calls = {
        "forward": forward,
        "forward_backward": forward_backward,
    }
    phases: dict[str, object] = {}
    for phase_name, phase_call in phase_calls.items():
        latency = measure_cuda_latency(
            phase_call,
            warmup=args.warmup,
            rounds=args.rounds,
            min_iterations=args.min_iterations,
            min_seconds=args.min_seconds,
        )
        _add_throughput(
            latency,
            query_tokens=query_tokens,
            effective_qk_pairs=effective_pairs,
        )
        latency["memory"] = measure_cuda_memory(phase_call)
        phases[phase_name] = latency

    record = {
        "case_id": case.case_id,
        "case": asdict(case),
        "status": "ok",
        "mask_or_metadata_mib": structural_bytes / 2**20,
        "effective_qk_pairs": effective_pairs,
        "dense_qk_pairs": dense_pairs,
        "effective_attention_density": effective_pairs / dense_pairs,
        "phases": phases,
    }
    return record


def run_worker(backend: str, args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("The SDPA/Triton comparison requires CUDA")
    records = []
    for case in _iter_cases(args):
        print(f"[{backend}] {case.case_id}", flush=True)
        try:
            records.append(_benchmark_case(backend, case, args))
        except torch.OutOfMemoryError as exc:
            records.append(
                {
                    "case_id": case.case_id,
                    "case": asdict(case),
                    "status": "capacity_limit",
                    "error": str(exc),
                }
            )
        except Exception as exc:
            records.append(
                {
                    "case_id": case.case_id,
                    "case": asdict(case),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return {
        "backend": backend,
        "environment": _environment(),
        "cases": records,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def combine_worker_payloads(
    workers: dict[str, dict[str, object]], *, command: str
) -> dict[str, object]:
    """Join isolated worker results and calculate direct comparison metrics."""
    indexed = {
        backend: {
            str(record["case_id"]): record
            for record in payload["cases"]  # type: ignore[index]
        }
        for backend, payload in workers.items()
    }
    case_ids = list(indexed["sdpa"])
    combined_cases = []
    for case_id in case_ids:
        backend_records = {
            backend: indexed[backend][case_id]
            for backend in BACKENDS
            if case_id in indexed[backend]
        }
        combined: dict[str, object] = {
            "case_id": case_id,
            "case": backend_records["sdpa"]["case"],
            "backends": backend_records,
        }
        if all(record.get("status") == "ok" for record in backend_records.values()):
            phase_comparisons = {}
            for phase_name in PHASES:
                sdpa = backend_records["sdpa"]["phases"][phase_name]
                triton = backend_records["triton"]["phases"][phase_name]
                sdpa_memory = sdpa["memory"]
                triton_memory = triton["memory"]
                phase_comparisons[phase_name] = {
                    "triton_latency_speedup": _safe_ratio(
                        float(sdpa["p50_ms"]), float(triton["p50_ms"])
                    ),
                    "triton_query_throughput_speedup": _safe_ratio(
                        float(triton["query_tokens_per_second"]),
                        float(sdpa["query_tokens_per_second"]),
                    ),
                    "triton_effective_qk_throughput_speedup": _safe_ratio(
                        float(triton["effective_qk_pairs_per_second"]),
                        float(sdpa["effective_qk_pairs_per_second"]),
                    ),
                    "triton_peak_allocated_saving_mib": (
                        float(sdpa_memory["peak_allocated_mib"])
                        - float(triton_memory["peak_allocated_mib"])
                    ),
                    "triton_peak_allocated_reduction_fraction": 1.0
                    - _safe_ratio(
                        float(triton_memory["peak_allocated_mib"]),
                        float(sdpa_memory["peak_allocated_mib"]),
                    ),
                    "triton_peak_allocated_delta_reduction_fraction": 1.0
                    - _safe_ratio(
                        float(triton_memory["peak_allocated_delta_mib"]),
                        float(sdpa_memory["peak_allocated_delta_mib"]),
                    ),
                }
            combined["comparison"] = phase_comparisons
        combined_cases.append(combined)

    return {
        "schema_version": 1,
        "environment": workers["sdpa"]["environment"],
        "command": command,
        "methodology": {
            "backend_process_isolation": True,
            "latency_statistic": "p50 of per-round CUDA-event means",
            "throughput_basis": "p50 latency",
            "qk_throughput_semantics": "effective visible QK pairs per training step",
            "memory_semantics": (
                "single phase after warmup; inputs and mask/metadata remain resident"
            ),
        },
        "cases": combined_cases,
    }


def flatten_comparisons(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return one side-by-side CSV row per case and phase."""
    rows: list[dict[str, object]] = []
    for case_record in payload["cases"]:  # type: ignore[index]
        case = case_record["case"]
        backends = case_record["backends"]
        comparison = case_record.get("comparison", {})
        for phase_name in PHASES:
            row: dict[str, object] = {
                **case,
                "case_id": case_record["case_id"],
                "phase": phase_name,
            }
            for backend in BACKENDS:
                backend_record = backends.get(backend, {})
                row[f"{backend}_status"] = backend_record.get("status", "missing")
                phase = backend_record.get("phases", {}).get(phase_name, {})
                memory = phase.get("memory", {})
                for key in (
                    "p50_ms",
                    "p90_ms",
                    "p99_ms",
                    "query_tokens_per_second",
                    "effective_qk_pairs_per_second",
                ):
                    row[f"{backend}_{key}"] = phase.get(key)
                for key in (
                    "baseline_allocated_mib",
                    "baseline_reserved_mib",
                    "peak_allocated_mib",
                    "peak_reserved_mib",
                    "peak_allocated_delta_mib",
                    "peak_reserved_delta_mib",
                ):
                    row[f"{backend}_{key}"] = memory.get(key)
            row.update(comparison.get(phase_name, {}))
            rows.append(row)
    return rows


def write_outputs(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    rows = flatten_comparisons(payload)
    csv_path = output.with_suffix(".csv")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _worker_command(
    args: argparse.Namespace, backend: str, worker_output: Path
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-backend",
        backend,
        "--worker-output",
        str(worker_output),
        "--batch-sizes",
        args.batch_sizes,
        "--context-lens",
        args.context_lens,
        "--block-sizes",
        args.block_sizes,
        "--num-anchors",
        str(args.num_anchors),
        "--query-heads",
        str(args.query_heads),
        "--kv-heads",
        str(args.kv_heads),
        "--head-dim",
        str(args.head_dim),
        "--dtype",
        args.dtype,
        "--anchor-distribution",
        args.anchor_distribution,
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--min-iterations",
        str(args.min_iterations),
        "--min-seconds",
        str(args.min_seconds),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DFlash SDPA and Triton forward/F+B performance."
    )
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument(
        "--context-lens", default="512,2048,8192,16384,32768,65536"
    )
    parser.add_argument("--block-sizes", default="16")
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--anchor-distribution", choices=("early", "uniform", "late"), default="uniform"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-iterations", type=int, default=20)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/dflash_attention_sdpa_vs_triton.json"),
    )
    parser.add_argument("--worker-backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.worker_backend:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker-backend")
        worker_payload = run_worker(args.worker_backend, args)
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(
            json.dumps(worker_payload, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return

    workers: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="dflash_sdpa_triton_") as temp_dir:
        for backend in BACKENDS:
            worker_output = Path(temp_dir) / f"{backend}.json"
            print(f"[compare] launching isolated {backend} worker", flush=True)
            subprocess.run(
                _worker_command(args, backend, worker_output),
                cwd=REPO_ROOT,
                check=True,
            )
            workers[backend] = json.loads(worker_output.read_text(encoding="utf-8"))

    payload = combine_worker_payloads(workers, command=" ".join(sys.argv))
    write_outputs(payload, args.output)
    print(f"[compare] wrote {args.output}", flush=True)
    print(f"[compare] wrote {args.output.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
