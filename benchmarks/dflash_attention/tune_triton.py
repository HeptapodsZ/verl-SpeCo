# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Correctness-gated offline autotuner for the five DFlash Triton variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.dflash_attention.benchmark import (  # noqa: E402
    BACKWARD_LIMITS,
    FORWARD_LIMITS,
    Case,
    Inputs,
    _bootstrap_median_ci,
    _bootstrap_ratio_ci,
    _percentile,
    backend_callable,
    compute_accuracy,
    environment,
    make_inputs,
    make_sdpa_reference,
    measure_cuda,
    parse_ints,
)
from verl_speco.models.dflash.kernels import tensor_error_metrics  # noqa: E402
from verl_speco.models.dflash.kernels.tuning import (  # noqa: E402
    DFlashKernelTuning,
    DFlashTuningKey,
    current_device_profile,
    get_triton_backward_tuning,
    get_triton_tuning,
    override_triton_tuning,
    triton_tuning_candidates,
)


VARIANT_TO_BACKEND = {
    "baseline": "triton",
    "two_anchor": "triton_two_anchor",
    "persistent": "triton_persistent",
    "one_grid": "triton_one_grid",
    "one_fixed_grid": "triton_one_fixed_grid",
}
TUNING_PROFILE_ENV = "VERL_SPECO_DFLASH_TUNING_PROFILE"
PROVENANCE_FILES = (
    "verl_speco/models/dflash/kernels/tuning.py",
    "verl_speco/models/dflash/kernels/triton_attention.py",
    "benchmarks/dflash_attention/benchmark.py",
    "benchmarks/dflash_attention/tune_triton.py",
)


def _valid_metrics(metrics: dict[str, float], limits: dict[str, float]) -> bool:
    return bool(
        metrics["actual_nan"] == metrics["reference_nan"] == 0
        and metrics["actual_inf"] == metrics["reference_inf"] == 0
        and metrics["allclose"]
        and metrics["relative_l2"] <= limits["relative_l2"]
        and metrics["cosine"] >= limits["cosine"]
    )


def _accuracy_once(
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: Inputs,
    reference: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, float]], bool]:
    query = inputs.query.detach().clone().requires_grad_(True)
    key = inputs.key.detach().clone().requires_grad_(True)
    value = inputs.value.detach().clone().requires_grad_(True)
    output = call(query, key, value)
    grads = torch.autograd.grad(output, (query, key, value), inputs.grad_output)
    metrics = {
        "output": tensor_error_metrics(
            output.detach(),
            reference["output"],
            atol=FORWARD_LIMITS["atol"],
            rtol=FORWARD_LIMITS["rtol"],
        ),
        "dQ": tensor_error_metrics(
            grads[0],
            reference["dQ"],
            atol=BACKWARD_LIMITS["atol"],
            rtol=BACKWARD_LIMITS["rtol"],
        ),
        "dK": tensor_error_metrics(
            grads[1],
            reference["dK"],
            atol=BACKWARD_LIMITS["atol"],
            rtol=BACKWARD_LIMITS["rtol"],
        ),
        "dV": tensor_error_metrics(
            grads[2],
            reference["dV"],
            atol=BACKWARD_LIMITS["atol"],
            rtol=BACKWARD_LIMITS["rtol"],
        ),
    }
    valid = _valid_metrics(metrics["output"], FORWARD_LIMITS) and all(
        _valid_metrics(metrics[name], BACKWARD_LIMITS) for name in ("dQ", "dK", "dV")
    )
    return metrics, valid


def _measure_phase(
    phase: str,
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: Inputs,
    args: argparse.Namespace,
) -> dict[str, object]:
    if phase == "forward":
        with torch.no_grad():
            return measure_cuda(
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
    return measure_cuda(
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


def _measure_pair(
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: Inputs,
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    result = {
        "forward": _measure_phase("forward", call, inputs, args),
        "backward": _measure_phase("backward", call, inputs, args),
    }

    def forward_backward():
        query = inputs.query.detach().requires_grad_(True)
        key = inputs.key.detach().requires_grad_(True)
        value = inputs.value.detach().requires_grad_(True)
        output = call(query, key, value)
        return torch.autograd.grad(output, (query, key, value), inputs.grad_output)

    result["forward_backward"] = measure_cuda(
        forward_backward,
        warmup=args.warmup,
        rounds=args.rounds,
        min_iterations=args.min_iterations,
        min_seconds=args.min_seconds,
    )
    return result


def _shape_kwargs(case: Case, variant: str, fixed_grid_size: int) -> dict[str, object]:
    return {
        "forward_variant": variant,
        "block_size": case.block_size,
        "ctx_len": case.context_len,
        "device": torch.device("cuda"),
        "batch_size": case.batch_size,
        "num_anchors": case.num_anchors,
        "query_heads": case.query_heads,
        "kv_heads": case.kv_heads,
        "head_dim": case.head_dim,
        "dtype": case.dtype,
        "fixed_grid_size": fixed_grid_size,
    }


def _tuning_key(case: Case, variant: str, fixed_grid_size: int) -> DFlashTuningKey:
    return DFlashTuningKey(
        device_profile=current_device_profile(),
        forward_variant=variant,
        batch_size=case.batch_size,
        context_len=case.context_len,
        block_size=case.block_size,
        num_anchors=case.num_anchors,
        query_heads=case.query_heads,
        kv_heads=case.kv_heads,
        head_dim=case.head_dim,
        dtype=case.dtype,
        fixed_grid_size=fixed_grid_size,
    )


def tuning_environment() -> dict[str, object]:
    result = environment()
    result["triton"] = importlib.metadata.version("triton")
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_status = ["unavailable"]
    result["git_status_short"] = git_status
    result["source_sha256"] = {
        relative_path: hashlib.sha256(
            (REPO_ROOT / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in PROVENANCE_FILES
    }
    return result


def _config_trial(
    *,
    phase: str,
    config: DFlashKernelTuning,
    forward_config: DFlashKernelTuning,
    backward_config: DFlashKernelTuning,
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: Inputs,
    reference: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, object]:
    result: dict[str, object] = {"config": asdict(config), "status": "running"}
    try:
        with override_triton_tuning(forward=forward_config, backward=backward_config):
            accuracy, accuracy_valid = _accuracy_once(call, inputs, reference)
            result["accuracy"] = accuracy
            result["accuracy_valid"] = accuracy_valid
            if not accuracy_valid:
                result["status"] = "invalid_accuracy"
                return result
            result["timing"] = _measure_phase(phase, call, inputs, args)
        result["status"] = "ok"
    except torch.OutOfMemoryError as exc:
        result.update(status="capacity_limit", error=str(exc))
        torch.cuda.empty_cache()
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def _best_config(
    trials: list[dict[str, object]],
    baseline: DFlashKernelTuning,
    baseline_ms: float,
    min_speedup: float,
) -> DFlashKernelTuning:
    valid = [trial for trial in trials if trial["status"] == "ok"]
    if not valid:
        return baseline
    baseline_dict = asdict(baseline)
    baseline_trial = next(
        (trial for trial in valid if trial["config"] == baseline_dict), None
    )
    if baseline_trial is not None:
        baseline_ms = float(baseline_trial["timing"]["p50_ms"])
    best = min(valid, key=lambda trial: float(trial["timing"]["p50_ms"]))
    best_ms = float(best["timing"]["p50_ms"])
    if baseline_ms / best_ms < min_speedup:
        return baseline
    return DFlashKernelTuning(**best["config"])


def _evaluate_pair(
    *,
    forward_config: DFlashKernelTuning,
    backward_config: DFlashKernelTuning,
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: Inputs,
    reference: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, object]:
    result: dict[str, object] = {
        "forward": asdict(forward_config),
        "backward": asdict(backward_config),
        "status": "running",
    }
    try:
        with override_triton_tuning(forward=forward_config, backward=backward_config):
            accuracy, accuracy_valid = compute_accuracy(call, inputs, reference)
            result["accuracy"] = accuracy
            result["accuracy_valid"] = accuracy_valid
            if not accuracy_valid:
                result["status"] = "invalid_accuracy"
                return result
            result["timing"] = _measure_pair(call, inputs, args)
        result["status"] = "ok"
    except torch.OutOfMemoryError as exc:
        result.update(status="capacity_limit", error=str(exc))
        torch.cuda.empty_cache()
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def _summarize_samples(samples: list[float], iterations: int) -> dict[str, object]:
    return {
        "rounds": len(samples),
        "iterations_per_round": iterations,
        "p50_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "bootstrap_median_ci95_ms": _bootstrap_median_ci(samples),
        "round_ms": samples,
    }


def _measure_interleaved_forward_backward(
    *,
    baseline_forward: DFlashKernelTuning,
    baseline_backward: DFlashKernelTuning,
    candidate_forward: DFlashKernelTuning,
    candidate_backward: DFlashKernelTuning,
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: Inputs,
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    configs = (
        ("baseline", baseline_forward, baseline_backward),
        ("candidate", candidate_forward, candidate_backward),
    )

    def forward_backward():
        query = inputs.query.detach().requires_grad_(True)
        key = inputs.key.detach().requires_grad_(True)
        value = inputs.value.detach().requires_grad_(True)
        output = call(query, key, value)
        return torch.autograd.grad(output, (query, key, value), inputs.grad_output)

    estimates = {}
    for name, forward_config, backward_config in configs:
        with override_triton_tuning(forward=forward_config, backward=backward_config):
            for _ in range(args.warmup):
                forward_backward()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward_backward()
            end.record()
            end.synchronize()
            estimates[name] = start.elapsed_time(end)
    iterations = max(
        args.min_iterations,
        int(
            math.ceil(
                args.min_seconds
                * 1000.0
                / max(max(estimates.values()) * args.rounds, 1e-6)
            )
        ),
    )
    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for round_index in range(args.rounds):
        ordered_configs = configs if round_index % 2 == 0 else tuple(reversed(configs))
        for name, forward_config, backward_config in ordered_configs:
            with override_triton_tuning(
                forward=forward_config, backward=backward_config
            ):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    forward_backward()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / iterations)
    return {
        name: _summarize_samples(values, iterations) for name, values in samples.items()
    }


def _select_pair(
    pair_trials: list[dict[str, object]], min_speedup: float
) -> dict[str, object]:
    baseline = pair_trials[0]
    valid = [trial for trial in pair_trials if trial["status"] == "ok"]
    if baseline["status"] != "ok" or not valid:
        raise ValueError("A valid baseline pair is required for final selection")
    eligible = []
    for trial in valid:
        if trial is baseline:
            trial["forward_backward_speedup"] = 1.0
            trial["forward_backward_speedup_ci95"] = [1.0, 1.0]
            continue
        comparison = trial.get("paired_forward_backward")
        if comparison is None:
            continue
        baseline_timing = comparison["baseline"]
        candidate_timing = comparison["candidate"]
        speedup = float(baseline_timing["p50_ms"]) / float(candidate_timing["p50_ms"])
        speedup_ci95 = _bootstrap_ratio_ci(
            baseline_timing["round_ms"], candidate_timing["round_ms"]
        )
        trial["forward_backward_speedup"] = speedup
        trial["forward_backward_speedup_ci95"] = speedup_ci95
        if speedup >= min_speedup and speedup_ci95[0] > 1.0:
            eligible.append(trial)
    return (
        max(eligible, key=lambda trial: float(trial["forward_backward_speedup"]))
        if eligible
        else baseline
    )


def tune_variant(
    case: Case,
    variant: str,
    inputs: Inputs,
    reference: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, object]:
    backend = VARIANT_TO_BACKEND[variant]
    call, _ = backend_callable(
        backend, case, inputs, fixed_grid_size=args.fixed_grid_size
    )
    shape_kwargs = _shape_kwargs(case, variant, args.fixed_grid_size)
    baseline_forward = get_triton_tuning(**shape_kwargs)
    baseline_backward = get_triton_backward_tuning(**shape_kwargs)

    with override_triton_tuning(forward=baseline_forward, backward=baseline_backward):
        baseline_accuracy, baseline_valid = compute_accuracy(call, inputs, reference)
        if not baseline_valid:
            return {
                "variant": variant,
                "backend": backend,
                "status": "invalid_baseline_accuracy",
                "baseline_accuracy": baseline_accuracy,
            }
        baseline_timing = _measure_pair(call, inputs, args)

    forward_trials = []
    for config in triton_tuning_candidates(
        phase="forward",
        forward_variant=variant,
        block_size=case.block_size,
        ctx_len=case.context_len,
        head_dim=case.head_dim,
        full=args.full_search,
    ):
        print(f"    forward {asdict(config)}", flush=True)
        forward_trials.append(
            _config_trial(
                phase="forward",
                config=config,
                forward_config=config,
                backward_config=baseline_backward,
                call=call,
                inputs=inputs,
                reference=reference,
                args=args,
            )
        )
    best_forward = _best_config(
        forward_trials,
        baseline_forward,
        float(baseline_timing["forward"]["p50_ms"]),
        args.min_speedup,
    )

    backward_trials = []
    for config in triton_tuning_candidates(
        phase="backward",
        forward_variant=variant,
        block_size=case.block_size,
        ctx_len=case.context_len,
        head_dim=case.head_dim,
        full=args.full_search,
    ):
        print(f"    backward {asdict(config)}", flush=True)
        backward_trials.append(
            _config_trial(
                phase="backward",
                config=config,
                forward_config=best_forward,
                backward_config=config,
                call=call,
                inputs=inputs,
                reference=reference,
                args=args,
            )
        )
    best_backward = _best_config(
        backward_trials,
        baseline_backward,
        float(baseline_timing["backward"]["p50_ms"]),
        args.min_speedup,
    )

    # Re-measure every unique finalist after compilation, then validate each
    # non-baseline finalist with alternating baseline/candidate round order.
    pair_configs = dict.fromkeys(
        (
            (baseline_forward, baseline_backward),
            (best_forward, baseline_backward),
            (baseline_forward, best_backward),
            (best_forward, best_backward),
        )
    )
    pair_trials = [
        _evaluate_pair(
            forward_config=forward_config,
            backward_config=backward_config,
            call=call,
            inputs=inputs,
            reference=reference,
            args=args,
        )
        for forward_config, backward_config in pair_configs
    ]
    baseline_pair = pair_trials[0]
    valid_pairs = [trial for trial in pair_trials if trial["status"] == "ok"]
    if baseline_pair["status"] != "ok" or not valid_pairs:
        return {
            "variant": variant,
            "backend": backend,
            "status": "invalid_validation_baseline",
            "pair_trials": pair_trials,
        }
    for pair_trial in valid_pairs[1:]:
        try:
            pair_trial["paired_forward_backward"] = (
                _measure_interleaved_forward_backward(
                    baseline_forward=baseline_forward,
                    baseline_backward=baseline_backward,
                    candidate_forward=DFlashKernelTuning(**pair_trial["forward"]),
                    candidate_backward=DFlashKernelTuning(**pair_trial["backward"]),
                    call=call,
                    inputs=inputs,
                    args=args,
                )
            )
        except Exception as exc:
            pair_trial["paired_validation_error"] = f"{type(exc).__name__}: {exc}"
    selected_pair = _select_pair(pair_trials, args.min_speedup)

    baseline_report_timing = dict(baseline_pair["timing"])
    selected_report_timing = dict(selected_pair["timing"])
    if selected_pair is not baseline_pair:
        comparison = selected_pair["paired_forward_backward"]
        baseline_report_timing["forward_backward"] = comparison["baseline"]
        selected_report_timing["forward_backward"] = comparison["candidate"]
    speedups = {}
    for phase in ("forward", "backward", "forward_backward"):
        speedups[phase] = float(baseline_report_timing[phase]["p50_ms"]) / float(
            selected_report_timing[phase]["p50_ms"]
        )
    return {
        "variant": variant,
        "backend": backend,
        "status": "ok",
        "key": asdict(_tuning_key(case, variant, args.fixed_grid_size)),
        "baseline": {
            "forward": asdict(baseline_forward),
            "backward": asdict(baseline_backward),
            "timing": baseline_report_timing,
            "accuracy": baseline_pair["accuracy"],
            "initial_timing": baseline_timing,
            "initial_accuracy": baseline_accuracy,
        },
        "forward_trials": forward_trials,
        "backward_trials": backward_trials,
        "pair_trials": pair_trials,
        "tuned": {
            "forward": selected_pair["forward"],
            "backward": selected_pair["backward"],
            "timing": selected_report_timing,
            "accuracy": selected_pair["accuracy"],
        },
        "speedup": speedups,
        "forward_backward_speedup_ci95": selected_pair["forward_backward_speedup_ci95"],
    }


def estimate_peak_bytes(case: Case) -> int:
    """Conservative preflight estimate including the SDPA reference."""
    element_size = 2
    query = case.batch_size * case.query_heads * case.query_len * case.head_dim
    kv = (
        case.batch_size
        * case.kv_heads
        * (case.context_len + case.query_len)
        * case.head_dim
    )
    groups = case.query_heads // case.kv_heads
    dense_mask = case.batch_size * case.query_len * (case.context_len + case.query_len)
    return int(
        12 * (query + 2 * kv) * element_size
        + 6 * groups * kv * element_size
        + dense_mask
    )


def iter_cases(args: argparse.Namespace) -> Iterable[Case]:
    for batch_size in parse_ints(args.batch_sizes):
        for context_len in parse_ints(args.context_lens):
            for head_dim in parse_ints(args.head_dims):
                yield Case(
                    batch_size=batch_size,
                    context_len=context_len,
                    block_size=16,
                    num_anchors=args.num_anchors,
                    query_heads=args.query_heads,
                    kv_heads=args.kv_heads,
                    head_dim=head_dim,
                    dtype=args.dtype,
                    anchor_distribution=args.anchor_distribution,
                )


def _summary_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for case_record in payload["cases"]:
        case = case_record["case"]
        for result in case_record["results"]:
            row = {**case, "variant": result["variant"], "status": result["status"]}
            if result["status"] == "ok":
                for phase in ("forward", "backward", "forward_backward"):
                    row[f"baseline_{phase}_p50_ms"] = result["baseline"]["timing"][
                        phase
                    ]["p50_ms"]
                    row[f"tuned_{phase}_p50_ms"] = result["tuned"]["timing"][phase][
                        "p50_ms"
                    ]
                    row[f"{phase}_speedup"] = result["speedup"][phase]
                row["forward_backward_speedup_ci95"] = json.dumps(
                    result["forward_backward_speedup_ci95"]
                )
                row["baseline_forward_config"] = json.dumps(
                    result["baseline"]["forward"], sort_keys=True
                )
                row["baseline_backward_config"] = json.dumps(
                    result["baseline"]["backward"], sort_keys=True
                )
                row["tuned_forward_config"] = json.dumps(
                    result["tuned"]["forward"], sort_keys=True
                )
                row["tuned_backward_config"] = json.dumps(
                    result["tuned"]["backward"], sort_keys=True
                )
            rows.append(row)
    return rows


def write_outputs(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    rows = _summary_rows(payload)
    fieldnames = sorted({key for row in rows for key in row})
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--context-lens", default="512,2048,8192")
    parser.add_argument("--head-dims", default="64,128")
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--fixed-grid-size", type=int, default=40)
    parser.add_argument("--variants", default=",".join(VARIANT_TO_BACKEND))
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--anchor-distribution", choices=("early", "uniform", "late"), default="uniform"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-iterations", type=int, default=20)
    parser.add_argument("--min-seconds", type=float, default=0.2)
    parser.add_argument("--min-speedup", type=float, default=1.01)
    parser.add_argument("--full-search", action="store_true")
    parser.add_argument("--max-memory-fraction", type=float, default=0.75)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/dflash_triton_tuning.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The DFlash Triton autotuner requires CUDA")
    if os.environ.get(TUNING_PROFILE_ENV):
        raise RuntimeError(
            f"Unset {TUNING_PROFILE_ENV} before tuning so the baseline is reproducible"
        )
    if args.fixed_grid_size <= 0:
        raise ValueError("--fixed-grid-size must be positive")
    if args.num_anchors <= 0:
        raise ValueError("--num-anchors must be positive")
    if args.query_heads <= 0 or args.kv_heads <= 0:
        raise ValueError("--query-heads and --kv-heads must be positive")
    if args.query_heads % args.kv_heads:
        raise ValueError("--query-heads must be divisible by --kv-heads")
    if args.warmup < 0 or args.rounds <= 0 or args.min_iterations <= 0:
        raise ValueError(
            "warmup must be nonnegative; rounds/iterations must be positive"
        )
    if args.min_seconds < 0 or args.min_speedup < 1:
        raise ValueError("--min-seconds must be nonnegative and --min-speedup >= 1")
    if not 0 < args.max_memory_fraction <= 1:
        raise ValueError("--max-memory-fraction must be in (0, 1]")
    batch_sizes = parse_ints(args.batch_sizes)
    context_lens = parse_ints(args.context_lens)
    head_dims = parse_ints(args.head_dims)
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if not context_lens or any(not 0 <= value <= 65536 for value in context_lens):
        raise ValueError("--context-lens must contain integers in [0, 65536]")
    if not head_dims or any(value not in (64, 128) for value in head_dims):
        raise ValueError("--head-dims must contain only 64 or 128")
    variants = tuple(item for item in args.variants.split(",") if item)
    if not variants:
        raise ValueError("--variants must not be empty")
    unknown = set(variants) - set(VARIANT_TO_BACKEND)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "dflash_triton_tuning",
        "environment": tuning_environment(),
        "search": {
            "space": "full" if args.full_search else "standard",
            "correctness_reference": "torch_sdpa",
            "seed": args.seed,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "min_iterations": args.min_iterations,
            "min_seconds": args.min_seconds,
            "min_speedup": args.min_speedup,
        },
        "profiles": [],
        "cases": [],
    }
    total_memory = torch.cuda.get_device_properties(0).total_memory
    for case in iter_cases(args):
        print(f"[dflash-tune] {case.case_id}", flush=True)
        estimate = estimate_peak_bytes(case)
        case_record: dict[str, object] = {
            "case": asdict(case),
            "estimated_peak_bytes": estimate,
            "results": [],
        }
        payload["cases"].append(case_record)
        if estimate > total_memory * args.max_memory_fraction:
            case_record["status"] = "capacity_preflight"
            write_outputs(payload, args.output)
            continue
        try:
            inputs = make_inputs(case, args.seed)
            reference = make_sdpa_reference(case, inputs)
            for variant in variants:
                print(f"  {variant}", flush=True)
                try:
                    result = tune_variant(case, variant, inputs, reference, args)
                except torch.OutOfMemoryError as exc:
                    result = {
                        "variant": variant,
                        "backend": VARIANT_TO_BACKEND[variant],
                        "status": "capacity_limit",
                        "error": str(exc),
                    }
                    torch.cuda.empty_cache()
                except Exception as exc:
                    result = {
                        "variant": variant,
                        "backend": VARIANT_TO_BACKEND[variant],
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                case_record["results"].append(result)
                if result["status"] == "ok":
                    payload["profiles"].append(
                        {
                            "key": result["key"],
                            "forward": result["tuned"]["forward"],
                            "backward": result["tuned"]["backward"],
                        }
                    )
                write_outputs(payload, args.output)
        except torch.OutOfMemoryError as exc:
            case_record.update(status="capacity_limit_reference", error=str(exc))
            write_outputs(payload, args.output)
        except Exception as exc:
            case_record.update(
                status="reference_error", error=f"{type(exc).__name__}: {exc}"
            )
            write_outputs(payload, args.output)
        finally:
            if "reference" in locals():
                del reference
            if "inputs" in locals():
                del inputs
            torch.cuda.empty_cache()
        case_record.setdefault("status", "complete")
        write_outputs(payload, args.output)


if __name__ == "__main__":
    main()
