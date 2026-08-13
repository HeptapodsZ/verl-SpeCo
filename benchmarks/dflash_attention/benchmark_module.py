# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Accuracy-gated benchmark for the complete DFlashAttention module."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch._functorch import config as functorch_config

# FlexAttention's compiled backward donates temporary buffers by default. A
# backward-only benchmark intentionally reuses one graph, so donation must be
# disabled in this standalone measurement process.
functorch_config.donated_buffer = False

from benchmark import (
    BACKWARD_LIMITS,
    FORWARD_LIMITS,
    Case,
    environment,
    iter_cases,
    make_block_mask,
    measure_cuda,
    write_outputs,
)
from verl_speco.models.dflash.configuration_dflash import DFlashConfig
from verl_speco.models.dflash.kernels import (
    build_dflash_dense_attention_mask,
    tensor_error_metrics,
)
from verl_speco.models.dflash.modeling_dflash import DFlashAttention


def module_inputs(case: Case, hidden_size: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    dtype = torch.bfloat16 if case.dtype == "bfloat16" else torch.float16
    anchors = torch.linspace(0, case.context_len, case.num_anchors, device="cuda")
    anchors = anchors.round().to(torch.int32).unsqueeze(0)
    anchors = anchors.expand(case.batch_size, -1).contiguous()
    keep = torch.ones_like(anchors, dtype=torch.bool)
    offsets = torch.arange(case.block_size, device="cuda").view(1, 1, -1)
    draft_positions = (anchors.to(torch.long).unsqueeze(-1) + offsets).reshape(
        case.batch_size, case.query_len
    )
    context_positions = torch.arange(case.context_len, device="cuda").view(1, -1)
    context_positions = context_positions.expand(case.batch_size, -1)
    return {
        "draft": torch.randn(
            (case.batch_size, case.query_len, hidden_size),
            generator=generator,
            device="cuda",
            dtype=dtype,
        ),
        "context": torch.randn(
            (case.batch_size, case.context_len, hidden_size),
            generator=generator,
            device="cuda",
            dtype=dtype,
        ),
        # Model losses are mean-reduced over draft tokens.  Matching that scale
        # keeps accumulated projection-weight gradients inside the same
        # elementwise atol/rtol contract used for attention-core gradients.
        "grad_output": torch.randn(
            (case.batch_size, case.query_len, hidden_size),
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        / case.query_len,
        "anchors": anchors,
        "keep": keep,
        "draft_positions": draft_positions,
        "context_positions": context_positions,
    }


def backend_kwargs(backend: str, case: Case, values: dict[str, torch.Tensor]) -> dict:
    kwargs = {"attention_backend": backend}
    if backend == "flex":
        class CoreInputs:
            anchors = values["anchors"]
            keep = values["keep"]

        kwargs["block_mask"] = make_block_mask(case, CoreInputs())
    elif backend == "sdpa":
        kwargs["dense_attention_mask"] = build_dflash_dense_attention_mask(
            values["anchors"], values["keep"], case.context_len, case.block_size
        )
    else:
        kwargs.update(
            anchor_positions=values["anchors"],
            block_keep_mask=values["keep"],
            block_size=case.block_size,
        )
    return kwargs


def run_module(
    module: DFlashAttention,
    values: dict[str, torch.Tensor],
    kwargs: dict,
    *,
    backward: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    draft = values["draft"].detach().requires_grad_(backward)
    context = values["context"].detach().requires_grad_(backward)
    output = module(
        draft,
        context,
        values["draft_positions"],
        values["context_positions"],
        **kwargs,
    )
    if not backward:
        return output, {}
    output.backward(values["grad_output"])
    grads = {
        "draft_hidden": draft.grad,
        "context_hidden": context.grad,
        **{
            f"parameter/{name}": parameter.grad
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        },
    }
    return output, grads


def metrics_valid(metrics: dict[str, dict[str, float]]) -> bool:
    for name, value in metrics.items():
        forward = name == "output"
        if value["actual_nan"] != 0 or value["actual_inf"] != 0:
            return False
        if value["reference_nan"] != 0 or value["reference_inf"] != 0:
            return False
        if not value.get("allclose", False):
            return False
        if value["relative_l2"] > (5e-3 if forward else 1e-2):
            return False
        if value["cosine"] < (0.9999 if forward else 0.999):
            return False
    return True


def benchmark_one(
    backend: str,
    case: Case,
    values: dict[str, torch.Tensor],
    config: DFlashConfig,
    state: dict[str, torch.Tensor],
    reference: tuple[torch.Tensor, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> dict:
    dtype = torch.bfloat16 if case.dtype == "bfloat16" else torch.float16
    module = DFlashAttention(config).to(device="cuda", dtype=dtype)
    module.load_state_dict(state)
    kwargs = backend_kwargs(backend, case, values)
    output, grads = run_module(module, values, kwargs, backward=True)
    reference_output, reference_grads = reference
    metrics = {
        "output": tensor_error_metrics(
            output,
            reference_output,
            atol=FORWARD_LIMITS["atol"],
            rtol=FORWARD_LIMITS["rtol"],
        )
    }
    metrics.update(
        {
            name: tensor_error_metrics(
                value,
                reference_grads[name],
                atol=BACKWARD_LIMITS["atol"],
                rtol=BACKWARD_LIMITS["rtol"],
            )
            for name, value in grads.items()
        }
    )
    result = {"backend": backend, "accuracy": metrics, "accuracy_valid": metrics_valid(metrics)}
    if not result["accuracy_valid"]:
        result["status"] = "invalid_accuracy"
        return result
    if args.accuracy_only:
        result["status"] = "accuracy_only"
        return result

    module.zero_grad(set_to_none=True)
    result["forward"] = measure_cuda(
        lambda: run_module(module, values, kwargs, backward=False)[0],
        warmup=args.warmup,
        rounds=args.rounds,
        min_iterations=args.min_iterations,
        min_seconds=args.min_seconds,
    )

    backward_draft = values["draft"].detach().requires_grad_(True)
    backward_context = values["context"].detach().requires_grad_(True)
    backward_output = module(
        backward_draft,
        backward_context,
        values["draft_positions"],
        values["context_positions"],
        **kwargs,
    )
    backward_targets = (
        backward_draft,
        backward_context,
        *tuple(module.parameters()),
    )
    result["backward"] = measure_cuda(
        lambda: torch.autograd.grad(
            backward_output,
            backward_targets,
            values["grad_output"],
            retain_graph=True,
        ),
        warmup=args.warmup,
        rounds=args.rounds,
        min_iterations=args.min_iterations,
        min_seconds=args.min_seconds,
    )

    def forward_backward():
        module.zero_grad(set_to_none=True)
        return run_module(module, values, kwargs, backward=True)

    result["forward_backward"] = measure_cuda(
        forward_backward,
        warmup=args.warmup,
        rounds=args.rounds,
        min_iterations=args.min_iterations,
        min_seconds=args.min_seconds,
    )
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    memory_output, memory_grads = forward_backward()
    torch.cuda.synchronize()
    del memory_output, memory_grads
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
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    peak_allocated = torch.cuda.max_memory_allocated()
    result["actual_free_fraction_after_run"] = free_bytes / total_bytes
    result["free_headroom_fraction"] = (total_bytes - peak_allocated) / total_bytes
    result["status"] = "ok" if result["free_headroom_fraction"] >= 0.20 else "low_headroom"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--context-lens", default="65536")
    parser.add_argument("--block-sizes", default="16")
    parser.add_argument("--backends", default="flex,sdpa,triton,tilelang")
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--anchor-distribution", default="uniform")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-iterations", type=int, default=20)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/dflash_attention_module.json")
    )
    args = parser.parse_args()
    payload = {"schema_version": 1, "environment": environment(), "cases": []}
    for case in iter_cases(args):
        record = {"case": asdict(case) | {"hidden_size": args.hidden_size}, "results": []}
        try:
            values = module_inputs(case, args.hidden_size, args.seed)
            config = DFlashConfig(
                hidden_size=args.hidden_size,
                intermediate_size=max(128, args.hidden_size * 3),
                num_attention_heads=case.query_heads,
                num_key_value_heads=case.kv_heads,
                max_position_embeddings=case.context_len + case.query_len + 20,
            )
            dtype = torch.bfloat16 if case.dtype == "bfloat16" else torch.float16
            base = DFlashAttention(config).to(device="cuda", dtype=dtype)
            state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in base.state_dict().items()
            }
            reference_module = DFlashAttention(config).to(device="cuda", dtype=dtype)
            reference_module.load_state_dict(state)
            reference = run_module(
                reference_module,
                values,
                backend_kwargs("sdpa", case, values),
                backward=True,
            )
            reference = (
                reference[0].detach(),
                {name: value.detach() for name, value in reference[1].items()},
            )
            del base, reference_module
            gc.collect()
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError as exc:
            record["results"] = [
                {"backend": backend, "status": "capacity_limit_reference", "error": str(exc)}
                for backend in args.backends.split(",")
            ]
            payload["cases"].append(record)
            torch.cuda.empty_cache()
            write_outputs(payload, args.output)
            continue
        for backend in args.backends.split(","):
            try:
                result = benchmark_one(
                    backend, case, values, config, state, reference, args
                )
            except torch.OutOfMemoryError as exc:
                result = {"backend": backend, "status": "capacity_limit", "error": str(exc)}
                torch.cuda.empty_cache()
            except Exception as exc:
                result = {
                    "backend": backend,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            print(f"[module] {case.case_id} {backend}: {result['status']}", flush=True)
            record["results"].append(result)
            gc.collect()
            torch.cuda.empty_cache()
        payload["cases"].append(record)
        write_outputs(payload, args.output)


if __name__ == "__main__":
    main()
