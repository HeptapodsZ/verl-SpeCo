# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Short NVTX-marked DFlash attention workload for Nsight Systems/Compute."""

from __future__ import annotations

import argparse

import torch

from benchmark import Case, backend_callable, make_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("flex", "sdpa", "triton", "tilelang"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--phase", choices=("forward", "backward", "forward_backward"), default="forward_backward")
    args = parser.parse_args()
    case = Case(args.batch_size, args.context_len, args.block_size)
    inputs = make_inputs(case, seed=0)
    call, _ = backend_callable(args.backend, case, inputs)

    def graph():
        query = inputs.query.detach().requires_grad_(True)
        key = inputs.key.detach().requires_grad_(True)
        value = inputs.value.detach().requires_grad_(True)
        output = call(query, key, value)
        return output, (query, key, value)

    def run() -> None:
        if args.phase == "forward":
            with torch.no_grad():
                call(inputs.query, inputs.key, inputs.value)
        else:
            output, targets = graph()
            torch.autograd.grad(output, targets, inputs.grad_output)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.autograd.profiler.emit_nvtx():
        for iteration in range(args.iterations):
            if args.phase == "backward":
                output, targets = graph()
                torch.cuda.synchronize()
                with torch.cuda.nvtx.range(
                    f"dflash/{args.backend}/backward/{iteration}"
                ):
                    torch.autograd.grad(output, targets, inputs.grad_output)
            else:
                with torch.cuda.nvtx.range(
                    f"dflash/{args.backend}/{args.phase}/{iteration}"
                ):
                    run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
