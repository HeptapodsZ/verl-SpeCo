# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Create a clean JSON/CSV artifact for a selected benchmark submatrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import add_acceptance, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--context-lens")
    parser.add_argument("--replace-from", type=Path)
    parser.add_argument("--replace-accuracy-only", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    selected_contexts = (
        {int(value) for value in args.context_lens.split(",")}
        if args.context_lens
        else None
    )
    payload["cases"] = [
        case
        for case in payload["cases"]
        if case["case"]["batch_size"] == args.batch_size
        and case["case"]["block_size"] == args.block_size
        and (
            selected_contexts is None
            or case["case"]["context_len"] in selected_contexts
        )
    ]
    payload["selection"] = {
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "context_lens": sorted(selected_contexts) if selected_contexts else "all",
    }
    if args.replace_from is not None:
        replacement_payload = json.loads(args.replace_from.read_text(encoding="utf-8"))
        replacements = {
            (
                case["case"]["batch_size"],
                case["case"]["context_len"],
                case["case"]["block_size"],
                result["backend"],
            ): result
            for case in replacement_payload["cases"]
            for result in case["results"]
        }
        for case in payload["cases"]:
            shape = case["case"]
            updated_results = []
            for result in case["results"]:
                replacement = replacements.get(
                    (
                        shape["batch_size"],
                        shape["context_len"],
                        shape["block_size"],
                        result["backend"],
                    )
                )
                if replacement is None:
                    updated_results.append(result)
                elif args.replace_accuracy_only:
                    result["accuracy"] = replacement["accuracy"]
                    result["accuracy_valid"] = replacement["accuracy_valid"]
                    updated_results.append(result)
                else:
                    updated_results.append(replacement)
            case["results"] = updated_results
            add_acceptance(case)
        payload["replacement_source"] = str(args.replace_from)
    write_outputs(payload, args.output)


if __name__ == "__main__":
    main()
