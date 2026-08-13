# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Extract a compact, machine-readable table from Nsight Compute reports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path


METRICS = {
    ("GPU Speed Of Light Throughput", "Duration"): "duration",
    ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"): "compute_sm_pct",
    ("GPU Speed Of Light Throughput", "DRAM Throughput"): "dram_pct",
    ("GPU Speed Of Light Throughput", "L2 Cache Throughput"): "l2_throughput_pct",
    ("Memory Workload Analysis", "Memory Throughput"): "memory_throughput_gbps",
    ("Memory Workload Analysis", "L2 Hit Rate"): "l2_hit_pct",
    ("Compute Workload Analysis", "Issue Slots Busy"): "issue_slots_busy_pct",
    ("Scheduler Statistics", "No Eligible"): "no_eligible_pct",
    ("Warp State Statistics", "Warp Cycles Per Issued Instruction"): "warp_cycles_per_issued",
    ("Launch Statistics", "Registers Per Thread"): "registers_per_thread",
    ("Launch Statistics", "Dynamic Shared Memory Per Block"): "dynamic_shared_memory",
    ("Occupancy", "Theoretical Occupancy"): "theoretical_occupancy_pct",
    ("Occupancy", "Achieved Occupancy"): "achieved_occupancy_pct",
    ("Source Counters", "Branch Efficiency"): "branch_efficiency_pct",
}


def _as_float(value: str) -> float:
    return float(value.replace(",", ""))


def _duration_us(value: str, unit: str) -> float:
    scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[unit]
    return _as_float(value) * scale


def _shared_kib(value: str, unit: str) -> float:
    if unit == "byte/block":
        return _as_float(value) / 1024.0
    if unit == "Kbyte/block":
        return _as_float(value)
    raise ValueError(f"Unsupported shared-memory unit: {unit}")


def summarize(report: Path) -> dict[str, str | float]:
    completed = subprocess.run(
        ["ncu", "--import", str(report), "--page", "details", "--csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    candidates: dict[str, dict[str, str | float]] = {}
    for row in rows:
        kernel = row["Kernel Name"]
        if "elementwise_kernel" in kernel:
            continue
        kernel_id = row["ID"]
        record = candidates.setdefault(
            kernel_id,
            {
                "source_report": report.name,
                "kernel": kernel,
                "block_size": row["Block Size"],
                "grid_size": row["Grid Size"],
            },
        )
        output_name = METRICS.get((row["Section Name"], row["Metric Name"]))
        if output_name is None or not row["Metric Value"]:
            continue
        value = row["Metric Value"]
        unit = row["Metric Unit"]
        if output_name == "duration":
            record["duration_us"] = _duration_us(value, unit)
        elif output_name == "dynamic_shared_memory":
            record["dynamic_shared_memory_kib"] = _shared_kib(value, unit)
        else:
            record[output_name] = _as_float(value)
    if not candidates:
        raise RuntimeError(f"No non-elementwise CUDA kernel found in {report}")
    return max(candidates.values(), key=lambda row: float(row.get("duration_us", 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"schema_version": 1, "kernels": [summarize(path) for path in args.reports]}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    rows = payload["kernels"]
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
