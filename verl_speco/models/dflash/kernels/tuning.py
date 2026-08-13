# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Versioned device-profile registry for DFlash kernel launch parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DFlashKernelTuning:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    backward_schedule: str


# Only SM120 has been measured in this workspace. SM90 is intentionally empty:
# it is an extension point, not an unverified H100 performance claim.
_DEVICE_PROFILES: dict[str, dict[str, DFlashKernelTuning]] = {
    "sm120": {
        "short": DFlashKernelTuning(16, 64, 4, 2, "pull"),
        "long": DFlashKernelTuning(16, 64, 4, 2, "pull"),
    },
    "sm90": {},
}


def current_device_profile(device: torch.device | None = None) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def get_triton_tuning(
    *, block_size: int, ctx_len: int, device: torch.device | None = None
) -> DFlashKernelTuning:
    profile = current_device_profile(device)
    bucket = "short" if int(ctx_len) <= 2048 else "long"
    base = _DEVICE_PROFILES.get(profile, {}).get(bucket)
    block_m = 16 if int(block_size) <= 16 else 32
    if base is None:
        return DFlashKernelTuning(block_m, 64, 4, 2, "pull")
    return DFlashKernelTuning(
        block_m=block_m,
        block_n=base.block_n,
        num_warps=base.num_warps,
        num_stages=base.num_stages,
        backward_schedule=base.backward_schedule,
    )


def device_profile_registry() -> dict[str, dict[str, dict[str, int | str]]]:
    return {
        device: {bucket: asdict(config) for bucket, config in buckets.items()}
        for device, buckets in _DEVICE_PROFILES.items()
    }
