# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Versioned device-profile registry for DFlash kernel launch parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch


ForwardVariant = Literal["baseline", "two_anchor", "persistent"]
_FORWARD_VARIANTS = frozenset({"baseline", "two_anchor", "persistent"})


@dataclass(frozen=True)
class DFlashKernelTuning:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    backward_schedule: str


# A context bucket is selected by the smallest upper bound greater than or
# equal to ctx_len. Keep the benchmarked lengths explicit so each variant and
# length can be tuned independently without changing dispatch code.
_DEFAULT_FORWARD_TUNING: dict[str, DFlashKernelTuning] = {
    "baseline": DFlashKernelTuning(16, 64, 4, 2, "pull"),
    "two_anchor": DFlashKernelTuning(16, 32, 4, 2, "pull"),
    "persistent": DFlashKernelTuning(16, 64, 4, 2, "pull"),
}
_DEFAULT_BACKWARD_TUNING = DFlashKernelTuning(16, 64, 4, 2, "pull")


# Only SM120 has been measured in this workspace. The values below preserve
# the currently validated launch behavior; replace individual entries only
# after correctness-gated measurements for that variant and context bucket.
_FORWARD_DEVICE_PROFILES: dict[
    str, dict[str, dict[int, DFlashKernelTuning]]
] = {
    "sm120": {
        "baseline": {
            512: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            2048: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            8192: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            16384: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            32768: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            65536: DFlashKernelTuning(16, 64, 4, 2, "pull"),
        },
        "two_anchor": {
            512: DFlashKernelTuning(16, 32, 4, 2, "pull"),
            2048: DFlashKernelTuning(16, 32, 4, 2, "pull"),
            8192: DFlashKernelTuning(16, 32, 4, 2, "pull"),
            16384: DFlashKernelTuning(16, 32, 4, 2, "pull"),
            32768: DFlashKernelTuning(16, 32, 4, 2, "pull"),
            65536: DFlashKernelTuning(16, 32, 4, 2, "pull"),
        },
        "persistent": {
            512: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            2048: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            8192: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            16384: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            32768: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            65536: DFlashKernelTuning(16, 64, 4, 2, "pull"),
        },
    },
    # Extension point, not an unverified H100 performance claim.
    "sm90": {},
}


# All three forward variants share the same backward kernels, so backward
# tuning is intentionally independent of forward_variant.
_BACKWARD_DEVICE_PROFILES: dict[str, dict[int, DFlashKernelTuning]] = {
    "sm120": {
        512: DFlashKernelTuning(16, 64, 4, 2, "pull"),
        2048: DFlashKernelTuning(16, 64, 4, 2, "pull"),
        8192: DFlashKernelTuning(16, 64, 4, 2, "pull"),
        16384: DFlashKernelTuning(16, 64, 4, 2, "pull"),
        32768: DFlashKernelTuning(16, 64, 4, 2, "pull"),
        65536: DFlashKernelTuning(16, 64, 4, 2, "pull"),
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


def _select_context_tuning(
    buckets: dict[int, DFlashKernelTuning],
    *,
    ctx_len: int,
    fallback: DFlashKernelTuning,
) -> DFlashKernelTuning:
    for max_ctx_len in sorted(buckets):
        if int(ctx_len) <= max_ctx_len:
            return buckets[max_ctx_len]
    return fallback


def _validate_tuning(
    config: DFlashKernelTuning, *, block_size: int
) -> DFlashKernelTuning:
    if config.block_m < int(block_size):
        raise ValueError(
            f"DFlash tuning block_m={config.block_m} must be at least "
            f"block_size={block_size}"
        )
    if config.backward_schedule != "pull":
        raise ValueError(
            "DFlash Triton currently supports only the pull backward schedule, "
            f"got {config.backward_schedule!r}"
        )
    return config


def get_triton_tuning(
    *,
    forward_variant: ForwardVariant = "baseline",
    block_size: int,
    ctx_len: int,
    device: torch.device | None = None,
) -> DFlashKernelTuning:
    """Return forward tuning for a device, implementation, and context bucket."""
    variant = str(forward_variant)
    if variant not in _FORWARD_VARIANTS:
        raise ValueError(f"Unknown DFlash Triton forward variant {variant!r}")
    profile = current_device_profile(device)
    buckets = _FORWARD_DEVICE_PROFILES.get(profile, {}).get(variant, {})
    config = _select_context_tuning(
        buckets,
        ctx_len=ctx_len,
        fallback=_DEFAULT_FORWARD_TUNING[variant],
    )
    return _validate_tuning(config, block_size=block_size)


def get_triton_backward_tuning(
    *, block_size: int, ctx_len: int, device: torch.device | None = None
) -> DFlashKernelTuning:
    """Return tuning for the backward kernels shared by all forward variants."""
    profile = current_device_profile(device)
    config = _select_context_tuning(
        _BACKWARD_DEVICE_PROFILES.get(profile, {}),
        ctx_len=ctx_len,
        fallback=_DEFAULT_BACKWARD_TUNING,
    )
    return _validate_tuning(config, block_size=block_size)


def device_profile_registry() -> dict[str, dict[str, object]]:
    devices = set(_FORWARD_DEVICE_PROFILES) | set(_BACKWARD_DEVICE_PROFILES)
    return {
        device: {
            "forward": {
                variant: {
                    str(max_ctx_len): asdict(config)
                    for max_ctx_len, config in sorted(buckets.items())
                }
                for variant, buckets in _FORWARD_DEVICE_PROFILES.get(device, {}).items()
            },
            "backward": {
                str(max_ctx_len): asdict(config)
                for max_ctx_len, config in sorted(
                    _BACKWARD_DEVICE_PROFILES.get(device, {}).items()
                )
            },
        }
        for device in sorted(devices)
    }
