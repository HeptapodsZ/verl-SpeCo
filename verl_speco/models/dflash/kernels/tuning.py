# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Versioned profiles and bounded search spaces for DFlash Triton kernels."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Literal

import torch


ForwardVariant = Literal[
    "baseline", "two_anchor", "persistent", "one_grid", "one_fixed_grid"
]
TuningPhase = Literal["forward", "backward"]
_FORWARD_VARIANTS = frozenset(
    {"baseline", "two_anchor", "persistent", "one_grid", "one_fixed_grid"}
)
_TUNING_PROFILE_ENV = "VERL_SPECO_DFLASH_TUNING_PROFILE"


@dataclass(frozen=True)
class DFlashKernelTuning:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    backward_schedule: str


@dataclass(frozen=True)
class DFlashTuningKey:
    """Exact workload identity used by generated tuning profiles.

    ``head_dim`` is the per-head kernel dimension. The model hidden size is
    represented by ``query_heads * head_dim``; keeping both fields avoids
    applying a result across different GQA layouts with the same hidden size.
    """

    device_profile: str
    forward_variant: str
    batch_size: int
    context_len: int
    block_size: int
    num_anchors: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    fixed_grid_size: int


_TuningOverride = tuple[DFlashKernelTuning | None, DFlashKernelTuning | None]
_TUNING_OVERRIDE: ContextVar[_TuningOverride] = ContextVar(
    "dflash_triton_tuning_override", default=(None, None)
)


# A context bucket is selected by the smallest upper bound greater than or
# equal to ctx_len. Keep the benchmarked lengths explicit so each variant and
# length can be tuned independently without changing dispatch code.
_DEFAULT_FORWARD_TUNING: dict[str, DFlashKernelTuning] = {
    "baseline": DFlashKernelTuning(16, 64, 4, 2, "pull"),
    "two_anchor": DFlashKernelTuning(16, 32, 4, 2, "pull"),
    "persistent": DFlashKernelTuning(16, 64, 4, 2, "pull"),
    "one_grid": DFlashKernelTuning(16, 64, 4, 2, "pull"),
    "one_fixed_grid": DFlashKernelTuning(16, 64, 4, 2, "pull"),
}
_DEFAULT_BACKWARD_TUNING = DFlashKernelTuning(16, 64, 4, 2, "pull")


# Only SM120 has been measured in this workspace. The values below preserve
# the currently validated launch behavior; replace individual entries only
# after correctness-gated measurements for that variant and context bucket.
_FORWARD_DEVICE_PROFILES: dict[str, dict[str, dict[int, DFlashKernelTuning]]] = {
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
        "one_grid": {
            512: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            2048: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            8192: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            16384: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            32768: DFlashKernelTuning(16, 64, 4, 2, "pull"),
            65536: DFlashKernelTuning(16, 64, 4, 2, "pull"),
        },
        "one_fixed_grid": {
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


# All forward variants share the same backward math and tuning. The one-grid
# specialization changes only the launch topology, so backward tuning remains
# intentionally independent of forward_variant.
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


def validate_triton_tuning(
    config: DFlashKernelTuning, *, block_size: int
) -> DFlashKernelTuning:
    """Fail closed before a malformed config reaches Triton compilation."""
    if config.block_m <= 0 or config.block_m & (config.block_m - 1):
        raise ValueError(
            f"DFlash tuning block_m must be a power of two, got {config.block_m}"
        )
    if config.block_m < int(block_size):
        raise ValueError(
            f"DFlash tuning block_m={config.block_m} must be at least "
            f"block_size={block_size}"
        )
    if config.block_n < 16 or config.block_n & (config.block_n - 1):
        raise ValueError(
            f"DFlash tuning block_n must be a power of two >= 16, got {config.block_n}"
        )
    if config.num_warps not in (2, 4, 8):
        raise ValueError(
            f"DFlash tuning num_warps must be one of (2, 4, 8), got {config.num_warps}"
        )
    if config.num_stages not in (1, 2, 3, 4):
        raise ValueError(
            "DFlash tuning num_stages must be one of (1, 2, 3, 4), "
            f"got {config.num_stages}"
        )
    if config.backward_schedule != "pull":
        raise ValueError(
            "DFlash Triton currently supports only the pull backward schedule, "
            f"got {config.backward_schedule!r}"
        )
    return config


def triton_tuning_candidates(
    *,
    phase: TuningPhase,
    forward_variant: ForwardVariant,
    block_size: int,
    ctx_len: int,
    head_dim: int,
    full: bool = False,
) -> tuple[DFlashKernelTuning, ...]:
    """Return a deterministic, pruned launch-config search space.

    The current kernels use ``BLOCK_M`` to represent an exact 16-token draft
    block. Larger values are numerically masked but add guaranteed work, so the
    search fixes it to ``block_size``. The standard space explores every other
    launch field with eight or fewer configs. ``full=True`` enables a bounded
    Cartesian search for deliberate offline sweeps.
    """
    if phase not in ("forward", "backward"):
        raise ValueError(f"Unknown DFlash tuning phase {phase!r}")
    variant = str(forward_variant)
    if variant not in _FORWARD_VARIANTS:
        raise ValueError(f"Unknown DFlash Triton forward variant {variant!r}")
    if int(block_size) != 16:
        raise ValueError(
            "DFlash Triton tuning currently supports only block_size=16, "
            f"got {block_size}"
        )
    if int(head_dim) not in (64, 128):
        raise ValueError(
            f"DFlash Triton tuning supports head_dim 64 or 128, got {head_dim}"
        )

    default = (
        _DEFAULT_FORWARD_TUNING[variant]
        if phase == "forward"
        else _DEFAULT_BACKWARD_TUNING
    )
    block_ns = (16, 32, 64, 128) if full else (32, 64, 128)
    if variant == "two_anchor" and phase == "forward":
        # This variant doubles its query/local tiles. BLOCK_N=128 compounds
        # register pressure without increasing useful local work.
        block_ns = tuple(value for value in block_ns if value <= 64)

    candidates: list[DFlashKernelTuning] = [default]
    if int(ctx_len) == 0:
        return tuple(candidates)
    if full:
        for block_n in block_ns:
            for num_warps in (2, 4, 8):
                for num_stages in (1, 2, 3):
                    # Narrow tiles cannot usefully feed eight warps, while a
                    # 128x128 tile with three stages is an avoidable resource
                    # cliff on the currently supported head dimensions.
                    if block_n == 16 and num_warps == 8:
                        continue
                    if block_n == 128 and int(head_dim) == 128 and num_stages == 3:
                        continue
                    candidates.append(
                        DFlashKernelTuning(
                            int(block_size), block_n, num_warps, num_stages, "pull"
                        )
                    )
    else:
        for block_n in block_ns:
            for num_warps in (4, 8):
                candidates.append(
                    DFlashKernelTuning(int(block_size), block_n, num_warps, 2, "pull")
                )
        for num_stages in (1, 3):
            candidates.append(
                DFlashKernelTuning(
                    int(block_size), default.block_n, 4, num_stages, "pull"
                )
            )

    unique = dict.fromkeys(candidates)
    return tuple(
        validate_triton_tuning(config, block_size=block_size) for config in unique
    )


@contextmanager
def override_triton_tuning(
    *,
    forward: DFlashKernelTuning | None = None,
    backward: DFlashKernelTuning | None = None,
) -> Iterator[None]:
    """Temporarily override launch configs for an offline benchmark.

    Context-local state keeps concurrent callers isolated. Production code
    never enters this context, so normal dispatch and first-call behavior are
    unchanged.
    """
    token = _TUNING_OVERRIDE.set((forward, backward))
    try:
        yield
    finally:
        _TUNING_OVERRIDE.reset(token)


def _tuning_key(
    *,
    forward_variant: str,
    block_size: int,
    ctx_len: int,
    device: torch.device | None,
    batch_size: int | None,
    num_anchors: int | None,
    query_heads: int | None,
    kv_heads: int | None,
    head_dim: int | None,
    dtype: str | None,
    fixed_grid_size: int | None,
) -> DFlashTuningKey | None:
    shape = (
        batch_size,
        num_anchors,
        query_heads,
        kv_heads,
        head_dim,
        dtype,
        fixed_grid_size,
    )
    if any(value is None for value in shape):
        return None
    return DFlashTuningKey(
        device_profile=current_device_profile(device),
        forward_variant=forward_variant,
        batch_size=int(batch_size),
        context_len=int(ctx_len),
        block_size=int(block_size),
        num_anchors=int(num_anchors),
        query_heads=int(query_heads),
        kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        dtype=str(dtype),
        fixed_grid_size=int(fixed_grid_size),
    )


def _config_from_dict(value: object, *, source: str) -> DFlashKernelTuning:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be a JSON object")
    expected = {"block_m", "block_n", "num_warps", "num_stages", "backward_schedule"}
    if set(value) != expected:
        raise ValueError(f"{source} fields must be exactly {sorted(expected)}")
    return DFlashKernelTuning(
        block_m=int(value["block_m"]),
        block_n=int(value["block_n"]),
        num_warps=int(value["num_warps"]),
        num_stages=int(value["num_stages"]),
        backward_schedule=str(value["backward_schedule"]),
    )


@lru_cache(maxsize=8)
def _load_generated_profile(
    path_string: str,
) -> dict[tuple[DFlashTuningKey, str], DFlashKernelTuning]:
    path = Path(path_string)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to load DFlash tuning profile {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"DFlash tuning profile {path} must contain a JSON object")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "dflash_triton_tuning"
    ):
        raise ValueError(
            f"DFlash tuning profile {path} has an unsupported schema or kind"
        )
    records: dict[tuple[DFlashTuningKey, str], DFlashKernelTuning] = {}
    for index, profile in enumerate(payload.get("profiles", ())):
        if not isinstance(profile, dict) or not isinstance(profile.get("key"), dict):
            raise ValueError(f"profiles[{index}] must contain a key object")
        try:
            key = DFlashTuningKey(**profile["key"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid DFlash tuning key at profiles[{index}]") from exc
        if key.forward_variant not in _FORWARD_VARIANTS:
            raise ValueError(
                f"Unknown forward variant {key.forward_variant!r} at profiles[{index}]"
            )
        for phase in ("forward", "backward"):
            config = _config_from_dict(
                profile.get(phase), source=f"profiles[{index}].{phase}"
            )
            config = validate_triton_tuning(config, block_size=key.block_size)
            record_key = (key, phase)
            if record_key in records:
                raise ValueError(
                    f"Duplicate DFlash tuning profile for {key} phase={phase}"
                )
            records[record_key] = config
    return records


def _generated_profile_tuning(
    key: DFlashTuningKey | None, phase: TuningPhase
) -> DFlashKernelTuning | None:
    path = os.environ.get(_TUNING_PROFILE_ENV)
    if not path or key is None:
        return None
    normalized_path = os.path.abspath(os.path.expanduser(path))
    return _load_generated_profile(normalized_path).get((key, phase))


def get_triton_tuning(
    *,
    forward_variant: ForwardVariant = "baseline",
    block_size: int,
    ctx_len: int,
    device: torch.device | None = None,
    batch_size: int | None = None,
    num_anchors: int | None = None,
    query_heads: int | None = None,
    kv_heads: int | None = None,
    head_dim: int | None = None,
    dtype: str | None = None,
    fixed_grid_size: int | None = None,
) -> DFlashKernelTuning:
    """Return forward tuning for a device, implementation, and context bucket."""
    variant = str(forward_variant)
    if variant not in _FORWARD_VARIANTS:
        raise ValueError(f"Unknown DFlash Triton forward variant {variant!r}")
    override, _ = _TUNING_OVERRIDE.get()
    if override is not None:
        return validate_triton_tuning(override, block_size=block_size)
    key = _tuning_key(
        forward_variant=variant,
        block_size=block_size,
        ctx_len=ctx_len,
        device=device,
        batch_size=batch_size,
        num_anchors=num_anchors,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        fixed_grid_size=fixed_grid_size,
    )
    generated = _generated_profile_tuning(key, "forward")
    if generated is not None:
        return generated
    profile = current_device_profile(device)
    buckets = _FORWARD_DEVICE_PROFILES.get(profile, {}).get(variant, {})
    config = _select_context_tuning(
        buckets,
        ctx_len=ctx_len,
        fallback=_DEFAULT_FORWARD_TUNING[variant],
    )
    return validate_triton_tuning(config, block_size=block_size)


def get_triton_backward_tuning(
    *,
    block_size: int,
    ctx_len: int,
    device: torch.device | None = None,
    forward_variant: ForwardVariant = "baseline",
    batch_size: int | None = None,
    num_anchors: int | None = None,
    query_heads: int | None = None,
    kv_heads: int | None = None,
    head_dim: int | None = None,
    dtype: str | None = None,
    fixed_grid_size: int | None = None,
) -> DFlashKernelTuning:
    """Return tuning for the backward kernels shared by all forward variants."""
    variant = str(forward_variant)
    if variant not in _FORWARD_VARIANTS:
        raise ValueError(f"Unknown DFlash Triton forward variant {variant!r}")
    _, override = _TUNING_OVERRIDE.get()
    if override is not None:
        return validate_triton_tuning(override, block_size=block_size)
    key = _tuning_key(
        forward_variant=variant,
        block_size=block_size,
        ctx_len=ctx_len,
        device=device,
        batch_size=batch_size,
        num_anchors=num_anchors,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        fixed_grid_size=fixed_grid_size,
    )
    generated = _generated_profile_tuning(key, "backward")
    if generated is not None:
        return generated
    profile = current_device_profile(device)
    config = _select_context_tuning(
        _BACKWARD_DEVICE_PROFILES.get(profile, {}),
        ctx_len=ctx_len,
        fallback=_DEFAULT_BACKWARD_TUNING,
    )
    return validate_triton_tuning(config, block_size=block_size)


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
