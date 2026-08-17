# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Custom CUDA attention kernels for DFlash training."""

from .dispatch import DFLASH_ATTENTION_BACKENDS, dflash_sparse_attention
from .reference import (
    build_dflash_dense_attention_mask,
    dense_dflash_attention_reference,
    tensor_error_metrics,
)
from .tuning import (
    DFlashKernelTuning,
    current_device_profile,
    device_profile_registry,
    get_triton_backward_tuning,
    get_triton_tuning,
)

__all__ = [
    "DFLASH_ATTENTION_BACKENDS",
    "DFlashKernelTuning",
    "build_dflash_dense_attention_mask",
    "dense_dflash_attention_reference",
    "dflash_sparse_attention",
    "current_device_profile",
    "device_profile_registry",
    "get_triton_backward_tuning",
    "get_triton_tuning",
    "tensor_error_metrics",
]
