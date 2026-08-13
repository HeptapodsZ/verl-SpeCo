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
from .tuning import current_device_profile, device_profile_registry

__all__ = [
    "DFLASH_ATTENTION_BACKENDS",
    "build_dflash_dense_attention_mask",
    "dense_dflash_attention_reference",
    "dflash_sparse_attention",
    "current_device_profile",
    "device_profile_registry",
    "tensor_error_metrics",
]
