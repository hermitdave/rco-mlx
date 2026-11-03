#!/usr/bin/env python3
"""
Deprecated shim: model adapters moved to rco_mlx.adapters.

This module re-exports everything from the new location for backward
compatibility. New code should import from ``rco_mlx.adapters`` instead.
"""

from .adapters import (  # noqa: F401
    ModelAdapter,
    OLMoEAdapter,
    Qwen3NextMoEAdapter,
    get,
    register,
)

__all__ = [
    "ModelAdapter",
    "OLMoEAdapter",
    "Qwen3NextMoEAdapter",
    "get",
    "register",
]
