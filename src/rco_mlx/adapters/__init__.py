#!/usr/bin/env python3
"""
Model adapters package for the RCO quantization pipeline.

Each supported model gets a small adapter class that describes:
  - how to enumerate its layers and tensors
  - which tensors to quantize vs keep BF16
  - how to group layers for RCO allocation
  - how to load calibration data

The pipeline (rco_pipeline.py) calls the adapter; it does not need to
know which model it is running on.
"""

from .base import ModelAdapter, get, register
from .bailing_moe_v3 import BailingMoeV3Adapter
from .default import DefaultAdapter
from .granitemoehybrid import GraniteMoeHybridAdapter
from .flint import FlintAdapter
from .inkling2_loop import Inkling2LoopAdapter
from .lfm2_moe import Lfm2MoeAdapter
from .nemotron_h_puzzle import NemotronHPuzzleAdapter
from .olmoe import OLMoEAdapter
from .qwen3_5 import Qwen35DenseAdapter
from .qwen3_5_moe import Qwen3NextMoEAdapter
from .qwen4_exp import Qwen4ExpAdapter

__all__ = [
    "ModelAdapter",
    "BailingMoeV3Adapter",
    "DefaultAdapter",
    "GraniteMoeHybridAdapter",
    "Inkling2LoopAdapter",
    "Lfm2MoeAdapter",
    "NemotronHPuzzleAdapter",
    "OLMoEAdapter",
    "Qwen35DenseAdapter",
    "Qwen3NextMoEAdapter",
    "Qwen4ExpAdapter",
    "register",
    "get",
]
