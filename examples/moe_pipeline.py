#!/usr/bin/env python3
"""Deprecated alias for ``examples/rco_pipeline.py`` (renamed 2026-11-23).

The pipeline was built for MoE models and later generalised to dense ones, so
the old filename was misleading. This shim keeps existing commands, drivers and
notebooks working; use ``rco_pipeline.py`` in new work.
"""
import runpy
import sys
from pathlib import Path

_impl = Path(__file__).with_name("rco_pipeline.py")
sys.argv[0] = str(_impl)
runpy.run_path(str(_impl), run_name="__main__")
