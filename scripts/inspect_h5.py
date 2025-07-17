#!/usr/bin/env python
"""
inspect_h5.py - print header + min/max summary of an HDF5 trajectory file.

Usage
-----
    python scripts/inspect_h5.py  path/to/file.h5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fastdep.io import h5_summary

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Inspect HDF5 dataset header.")
parser.add_argument("h5_file", type=Path, help="Path to .h5 file")
args = parser.parse_args()

h5_summary(args.h5_file)
