#!/usr/bin/env python3
"""Alias CLI: `python slicer.py slice <in> <out>` == `sample-slicer slice <in> <out>`.
Logika žije v balíčku sample_slicer (viz README.md)."""
import sys
from sample_slicer.cli import main

if __name__ == "__main__":
    sys.exit(main())
