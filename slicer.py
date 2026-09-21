#!/usr/bin/env python3
"""Zpětně kompatibilní vstup: `python slicer.py --input-dir A --output-dir B`.

Logika žije v balíčku sample_slicer (viz `sample-slicer slice --help`). Staré
přepínače detekce (--threshold_db, --min_length, …) už nemají význam — nový
algoritmus pracuje s lokálním šumovým dnem a přirozeným dozvukem — a jsou
ignorovány s varováním.
"""
import argparse, sys
from sample_slicer.cli import main

IGNORED = ["--threshold_db", "--min_length", "--min_length_after_trim", "--trim_threshold_offset",
           "--fade_ms", "--no_fades", "--resume", "--preview", "--log_level"]

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    for opt in IGNORED:
        if opt in ("--no_fades", "--resume", "--preview"):
            p.add_argument(opt, action="store_true")
        else:
            p.add_argument(opt, default=None)
    a, rest = p.parse_known_args()
    used = [o for o in IGNORED if getattr(a, o.lstrip("-")) not in (None, False)]
    if used:
        print(f"UPOZORNĚNÍ: přepínače {', '.join(used)} se v nové verzi ignorují", file=sys.stderr)
    sys.exit(main(["slice", a.input_dir, a.output_dir] + rest))
