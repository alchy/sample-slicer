"""CLI: sample-slicer slice | analyze | build."""
from __future__ import annotations
import argparse, sys
from dataclasses import fields
from .detect import DetectParams
from . import slicing


def add_detect_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("detekce (výchozí hodnoty ze specu)")
    for f in fields(DetectParams):
        g.add_argument("--" + f.name.replace("_", "-"), type=float, default=None, metavar="X",
                       help=f"výchozí {getattr(DetectParams(), f.name)}")
    g.add_argument("--tail-s", type=float, default=2.0, help="délka přirozeného dozvuku (s)")
    g.add_argument("--fade-in-ms", type=float, default=2.0)


def detect_params_from_args(args) -> DetectParams:
    kw = {f.name: getattr(args, f.name) for f in fields(DetectParams) if getattr(args, f.name) is not None}
    return DetectParams(**kw)


def cmd_slice(args) -> int:
    n = slicing.slice_dir(args.in_dir, args.out_dir, detect_params_from_args(args), args.tail_s, args.fade_in_ms)
    print(f"Uloženo {n} úderů do {args.out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sample-slicer")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice", help="generický střih: adresář WAV → ořezané údery")
    s.add_argument("in_dir"); s.add_argument("out_dir")
    add_detect_args(s)
    s.set_defaults(func=cmd_slice)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
