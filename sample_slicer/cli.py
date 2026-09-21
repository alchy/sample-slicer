"""CLI: sample-slicer slice | analyze | build."""
from __future__ import annotations
import argparse, sys
from dataclasses import fields
from .detect import DetectParams
from . import slicing


def add_detect_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("detekce (význam a výchozí hodnoty: README, docs/algorithm.md)")
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


def cmd_analyze(args) -> int:
    import json
    from . import analyze as an
    truth = json.loads(open(args.truth).read()) if args.truth else None
    rows = an.analyze_dir(args.src_dir, detect_params_from_args(args), args.tail_s, truth)
    print(an.format_rows(rows))
    if truth:
        acc = an.accuracy_by_octave(rows)
        print("\npřesnost po oktávách (ok / oktávová chyba / jiná):")
        for k, v in acc.items():
            print(f"  {k:<8} ok {v['ok']:3d}  octave {v['octave']:3d}  other {v['other']:3d}")
    return 0


def cmd_build(args) -> int:
    import json
    from pathlib import Path
    from .bank import build_bank
    from .tuning import TuningParams
    ov_path = Path(args.overrides) if args.overrides else Path(args.src_dir) / "overrides.json"
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else None
    try:
        res = build_bank(args.src_dir, args.original, args.out, detect_params_from_args(args), args.tail_s,
                         TuningParams(), retune=args.retune, overrides=overrides)
    except RuntimeError as e:
        print(f"CHYBA: {e}", file=sys.stderr)
        return 2
    print(f"Zapsáno {res.written} úderů, odmítnuto {res.rejected}, přeskočeno zdrojů {res.skipped_sources}. "
          f"Report: {Path(args.original) / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sample-slicer")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice", help="generický střih: adresář WAV → ořezané údery")
    s.add_argument("in_dir"); s.add_argument("out_dir")
    add_detect_args(s)
    s.set_defaults(func=cmd_slice)
    a = sub.add_parser("analyze", help="dry-run: segmenty, noty, confidence; s --truth přesnost")
    a.add_argument("src_dir")
    a.add_argument("--truth", default=None, help="JSON se známým pořadím nahrávání (jen k měření přesnosti)")
    add_detect_args(a)
    a.set_defaults(func=cmd_analyze)
    b = sub.add_parser("build", help="celý workflow: zdroje → original (96k/24b) → out (48k/16b)")
    b.add_argument("src_dir")
    b.add_argument("--original", required=True); b.add_argument("--out", required=True)
    b.add_argument("--retune", action="store_true", help="dolaď výšku na temperované ladění při převodu")
    b.add_argument("--overrides", default=None, help="JSON s ručními zásahy (výchozí <src>/overrides.json)")
    add_detect_args(b)
    b.set_defaults(func=cmd_build)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
