"""music-reader command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .bundle import write_bundle
from .backends import BACKENDS, RecognitionOptions, get_backend
from .crops import write_crops
from .debug import write_debug
from .export import EXPORTERS, EXTENSIONS, export
from .pdfinput import Document


def _pages(text: str | None):
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="music-reader", description="Extract a melody from a sheet-music PDF.")
    p.add_argument("input", help="PDF (or PNG/JPG) file")
    p.add_argument("-o", "--output", help="output file (default: standard output)")
    p.add_argument("-f", "--format", default=None, choices=sorted(EXPORTERS), help="output format (default: abc, or from --output extension)")
    p.add_argument("--debug", metavar="DIR", help="write annotated page images, debug.pdf and diagnostics.txt to DIR")
    p.add_argument("--crops", metavar="DIR", help="write one image per detected system (notation as printed) and crops.json to DIR")
    p.add_argument("--crop-dpi", type=float, help="resolution of the system crops (re-renders vector PDFs)")
    p.add_argument("--bundle", metavar="DIR", help="write melody.abc, melody.musicxml, score.json, crops/ and manifest.json to DIR (for anno-api)")
    p.add_argument("--staff", type=int, help="1-based staff within each system that carries the melody")
    p.add_argument("--meter", help="time signature, e.g. 3/4 (digits are not read from the image)")
    p.add_argument("--key", help="key, e.g. G or Em (overrides the detected key signature)")
    p.add_argument("--title", help="title for the output")
    p.add_argument("--pages", help="pages to read, e.g. 1-3 or 1,3")
    p.add_argument("--dpi", type=float, help="render resolution (default: chosen from the staff size)")
    p.add_argument("--backend", default="classical", choices=sorted(BACKENDS))
    p.add_argument("--strict", action="store_true", help="exit with status 2 if any error-level diagnostic remains")
    p.add_argument("--version", action="version", version=f"music-reader {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fmt = args.format
    if fmt is None and args.output:
        ext = Path(args.output).suffix.lstrip(".").lower()
        fmt = next((n for n, e in EXTENSIONS.items() if e == ext), None)
    fmt = fmt or "abc"
    doc = Document(args.input)
    opts = RecognitionOptions(staff=args.staff, meter=args.meter, key=args.key, title=args.title or Path(args.input).stem,
                              dpi=args.dpi, pages=_pages(args.pages))
    result = get_backend(args.backend).recognize(doc, opts)
    text = export(result.score, fmt, title=opts.title)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.debug:
        write_debug(result, args.debug)
    if args.crops:
        write_crops(result, doc, args.crops, dpi=args.crop_dpi)
    if args.bundle:
        write_bundle(result, doc, args.bundle, title=opts.title, crop_dpi=args.crop_dpi or 300)
    for d in result.score.diagnostics:
        if d.severity != "info":
            print(d, file=sys.stderr)
    return 2 if args.strict and result.score.has_errors() else 0


if __name__ == "__main__":
    raise SystemExit(main())
