# Development

```bash
pip install -e ".[test]"
python -m pytest                      # unit + integration tests on generated scores (about 10 s)
python -m pytest -m "not integration" # pure unit tests only
python -m pytest -m real_pdf -s       # optional, needs MUSIC_READER_FIXTURES (see tests/corpus/README.md)
```

## Test layers

| Layer | Files | Needs |
|---|---|---|
| deterministic unit tests | `test_ir.py`, `test_abc.py`, `test_validate.py`, `test_musicxml.py`, `test_cv_units.py` | nothing but the test extra |
| integration (PDF to ABC) | `test_recognition.py`, `test_pipeline.py` (marked `integration`) | generated PDFs |
| optional real PDFs | `test_real_pdfs.py` (marked `real_pdf`) | your own fixtures, never committed |

Coverage of the topics in the brief: staff detection (`test_cv_units.py`), pitch (`test_recognition.py`, ledger lines,
clefs), durations and dots, accidentals and key signatures, barlines, ABC generation (with a small ABC reader in
`tests/helpers.py` that checks the ABC *means* what the IR says), multi-page and multi-system, malformed/ambiguous input
(wrong meter, missing staves, bad page, low resolution, ambiguous melody staff), validation.

## The synthetic score generator

`tests/synth.py` draws scores from a tiny text notation with reportlab:

```python
from tests.synth import SystemSpec, StaffSpec, parse_measures, render_pdf, rasterize_pdf
render_pdf("out.pdf", [[SystemSpec([StaffSpec(parse_measures("E4q F#4e G4e A4h~ | A4w"), key_fifths=1)])]], sp=5.0)
rasterize_pdf("out.pdf", "scan.pdf", dpi=220, rotate_deg=0.8, noise=6)   # makes a "scanned" PDF
```

Tokens: pitch + duration letter (`w h q e s`), `.` dot, `~` tie, prefix `#`, `b`, `n` for a printed accidental, `rq` rest.
It draws a crude engraving (ellipse heads, line-art clefs), so passing tests prove the logic, **not** that real
engraving fonts work. When you meet a real PDF that fails, reduce it to the smallest failing symbol, add the
equivalent to the generator or a unit test, and fix it there.

## Debugging a PDF

```bash
music-reader hymn.pdf --debug debug/
```

`debug/page-N.png` (and `debug.pdf`) show staff lines (blue), system boxes, clef/key/time (blue), notes (green, red when
confidence < 0.6) with pitch, duration and confidence, rests, barlines (orange), accidentals/dots, ties, unknown symbols
(red `?`) and rejected candidates (grey, "no ledger"). `diagnostics.txt` lists every diagnostic.

## Conventions

* Python 3.10+, type hints, dataclasses. No linter is configured; match the surrounding style.
* Sizes in `cv/` are multiples of `Staff.spacing` (`sp`), never pixels.
* Recognition emits `Diagnostic`s instead of guessing silently; add a code to `validate.py` or `symbols.py`.
* Exporters read only the IR. Backends never write ABC.
* Never commit copyrighted PDFs or scans (`tests/corpus/.gitignore`d patterns).
