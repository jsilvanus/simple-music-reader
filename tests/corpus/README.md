# Test corpus

Three kinds of test material, from safest to most fragile:

1. **Generated scores** (`tests/synth.py`): vector PDFs drawn from a text notation at several scales, rasterised
   and rotated on demand. Used by the unit and integration tests; they run in CI and contain no copyrighted music.
2. **Local real PDFs** (this directory is git-ignored for `*.pdf`, `*.png`, `*.jpg`): hymn and liturgy scans you
   are allowed to use locally. Never commit them.
3. **Expected melodies**: hand-written ABC for a real PDF, kept next to it.

To run the optional real-PDF tests:

```bash
export MUSIC_READER_FIXTURES=/path/to/private/fixtures   # or tests/corpus
# name.pdf, name.expected.abc, optional name.options containing e.g. "staff=2 meter=3/4"
python -m pytest -m real_pdf -s
```

`MUSIC_READER_MIN_SIMILARITY` (default `0.8`) is the required sequence similarity of pitch+duration. The test
output lists every diagnostic so a failing PDF can be debugged with `music-reader x.pdf --debug out/`.
