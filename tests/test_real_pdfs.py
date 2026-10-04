"""Optional tests on real church PDFs. Real hymn PDFs are copyrighted and must NOT be committed.

Point MUSIC_READER_FIXTURES at a local directory containing ``name.pdf`` and, next to it, the expected
melody ``name.expected.abc`` (one staff, written by a human). Each pair is read with the classical backend
and the pitch/duration sequence is compared with the expectation. See tests/corpus/README.md.
"""
import difflib
import os
from pathlib import Path

import pytest

from music_reader.backends import RecognitionOptions, get_backend
from music_reader.export import export
from music_reader.pdfinput import Document

from tests.helpers import parse_abc

pytestmark = pytest.mark.real_pdf
ROOT = os.environ.get("MUSIC_READER_FIXTURES")
PAIRS = sorted(Path(ROOT).glob("*.pdf")) if ROOT and Path(ROOT).is_dir() else []
MIN_SIMILARITY = float(os.environ.get("MUSIC_READER_MIN_SIMILARITY", "0.8"))


@pytest.mark.skipif(not PAIRS, reason="set MUSIC_READER_FIXTURES to a directory with real PDFs")
@pytest.mark.parametrize("pdf", PAIRS, ids=[p.stem for p in PAIRS])
def test_real_pdf_melody(pdf):
    expected = pdf.with_suffix(".expected.abc")
    if not expected.exists():
        pytest.skip(f"no {expected.name}")
    kw = {}
    opts_file = pdf.with_suffix(".options")
    if opts_file.exists():  # e.g. "staff=2 meter=3/4"
        kw = dict(tok.split("=") for tok in opts_file.read_text().split())
        if "staff" in kw:
            kw["staff"] = int(kw["staff"])
    res = get_backend("classical").recognize(Document(pdf), RecognitionOptions(**kw))
    got = [(p, d) for p, d, _ in parse_abc(export(res.score, "abc"))]
    want = [(p, d) for p, d, _ in parse_abc(expected.read_text())]
    ratio = difflib.SequenceMatcher(None, got, want).ratio()
    print(f"{pdf.name}: similarity {ratio:.2f}, {len(got)} events vs {len(want)} expected")
    for d in res.score.diagnostics:
        print("  ", d)
    assert ratio >= MIN_SIMILARITY
