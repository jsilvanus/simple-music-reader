"""End-to-end recognition on generated PDFs: PDF -> Music IR -> ABC."""
from fractions import Fraction as F

import pytest

from music_reader.backends import RecognitionOptions, get_backend
from music_reader.export import export
from music_reader.pdfinput import Document

from tests.helpers import events, measures, parse_abc, spec

pytestmark = pytest.mark.integration


def read(path, **opt):
    return get_backend("classical").recognize(Document(path), RecognitionOptions(**opt))


def codes(score):
    return [d.code for d in score.diagnostics]


def test_first_vertical_slice_pdf_to_valid_abc(make_pdf):
    """One simple PDF -> one staff -> simple notes -> Music IR -> valid ABC."""
    res = read(make_pdf("slice", "E4q G4q B4q D5q | C5h A4q rq | E4q F4e G4e A4h"))
    assert events(res.score) == spec("E4q G4q B4q D5q C5h A4q rq E4q F4e G4e A4h")
    assert res.score.meter.numerator == 4 and res.score.meter.denominator == 4
    abc = export(res.score, "abc")
    assert [(p, d) for p, d, _ in parse_abc(abc)] == events(res.score)
    assert not res.score.has_errors()


def test_note_durations_and_dots(make_pdf):
    text = "C5w | B4h A4h | G4q. F4e E4h | D4q. E4e F4q G4q"
    res = read(make_pdf("durs", text))
    assert events(res.score) == spec(text)
    assert measures(res.score)[2][0] == ("G4", F(3, 8))


def test_sixteenths(make_pdf):
    res = read(make_pdf("sixteenth", "E4q F4s G4s A4e B4h"))
    assert events(res.score) == spec("E4q F4s G4s A4e B4h")


def test_rests(make_pdf):
    res = read(make_pdf("rests", "rw | rh E4h | E4q rq rh | E4q re re rh", meter="4/4"))
    got = events(res.score)
    assert got[0] == ("rest", F(1)) and got[1] == ("rest", F(1, 2))
    assert ("rest", F(1, 4)) in got and ("rest", F(1, 8)) in got


def test_key_signature_applies_to_pitches(make_pdf):
    res = read(make_pdf("keyG", "G4q A4q B4q F5q | F5h G5h", key_fifths=1))
    assert res.score.key.fifths == 1
    assert [p for p, _ in events(res.score)] == ["G4", "A4", "B4", "F#5", "F#5", "G5"]
    assert "K:G" in export(res.score, "abc")


def test_flat_key_signature(make_pdf):
    res = read(make_pdf("keyBb", "B4q E5q A4q D5q | B4w", key_fifths=-2))
    assert res.score.key.fifths == -2
    assert [p for p, _ in events(res.score)] == ["Bb4", "Eb5", "Ab4" if False else "A4", "D5", "Bb4"]


def test_accidentals_in_measure_scope(make_pdf):
    # sharp on F, persists in the bar (second F), cancelled by the barline; natural cancels a key flat
    res = read(make_pdf("acc", "E4q #F4q F4q G4q | F4h bB4h | nB4h A4h", key_fifths=0))
    assert [p for p, _ in events(res.score)] == ["E4", "F#4", "F#4", "G4", "F4", "Bb4", "B4", "A4"]
    abc = export(res.score, "abc")
    assert [p for p, *_ in parse_abc(abc)] == [p for p, _ in events(res.score)]


def test_natural_cancels_key_signature(make_pdf):
    res = read(make_pdf("nat", "F4q nF4q F4h | F4w", key_fifths=1))
    assert [p for p, _ in events(res.score)] == ["F#4", "F4", "F4", "F#4"]


def test_ledger_lines_high_and_low(make_pdf):
    text = "C4q D4q A5q B5q | C6h A3q B3q"
    res = read(make_pdf("ledgerhi", text))
    assert [p for p, _ in events(res.score)] == ["C4", "D4", "A5", "B5", "C6", "A3", "B3"]


def test_ties(make_pdf):
    res = read(make_pdf("ties", "E4h~ E4h | G4q A4q~ A4q B4h"))
    ev = list(res.score.melody.events())
    assert [e.tie_start for e in ev] == [True, False, False, True, False, False]
    assert [e.tie_stop for e in ev] == [False, True, False, False, True, False]
    assert "E2- E2" in export(res.score, "abc")


def test_barlines_make_measures(make_pdf):
    res = read(make_pdf("bars", "E4h F4h | G4h A4h | B4w"))
    assert [len(m) for m in measures(res.score)] == [2, 2, 1]
    assert not res.score.has_errors()


def test_lyrics_and_title_are_ignored(make_pdf):
    res = read(make_pdf("lyr", "E4q G4q B4q D5q"))
    assert len(events(res.score)) == 4


@pytest.mark.parametrize("sp,dpi", [(4.0, None), (6.0, None), (8.0, None)])
def test_independent_of_page_scale(make_pdf, sp, dpi):
    text = "E4q G4q B4q D5q | C5h A4q rq | E4q F4e G4e A4h"
    res = read(make_pdf(f"scale{sp}", text, sp=sp))
    assert events(res.score) == spec(text)


def test_scanned_rotated_page(make_pdf, pdfdir):
    from tests.synth import rasterize_pdf
    src = make_pdf("scan-src", "E4q G4q B4q D5q | C5h A4q rq | E4q F4e G4e A4h")
    dst = pdfdir / "scan.pdf"
    rasterize_pdf(src, dst, dpi=220, rotate_deg=0.8, noise=6.0)
    doc = Document(dst)
    assert doc.analyze(0).kind == "raster"
    res = get_backend("classical").recognize(doc, RecognitionOptions())
    assert res.pages[0].skew_degrees == pytest.approx(-0.8, abs=0.25)
    assert events(res.score) == spec("E4q G4q B4q D5q C5h A4q rq E4q F4e G4e A4h")


def test_meter_override_and_inference(make_pdf):
    path = make_pdf("waltz", "E4q G4q B4q | C5h A4q | E4q F4q G4q", meter="3/4")
    res = read(path)
    assert str(res.score.meter) == "3/4" and res.score.meter.inferred
    assert "METER_INFERRED" in codes(res.score)
    res = read(path, meter="3/4")
    assert not res.score.meter.inferred and "METER_INFERRED" not in codes(res.score)


def test_wrong_meter_is_flagged_not_hidden(make_pdf):
    res = read(make_pdf("wrong", "E4q G4q B4q | C5h A4q | E4q F4q G4q"), meter="4/4")
    assert codes(res.score).count("MEASURE_INCOMPLETE") + codes(res.score).count("MEASURE_OVERFULL") >= 2
    assert measures(res.score)[0] == spec("E4q G4q B4q")  # untouched


def test_key_override(make_pdf):
    res = read(make_pdf("keyover", "G4q A4q B4q F5q | F5h G5h"), key="G")
    assert res.score.key.fifths == 1 and events(res.score)[3][0] == "F#5"


def test_bass_clef_is_a_flagged_guess(make_pdf):
    res = read(make_pdf("bass", "G2q A2q B2q C3q", clef="bass"))
    assert "CLEF_BASS_GUESSED" in codes(res.score)
    assert [p for p, _ in events(res.score)] == ["G2", "A2", "B2", "C3"]


def test_no_staves(pdfdir):
    from reportlab.pdfgen import canvas
    p = pdfdir / "blank.pdf"
    c = canvas.Canvas(str(p))
    c.drawString(100, 700, "no music here")
    c.save()
    res = read(p)
    assert "NO_STAVES" in codes(res.score) and "NO_MUSIC" in codes(res.score)
    assert res.score.has_errors()


def test_bad_page_number(make_pdf):
    res = read(make_pdf("one", "E4w"), pages=[3])
    assert "BAD_PAGE" in codes(res.score)


def test_pdf_classification(make_pdf):
    a = Document(make_pdf("cls", "E4w")).analyze(0)
    assert a.kind == "vector" and a.n_paths >= 5 and a.n_images == 0
