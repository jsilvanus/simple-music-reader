"""Grace notes (small slashed notes that take no time) and triplets (a 3 over three notes, bracketed unless beamed)."""
from fractions import Fraction as F

import pytest

from music_reader.backends import RecognitionOptions, get_backend
from music_reader.export import export
from music_reader.pdfinput import Document

from tests.helpers import events, parse_abc, spec
from tests.synth import StaffSpec, SystemSpec, parse_measures, render_pdf

pytestmark = pytest.mark.integration

TWO_THIRDS = F(2, 3)


def make(tmp_path_factory, name, text, sp=5.0, **staff):
    path = tmp_path_factory.mktemp(name) / f"{name}.pdf"
    render_pdf(path, [[SystemSpec([StaffSpec(parse_measures(text), **staff)])]], sp=sp, lyrics=False)
    return path


def read(path, **opt):
    return get_backend("classical").recognize(Document(path), RecognitionOptions(meter="4/4", **opt))


def triplet(text, n=3):
    """spec() of the first ``n`` tokens with the two-thirds rule applied, then the rest as written."""
    toks = text.split()
    head, tail = " ".join(toks[:n]), " ".join(toks[n:])
    return [(p, d * TWO_THIRDS) for p, d in spec(head)] + spec(tail)


def codes(score):
    return [d.code for d in score.diagnostics]


# ---- triplets ----------------------------------------------------------------------

BEAMED = "E4e G4e B4e C5q D5h"


def test_beamed_triplet_with_a_number(tmp_path_factory):
    res = read(make(tmp_path_factory, "tri-beam", "(3 " + BEAMED, beam_group_size=3))
    assert events(res.score) == triplet(BEAMED)
    assert [e.tuplet for e in res.score.melody.events()] == [(3, 2)] * 3 + [None] * 2
    assert not res.score.has_errors() and "MEASURE_INCOMPLETE" not in codes(res.score)


def test_bracketed_quarter_triplet_and_repeated_pitch_is_not_a_tie(tmp_path_factory):
    text = "G4q G4q G4q A4h"
    res = read(make(tmp_path_factory, "tri-bracket", "(3 " + text))
    assert events(res.score) == triplet(text)
    assert not any(e.tie_start for e in res.score.melody.events())
    assert not res.score.has_errors()


def test_triplet_with_stems_down_has_its_number_below(tmp_path_factory):
    text = "B4e C5e D5e E5q F5h"
    res = read(make(tmp_path_factory, "tri-down", "(3 " + text, beam_group_size=3))
    assert events(res.score) == triplet(text)


def test_triplet_with_a_rest(tmp_path_factory):
    text = "E4e re G4e C5q D5h"
    res = read(make(tmp_path_factory, "tri-rest", "(3 " + text, beam_group_size=3))
    assert events(res.score) == triplet(text)


def test_number_that_crosses_a_staff_line(tmp_path_factory):
    text = "A3e B3e C4e C5q D5h"
    res = read(make(tmp_path_factory, "tri-cross", "(3 " + text, beam_group_size=3))
    assert events(res.score) == triplet(text)


def test_two_triplets_in_a_row(tmp_path_factory):
    text = "E4q G4q B4q C5q D5q E5q"
    res = read(make(tmp_path_factory, "tri-two", "(3 E4q G4q B4q (3 C5q D5q E5q"))
    assert events(res.score) == triplet(text, 3)[:3] + [(p, d * TWO_THIRDS) for p, d in spec("C5q D5q E5q")]


def test_a_three_that_is_no_triplet_is_not_one(tmp_path_factory):
    # nothing is printed over these notes: no tuplet, plain durations
    res = read(make(tmp_path_factory, "tri-none", "E4q G4q B4q D5q | C5w"))
    assert all(e.tuplet is None for e in res.score.melody.events())


def test_triplet_exports(tmp_path_factory):
    res = read(make(tmp_path_factory, "tri-exp", "(3 " + BEAMED, beam_group_size=3))
    abc = export(res.score, "abc")
    assert "(3" in abc
    assert [(p, d) for p, d, _ in parse_abc(abc)] == events(res.score)
    xml = export(res.score, "musicxml")
    assert xml.count("<time-modification>") == 3 and 'tuplet type="start"' in xml and 'tuplet type="stop"' in xml
    melody = __import__("json").loads(export(res.score, "melody"))
    ev = melody["measures"][0]["events"]
    assert [e.get("tuplet") for e in ev[:4]] == ["3:2"] * 3 + [None] and ev[0]["duration"] == "1/12"


# ---- grace notes -------------------------------------------------------------------

def test_grace_note_takes_no_time_and_is_attached(tmp_path_factory):
    res = read(make(tmp_path_factory, "grace1", "{G4}A4q B4q C5h | {E5}D5q C5q B4h"))
    assert events(res.score) == spec("A4q B4q C5h D5q C5q B4h")
    assert [[g.name for g in e.graces] for e in res.score.melody.events()] == [["G4"], [], [], ["E5"], [], []]
    assert not res.score.has_errors() and "UNCLASSIFIED_SYMBOL" not in codes(res.score)


def test_grace_notes_high_low_and_two_in_a_row(tmp_path_factory):
    res = read(make(tmp_path_factory, "grace2", "{C5,D5}E5q F5q G5h | {A3}C4q D4q E4h"))
    assert events(res.score) == spec("E5q F5q G5h C4q D4q E4h")
    assert [[g.name for g in e.graces] for e in res.score.melody.events()] == [["C5", "D5"], [], [], ["A3"], [], []]


def test_grace_note_follows_the_key_signature(tmp_path_factory):
    res = read(make(tmp_path_factory, "grace-key", "{F5}G5q A5q B5h | G5w", key_fifths=1))
    assert [g.name for g in next(res.score.melody.events()).graces] == ["F#5"]


def test_grace_notes_leave_everything_else_alone(tmp_path_factory):
    plain = "E4q G4q B4q D5q | C5h A4q rq | E4q F4e G4e A4h"
    withgrace = "{D4}E4q G4q B4q D5q | C5h {B4}A4q rq | E4q F4e G4e A4h"
    a = read(make(tmp_path_factory, "g-plain", plain))
    b = read(make(tmp_path_factory, "g-with", withgrace))
    assert events(a.score) == events(b.score) == spec(plain)


def test_grace_exports(tmp_path_factory):
    res = read(make(tmp_path_factory, "grace-exp", "{G4}A4q B4q C5h"))
    abc = export(res.score, "abc")
    assert "{G}" in abc and [(p, d) for p, d, _ in parse_abc(abc)] == events(res.score)
    assert '<grace slash="yes"' in export(res.score, "musicxml")
    first = __import__("json").loads(export(res.score, "melody"))["measures"][0]["events"][0]
    assert first["grace_notes"] == ["G4"] and first["onset"] == "0"


# ---- scanned pages -----------------------------------------------------------------

@pytest.mark.parametrize("text,kw", [("(3 E4e G4e B4e C5q D5h", {"beam_group_size": 3}), ("(3 G4q G4q G4q A4h", {}),
                                      ("{G4}A4q B4q C5h | {E5}D5q C5q B4h", {})])
def test_survives_a_rotated_noisy_scan(tmp_path_factory, text, kw):
    from tests.synth import rasterize_pdf
    src = make(tmp_path_factory, "scan-src", text, **kw)
    dst = src.with_name("scan.pdf")
    rasterize_pdf(src, dst, dpi=220, rotate_deg=0.8, noise=6.0)
    got = read(dst).score
    ref = read(src).score
    assert events(got) == events(ref)
    assert [e.tuplet for e in got.melody.events()] == [e.tuplet for e in ref.melody.events()]
    assert [[g.name for g in e.graces] for e in got.melody.events()] == [[g.name for g in e.graces] for e in ref.melody.events()]
