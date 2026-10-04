from fractions import Fraction as F

from music_reader.export import export
from music_reader.export.abc import export_abc
from music_reader.ir import Event, KeySignature, Measure, Meter, Pitch, Score, Voice

from tests.helpers import parse_abc


def score_of(bars, meter="4/4", key=KeySignature(0)):
    voice = Voice("1", measures=[Measure(i, evs) for i, evs in enumerate(bars)])
    return Score(meter=Meter.parse(meter), key=key, voices=[voice])


def n(name, dur, **kw):
    return Event("note", F(dur), Pitch.parse(name), **kw)


def r(dur):
    return Event("rest", F(dur))


def test_example_from_the_brief():
    s = score_of([[n("G4", "1/4"), n("A4", "1/4"), n("B4", "1/4"), n("C5", "1/4")],
                  [n("D5", "1/2"), n("B4", "1/2")],
                  [n("A4", "1/4"), n("G4", "1/4"), n("F#4", "1/4"), n("E4", "1/4")],
                  [n("D4", "1")]], key=KeySignature(1))
    abc = export_abc(s, title="Example")
    assert "X:1" in abc and "T:Example" in abc and "M:4/4" in abc and "L:1/4" in abc and "K:G" in abc
    assert "G A B c | d2 B2 | A G F E | D4 |]" in abc  # F is sharp in G major: no accidental needed


def test_pitch_and_octave_notation():
    s = score_of([[n("C3", "1/4"), n("C4", "1/4"), n("C5", "1/4"), n("C6", "1/4")]])
    assert "C, C c c'" in export_abc(s)


def test_accidentals_follow_key_and_bar_scope():
    # G major: F# is implicit; F natural needs "=", then stays natural for the rest of the bar
    s = score_of([[n("F#4", "1/4"), n("F4", "1/4"), n("F4", "1/4"), n("F#4", "1/4")],
                  [n("F#4", "1/4"), n("G#4", "1/4"), n("G#4", "1/4"), n("G4", "1/4")]], key=KeySignature(1))
    abc = export_abc(s)
    assert "F =F F ^F | F ^G G =G |]" in abc  # accidentals persist to the end of the bar
    assert [p for p, *_ in parse_abc(abc)] == [e.pitch.name for m in s.voices[0].measures for e in m.events]


def test_rests_and_lengths():
    s = score_of([[r("1/2"), n("E4", "1/4"), r("1/4")], [r("1")]])
    abc = export_abc(s)
    assert "z2 E z |" in abc and "z4 |]" in abc


def test_eighths_switch_unit_length_and_fractions():
    s = score_of([[n("E4", "1/8"), n("F4", "1/8"), n("G4", "3/8"), n("A4", "1/4")]])
    abc = export_abc(s)
    assert "L:1/8" in abc
    assert [d for _, d, _ in parse_abc(abc)] == [F(1, 8), F(1, 8), F(3, 8), F(1, 4)]


def test_tie_marker():
    s = score_of([[n("E4", "1/2", tie_start=True), n("E4", "1/2", tie_stop=True)]])
    assert "E2-" in export_abc(s)


def test_roundtrip_semantics():
    bars = [[n("Bb4", "1/4"), n("B4", "1/4"), n("B4", "1/4"), r("1/4")],
            [n("E4", "1/8"), n("Eb4", "1/8"), n("E5", "1/4"), n("E4", "1/2")]]
    s = score_of(bars, key=KeySignature(-2))
    got = parse_abc(export_abc(s))
    want = [("rest" if e.kind == "rest" else e.pitch.name, e.duration) for m in s.voices[0].measures for e in m.events]
    assert [(p, d) for p, d, _ in got] == want


def test_diagnostics_are_kept_as_comments():
    from music_reader.ir import Diagnostic
    s = score_of([[n("C4", "1")]])
    s.diagnostics.append(Diagnostic("X", "warning", "check me"))
    assert "% WARNING X: check me" in export_abc(s)


def test_unknown_format():
    import pytest
    with pytest.raises(ValueError):
        export(score_of([[n("C4", "1")]]), "nope")
