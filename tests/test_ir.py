from fractions import Fraction

import pytest

from music_reader.ir import Event, KeySignature, Measure, Meter, Pitch


def test_pitch_names_and_midi():
    assert Pitch("C", 4).midi == 60
    assert Pitch("F", 4, 1).name == "F#4"
    assert Pitch("E", 4, -1).name == "Eb4"
    assert Pitch.parse("Bb3") == Pitch("B", 3, -1)
    assert Pitch.parse("F#5").midi == 78


def test_pitch_from_diatonic_roundtrip():
    for n in range(20, 50):
        assert Pitch.from_diatonic(n).diatonic == n


def test_treble_clef_bottom_line_is_e4():
    assert Pitch.from_diatonic(4 * 7 + 2).name == "E4"


def test_exact_durations():
    m = Measure(0, [Event("rest", Fraction(1, 3)), Event("rest", Fraction(1, 6))])
    assert m.duration == Fraction(1, 2)  # no float drift


def test_event_validation():
    with pytest.raises(ValueError):
        Event("note", Fraction(1, 4))  # note without pitch
    with pytest.raises(ValueError):
        Event("rest", Fraction(0))


@pytest.mark.parametrize("fifths,sharp_steps", [(0, ""), (1, "F"), (2, "FC"), (3, "FCG")])
def test_key_signature_sharps(fifths, sharp_steps):
    k = KeySignature(fifths)
    assert {s for s in "CDEFGAB" if k.alter_for(s) == 1} == set(sharp_steps)


def test_key_signature_flats_and_names():
    k = KeySignature(-2)
    assert k.tonic == "Bb" and k.alter_for("B") == -1 and k.alter_for("E") == -1 and k.alter_for("A") == 0
    assert KeySignature.parse("Em").fifths == 1
    assert KeySignature.parse("F#").fifths == 6


def test_meter():
    assert Meter.parse("6/8").measure_duration == Fraction(3, 4)
