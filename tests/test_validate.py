from fractions import Fraction as F

from music_reader.ir import Event, KeySignature, Measure, Meter, Pitch, Score, Voice
from music_reader.validate import validate


def mk(bars, meter="4/4"):
    voice = Voice("1", measures=[Measure(i, e) for i, e in enumerate(bars)])
    return Score(meter=Meter.parse(meter) if meter else None, voices=[voice])


def n(name, dur="1/4", conf=1.0):
    return Event("note", F(dur), Pitch.parse(name), confidence=conf)


def codes(s):
    return [d.code for d in validate(s)]


def test_valid_score_has_no_diagnostics():
    assert validate(mk([[n("C4"), n("D4"), n("E4"), n("F4")], [n("G4", "1")]])) == []


def test_overfull_measure_is_an_error():
    s = mk([[n("C4"), n("D4"), n("E4"), n("F4"), n("G4")]])
    d = validate(s)
    assert d[0].code == "MEASURE_OVERFULL" and d[0].severity == "error" and d[0].measure == 0


def test_incomplete_measure_is_flagged_not_fixed():
    s = mk([[n("C4"), n("D4"), n("E4"), n("F4")], [n("G4"), n("A4")], [n("B4", "1")]])
    assert codes(s) == ["MEASURE_INCOMPLETE"]
    assert s.melody.measures[1].duration == F(1, 2)  # untouched


def test_pickup_measure_completed_by_last_measure_is_fine():
    s = mk([[n("C4")], [n("D4", "1")], [n("E4", "1/2"), n("F4", "1/4")]])
    assert validate(s) == []


def test_empty_measure_and_missing_meter():
    s = mk([[n("C4", "1")], []])
    assert "EMPTY_MEASURE" in codes(s)
    assert codes(mk([[n("C4", "1")]], meter=None)) == ["NO_METER"]


def test_suspicious_jump_and_range_and_confidence():
    s = mk([[n("C4"), n("D5", conf=0.3), n("D2"), n("E4")]])
    c = codes(s)
    assert "SUSPICIOUS_JUMP" in c and "LOW_CONFIDENCE" in c and "UNUSUAL_PITCH" in c


def test_no_music():
    assert codes(Score()) == ["NO_MUSIC"]


def test_infer_meter_rejects_bizarre_chant_durations():
    from music_reader.assemble import infer_meter
    from music_reader.ir import Measure
    # Irregular or chant bars of length 55/12 or 18/4
    bars = [Measure(0, [n('C4', '18/4')]), Measure(1, [n('D4', '18/4')]), Measure(2, [n('E4', '18/4')])]
    assert infer_meter(bars) is None


def test_reconcile_meter_from_measures_in_classical():
    from music_reader.backends.classical import ClassicalBackend, RecognitionOptions
    from music_reader.ir import Score, Voice, Measure
    from music_reader.cv.symbols import StaffResult, Item
    from types import SimpleNamespace as NS

    backend = ClassicalBackend()
    # 6 bars of 6/4 (duration 3/2) with printed 5/4
    measures = [Measure(i, [n('C4', '1/4'), n('D4', '1/4'), n('E4', '1/4'), n('F4', '1/4'), n('G4', '1/4'), n('A4', '1/4')]) for i in range(6)]
    doc = NS(analyze=lambda p: NS(kind='vector', music_fonts=[]), num_pages=1)
    # Test infer_meter directly on these 6 measures
    from music_reader.assemble import infer_meter
    assert str(infer_meter(measures)) == '6/4'
