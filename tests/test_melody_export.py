import json
from fractions import Fraction

import pytest

from music_reader.backends import RecognitionOptions, get_backend
from music_reader.export import export
from music_reader.export.melody import SCHEMA
from music_reader.pdfinput import Document

pytestmark = pytest.mark.integration


def test_melody_export_has_a_stable_shape(make_pdf):
    res = get_backend("classical").recognize(Document(make_pdf("melody-export", "E4q G4q B4q | C5h A4q | E4q F4q G4q", meter="3/4")),
                                             RecognitionOptions())
    data = json.loads(export(res.score, "melody", title="Test"))
    assert data["schema"] == SCHEMA and data["title"] == "Test"
    assert data["meter"] == {"numerator": 3, "denominator": 4, "source": "printed"}
    assert data["key"]["fifths"] == 0 and data["key"]["tonic"] == "C"
    assert [m["index"] for m in data["measures"]] == [0, 1, 2]
    first = data["measures"][0]
    assert [e["pitch"] for e in first["events"]] == ["E4", "G4", "B4"]
    assert [e["midi"] for e in first["events"]] == [64, 67, 71]
    assert sum(Fraction(e["duration"]) for e in first["events"]) == Fraction(first["duration"]) == Fraction(3, 4)
    assert 0 < first["confidence"] <= 1 and 0 < data["confidence"] <= 1
    assert data["needs_review"] in (True, False)


def test_melody_export_says_where_the_meter_came_from(make_pdf):
    path = make_pdf("melody-export-inferred", "E4q G4q B4q | C5h A4q | E4q F4q G4q", meter=None)
    res = get_backend("classical").recognize(Document(path), RecognitionOptions())
    assert json.loads(export(res.score, "melody"))["meter"]["source"] == "inferred"
    res = get_backend("classical").recognize(Document(path), RecognitionOptions(meter="3/4"))
    assert json.loads(export(res.score, "melody"))["meter"]["source"] == "option"
