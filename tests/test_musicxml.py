from fractions import Fraction as F
from xml.etree import ElementTree as ET

from music_reader.export import export
from music_reader.ir import Event, KeySignature, Measure, Meter, Pitch, Score, Voice


def test_musicxml_is_wellformed_and_complete():
    ev = [Event("note", F(3, 8), Pitch.parse("F#4"), dots=1, tie_start=True), Event("note", F(1, 8), Pitch.parse("F#4"), tie_stop=True),
          Event("rest", F(1, 2))]
    s = Score(meter=Meter(4, 4), key=KeySignature(2), voices=[Voice("1", measures=[Measure(0, ev)])])
    root = ET.fromstring(export(s, "musicxml", title="T").split("?>", 1)[1].split("]>", 1)[-1].split("dtd\">", 1)[-1])
    assert root.find("work/work-title").text == "T"
    notes = root.findall("part/measure/note")
    assert [n.find("duration").text for n in notes] == ["24", "8", "32"]
    assert notes[0].find("pitch/step").text == "F" and notes[0].find("pitch/alter").text == "1"
    assert notes[0].find("type").text == "quarter" and notes[0].find("dot") is not None
    assert notes[2].find("rest") is not None
    assert root.find("part/measure/attributes/key/fifths").text == "2"
    assert root.find("part/measure/attributes/time/beats").text == "4"
    assert notes[0].find("tie").get("type") == "start" and notes[1].find("tie").get("type") == "stop"
