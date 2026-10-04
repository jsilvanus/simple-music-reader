"""MusicXML 4.0 (score-partwise) exporter: lets another program (or MuseScore, anno-api's renderer)
re-engrave the melody in any layout. Works purely from the Music IR."""
from __future__ import annotations

from fractions import Fraction
from xml.etree import ElementTree as ET

from ..ir import Score

_TYPES = {Fraction(1): "whole", Fraction(1, 2): "half", Fraction(1, 4): "quarter", Fraction(1, 8): "eighth",
          Fraction(1, 16): "16th", Fraction(1, 32): "32nd"}
DIVISIONS = 16  # per quarter note (resolves 64th notes and dotted 32nds)


def _base_and_dots(d: Fraction):
    for dots in range(0, 3):
        base = d / (2 - Fraction(1, 2**dots))
        if base in _TYPES:
            return _TYPES[base], dots
    return None, 0


def export_musicxml(score: Score, title: str | None = None) -> str:
    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title or score.metadata.get("title") or "Untitled"
    plist = ET.SubElement(root, "part-list")
    sp = ET.SubElement(plist, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Melody"
    part = ET.SubElement(root, "part", id="P1")
    voice = score.melody
    meter = score.meter
    for m in (voice.measures if voice else []):
        me = ET.SubElement(part, "measure", number=str(m.index + 1))
        if m.index == 0:
            at = ET.SubElement(me, "attributes")
            ET.SubElement(at, "divisions").text = str(DIVISIONS)
            key = ET.SubElement(at, "key")
            ET.SubElement(key, "fifths").text = str(score.key.fifths)
            ET.SubElement(key, "mode").text = score.key.mode
            if meter:
                t = ET.SubElement(at, "time")
                ET.SubElement(t, "beats").text = str(meter.numerator)
                ET.SubElement(t, "beat-type").text = str(meter.denominator)
            clef = ET.SubElement(at, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"
        for e in m.events:
            n = ET.SubElement(me, "note")
            if e.kind == "rest":
                ET.SubElement(n, "rest")
            else:
                p = ET.SubElement(n, "pitch")
                ET.SubElement(p, "step").text = e.pitch.step
                if e.pitch.alter:
                    ET.SubElement(p, "alter").text = str(e.pitch.alter)
                ET.SubElement(p, "octave").text = str(e.pitch.octave)
            ticks = e.duration * 4 * DIVISIONS
            ET.SubElement(n, "duration").text = str(int(ticks)) if ticks.denominator == 1 else str(round(float(ticks)))
            if e.tie_stop:
                ET.SubElement(n, "tie", type="stop")
            if e.tie_start:
                ET.SubElement(n, "tie", type="start")
            tname, dots = _base_and_dots(e.duration)
            if tname:
                ET.SubElement(n, "type").text = tname
                for _ in range(dots):
                    ET.SubElement(n, "dot")
            if e.tie_start or e.tie_stop:
                nt = ET.SubElement(n, "notations")
                if e.tie_stop:
                    ET.SubElement(nt, "tied", type="stop")
                if e.tie_start:
                    ET.SubElement(nt, "tied", type="start")
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">\n' + body + "\n")
