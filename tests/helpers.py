from __future__ import annotations

import re
from fractions import Fraction

from music_reader.ir import KeySignature, Pitch

_LEN = {"w": Fraction(1), "h": Fraction(1, 2), "q": Fraction(1, 4), "e": Fraction(1, 8), "s": Fraction(1, 16)}


def events(score):
    """[(pitch name or 'rest', duration)] of the melody."""
    return [("rest" if e.kind == "rest" else e.pitch.name, e.duration) for e in score.melody.events()]


def spec(text: str):
    """'E4q G4h r q' style list -> [(pitch, Fraction)] with accidentals written as sounding pitch: 'F#4q'."""
    out = []
    for tok in text.replace("|", " ").split():
        if tok[0] == "r":
            name, rest = "rest", tok[1:]
        else:
            m = re.match(r"([A-G][#b]?\d)(.*)", tok)
            name, rest = m.group(1), m.group(2)
        dots = rest.count(".")
        base = _LEN[rest.replace(".", "").replace("~", "")]
        out.append((name, base * (2 - Fraction(1, 2**dots))))
    return out


def measures(score):
    return [[("rest" if e.kind == "rest" else e.pitch.name, e.duration) for e in m.events] for m in score.melody.measures]


def parse_abc(text: str):
    """Minimal ABC reader used to check that exported ABC means what the IR says."""
    key = KeySignature(0)
    unit = Fraction(1, 8)
    body = []
    for line in text.splitlines():
        if line.startswith("%"):
            continue
        if line.startswith("K:"):
            key = KeySignature.parse(line[2:].strip())
        elif line.startswith("L:"):
            n, d = line[2:].strip().split("/")
            unit = Fraction(int(n), int(d))
        elif re.match(r"^[A-Z]:", line):
            continue
        else:
            body.append(line)
    out, state = [], {}
    for tok in re.findall(r"\|\]|\||[_=^]*[A-Ga-gz][,']*\d*/?\d*-?", " ".join(body)):
        if tok.startswith("|"):
            state = {}
            continue
        m = re.match(r"([_=^]*)([A-Ga-gz])([,']*)(\d*)(/?)(\d*)(-?)$", tok)
        acc, letter, octs, num, slash, den, tie = m.groups()
        mult = Fraction(int(num) if num else 1, int(den) if den else (2 if slash else 1))
        dur = unit * mult
        if letter == "z":
            out.append(("rest", dur, False))
            continue
        step = letter.upper()
        octave = 4 if letter.isupper() else 5
        octave += octs.count("'") - octs.count(",")
        if acc:
            state[(step, octave)] = {"^": 1, "_": -1, "=": 0}[acc[-1]]
        alter = state.get((step, octave), key.alter_for(step))
        out.append((Pitch(step, octave, alter).name, dur, bool(tie)))
    return out
