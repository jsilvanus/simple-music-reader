"""ABC exporter. Works purely from the Music IR; knows nothing about recognition."""
from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..ir import Event, Pitch, Score

_ACC = {1: "^", -1: "_", 0: "="}


def _unit_length(score: Score) -> Fraction:
    durs = {_written(e) for v in score.voices for e in v.events()}
    if not durs:
        return Fraction(1, 4)
    # 1/8 if any duration is shorter than a quarter, else 1/4
    return Fraction(1, 8) if min(durs) < Fraction(1, 4) else Fraction(1, 4)


def _written(e: Event) -> Fraction:
    """The length as printed: a triplet eighth sounds 1/12 but is written as an eighth."""
    return e.duration * e.tuplet[0] / e.tuplet[1] if e.tuplet else e.duration


def _pitch_token(p: Pitch) -> str:
    if p.octave >= 5:
        return p.step.lower() + "'" * (p.octave - 5)
    return p.step + "," * (4 - p.octave)


def _length_token(dur: Fraction, unit: Fraction) -> str:
    r = dur / unit
    n, d = r.numerator, r.denominator
    if d == 1:
        return "" if n == 1 else str(n)
    if n == 1:
        return "/" if d == 2 else f"/{d}"
    return f"{n}/{d}"


def export_abc(score: Score, title: str | None = None) -> str:
    voice = score.melody
    meter = score.meter
    lines = ["X:1", f"T:{title or score.metadata.get('title') or 'Untitled'}"]
    if meter is not None:
        lines.append(f"M:{meter.numerator}/{meter.denominator}")
    unit = _unit_length(score)
    lines.append(f"L:{unit.numerator}/{unit.denominator}")
    lines.append(f"K:{score.key.tonic}{'m' if score.key.mode == 'minor' else ''}")
    header_comments = [f"% {d}" for d in score.diagnostics if d.severity != "info"]
    if score.selection:
        header_comments.insert(0, f"% melody source: {score.selection.get('description', '')}")
    out = header_comments + lines[:]
    if voice is None:
        return "\n".join(out) + "\n"

    key = score.key
    bars: list[str] = []
    for m in voice.measures:
        # accidental state is per bar: (step, octave) -> alter in force
        state: dict[tuple[str, int], int] = {}
        toks: list[str] = []
        run = 0  # events of the tuplet that is open
        for e in m.events:
            if e.tuplet and run == 0:
                toks.append(f"({e.tuplet[0]}")
                run = e.tuplet[0]
            if run:
                run -= 1
            toks.append(_event_token(e, key, state, unit))
        bars.append(" ".join(toks))
    body: list[str] = []
    for i in range(0, len(bars), 4):
        chunk = " | ".join(b for b in bars[i : i + 4])
        body.append(chunk + (" |]" if i + 4 >= len(bars) else " |"))
    out.extend(body)
    return "\n".join(out) + "\n"


def _event_token(e: Event, key, state, unit: Fraction) -> str:
    length = _length_token(_written(e), unit)
    if e.kind == "rest":
        return f"z{length}"
    graces = ""
    if e.graces:
        gt = []
        for g in e.graces:  # an accidental on a grace note is written, but does not stay in force
            inf = state.get((g.step, g.octave), key.alter_for(g.step))
            gt.append(f"{_ACC[g.alter] if g.alter != inf else ''}{_pitch_token(g)}")
        graces = "{" + "".join(gt) + "}"
    p = e.pitch
    in_force = state.get((p.step, p.octave), key.alter_for(p.step))
    acc = ""
    if p.alter != in_force:
        acc = _ACC[p.alter]
    state[(p.step, p.octave)] = p.alter
    return f"{graces}{acc}{_pitch_token(p)}{length}{'-' if e.tie_start else ''}"
