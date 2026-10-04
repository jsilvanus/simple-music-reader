"""ABC exporter. Works purely from the Music IR; knows nothing about recognition."""
from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..ir import Event, Pitch, Score

_ACC = {1: "^", -1: "_", 0: "="}


def _unit_length(score: Score) -> Fraction:
    durs = {e.duration for v in score.voices for e in v.events()}
    if not durs:
        return Fraction(1, 4)
    # 1/8 if any duration is shorter than a quarter, else 1/4
    return Fraction(1, 8) if min(durs) < Fraction(1, 4) else Fraction(1, 4)


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
        for e in m.events:
            toks.append(_event_token(e, key, state, unit))
        bars.append(" ".join(toks))
    body: list[str] = []
    for i in range(0, len(bars), 4):
        chunk = " | ".join(b for b in bars[i : i + 4])
        body.append(chunk + (" |]" if i + 4 >= len(bars) else " |"))
    out.extend(body)
    return "\n".join(out) + "\n"


def _event_token(e: Event, key, state, unit: Fraction) -> str:
    length = _length_token(e.duration, unit)
    if e.kind == "rest":
        return f"z{length}"
    p = e.pitch
    in_force = state.get((p.step, p.octave), key.alter_for(p.step))
    acc = ""
    if p.alter != in_force:
        acc = _ACC[p.alter]
    state[(p.step, p.octave)] = p.alter
    return f"{acc}{_pitch_token(p)}{length}{'-' if e.tie_start else ''}"
