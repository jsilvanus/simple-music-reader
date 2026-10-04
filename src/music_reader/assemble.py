"""Turn recognised staff items into the Music IR (Score): pitches, measures, key, meter, ties."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Optional

from .cv.symbols import BASE_BASS, BASE_TREBLE, Item, StaffResult
from .ir import Diagnostic, Event, KeySignature, Measure, Meter, Pitch, Score, Source, Voice

_ALTER = {"#": 1, "b": -1, "n": 0}


def infer_meter(measures: list[Measure]) -> Optional[Meter]:
    full = [m.duration for m in measures if m.events]
    if len(full) >= 3:
        full = full[1:-1]  # first and last bars may be pick-up / ending
    if not full:
        return None
    s = Counter(full).most_common(1)[0][0]
    if s.denominator <= 4:
        return Meter(int(s * 4), 4, inferred=True)
    return Meter(s.numerator, s.denominator, inferred=True)


def build_voice(selected: list[tuple[StaffResult, int, int]], key: KeySignature, diagnostics: list[Diagnostic]) -> Voice:
    """``selected``: (staff result, page number, system index) in reading order."""
    voice = Voice("1", "melody")
    measure = Measure(0)
    state: dict[tuple[str, int], int] = {}
    prev_tied: Optional[Event] = None

    def close():
        nonlocal measure, state
        if measure.events:
            voice.measures.append(measure)
            measure = Measure(len(voice.measures))
        state = {}

    for res, page, system in selected:
        base = BASE_BASS if res.clef == "bass" else BASE_TREBLE
        for it in res.items:
            src = Source(page, it.bbox, system, res.staff.index, it.comp)
            if it.kind == "bar":
                close()
                continue
            onset = measure.duration
            if it.kind == "rest":
                ev = Event("rest", it.duration, onset=onset, measure=measure.index, confidence=it.conf, source=src)
            else:
                d = base + it.pos
                step = "CDEFGAB"[d % 7]
                octave = d // 7
                if it.accidental:
                    alter = _ALTER[it.accidental]
                else:
                    alter = state.get((step, octave), key.alter_for(step))
                state[(step, octave)] = alter
                ev = Event("note", it.duration, Pitch(step, octave, alter), onset, measure.index,
                           accidental=it.accidental, dots=it.dots, tie_start=it.tie_start,
                           confidence=it.conf, source=src)
                if prev_tied is not None:
                    if prev_tied.pitch == ev.pitch:
                        ev.tie_stop = True
                    else:
                        diagnostics.append(Diagnostic("TIE_MISMATCH", "warning", "tied notes have different pitches", src, measure.index))
                    prev_tied = None
                if ev.tie_start:
                    prev_tied = ev
            if it.kind == "rest":
                prev_tied = None
            measure.events.append(ev)
            if measure.source is None:
                measure.source = src
    close()
    return voice
