"""Musical-structure validation. Flags problems; never silently repairs them."""
from __future__ import annotations

from fractions import Fraction

from .ir import Diagnostic, Score

LOW_CONFIDENCE = 0.6
MAX_JUMP_SEMITONES = 12
LOW_MIDI, HIGH_MIDI = 43, 88  # G2 .. E6


def validate(score: Score) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    voice = score.melody
    if voice is None or not voice.measures:
        out.append(Diagnostic("NO_MUSIC", "error", "no measures were recognised"))
        return out
    if score.meter is None:
        out.append(Diagnostic("NO_METER", "warning", "no time signature known; durations are not checked"))
    else:
        md = score.meter.measure_duration
        last = len(voice.measures) - 1
        for m in voice.measures:
            d = m.duration
            if not m.events:
                out.append(Diagnostic("EMPTY_MEASURE", "warning", "measure has no events", m.source, m.index))
            elif d > md:
                out.append(Diagnostic("MEASURE_OVERFULL", "error", f"measure holds {d} but meter {score.meter} allows {md}", m.source, m.index))
            elif d < md:
                pickup_ok = m.index == 0 and last > 0 and voice.measures[last].duration + d == md
                final_ok = m.index == last and _pickup_partner(voice, md)
                if not (pickup_ok or final_ok):
                    out.append(Diagnostic("MEASURE_INCOMPLETE", "warning", f"measure holds {d} but meter {score.meter} expects {md}", m.source, m.index))
    prev = None
    for e in voice.events():
        if e.confidence < LOW_CONFIDENCE:
            out.append(Diagnostic("LOW_CONFIDENCE", "warning", f"low confidence ({e.confidence:.2f}) for {e.kind}", e.source, e.measure))
        if e.kind != "note":
            continue
        if not LOW_MIDI <= e.pitch.midi <= HIGH_MIDI:
            out.append(Diagnostic("UNUSUAL_PITCH", "warning", f"{e.pitch} is outside the usual melody range", e.source, e.measure))
        if prev is not None and abs(e.pitch.midi - prev.midi) > MAX_JUMP_SEMITONES:
            out.append(Diagnostic("SUSPICIOUS_JUMP", "warning", f"jump of {abs(e.pitch.midi - prev.midi)} semitones ({prev} to {e.pitch})", e.source, e.measure))
        prev = e.pitch
    return out


def _pickup_partner(voice, md: Fraction) -> bool:
    first = voice.measures[0]
    return len(voice.measures) > 1 and first.duration < md and first.duration + voice.measures[-1].duration == md
