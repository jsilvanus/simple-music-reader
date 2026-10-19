"""Melody export for other programs (liturgos-musicus and similar): the recognised melody and nothing else.

A small, versioned JSON document: the measures with pitch, duration and confidence per event, the meter and key with
where they came from, and the diagnostics that say how far to trust it. It carries no page images and no pixel
positions. Additions to it keep ``schema`` at 1; a change that breaks a reader gets a new number.
"""
from __future__ import annotations

import json

from ..ir import Event, Score

SCHEMA = "music-reader.melody/1"


def _event(e: Event) -> dict:
    out = {
        "kind": e.kind,
        "duration": str(e.duration),
        "onset": str(e.onset),
        "confidence": round(e.confidence, 3),
    }
    if e.kind == "note":
        out.update(pitch=e.pitch.name, midi=e.pitch.midi, tie_to_next=e.tie_start)
    return out


def melody_dict(score: Score, title: str | None = None) -> dict:
    voice = score.melody
    measures = []
    for m in (voice.measures if voice else []):
        confs = [e.confidence for e in m.events]
        measures.append({
            "index": m.index,
            "duration": str(m.duration),
            "confidence": round(min(confs), 3) if confs else 0.0,
            "events": [_event(e) for e in m.events],
        })
    events = [e for m in (voice.measures if voice else []) for e in m.events]
    return {
        "schema": SCHEMA,
        "title": title or score.metadata.get("title"),
        "meter": None if score.meter is None else {
            "numerator": score.meter.numerator, "denominator": score.meter.denominator,
            "source": score.meter_source or ("inferred" if score.meter.inferred else "printed"),
        },
        "key": {"fifths": score.key.fifths, "tonic": score.key.tonic, "mode": score.key.mode, "inferred": score.key.inferred},
        "measures": measures,
        "confidence": round(sum(e.confidence for e in events) / len(events), 3) if events else 0.0,
        "needs_review": any(d.severity != "info" for d in score.diagnostics),
        "diagnostics": [{"code": d.code, "severity": d.severity, "measure": d.measure}
                        for d in score.diagnostics if d.severity != "info"],
    }


def export_melody(score: Score, title: str | None = None) -> str:
    return json.dumps(melody_dict(score, title), indent=2, ensure_ascii=False) + "\n"
