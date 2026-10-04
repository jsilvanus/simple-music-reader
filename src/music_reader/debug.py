"""Annotated debug images: staff lines, systems, symbols, pitch/duration labels and confidence."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .backends.base import RecognitionResult
from .ir import Score

# BGR
_COLORS = {
    "note": (0, 160, 0), "rest": (200, 0, 200), "bar": (0, 128, 255), "clef": (200, 120, 0), "key": (200, 120, 0),
    "time": (200, 120, 0), "accidental": (0, 160, 160), "dot": (0, 160, 160), "tie": (160, 0, 160),
    "slur": (160, 100, 160), "unknown": (0, 0, 255), "rejected": (150, 150, 150),
}


def _color(kind: str, conf: float):
    base = _COLORS.get(kind.removeprefix("other-"), (90, 90, 90))
    if kind.startswith("other-"):
        return (170, 170, 170)
    if conf < 0.6 and kind in ("note", "rest"):
        return (0, 0, 255)  # low confidence: red
    return base


def annotate(result: RecognitionResult) -> list[np.ndarray]:
    """One BGR image per page. Note labels are filled in from the score's events."""
    labels = _event_labels(result.score)
    out = []
    for dbg in result.pages:
        img = cv2.cvtColor(dbg.image, cv2.COLOR_GRAY2BGR)
        for (x0, y0, x1, y1) in dbg.system_boxes:
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 160, 60), 1)
        for s in dbg.staves:
            for y in s.lines:
                cv2.line(img, (s.x0, int(round(y))), (s.x1, int(round(y))), (255, 0, 0), 1)
        for m in dbg.marks:
            x0, y0, x1, y1 = m["bbox"]
            col = _color(m["kind"], m["conf"])
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
            label = labels.get((dbg.page, m["bbox"]), m["label"])
            if label and m["kind"] in ("note", "rest", "clef", "time", "unknown", "rejected"):
                text = f"{label} {m['conf']:.2f}" if m["kind"] in ("note", "rest") else label
                cv2.putText(img, text, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        out.append(img)
    return out


def _event_labels(score: Score) -> dict:
    labels = {}
    voice = score.melody
    if voice is None:
        return labels
    for e in voice.events():
        if e.source and e.source.bbox:
            name = e.pitch.name if e.kind == "note" else "rest"
            labels[(e.source.page, tuple(e.source.bbox))] = f"{name} {e.duration}"
    return labels


def write_debug(result: RecognitionResult, directory: str | Path) -> list[Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths, pil = [], []
    for dbg, img in zip(result.pages, annotate(result)):
        p = d / f"page-{dbg.page}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
        pil.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if pil:
        pdf = d / "debug.pdf"
        pil[0].save(str(pdf), save_all=True, append_images=pil[1:])
        paths.append(pdf)
    (d / "diagnostics.txt").write_text("\n".join(str(x) for x in result.score.diagnostics) + "\n", encoding="utf-8")
    return paths
