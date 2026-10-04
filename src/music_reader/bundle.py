"""One-call output for downstream consumers (e.g. anno-api): every format plus system images and a manifest.

    <dir>/melody.abc  melody.musicxml  score.json  crops/page-P-system-S.png  crops/crops.json  manifest.json

The images reproduce the notation as printed (goal: show the original engraving); the ABC/MusicXML/JSON
carry the recognised melody (goal: re-render it in any layout). The two are linked by page/system numbers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import __version__
from .backends.base import RecognitionResult
from .crops import write_crops
from .export import export
from .pdfinput import Document


def write_bundle(result: RecognitionResult, doc: Document, directory: str | Path, title: str | None = None,
                 crop_dpi: float | None = 300) -> dict:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for fmt, name in (("abc", "melody.abc"), ("musicxml", "melody.musicxml"), ("json", "score.json")):
        (d / name).write_text(export(result.score, fmt, title=title), encoding="utf-8")
    crops = write_crops(result, doc, d / "crops", dpi=crop_dpi)
    score = result.score
    manifest = {
        "generator": f"music-reader {__version__}",
        "source": {"file": doc.path.name, "sha256": hashlib.sha256(doc.path.read_bytes()).hexdigest()},
        "title": title or score.metadata.get("title"),
        "pages": [{"page": p.number, "kind": p.kind, "systems": len(p.systems), "music_fonts": p.music_fonts} for p in score.pages],
        "melody": {
            "measures": len(score.melody.measures) if score.melody else 0,
            "meter": str(score.meter) if score.meter else None,
            "meter_inferred": bool(score.meter and score.meter.inferred),
            "key_fifths": score.key.fifths,
            "selection": score.selection,
        },
        "crops": crops,
        "diagnostics": [{"code": x.code, "severity": x.severity, "message": x.message} for x in score.diagnostics],
        "needs_review": any(x.severity != "info" for x in score.diagnostics),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
