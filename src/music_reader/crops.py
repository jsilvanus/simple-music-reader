"""Cut the notation of each system out of the page as an image (goal: reuse the original engraving).

Independent of recognition success: it only needs the staff/system geometry. Crops can be taken at a
higher resolution than recognition used, by re-rendering the (vector) PDF page.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from .backends.base import RecognitionResult
from .cv.staff import rotate
from .pdfinput import Document


def write_crops(result: RecognitionResult, doc: Document, directory: str | Path, dpi: float | None = None,
                padding_px: int = 0) -> list[dict]:
    """Write ``page-P-system-S.png`` for every detected system plus ``crops.json``; return the manifest."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dbg in result.pages:
        scale = 1.0
        img = dbg.image
        if dpi and doc.path.suffix.lower() == ".pdf" and abs(dpi - dbg.dpi) > 1:
            img, _ = doc.render(dbg.page - 1, dpi)
            if abs(dbg.skew_degrees) >= 0.15:
                img = rotate(img, dbg.skew_degrees, border=255)
            scale = dpi / dbg.dpi
        for si, (x0, y0, x1, y1) in enumerate(dbg.system_boxes):
            bx0, by0 = max(0, int(x0 * scale) - padding_px), max(0, int(y0 * scale) - padding_px)
            bx1, by1 = min(img.shape[1], int(x1 * scale) + padding_px), min(img.shape[0], int(y1 * scale) + padding_px)
            name = f"page-{dbg.page}-system-{si + 1}.png"
            cv2.imwrite(str(d / name), img[by0:by1, bx0:bx1])
            manifest.append({
                "file": name, "page": dbg.page, "system": si + 1, "bbox_px": [bx0, by0, bx1, by1],
                "bbox_norm": [bx0 / img.shape[1], by0 / img.shape[0], bx1 / img.shape[1], by1 / img.shape[0]],
                "dpi": dpi or dbg.dpi, "page_size_px": [img.shape[1], img.shape[0]],
            })
    (d / "crops.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
