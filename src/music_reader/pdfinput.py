"""PDF (and plain image) input: classify pages and render them to grayscale arrays.

Uses pypdfium2 (BSD-3/Apache-2.0 bindings to PDFium). A page is:

* ``vector``: notation is drawn with path/text objects (engraved by Sibelius, MuseScore, ...);
* ``raster``: a scan, i.e. one large image covers the page;
* ``image``: the input was an image file, not a PDF.

For vector pages we also report embedded music fonts (Bravura, Emmentaler, Opus, ...). Today
both kinds go through the same render-then-CV path; the classification is kept so a future
backend can read staff lines/glyphs directly from the PDF objects (see OMR.md).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

MUSIC_FONT_HINTS = (
    "bravura", "emmentaler", "opus", "maestro", "petrucci", "leland", "finale", "sonata",
    "musescore", "gonville", "lilyglyphs", "profondo", "sebastian", "november", "musejazz",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class PageAnalysis:
    number: int  # 1-based
    kind: str
    width_pt: float
    height_pt: float
    n_images: int = 0
    n_paths: int = 0
    n_text: int = 0
    image_coverage: float = 0.0
    fonts: list[str] = field(default_factory=list)
    music_fonts: list[str] = field(default_factory=list)


class Document:
    """Uniform page source over a PDF or an image file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._pdf = None
        self._image = None
        if self.path.suffix.lower() in IMAGE_SUFFIXES:
            img = cv2.imread(str(self.path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"cannot read image {self.path}")
            self._image = img
        else:
            self._pdf = pdfium.PdfDocument(str(self.path))

    def __len__(self) -> int:
        return 1 if self._image is not None else len(self._pdf)

    def analyze(self, index: int) -> PageAnalysis:
        if self._image is not None:
            h, w = self._image.shape
            return PageAnalysis(1, "image", w, h)
        return _analyze_pdf_page(self._pdf[index], index + 1)

    def render(self, index: int, dpi: float) -> tuple[np.ndarray, float]:
        """Return (grayscale uint8 image, effective dpi)."""
        if self._image is not None:
            return self._image, dpi
        page = self._pdf[index]
        bitmap = page.render(scale=dpi / 72.0, grayscale=True)
        arr = bitmap.to_numpy()
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return np.ascontiguousarray(arr), dpi


def _analyze_pdf_page(page, number: int) -> PageAnalysis:
    w, h = page.get_size()
    a = PageAnalysis(number, "vector", w, h)
    fonts: set[str] = set()
    covered = 0.0
    for obj in page.get_objects():
        t = obj.type
        if t == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            a.n_images += 1
            l, b, r, tp = obj.get_bounds()
            covered = max(covered, max(0.0, (r - l) * (tp - b)) / (w * h))
        elif t == pdfium_c.FPDF_PAGEOBJ_PATH:
            a.n_paths += 1
        elif t == pdfium_c.FPDF_PAGEOBJ_TEXT:
            a.n_text += 1
            name = _font_name(obj)
            if name:
                fonts.add(name)
    a.image_coverage = covered
    a.fonts = sorted(fonts)
    a.music_fonts = [f for f in a.fonts if any(k in f.lower() for k in MUSIC_FONT_HINTS)]
    if covered >= 0.6 and a.n_paths < 50:
        a.kind = "raster"
    return a


def _font_name(text_obj) -> str:
    try:
        font = pdfium_c.FPDFTextObj_GetFont(text_obj.raw)
        n = pdfium_c.FPDFFont_GetBaseFontName(font, None, 0)
        if n <= 1:
            return ""
        buf = ctypes.create_string_buffer(n)
        pdfium_c.FPDFFont_GetBaseFontName(font, buf, n)
        return buf.value.decode("latin-1")
    except Exception:  # font introspection is best effort
        return ""
