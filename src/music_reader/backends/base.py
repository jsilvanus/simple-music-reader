"""Recognition backend interface. Every backend turns a Document into a Score.

To add a backend (e.g. one that runs Audiveris or homr as an external process and parses its
MusicXML into the Music IR): implement ``recognize`` and register it in ``backends/__init__.py``.
Backends must not emit ABC; output formats live in ``export/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from ..ir import Score
from ..pdfinput import Document


@dataclass
class RecognitionOptions:
    staff: Optional[int] = None  # 1-based staff inside each system to read as the melody
    meter: Optional[str] = None  # e.g. "3/4"; overrides inference
    key: Optional[str] = None  # e.g. "G", "Em"; overrides detection
    title: Optional[str] = None
    dpi: Optional[float] = None
    pages: Optional[list[int]] = None  # 1-based page numbers; default all


@dataclass
class PageDebug:
    """Everything needed to draw an annotated page and to crop systems."""

    page: int
    image: np.ndarray  # deskewed grayscale working image
    staves: list  # cv.staff.Staff
    marks: list[dict] = field(default_factory=list)  # {kind, bbox, label, conf}
    system_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # crop box per system
    skew_degrees: float = 0.0
    dpi: float = 0.0


@dataclass
class RecognitionResult:
    score: Score
    pages: list[PageDebug] = field(default_factory=list)


class Backend(Protocol):
    name: str

    def recognize(self, doc: Document, options: RecognitionOptions) -> RecognitionResult: ...
