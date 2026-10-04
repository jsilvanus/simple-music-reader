"""Page normalisation and staff geometry.

Everything downstream is expressed in units of the staff spacing (distance between two
adjacent staff lines), so the pipeline does not depend on render resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Staff:
    lines: list[float]  # y of the five line centres, top to bottom
    x0: int
    x1: int
    spacing: float
    thickness: float
    system: int = 0
    index: int = 0  # position inside its system, top to bottom
    uid: int = 0  # position on the page, top to bottom

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def bottom(self) -> float:
        return self.lines[4]

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def position(self, y: float) -> float:
        """Vertical position in half-spaces above the bottom line (0 = bottom line, 8 = top line)."""
        return (self.bottom - y) / (self.spacing / 2.0)

    def y_of(self, position: int) -> float:
        return self.bottom - position * self.spacing / 2.0


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink = 255, paper = 0."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def estimate_skew(bw: np.ndarray, max_deg: float = 3.0, step: float = 0.1) -> float:
    """Angle (degrees, counter-clockwise positive) to rotate the page by to level the staff lines."""
    scale = 1200.0 / max(bw.shape)
    small = cv2.resize(bw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bw
    h, w = small.shape
    centre = (w / 2, h / 2)
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + step / 2, step):
        m = cv2.getRotationMatrix2D(centre, float(a), 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_LINEAR)
        score = float(np.var(rot.sum(axis=1, dtype=np.float64)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def rotate(img: np.ndarray, degrees: float, border: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def detect_staves(bw: np.ndarray) -> tuple[list[Staff], np.ndarray]:
    """Find five-line staves. Returns (staves sorted top to bottom, staff-line mask)."""
    h, w = bw.shape
    klen = max(30, int(0.12 * w))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    line_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    rows = line_mask.sum(axis=1, dtype=np.float64) / 255.0
    if rows.max() <= 0:
        return [], line_mask
    thresh = 0.35 * rows.max()
    centres, thicks = [], []
    y = 0
    while y < h:
        if rows[y] >= thresh:
            y0 = y
            while y < h and rows[y] >= thresh:
                y += 1
            seg = rows[y0:y]
            centres.append(float((np.arange(y0, y) * seg).sum() / seg.sum()))
            thicks.append(y - y0)
        else:
            y += 1
    staves: list[Staff] = []
    i = 0
    while i + 4 < len(centres):
        c = centres[i : i + 5]
        d = np.diff(c)
        if d.min() >= 4 and d.max() <= 1.3 * d.min() + 2 and abs(d.max() - d.min()) <= 0.25 * d.mean():
            sp = float(d.mean())
            mid = int(round(c[2]))
            band = line_mask[max(0, mid - 1) : mid + 2].any(axis=0)
            xs = np.nonzero(band)[0]
            staves.append(Staff(c, int(xs.min()), int(xs.max()), sp, float(np.mean(thicks[i : i + 5]))))
            i += 5
        else:
            i += 1
    for k, s in enumerate(staves):
        s.index = s.uid = k
    return staves, line_mask


def group_systems(staves: list[Staff], bw: np.ndarray) -> list[list[Staff]]:
    """Group staves into systems: staves joined by a vertical line at the left edge."""
    systems: list[list[Staff]] = []
    for s in staves:
        if systems and _joined(systems[-1][-1], s, bw):
            systems[-1].append(s)
        else:
            systems.append([s])
    for si, group in enumerate(systems):
        for k, s in enumerate(group):
            s.system, s.index = si, k
    return systems


def _joined(a: Staff, b: Staff, bw: np.ndarray) -> bool:
    sp = a.spacing
    x0 = int(min(a.x0, b.x0))
    xa, xb = max(0, int(x0 - 0.6 * sp)), int(x0 + 0.5 * sp) + 1
    ya, yb = int(a.bottom) + 1, int(b.top)
    if yb - ya < 2:
        return False
    gap = bw[ya:yb, xa:xb].any(axis=1)
    return bool(gap.mean() >= 0.95)


def remove_staff_lines(bw: np.ndarray, staves: list[Staff], line_mask: np.ndarray) -> np.ndarray:
    """Erase staff lines, but give back pixels where a symbol crosses the line (ink above and below)."""
    res = bw.copy()
    for s in staves:
        for yc in s.lines:
            a, b = int(np.floor(yc - s.thickness / 2 - 0.5)), int(np.ceil(yc + s.thickness / 2 + 0.5))
            res = _erase_strip(res, bw, a, b, s.x0 - 2, s.x1 + 3)
    return res


def _erase_strip(res: np.ndarray, orig: np.ndarray, a: int, b: int, xa: int, xb: int, tol: int = 1) -> np.ndarray:
    """Erase rows a..b between columns xa..xb, except in columns where ink continues beyond the strip.

    A column that is only a line is exactly strip-high, so it is erased. If more than ``tol`` extra ink
    pixels touch the strip (a stem crossing it, the cap of a hollow head, an arc lying on the line) the
    column belongs to a symbol and is kept.
    """
    h, w = res.shape
    a, b = max(1, a), min(h - 2, b)
    xa, xb = max(0, xa), min(w, xb)
    ink = orig[:, xa:xb] > 0
    n = xb - xa
    extra = np.zeros(n, dtype=np.int32)
    for rows in (range(a - 1, max(-1, a - 2 - tol), -1), range(b + 1, min(h, b + 2 + tol))):
        alive = np.ones(n, dtype=bool)
        for y in rows:
            alive &= ink[y]
            extra += alive
    erase = extra <= tol
    res[a : b + 1, xa:xb][:, erase] = 0
    return res


@dataclass
class Ledger:
    x0: int
    x1: int
    y: float
    staff: int


def remove_ledger_lines(res: np.ndarray, staves: list[Staff]) -> tuple[np.ndarray, list[Ledger]]:
    """Detect short thin horizontal strokes above/below each staff (ledger lines) and erase them."""
    ledgers: list[Ledger] = []
    orig = res.copy()
    for s in staves:
        sp = s.spacing
        ya, yb = int(max(0, s.top - 5 * sp)), int(min(res.shape[0] - 1, s.bottom + 5 * sp))
        xa, xb = max(0, s.x0), min(res.shape[1], s.x1 + 1)
        roi = res[ya:yb, xa:xb].copy()
        # blank the staff itself: only look outside the five lines
        roi[int(s.top - ya - s.thickness) : int(s.bottom - ya + s.thickness) + 1, :] = 0
        klen = max(5, int(1.45 * sp))
        opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        for i in range(1, n):
            x, y, w, h, _a = stats[i]
            if h <= max(2.0, 0.5 * sp) and w <= 3.2 * sp:
                yc = ya + y + h / 2.0
                # must sit on the staff's half-space grid (even position) within tolerance
                pos = s.position(yc)
                if abs(pos - round(pos)) <= 0.45 and int(round(pos)) % 2 == 0:
                    ledgers.append(Ledger(xa + x, xa + x + w, yc, s.uid))
                    res = _erase_strip(res, orig, ya + y - 1, ya + y + h, xa + x - 1, xa + x + w + 1)
    return res, ledgers
