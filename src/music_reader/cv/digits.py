"""Read the digits of a time signature from a binary crop.

No learned model: the glyph is compared with digits rendered at run time from Pillow's bundled scalable font and from
OpenCV's Hershey fonts, with the stroke width of the template matched to the glyph's own. A hole count (a 6 has one, an
8 two, a 7 none) corrects the shapes that stroke-weight differences confuse. Church scores use many engraving fonts, so
this is tolerant by design and reports a score; the caller decides whether to trust it. Verified on generated scores
only (see OMR.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SQUEEZES = (85, 100, 115)  # template widths (percent): fonts differ in how wide their digits are
BLUR = 0.9
CANVAS = 32  # normalised digit is fitted into a CANVAS x CANVAS box
_HOLES = {"0": (1,), "1": (0,), "2": (0,), "3": (0,), "4": (0, 1), "5": (0,), "6": (1,), "7": (0,), "8": (2,), "9": (1,)}
_HERSHEY = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX)
VALID_DENOMINATORS = (1, 2, 4, 8, 16)
VALID_NUMERATORS = (2, 3, 4, 5, 6, 7, 8, 9, 12, 15)  # no 1: a lone stem would read as one


@dataclass
class Reading:
    numerator: int
    denominator: int
    score: float  # mean digit similarity, 0..1
    margin: float  # lead over the best other valid reading, 0..1

    @property
    def text(self) -> str:
        return f"{self.numerator}/{self.denominator}"


def stroke_ratio(mask: np.ndarray) -> float:
    """Stroke width relative to the glyph height, from area and contour length (area ~ width * length, perimeter ~ 2 * length)."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cnts)
    if perim <= 0:
        return 0.15
    return float(np.clip(2.0 * m.sum() / perim / max(1, m.shape[0]), 0.06, 0.45))


def hole_count(mask: np.ndarray, min_area: float) -> int:
    m = np.pad(mask.astype(np.uint8) * 255, 2)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return sum(1 for i, c in enumerate(cnts) if hier[0][i][3] != -1 and cv2.contourArea(c) >= min_area)


def _normalise(mask: np.ndarray, bands: tuple[tuple[int, int], ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Fit the glyph into the canvas keeping its aspect ratio and blur it. Also returns the weight of each canvas pixel:
    0 on rows where ``bands`` (row ranges of the crop that a removed staff line made unreliable) fall, else 1."""
    weight = np.ones((CANVAS, CANVAS), np.float32)
    h, w = mask.shape
    if h == 0 or w == 0 or not mask.any():
        return np.zeros((CANVAS, CANVAS), np.float32), weight
    s = (CANVAS - 4) / max(h, w)
    nh, nw = max(3, int(round(h * s))), max(3, int(round(w * s)))
    g = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((CANVAS, CANVAS), np.float32)
    y0, x0 = (CANVAS - nh) // 2, (CANVAS - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = g
    for b0, b1 in bands:
        r0, r1 = y0 + int(np.floor(b0 * nh / h)), y0 + int(np.ceil((b1 + 1) * nh / h))
        weight[max(0, r0) : max(0, r1)] = 0
    return cv2.GaussianBlur(out, (0, 0), BLUR), weight


def _correlation(f: np.ndarray, t: np.ndarray, weight: np.ndarray) -> float:
    """Normalised correlation over the pixels that have weight."""
    sw = weight.sum()
    if sw < 20:
        return 0.0
    fm, tm = (weight * f).sum() / sw, (weight * t).sum() / sw
    fc, tc = f - fm, t - tm
    den = np.sqrt((weight * fc * fc).sum() * (weight * tc * tc).sum())
    return float((weight * fc * tc).sum() / den) if den > 0 else 0.0


@lru_cache(maxsize=4096)
def _template(digit: str, font: int, ratio_bucket: int) -> np.ndarray:
    return _template_variant(digit, font, ratio_bucket, 100)


@lru_cache(maxsize=16384)
def _template_variant(digit: str, font: int, ratio_bucket: int, squeeze: int) -> np.ndarray:
    ratio = ratio_bucket / 100.0
    size = 120
    img = np.zeros((size * 2, size * 2), np.uint8)
    if font < 0:
        f = ImageFont.load_default(size)
        pil = Image.new("L", img.shape[::-1], 0)
        ImageDraw.Draw(pil).text((size // 2, size // 4), digit, fill=255, font=f, stroke_width=0)
        img = np.array(pil)
        # thicken (or thin) towards the target stroke width
        ys, xs = np.nonzero(img > 127)
        if len(ys):
            h = ys.max() - ys.min() + 1
            have = stroke_ratio((img > 127)[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
            delta = int(round((ratio - have) * h / 2))
            if delta > 0:
                img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * delta + 1,) * 2))
            elif delta < 0:
                img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * -delta + 1,) * 2))
    else:
        scale = 3.2
        cv2.putText(img, digit, (size // 2, size + size // 2), _HERSHEY[font], scale, 255, max(1, int(round(ratio * 38 * scale / 2))), cv2.LINE_AA)
    ink = img > 127
    if squeeze != 100:
        ys, xs = np.nonzero(ink)
        crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255
        crop = cv2.resize(crop, (max(3, int(crop.shape[1] * squeeze / 100)), crop.shape[0]), interpolation=cv2.INTER_AREA)
        ink = crop > 127
    return _normalise(ink)[0]


def classify_digit(mask: np.ndarray, bands: tuple[tuple[int, int], ...] = ()) -> list[tuple[str, float]]:
    """All digits ranked by similarity (best first); scores are correlations in [-1, 1] after the hole-count adjustment.

    ``bands`` are row ranges of ``mask`` (a crop of the glyph) that are ignored: a staff line crossing a digit takes its
    flat strokes with it when the line is removed."""
    if mask.sum() == 0:
        return []
    ratio_bucket = int(round(stroke_ratio(mask) * 100 / 3) * 3)
    feat, weight = _normalise(mask, bands)
    holes = hole_count(mask, min_area=max(2.0, 0.004 * mask.shape[0] * mask.shape[0]))
    ranked = []
    for d in "0123456789":
        best = max(_correlation(feat, _template_variant(d, f, ratio_bucket, q), weight)
                   for f in (-1, *range(len(_HERSHEY))) for q in SQUEEZES)
        if holes > max(_HOLES[d]):
            best -= 0.3  # a ring that is there cannot be a mistake...
        elif holes < min(_HOLES[d]):
            best -= 0.3 if not bands else 0.12  # ...but a removed staff line may have cut one open
        ranked.append((d, best))
    ranked.sort(key=lambda t: -t[1])
    return ranked


def split_digits(mask: np.ndarray) -> list[np.ndarray]:
    """Split a row of one or two touching digits at the emptiest column near the middle (a 1 is narrow)."""
    h, w = mask.shape
    if w <= 0.85 * h:  # one digit is about 0.5-0.8 of its height wide
        return [mask]
    col = mask.sum(axis=0).astype(np.float32)
    lo, hi = int(0.25 * w), int(0.75 * w) + 1
    cut = lo + int(np.argmin(col[lo:hi]))
    return [mask[:, :cut], mask[:, cut:]]


def _fit_rows(mask: np.ndarray, bands: tuple[tuple[int, int], ...]):
    """Crop to the ink; where an edge of the ink lies in or just beside a band, move it out to the band's edge (the line
    took the flat stroke there). Returns the crop and the bands relative to it."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask, ()
    ya, yb = int(ys.min()), int(ys.max())
    for b0, b1 in bands:
        if b0 < ya <= b1 + 2:
            ya = b0
        if b0 - 2 <= yb < b1:
            yb = b1
    ya, yb = max(0, ya), min(mask.shape[0] - 1, yb)
    rel = tuple((max(0, b0 - ya), min(yb - ya, b1 - ya)) for b0, b1 in bands if b1 >= ya and b0 <= yb)
    return mask[ya : yb + 1, xs.min() : xs.max() + 1], rel


def _candidates(row: np.ndarray, valid: tuple[int, ...], bands, n_best: int = 8) -> list[tuple[int, float]]:
    """Numbers a row of digits can be, with the mean digit score."""
    crop, rel = _fit_rows(row, bands)
    out_parts = []
    for part in split_digits(crop):
        pc, prel = _fit_rows(part, rel)
        out_parts.append(classify_digit(pc, prel))
    if any(not r for r in out_parts):
        return []
    out = []
    for combo in product(*[r[:n_best] for r in out_parts]):
        text = "".join(d for d, _ in combo)
        if text.startswith("0"):
            continue
        n = int(text)
        if n in valid:
            out.append((n, float(np.mean([s for _, s in combo]))))
    return out


def read_time_signature(num_mask: np.ndarray, den_mask: np.ndarray, num_bands=(), den_bands=()) -> Reading | None:
    """Numerator and denominator of a stacked time signature from the ink above and below the middle line.

    ``*_bands``: row ranges of each mask where a removed staff line makes the ink unreliable.
    Returns None when no valid meter is a convincing match."""
    nums = _candidates(num_mask, VALID_NUMERATORS, tuple(num_bands))
    dens = _candidates(den_mask, VALID_DENOMINATORS, tuple(den_bands))
    if not nums or not dens:
        return None
    combos = sorted(((n, d, (sn + sd) / 2) for n, sn in nums for d, sd in dens), key=lambda t: -t[2])
    n, d, score = combos[0]
    other = next((c[2] for c in combos[1:] if (c[0], c[1]) != (n, d)), -1.0)
    return Reading(n, d, score, max(0.0, score - other))
