"""Symbol recognition on a staff-line-free image, in units of the staff spacing."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import cv2
import numpy as np

from ..ir import Diagnostic, Source
from . import digits as cvdigits
from .staff import Ledger, Staff

BASE_TREBLE = 4 * 7 + 2  # E4: bottom line of a treble staff, as a diatonic number
BASE_BASS = 2 * 7 + 4  # G2


@dataclass
class Comp:
    id: int
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int
    area: int
    mask: np.ndarray  # bool crop

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)


def extract_components(res: np.ndarray, sp: float) -> list[Comp]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(res, connectivity=8)
    min_area = max(4, int(0.08 * sp * sp))
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < min_area:
            continue
        comps.append(Comp(i, x, y, x + w, y + h, a, lab[y : y + h, x : x + w] == i))
    comps.sort(key=lambda c: c.x0)
    return comps


@dataclass
class Head:
    cx: float  # page coordinates
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float
    ink: float  # fraction of the head blob that is ink: ~1 filled, ~0.6 hollow
    hollow: bool = False


def find_heads(c: Comp, sp: float) -> list[Head]:
    if c.area / (c.w * c.h) >= 0.93 and c.w / c.h >= 1.5 and c.h <= 0.9 * sp:
        return []  # a solid bar: half or whole rest
    pad = 4
    m = np.pad(c.mask.astype(np.uint8) * 255, pad)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = m.copy()
    if hier is not None:
        for i, cnt in enumerate(cnts):
            if hier[0][i][3] != -1 and cv2.contourArea(cnt) <= 0.8 * sp * sp:
                cv2.drawContours(filled, [cnt], -1, 255, -1)
    d = max(3, int(round(0.62 * sp)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(opened, connectivity=8)
    heads = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if 1.7 * sp < h <= 2.7 * sp and 0.7 * sp <= w <= 1.7 * sp and 1.0 * sp * sp <= a <= 4.2 * sp * sp:
            # two heads one above the other (a third in two voices on one stem column): split the blob in two
            hh = 0.92 * sp
            for top in (True, False):
                ya, yb = (y, int(y + hh)) if top else (int(y + h - hh), y + h)
                box = m[ya:yb, x : x + w] > 0
                inside = np.zeros(box.shape, np.uint8)  # ink is judged inside the ellipse of the head, not in the box corners
                cv2.ellipse(inside, (box.shape[1] // 2, box.shape[0] // 2), (max(1, box.shape[1] // 2 - 1), max(1, box.shape[0] // 2 - 1)), 0, 0, 360, 1, -1)
                ink = float(box[inside > 0].mean()) if inside.any() else 0.0
                heads.append(Head(c.x0 + x + w / 2 - pad, c.y0 + (ya + yb) / 2 - pad, c.x0 + x - pad, c.y0 + ya - pad,
                                  c.x0 + x + w - pad, c.y0 + yb - pad, ink, ink < 0.8))
            continue
        if not (0.45 * sp * sp <= a <= 2.4 * sp * sp and 0.70 * sp <= w <= 2.0 * sp and 0.70 * sp <= h <= 1.7 * sp):
            continue
        region = (lab == i).astype(np.uint8)
        if not _head_like_orientation(region):
            continue  # e.g. a steep flag stroke
        ink = float((m[lab == i] > 0).mean())
        heads.append(Head(c.x0 + cents[i][0] - pad, c.y0 + cents[i][1] - pad, c.x0 + x - pad, c.y0 + y - pad,
                          c.x0 + x + w - pad, c.y0 + y + h - pad, ink, ink < 0.8))
    return heads


def _head_like_orientation(region: np.ndarray) -> bool:
    """Note heads are round or elongated along (roughly) the horizontal; steep elongated blobs are not heads."""
    mo = cv2.moments(region, binaryImage=True)
    if mo["m00"] == 0:
        return False
    mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
    common = np.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4 * mu11**2))
    l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
    if l2 <= 0 or np.sqrt(l1 / l2) <= 1.4:
        return True
    theta = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    return abs(theta) <= 55.0


def _solidity(region: np.ndarray) -> float:
    cnts, _ = cv2.findContours(np.pad(region.astype(np.uint8), 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    return float(cv2.contourArea(cnt) / hull) if hull > 0 else 0.0


def find_holes(bw: np.ndarray, sp: float) -> list[Head]:
    """Hollow note heads found as small enclosed white regions of the page *with* staff lines in place.

    A hollow head sitting between two lines is tangent to both, so removing the lines breaks its ring;
    the white inside survives, which is why heads are also searched for this way. For a hollow head
    sitting ON a line, the line cuts its hole in two; the two halves are merged across the line.
    """
    n, lab, stats, cents = cv2.connectedComponentsWithStats(255 - bw, connectivity=4)
    out = []
    small_split = []
    pad_x, pad_y = 0.25 * sp, 0.2 * sp
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if (0.2 * sp * sp <= a <= 0.9 * sp * sp and 0.5 * sp <= w <= 1.3 * sp and 0.25 * sp <= h <= 0.8 * sp):
            if 0.48 <= a / (w * h) <= 0.9 and _solidity(lab[y : y + h, x : x + w] == i) >= 0.85:
                out.append(Head(float(cents[i][0]), float(cents[i][1]), x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y, 0.6, True))
                continue
        # Split half-hole cut by a staff line
        if 0.015 * sp * sp <= a <= 0.5 * sp * sp and 0.15 * sp <= w <= 1.3 * sp and h <= 0.6 * sp:
            small_split.append((x, y, w, h, a, float(cents[i][0]), float(cents[i][1])))

    # Pair up split half-holes across staff lines
    used = set()
    for j, h1 in enumerate(small_split):
        if j in used:
            continue
        for k in range(j + 1, len(small_split)):
            if k in used:
                continue
            h2 = small_split[k]
            if abs(h1[5] - h2[5]) > 0.6 * sp:
                continue
            top_h, bot_h = (h1, h2) if h1[1] < h2[1] else (h2, h1)
            gap = bot_h[1] - (top_h[1] + top_h[3])
            if not (0 <= gap <= 0.4 * sp):
                continue
            tot_x0 = min(top_h[0], bot_h[0])
            tot_x1 = max(top_h[0] + top_h[2], bot_h[0] + bot_h[2])
            tot_y0 = top_h[1]
            tot_y1 = bot_h[1] + bot_h[3]
            tot_w = tot_x1 - tot_x0
            tot_h = tot_y1 - tot_y0
            tot_a = top_h[4] + bot_h[4] + gap * min(top_h[2], bot_h[2])
            if (0.2 * sp * sp <= tot_a <= 0.9 * sp * sp and 0.5 * sp <= tot_w <= 1.3 * sp and 0.35 * sp <= tot_h <= 1.0 * sp):
                used.add(j)
                used.add(k)
                cx = (top_h[5] + bot_h[5]) / 2
                cy = (top_h[6] + bot_h[6]) / 2
                out.append(Head(cx, cy, tot_x0 - pad_x, tot_y0 - pad_y, tot_x1 + pad_x, tot_y1 + pad_y, 0.6, True))
                break
    return out


def _on_beam(beams: np.ndarray, h: "Head") -> bool:
    y0, y1 = max(0, int(h.y0 - 2)), min(beams.shape[0], int(h.y1 + 3))
    x0, x1 = max(0, int(h.x0 - 2)), min(beams.shape[1], int(h.x1 + 3))
    return bool((beams[y0:y1, x0:x1] > 0).any())


def find_beams(res: np.ndarray, sp: float) -> np.ndarray:
    """Mask of long, thick strokes of a staff-line-free image: beams, flat or slanted.

    A beam survives an erosion that takes away stems, slurs and flags, is long, and is thin across (a note head is not)."""
    d = max(2, int(round(0.3 * sp)))
    core = cv2.erode(res, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    out = np.zeros_like(res)
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if np.hypot(w, h) < 2.2 * sp:
            continue
        ys, xs = np.nonzero(lab[y : y + h, x : x + w] == i)
        cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else None
        if cov is None:
            continue
        ev, vec = np.linalg.eigh(cov)
        if ev[0] <= 0 or ev[1] / ev[0] < 16.0:  # length at least 4x the thickness
            continue
        if abs(vec[1, 1]) > abs(vec[0, 1]) * 0.85:  # steeper than about 40 degrees: a stem with its head, not a beam
            continue
        out[y : y + h, x : x + w][lab[y : y + h, x : x + w] == i] = 255
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d + 2, d + 2)))


def longest_run(col: np.ndarray) -> int:
    best = cur = 0
    for v in col:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def strokes(c: Comp, frac: float = 0.55):
    """Vertical strokes: groups of columns whose longest ink run is >= frac of the height."""
    runs = np.array([longest_run(c.mask[:, x]) for x in range(c.w)])
    on = runs >= frac * c.h
    out, x = [], 0
    while x < c.w:
        if on[x]:
            x0 = x
            while x < c.w and on[x]:
                x += 1
            out.append(((x0 + x - 1) / 2.0, int(runs[x0:x].max())))
        else:
            x += 1
    return out


def classify_accidental(c: Comp, sp: float) -> Optional[str]:
    # shorter than a note with stem, whose head+stem is taller (the bowl of a flat looks like a head)
    if not (1.6 * sp <= c.h <= 3.6 * sp and 0.5 * sp <= c.w <= 1.8 * sp):
        return None
    st = strokes(c)
    if len(st) >= 2 and all(r >= 0.8 * c.h for _, r in st[:2]):
        return "#"
    if len(st) >= 2 and all(0.45 * c.h <= r < 0.82 * c.h for _, r in st[:2]):
        return "n"
    if len(st) == 1 and st[0][1] >= 0.7 * c.h and st[0][0] <= 0.38 * c.w and c.w <= 1.5 * sp:
        top_right = c.mask[:max(1, int(0.5 * c.h)), max(1, int(0.35 * c.w)):]
        if not top_right.any() or int(top_right.sum()) <= 1:
            return "b"
    return None


def _merge_comps(c1: Comp, c2: Comp) -> Comp:
    x0, y0 = min(c1.x0, c2.x0), min(c1.y0, c2.y0)
    x1, y1 = max(c1.x1, c2.x1), max(c1.y1, c2.y1)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    mask[c1.y0 - y0 : c1.y1 - y0, c1.x0 - x0 : c1.x1 - x0] |= c1.mask
    mask[c2.y0 - y0 : c2.y1 - y0, c2.x0 - x0 : c2.x1 - x0] |= c2.mask
    return Comp(c1.id, x0, y0, x1, y1, int(mask.sum()), mask)


def _try_merge_accidental(c1: Comp, c2: Comp, sp: float) -> Optional[tuple[str, Comp]]:
    if not (0 <= c2.x0 - c1.x1 <= 0.6 * sp):
        return None
    if not (0.5 * sp <= (c2.x1 - c1.x0) <= 1.8 * sp):
        return None
    ov = min(c1.y1, c2.y1) - max(c1.y0, c2.y0)
    if ov < 0.4 * min(c1.h, c2.h):
        return None
    merged = _merge_comps(c1, c2)
    acc = classify_accidental(merged, sp)
    if acc in ("#", "b", "n"):
        return acc, merged
    return None


@dataclass
class Item:
    kind: str  # "note" | "rest" | "bar"
    x: float
    bbox: tuple
    conf: float
    comp: int
    pos: Optional[int] = None  # staff position in half-spaces above the bottom line
    duration: Optional[Fraction] = None
    dots: int = 0
    accidental: Optional[str] = None
    tie_start: bool = False
    hollow: bool = False
    stem: Optional[str] = None
    flags: int = 0
    notes: list = field(default_factory=list)
    graces: list = field(default_factory=list)  # grace-note Items leaning on this note
    tuplet: Optional[tuple] = None  # (actual, normal)


@dataclass
class StaffResult:
    staff: Staff
    clef: str = "treble"
    clef_confirmed: bool = False
    key_fifths: Optional[int] = None
    time_sig_seen: bool = False
    time_signature: Optional[str] = None
    time_signature_confidence: float = 0.0
    items: list[Item] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    marks: list[dict] = field(default_factory=list)  # for debug rendering


def _mark(res, kind, bbox, label="", conf=1.0):
    res.marks.append({"kind": kind, "bbox": tuple(int(v) for v in bbox), "label": label, "conf": float(conf)})


def analyze_staff(staff: Staff, comps: list[Comp], ledgers: list[Ledger], page: int, system: int,
                  holes: Optional[list[Head]] = None, beams: Optional[np.ndarray] = None,
                  voice: str = "top") -> StaffResult:
    sp = staff.spacing
    res = StaffResult(staff)
    src = lambda c=None: Source(page, c.bbox if c else None, system, staff.index, c.id if c else None)  # noqa: E731

    def diag(code, sev, msg, c=None):
        res.diagnostics.append(Diagnostic(code, sev, msg, src(c)))

    bars = [c for c in comps if _is_barline(c, staff)]
    rest = sorted((c for c in comps if c not in bars), key=lambda c: c.x0)

    # ---- header: clef, key signature, time signature -------------------------------
    header_end = float(staff.x0)
    clef = next((c for c in rest if c.x0 < staff.x0 + 8 * sp and c.h >= 2.2 * sp and 0.8 * sp <= c.w <= 3.5 * sp), None)
    if clef is None:
        diag("CLEF_MISSING", "warning", "no clef found at the start of the staff; treble clef assumed")
    else:
        header_end = clef.x1
        _mark(res, "clef", clef.bbox, f"clef h={clef.h / sp:.1f}sp")
        if clef.h >= 5.0 * sp:
            res.clef_confirmed = True
        elif 2.5 * sp <= clef.h < 4.5 * sp:
            res.clef = "bass"
            diag("CLEF_BASS_GUESSED", "warning", "clef looks like a bass clef; pitches are computed for a bass staff but this is a guess", clef)
        else:
            diag("CLEF_UNCERTAIN", "warning", "clef shape is ambiguous; treble clef assumed", clef)
    base = BASE_BASS if res.clef == "bass" else BASE_TREBLE

    tail = [c for c in rest if clef is None or c.x0 >= clef.x1 - 0.3 * sp]
    heads_cache: dict[int, list[Head]] = {}

    def heads_of(c):
        if c.id not in heads_cache:
            if classify_accidental(c, sp):
                heads_cache[c.id] = []
            else:
                found = find_heads(c, sp)
                if beams is not None:  # a blob where a stem meets a beam (and a staff line crosses it) is not a head
                    found = [h for h in found if not _on_beam(beams, h)]
                heads_cache[c.id] = found
        return heads_cache[c.id]

    def looks_like_time(c):
        return (3.0 * sp <= c.h <= 4.4 * sp and 0.8 * sp <= c.w <= 2.2 * sp and abs(c.cy - staff.lines[2]) <= 0.7 * sp)

    key_accs, last_x1, idx = [], header_end, 0
    while idx < len(tail):
        c = tail[idx]
        if c.cy < staff.top - 1.2 * sp or c.cy > staff.bottom + 1.2 * sp:
            idx += 1
            continue
        if c.x0 - last_x1 > 3.0 * sp:
            break
        if not key_accs and _read_time_block(tail, idx, last_x1, staff) is not None:
            break  # a narrow digit (1) looks like a flat; what follows reads as a time signature
        kind = classify_accidental(c, sp)
        advance = 1
        if kind is None and idx + 1 < len(tail):
            m = _try_merge_accidental(c, tail[idx + 1], sp)
            if m is not None:
                kind, c = m
                advance = 2
        if kind in ("#", "b"):
            nxt = tail[idx + advance] if idx + advance < len(tail) else None
            if nxt is not None and not looks_like_time(nxt) and heads_of(nxt) and nxt.x0 - c.x1 <= 1.5 * sp:
                break  # belongs to the first note, not to the key signature
            key_accs.append((kind, c))
            _mark(res, "key", c.bbox, kind)
            last_x1, idx = c.x1, idx + advance
            continue
        break
    if key_accs:
        kinds = {k for k, _ in key_accs}
        if len(kinds) == 1:
            n = len(key_accs)
            res.key_fifths = n if "#" in kinds else -n
        else:
            diag("KEY_MIXED", "warning", "key signature mixes sharps and flats; key not set")
    else:
        res.key_fifths = 0
    header_end = last_x1
    read = _read_time_block(tail, idx, header_end, staff) if idx < len(tail) else None
    if idx < len(tail):
        s0 = tail[idx]
        if read is not None:
            text, conf, label, bbox, header_end, used = read
            res.time_sig_seen, res.time_signature, res.time_signature_confidence = True, text, conf
            _mark(res, "time", bbox, label, conf)
            idx += used
        else:
            cluster = _time_cluster(tail, idx, header_end, staff)
            if cluster:
                res.time_sig_seen = True
                bx0, by0 = min(c.x0 for c in cluster), min(c.y0 for c in cluster)
                bx1, by1 = max(c.x1 for c in cluster), max(c.y1 for c in cluster)
                _mark(res, "time", (bx0, by0, bx1, by1), "time signature (digits not read)")
                header_end = bx1
                idx = tail.index(cluster[-1]) + 1
            elif s0.x0 - header_end <= 3.0 * sp and looks_like_time(s0):
                res.time_sig_seen = True  # two touching digits whose reading failed
                _mark(res, "time", s0.bbox, "time signature (digits not read)")
                header_end = s0.x1
                idx += 1
    if res.time_signature is None and idx + 1 < len(tail):
        a, b = tail[idx], tail[idx + 1]
        if (a.x0 - header_end <= 3.0 * sp and 1.2 * sp <= a.h <= 2.7 * sp and 1.2 * sp <= b.h <= 2.7 * sp
                and abs(a.cx - b.cx) <= 0.6 * sp and a.cy < staff.lines[2] < b.cy):
            res.time_sig_seen = True
            _mark(res, "time", (min(a.x0, b.x0), a.y0, max(a.x1, b.x1), b.y1), "time signature (digits not read)")
            header_end = max(a.x1, b.x1)
            idx += 2

    content = tail[idx:]

    # ---- content ------------------------------------------------------------------
    notes: list[Item] = []
    accs: list[tuple[str, Comp]] = []
    dots: list[Comp] = []
    ties: list[Comp] = []
    frags: list[Comp] = []
    # hollow heads found as enclosed white regions, attached to the component that holds their stem
    double_bars = [(min(b1.x0, b2.x0), max(b1.x1, b2.x1)) for i, b1 in enumerate(bars) for b2 in bars[i + 1:] if abs(b1.cx - b2.cx) <= 1.2 * sp]
    my_holes = [h for h in (holes or []) if staff.top - 5 * sp <= h.cy <= staff.bottom + 5 * sp and h.cx > header_end and not any(x0 - 0.2 * sp <= h.cx <= x1 + 0.2 * sp for x0, x1 in double_bars)]
    comp_heads: dict[int, list[Head]] = {c.id: ([] if classify_accidental(c, sp) else list(heads_of(c))) for c in content}
    standalone: list[Head] = []
    for hole in my_holes:
        if any(abs(hole.cx - b.cx) < 0.7 * sp and abs(hole.cy - b.cy) < 0.7 * sp for hs in comp_heads.values() for b in hs):
            for hs in comp_heads.values():
                for b in hs:
                    if abs(hole.cx - b.cx) < 0.7 * sp and abs(hole.cy - b.cy) < 0.7 * sp:
                        b.hollow = True
            continue
        owners = [c for c in content if c.x0 - 0.5 * sp <= hole.cx <= c.x1 + 0.5 * sp and c.y0 - 0.5 * sp <= hole.cy <= c.y1 + 0.5 * sp]
        owner = max(owners, key=lambda c: c.area, default=None)
        if owner is not None:
            if classify_accidental(owner, sp):
                continue  # the bowl of a flat sign, not a note head
            if any(not h.hollow for h in comp_heads[owner.id]):
                continue  # stem already has a solid head; cannot also hold a hollow head
            if owner.h >= 1.5 * sp:
                comp_heads[owner.id].append(hole)
            else:
                standalone.append(hole)
        else:
            standalone.append(hole)
    graces: list[Item] = []
    tri_digits: list[Comp] = []
    tri_brackets: list[Comp] = []

    def other(c, c_next=None):
        """A component that holds no note head: accidental, dot, tie, rest, grace note or leftover.
        Returns the number of components consumed (1 or 2)."""
        gh = find_grace_head(c, sp) if _classify_rest(c, staff) is None else None  # an eighth rest has a filled dot too
        if gh is not None:
            g = _grace_item(staff, c, gh, base, ledgers, res)
            if g is not None:
                graces.append(g)
            return 1
        acc = classify_accidental(c, sp) if staff.top - 2.5 * sp <= c.cy <= staff.bottom + 2.5 * sp else None
        acc_comp = c
        adv = 1
        if acc is None and c_next is not None and staff.top - 2.5 * sp <= c.cy <= staff.bottom + 2.5 * sp:
            m = _try_merge_accidental(c, c_next, sp)
            if m is not None:
                acc, acc_comp = m
                adv = 2
        if acc:
            accs.append((acc, acc_comp))
            return adv
        if _is_dot(c, sp):
            dots.append(c)
            return 1
        if _is_tie(c, sp):
            ties.append(c)
            return 1
        r = _classify_rest(c, staff)
        if r is not None:
            kind, dur = r
            res.items.append(Item("rest", c.cx, c.bbox, 0.7, c.id, duration=dur, notes=[kind]))
            _mark(res, "rest", c.bbox, kind, 0.7)
            return 1
        if c.w >= 0.3 * sp and c.h <= 1.3 * sp and c.w >= 1.2 * c.h:
            frags.append(c)  # maybe a piece of a tie/slur that a staff line cut apart
            return 1
        if staff.top - 0.2 * sp <= c.cy <= staff.bottom + 0.2 * sp and c.h >= 0.8 * sp:
            diag("UNCLASSIFIED_SYMBOL", "info", f"unrecognised symbol {c.w / sp:.1f}x{c.h / sp:.1f} sp", c)
            _mark(res, "unknown", c.bbox, "?", 0.0)
        return 1

    c_idx = 0
    while c_idx < len(content):
        c = content[c_idx]
        heads = comp_heads[c.id]
        if heads:
            for h in heads:
                it = _make_note(staff, c, h, base, ledgers, res, src)
                if it is not None:
                    notes.append(it)
            c_idx += 1
            continue
        if any(hh.x0 - 0.2 * sp <= c.x0 and c.x1 <= hh.x1 + 0.2 * sp and hh.y0 - 0.2 * sp <= c.y0 and c.y1 <= hh.y1 + 0.2 * sp
               for hh in standalone):
            c_idx += 1
            continue  # a fragment of a hollow head's broken ring
        if _is_three(c, staff):
            tri_digits.append(c)  # maybe the 3 of a triplet: decided once the notes are known
            c_idx += 1
            continue
        if _is_straight_stroke(c, sp):
            tri_brackets.append(c)  # maybe a triplet bracket (a tie is curved)
            c_idx += 1
            continue
        c_next = content[c_idx + 1] if c_idx + 1 < len(content) else None
        consumed = other(c, c_next)
        c_idx += consumed
    for h in standalone:
        it = _make_note(staff, None, h, base, ledgers, res, src)
        if it is not None:
            notes.append(it)

    tri_groups = _triplet_groups(notes + [i for i in res.items if i.kind == "rest"], tri_digits, tri_brackets, sp, other, res)

    for kind, c in accs:
        target = _nearest_note(notes, c, sp)
        if target is None:
            diag("ORPHAN_ACCIDENTAL", "warning", f"accidental {kind!r} with no note to its right", c)
            _mark(res, "accidental", c.bbox, kind + "?", 0.0)
        else:
            target.accidental = kind
            _mark(res, "accidental", c.bbox, kind, 1.0)
    for c in dots:
        target = _note_for_dot(notes, c, sp)
        if target is not None and target.duration is not None:
            target.dots += 1
            _mark(res, "dot", c.bbox, ".", 1.0)
    for it in notes:
        if it.duration is not None:
            it.duration = _dotted(it.duration, it.dots)
    ties.extend(_merge_arc_fragments(frags, sp))
    unmatched = [c for c in ties if not _apply_tie(res, notes, c, sp, diag)]
    _tie_from_halves(res, notes, [c for c in frags if c.w >= 1.0 * sp] + unmatched, sp)

    # several heads in one column: two voices on one staff (or a chord); keep the voice that was asked for, warn once
    notes.sort(key=lambda i: i.x)
    merged: list[Item] = []
    columns = 0
    for it in notes:
        if merged and abs(merged[-1].x - it.x) <= 0.8 * sp:
            prev = merged[-1]
            take_it = it.pos > prev.pos if voice == "top" else it.pos < prev.pos
            keep, drop = (it, prev) if take_it else (prev, it)
            if not prev.notes or not any(n.startswith("dropped") for n in prev.notes):
                columns += 1
            keep.notes.append(f"dropped {'lower' if voice == 'top' else 'upper'} note pos {drop.pos}")
            merged[-1] = keep
        else:
            merged.append(it)
    if columns:
        which = "highest" if voice == "top" else "lowest"
        diag("CHORD_OR_POLYPHONY", "warning",
             f"{columns} column(s) with several note heads (two voices on one staff?); the {which} note is taken as the melody "
             f"(--voice {'bottom' if voice == 'top' else 'top'} reads the other voice)", None)
    for group in tri_groups:  # three columns sound as two: every duration is two thirds of what is written
        for col in group:
            for it in col:
                if it.duration is not None:
                    it.duration = it.duration * Fraction(2, 3)
                    it.tuplet = (3, 2)
    following = sorted(merged + [i for i in res.items if i.kind == "rest"], key=lambda i: i.x)
    for g in sorted(graces, key=lambda i: i.x):
        nxt = next((i for i in following if i.x > g.x + 0.3 * sp), None)
        if nxt is not None and nxt.kind == "note" and nxt.x - g.x <= 5.0 * sp:
            nxt.graces.append(g)
        else:
            diag("GRACE_NOTE_ORPHAN", "info", "grace note with no note after it; ignored")
    res.items.extend(merged)
    for c in bars:
        if c.x0 < header_end - 0.5 * sp:
            continue  # system line or clef remnant
        res.items.append(Item("bar", c.cx, c.bbox, 1.0, c.id))
        _mark(res, "bar", c.bbox, "|")
    res.items.sort(key=lambda i: i.x)
    # collapse barlines closer than 1.5sp (double / final barline)
    out: list[Item] = []
    for it in res.items:
        if it.kind == "bar" and out and out[-1].kind == "bar" and it.x - out[-1].x <= 1.5 * sp:
            continue
        out.append(it)
    res.items = out
    return res



TRIPLET_MIN_SCORE = 0.6  # the digit reader's similarity for a "3" to count as a triplet number
TIME_MIN_SCORE = 0.65  # mean digit similarity below which a reading is not trusted
TIME_MIN_MARGIN = 0.003  # lead over the next valid reading


def _time_cluster(tail: list[Comp], idx: int, header_end: float, staff: Staff) -> list[Comp]:
    """Components that may make up a time signature: the next ones after the key signature that stand close together
    inside the staff, as tall together as two digits."""
    sp = staff.spacing
    lo, hi = staff.top - 0.6 * sp, staff.bottom + 0.6 * sp

    def fits(c):
        return c.y0 >= lo and c.y1 <= hi and c.w <= 2.4 * sp and c.h <= 4.6 * sp

    while idx < len(tail) and (tail[idx].cy < lo or tail[idx].cy > hi):
        idx += 1
    if idx >= len(tail) or tail[idx].x0 - header_end > 3.0 * sp or not fits(tail[idx]):
        return []
    cluster = [tail[idx]]
    k = idx + 1
    while k < len(tail):
        c = tail[k]
        if c.cy < lo or c.cy > hi:
            k += 1
            continue
        if fits(c) and c.x0 - max(x.x1 for x in cluster) <= 0.5 * sp:
            cluster.append(c)
            k += 1
        else:
            break
    # drop trailing components (a note after the time signature) until the rest is as tall as two digits, around the middle line
    while cluster:
        y0, y1 = min(c.y0 for c in cluster), max(c.y1 for c in cluster)
        if 3.0 * sp <= y1 - y0 <= 4.6 * sp and abs((y0 + y1) / 2 - staff.lines[2]) <= 0.7 * sp:
            return cluster
        cluster.pop()
    return []


def _read_time_block(tail: list[Comp], idx: int, header_end: float, staff: Staff):
    """Recognise a common-time, cut-time or stacked-digit time signature starting at tail[idx].

    Returns (meter text, confidence, label, bbox, x of its right edge, components used) or None."""
    sp = staff.spacing
    cluster = _time_cluster(tail, idx, header_end, staff)
    first = tail[idx] if idx < len(tail) else None
    if first is not None and first.x0 - header_end <= 3.0 * sp:
        if _is_common_time(first, staff):
            return "4/4", 0.8, "common time 4/4", first.bbox, first.x1, 1
        if len(cluster) == 1 and _is_cut_time(first, sp):
            return "2/2", 0.8, "cut time 2/2", first.bbox, first.x1, 1
    if not cluster:
        return None
    x0, y0 = min(c.x0 for c in cluster), min(c.y0 for c in cluster)
    x1, y1 = max(c.x1 for c in cluster), max(c.y1 for c in cluster)
    canvas = np.zeros((y1 - y0, x1 - x0), bool)
    for c in cluster:
        canvas[c.y0 - y0 : c.y1 - y0, c.x0 - x0 : c.x1 - x0] |= c.mask
    t = int(np.ceil(staff.thickness)) + 1
    bands = [(int(round(y)) - y0 - t // 2, int(round(y)) - y0 + t // 2) for y in staff.lines]
    _bridge_bands(canvas, bands)  # the flat stroke of a 2, 5 or 6 lying on a staff line went with the line
    # the two rows meet around the middle line: cut at the emptiest row near it (touching digits join at one point)
    mid = int(round(staff.lines[2])) - y0
    ink = canvas.sum(axis=1)
    gaps = [r for r in range(max(1, int(mid - 0.6 * sp)), min(canvas.shape[0] - 1, int(mid + 0.6 * sp)) + 1) if ink[r] == 0]
    if gaps:
        mid = min(gaps, key=lambda r: abs(r - mid))
    else:  # the rows touch: cut where they are joined, the thinnest row close to the middle line
        near = range(max(1, int(mid - 0.3 * sp)), min(canvas.shape[0] - 1, int(mid + 0.3 * sp)) + 1)
        if not near:
            return None
        mid = int(min(near, key=lambda r: (ink[r], abs(r - mid))))
    up = tuple((b0, min(b1, mid - 1)) for b0, b1 in bands if b0 < mid)
    down = tuple((max(b0, mid) - mid, b1 - mid) for b0, b1 in bands if b1 >= mid)
    reading = cvdigits.read_time_signature(canvas[:mid], canvas[mid:], up, down)
    if reading is None or reading.score < TIME_MIN_SCORE or reading.margin < TIME_MIN_MARGIN:
        return None
    return reading.text, round(min(1.0, reading.score), 2), f"time signature {reading.text}", (x0, y0, x1, y1), x1, len(cluster)


def _bridge_bands(canvas: np.ndarray, bands: list[tuple[int, int]], reach: int = 2) -> None:
    """Redraw the rows of each staff-line band: ink only where there is ink on both sides of the band (within ``reach``
    rows). That gives back a stroke thicker than the line that was erased with it (the foot of a 6, the bar of a 2) and
    clears the stubs of the line itself that the line removal left beside a stroke. In place."""
    h = canvas.shape[0]
    for b0, b1 in bands:
        if b0 < 1 or b1 > h - 2 or b0 > b1:
            continue
        above = canvas[max(0, b0 - reach) : b0].any(axis=0)
        below = canvas[b1 + 1 : min(h, b1 + 1 + reach)].any(axis=0)
        canvas[b0 : b1 + 1] = (above & below)[None, :]


def _is_common_time(c: Comp, staff: Staff) -> bool:
    """The C sign: one open curve, a long left spine, no ink in the middle of its right side, no hole, no cut stroke."""
    sp = staff.spacing
    if not (2.0 * sp <= c.h <= 3.6 * sp and 0.7 * c.h <= c.w <= 1.5 * c.h and abs(c.cy - staff.lines[2]) <= 0.7 * sp):
        return False
    m = c.mask
    h, w = m.shape
    if cvdigits.hole_count(m, min_area=max(2.0, 0.004 * h * h)) > 0:
        return False
    left = max((longest_run(m[:, x]) for x in range(max(1, int(0.3 * w)))), default=0)
    mid_right = m[int(0.35 * h) : int(0.65 * h), int(0.7 * w) :]
    top_right, bottom_right = m[: int(0.3 * h), int(0.5 * w) :], m[int(0.7 * h) :, int(0.5 * w) :]
    return bool(left >= 0.5 * h and not mid_right.any() and top_right.any() and bottom_right.any()
                and not _is_cut_time(c, sp))


def _is_cut_time(c: Comp, sp: float) -> bool:
    """Recognize a C-like alla breve glyph crossed by a near-vertical stroke.

    Staff-line removal can leave either the complete glyph or only its slash as
    a connected component, so test those shapes separately.
    """
    h, w = c.mask.shape
    if not (2.8 * sp <= c.h <= 4.8 * sp):
        return False

    # A disconnected slash is tall and narrow. Check it before applying the
    # wider bounds and body-shape checks used for a complete C| glyph.
    if c.w <= 1.5 * sp:
        central_run = 0
        max_drift = max(2, int(round(0.18 * h)))
        for drift in range(-max_drift, max_drift + 1):
            for origin in range(w):
                run = 0
                for y in range(h):
                    x = round(origin + drift * y / max(1, h - 1))
                    if 0 <= x < w and c.mask[y, x]:
                        run += 1
                        central_run = max(central_run, run)
                    else:
                        run = 0
        # A disconnected slash is a narrow component with a continuous
        # near-vertical path. Per-column edge runs are unreliable for a
        # slanted stroke: its ink can move between columns over its height.
        if central_run >= 0.8 * h and c.w <= 0.8 * sp:
            return True

    # Full glyph: require the C's left spine and a distinct slash through the
    # middle. Keep the central search away from the left spine, or an ordinary
    # common-time C would satisfy the slash test by itself.
    if not (0.65 * sp <= c.w <= 2.2 * sp):
        return False
    x0 = max(1, int(0.35 * w))
    x1 = max(x0 + 1, int(0.78 * w))
    central_run = 0
    max_drift = max(2, int(round(0.18 * h)))
    for drift in range(-max_drift, max_drift + 1):
        for origin in range(x0, x1):
            run = 0
            for y in range(h):
                x = round(origin + drift * y / max(1, h - 1))
                if x0 <= x < x1 and c.mask[y, x]:
                    run += 1
                    central_run = max(central_run, run)
                else:
                    run = 0

    left = c.mask[:, :max(1, int(0.3 * w))]
    right = c.mask[:, min(w - 1, int(0.78 * w)):]
    left_run = max((longest_run(left[:, x]) for x in range(left.shape[1])), default=0)
    right_run = max((longest_run(right[:, x]) for x in range(right.shape[1])), default=0)
    return (central_run >= 0.9 * h and left_run >= 0.3 * h
            and right_run < 0.72 * h)


def _is_barline(c: Comp, staff: Staff) -> bool:
    sp = staff.spacing
    if c.w > 1.0 * sp or c.h < 0.8 * staff.height:
        return False
    # a barline spans the whole staff (stems end somewhere inside or outside it)...
    if not (c.y0 <= staff.top + 0.6 * sp and c.y1 >= staff.bottom - 0.6 * sp):
        return False
    # ...and is one straight vertical stroke (a zigzag quarter rest is as tall but not straight)
    return max(longest_run(c.mask[:, x]) for x in range(c.w)) >= 0.9 * c.h


def _staff_pos(staff: Staff, h: Head, ledgers, res: StaffResult):
    """(rounded staff position, exact position) of a head; None when it sits off the staff without the ledger lines it needs."""
    sp = staff.spacing
    pos_f = staff.position(h.cy)
    pos = int(round(pos_f))
    if pos < -1 or pos > 9:  # outside the staff: needs ledger lines (rejects lyrics / text)
        need = pos if pos % 2 == 0 else (pos + 1 if pos < 0 else pos - 1)
        ok = all(
            any(abs(l.y - staff.y_of(p)) <= 0.35 * sp and l.x0 - 0.6 * sp <= h.cx <= l.x1 + 0.6 * sp and l.staff == staff.uid for l in ledgers)
            for p in (range(-2, need - 1, -2) if pos < 0 else range(10, need + 1, 2))
        )
        if not ok:
            _mark(res, "rejected", (h.x0, h.y0, h.x1, h.y1), "no ledger", 0.0)
            return None
    return pos, pos_f


def _dotted(d: Fraction, dots: int) -> Fraction:
    return d * (2 - Fraction(1, 2**dots))


def find_grace_head(c: Comp, sp: float) -> Optional[Head]:
    """The small filled head of a grace note: a short stem (with a flag and a slash) and a head about two thirds the size
    of an ordinary one. Stem, flag and slash are thinner than the head and vanish in an opening; what remains is the head."""
    if not (1.5 * sp <= c.h <= 3.8 * sp and c.w <= 2.6 * sp):
        return None
    pad = 3
    m = np.pad(c.mask.astype(np.uint8) * 255, pad)
    d = max(3, int(round(0.38 * sp)))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(opened, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if not (0.5 * sp <= w <= 1.2 * sp and 0.3 * sp <= h <= 0.85 * sp and w >= 0.9 * h):
            continue
        if a < 0.55 * w * h or not _head_like_orientation((lab == i).astype(np.uint8)):
            continue
        if best is None or a > best[0]:
            best = (a, Head(c.x0 + cents[i][0] - pad, c.y0 + cents[i][1] - pad, c.x0 + x - pad, c.y0 + y - pad,
                            c.x0 + x + w - pad, c.y0 + y + h - pad, 1.0))
    if best is None:
        return None
    head = best[1]
    # a grace note has a stem: ink of the component reaching well beyond the head
    reach = max(head.y0 - c.y0, c.y1 - head.y1)
    return head if reach >= 1.0 * sp else None


def _grace_item(staff: Staff, c: Comp, h: Head, base: int, ledgers, res: StaffResult) -> Optional[Item]:
    found = _staff_pos(staff, h, ledgers, res)
    if found is None:
        return None
    pos, pos_f = found
    it = Item("grace", h.cx, (int(h.x0), int(h.y0), int(h.x1), int(h.y1)), max(0.0, 0.8 - 2.0 * abs(pos_f - pos)), c.id, pos=pos)
    _mark(res, "grace", c.bbox, "grace", it.conf)
    return it


def _is_three(c: Comp, staff: Staff) -> bool:
    """A "3" of digit size (the number of a triplet), judged with the digit reader; the staff lines that cross it are bridged."""
    sp = staff.spacing
    if not (0.9 * sp <= c.h <= 2.1 * sp and 0.45 * sp <= c.w <= 1.5 * sp):
        return False
    canvas = c.mask.copy()
    t = int(np.ceil(staff.thickness)) + 1
    bands = [(int(round(y)) - c.y0 - t // 2, int(round(y)) - c.y0 + t // 2) for y in staff.lines if c.y0 - t <= y <= c.y1 + t]
    bands = [(max(0, b0), min(c.h - 1, b1)) for b0, b1 in bands if b1 >= 0 and b0 <= c.h - 1]
    _bridge_bands(canvas, bands)
    ranked = cvdigits.classify_digit(canvas, tuple(bands))
    return len(ranked) > 1 and ranked[0][0] == "3" and ranked[0][1] >= TRIPLET_MIN_SCORE and ranked[0][1] - ranked[1][1] >= 0.03


def _is_straight_stroke(c: Comp, sp: float) -> bool:
    """A long thin level stroke (the middle of a triplet bracket); a tie or slur is an arc and bends more."""
    if not (1.0 * sp <= c.w <= 40 * sp and c.h <= 1.2 * sp and c.w >= 2.5 * c.h):
        return False
    lo, hi = int(0.2 * c.w), max(int(0.2 * c.w) + 1, int(0.8 * c.w))
    centres = []
    for x in range(lo, hi):
        ys = np.nonzero(c.mask[:, x])[0]
        if len(ys) == 0:
            return False
        centres.append(ys.mean())
    return max(centres) - min(centres) <= max(1.5, 0.15 * sp)


def _triplet_groups(items: list[Item], digits: list[Comp], brackets: list[Comp], sp: float, other, res: StaffResult):
    """Find the three columns each "3" belongs to. A digit that does not sit beyond the stems of three neighbouring
    columns is handed back to ``other`` (and read as whatever else it may be); so is a stroke that is no bracket."""
    columns: list[list[Item]] = []
    for it in sorted(items, key=lambda i: i.x):
        if columns and abs(columns[-1][0].x - it.x) <= 0.8 * sp:
            columns[-1].append(it)
        else:
            columns.append([it])
    groups, used, taken = [], set(), []
    for d in sorted(digits, key=lambda c: c.x0):
        best = None
        for i in range(len(columns) - 2):
            if used & {i, i + 1, i + 2}:
                continue
            xa, xb = columns[i][0].x, columns[i + 2][0].x
            off = abs(d.cx - (xa + xb) / 2)
            if off > max(1.2 * sp, 0.35 * (xb - xa)):
                continue
            boxes = [it.bbox for k in range(i, i + 3) for it in columns[k]]
            top, bottom = min(b[1] for b in boxes), max(b[3] for b in boxes)
            gap = (top - d.cy) if d.cy < top else (d.cy - bottom) if d.cy > bottom else None
            if gap is None or gap < 0.3 * sp or gap > 3.0 * sp:
                continue
            if best is None or off < best[0]:
                best = (off, i)
        if best is None:
            other(d)
            continue
        i = best[1]
        used |= {i, i + 1, i + 2}
        groups.append([columns[k] for k in range(i, i + 3)])
        taken.append((d, columns[i][0].x, columns[i + 2][0].x))
        _mark(res, "tuplet", d.bbox, "3", 1.0)
    for b in brackets:
        if any(abs(b.cy - d.cy) <= 1.3 * sp and xa - 1.5 * sp <= b.cx <= xb + 1.5 * sp for d, xa, xb in taken):
            _mark(res, "tuplet", b.bbox, "bracket", 1.0)
        else:
            other(b)
    return groups


def _make_note(staff: Staff, c: Optional[Comp], h: Head, base: int, ledgers, res: StaffResult, src) -> Optional[Item]:
    sp = staff.spacing
    found = _staff_pos(staff, h, ledgers, res)
    if found is None:
        return None
    pos, pos_f = found
    conf = max(0.0, 1.0 - 2.0 * abs(pos_f - pos))
    if conf < 0.20:
        return None
    hollow = h.hollow
    stem, tip_len, tip_pt = _find_stem(c, h, sp) if c is not None else (None, 0, None)
    flags = _count_flags(c, h, stem, tip_pt, sp) if stem else 0
    if stem is None:
        dur = Fraction(1) if hollow else Fraction(1, 4)
        if not hollow:
            if conf < 0.25:
                return None
            conf = min(conf, 0.2)
            res.diagnostics.append(Diagnostic("STEMLESS_FILLED_HEAD", "warning", "filled note head without a stem; read as a quarter note", src(c)))
    elif hollow:
        dur = Fraction(1, 2)
    else:
        dur = Fraction(1, 4 * 2**flags)
    it = Item("note", h.cx, (int(h.x0), int(h.y0), int(h.x1), int(h.y1)), conf, c.id if c is not None else -1, pos=pos, duration=dur,
              hollow=hollow, stem=stem, flags=flags)
    it.notes.append(f"pos_f={pos_f:.2f}")
    it.diatonic = base + pos  # type: ignore[attr-defined]
    _mark(res, "note", it.bbox, "", conf)
    return it


def _runs(col: np.ndarray):
    """(start, end) of each ink run in a boolean column, end inclusive."""
    d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def _find_stem(c: Comp, h: Head, sp: float):
    """Return (direction, length, (x, tip_y)) in page coordinates, or (None, 0, None).

    A stem is a vertical ink run in a column just beside the head that reaches the head and
    extends well beyond it.
    """
    m = c.mask
    cy = h.cy - c.y0
    ya, yb = h.y0 - c.y0 - 0.3 * sp, h.y1 - c.y0 + 0.3 * sp
    half = (h.y1 - h.y0) / 2
    best = (0.0, None, 0, 0)
    xa, xb = int(h.x0 - 0.25 * sp - c.x0), int(h.x1 + 0.25 * sp - c.x0)
    for x in range(max(0, xa), min(c.w, xb + 1)):
        for r0, r1 in _runs(m[:, x]):
            if r1 < ya or r0 > yb:
                continue  # does not touch the head
            up, down = cy - r0, r1 - cy
            direction, ext, tip = ("up", up, r0) if up >= down else ("down", down, r1)
            if ext > best[0]:
                best = (ext, direction, x, tip)
    ext, direction, x, tip = best
    if direction is None or ext - half < 1.5 * sp:
        return None, 0, None
    return direction, ext - half, (c.x0 + x, c.y0 + tip)


def _count_flags(c: Comp, h: Head, direction: str, tip, sp: float) -> int:
    """Count flag/beam strokes beside the stem near its tip.

    A beam is half a spacing thick and the next one starts a quarter spacing further in; on a coarse or blurred scan
    the gap closes and two beams look like one run twice as thick, so a run counts as one stroke per 0.75 spacing."""
    sx, tip_y = tip
    m = c.mask
    best = 0
    for side in (-1, 1):
        x = int(round(sx + side * 0.55 * sp - c.x0))
        if not 0 <= x < c.w:
            continue
        y_tip = int(tip_y - c.y0)
        step = 1 if direction == "up" else -1  # walk from the tip toward the head
        n_runs, inrun, run_len = 0, False, 0

        def close(run_len):
            return max(1, int(round((run_len / sp + 0.25) / 0.75))) if run_len >= 0.15 * sp else 0

        for k in range(int(2.2 * sp)):
            y = y_tip + step * k
            if not 0 <= y < c.h:
                break
            if m[y, x]:
                run_len += 1
                inrun = True
            else:
                if inrun:
                    n_runs += close(run_len)
                inrun, run_len = False, 0
        if inrun:
            n_runs += close(run_len)
        best = max(best, n_runs)
    return min(best, 3)


def _nearest_note(notes: list[Item], c: Comp, sp: float) -> Optional[Item]:
    best, bd = None, 1e9
    for n in notes:
        gap = n.bbox[0] - c.x1
        if -0.3 * sp <= gap <= 1.5 * sp and abs(((n.bbox[1] + n.bbox[3]) / 2) - c.cy) <= 1.8 * sp and gap < bd:
            best, bd = n, gap
    return best


def _is_dot(c: Comp, sp: float) -> bool:
    return (0.12 * sp <= c.w <= 0.65 * sp and 0.12 * sp <= c.h <= 0.65 * sp and c.area / (c.w * c.h) >= 0.5)


def _note_for_dot(notes: list[Item], c: Comp, sp: float) -> Optional[Item]:
    best, bd = None, 1e9
    for n in notes:
        gap = c.x0 - n.bbox[2]
        ny = (n.bbox[1] + n.bbox[3]) / 2
        if 0.2 * sp <= gap <= 2.4 * sp and -0.2 * sp <= ny - c.cy <= 0.7 * sp + 0.2 * sp and gap < bd:
            best, bd = n, gap
        elif 0.2 * sp <= gap <= 2.4 * sp and abs(ny - c.cy) <= 0.7 * sp and gap < bd:
            best, bd = n, gap
    return best


def _is_tie(c: Comp, sp: float) -> bool:
    return c.w >= 1.5 * sp and c.h <= 1.4 * sp and c.w / c.h >= 2.0 and c.area / (c.w * c.h) <= 0.45


def _merge_arc_fragments(frags: list[Comp], sp: float) -> list[Comp]:
    """Join horizontally adjacent curved pieces into whole arcs; keep those long enough to be ties/slurs."""
    out, cluster = [], []

    def flush():
        if cluster:
            x0, y0 = min(c.x0 for c in cluster), min(c.y0 for c in cluster)
            x1, y1 = max(c.x1 for c in cluster), max(c.y1 for c in cluster)
            if x1 - x0 >= 1.5 * sp and y1 - y0 <= 1.6 * sp:
                out.append(Comp(-1, x0, y0, x1, y1, sum(c.area for c in cluster), None))
        cluster.clear()

    for c in sorted(frags, key=lambda c: c.x0):
        if cluster and (c.x0 - max(k.x1 for k in cluster) > 1.2 * sp or abs(c.cy - cluster[-1].cy) > 1.0 * sp):
            flush()
        cluster.append(c)
    flush()
    return out


def _apply_tie(res: StaffResult, notes: list[Item], c: Comp, sp: float, diag) -> bool:
    """Connect the two notes at the ends of an arc. Returns False when the arc does not join two notes."""
    def near(x):
        cand = [n for n in notes if abs(n.x - x) <= 1.0 * sp]
        return min(cand, key=lambda n: abs(n.x - x)) if cand else None

    a, b = near(c.x0 + 0.2 * sp), near(c.x1 - 0.2 * sp)
    if a is None or b is None or a is b:
        return False
    if a.pos == b.pos:
        a.tie_start = True
        _mark(res, "tie", c.bbox, "tie")
    else:
        diag("SLUR_IGNORED", "info", "slur between different pitches ignored", c)
        _mark(res, "slur", c.bbox, "slur")
    return True


def _tie_from_halves(res: StaffResult, notes: list[Item], arcs: list[Comp], sp: float) -> None:
    """A tie lying on a staff line is cut in the middle when the line is removed: its two halves still start at
    one note and end at the next note of the same pitch."""
    notes = sorted(notes, key=lambda n: n.x)
    for a, b in zip(notes, notes[1:]):
        if a.pos != b.pos or a.tie_start or a.kind != "note" or b.kind != "note":
            continue
        left = [c for c in arcs if a.x - 0.5 * sp <= c.x0 <= a.x + 1.3 * sp and c.x1 < b.x - 1.0 * sp]
        right = [c for c in arcs if b.x - 1.3 * sp <= c.x1 <= b.x + 0.5 * sp and c.x0 > a.x + 1.0 * sp]
        for l in left:
            for r in right:
                if l is not r and l.x1 <= r.x0 + 0.2 * sp and abs(l.cy - r.cy) <= 0.8 * sp:
                    a.tie_start = True
                    _mark(res, "tie", (l.x0, min(l.y0, r.y0), r.x1, max(l.y1, r.y1)), "tie (halves)")
                    break
            if a.tie_start:
                break


def _classify_rest(c: Comp, staff: Staff):
    sp = staff.spacing
    if not (staff.top - 0.3 * sp <= c.cy <= staff.bottom + 0.3 * sp):
        return None
    solid = c.area / (c.w * c.h)
    if 0.9 * sp <= c.w <= 1.8 * sp and 0.25 * sp <= c.h <= 0.8 * sp and solid >= 0.8:
        top_gap = abs(c.y0 - staff.lines[1])
        bottom_gap = abs(c.y1 - staff.lines[2])
        return ("whole rest", Fraction(1)) if top_gap < bottom_gap else ("half rest", Fraction(1, 2))
    if c.w <= 1.6 * sp and not strokes(c, 0.55):
        if c.h >= 2.4 * sp:
            return "quarter rest", Fraction(1, 4)
        if 1.2 * sp <= c.h < 2.4 * sp:
            return "eighth rest", Fraction(1, 8)
    return None
