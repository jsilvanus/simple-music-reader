"""Symbol recognition on a staff-line-free image, in units of the staff spacing."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import cv2
import numpy as np

from ..ir import Diagnostic, Source
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
        if not (0.4 * sp * sp <= a <= 2.4 * sp * sp and 0.7 * sp <= w <= 2.0 * sp and 0.6 * sp <= h <= 1.7 * sp):
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
    if l2 <= 0 or np.sqrt(l1 / l2) <= 1.25:
        return True
    theta = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    return abs(theta) <= 40.0


def find_holes(bw: np.ndarray, sp: float) -> list[Head]:
    """Hollow note heads found as small enclosed white regions of the page *with* staff lines in place.

    A hollow head sitting between two lines is tangent to both, so removing the lines breaks its ring;
    the white inside survives, which is why heads are also searched for this way.
    """
    n, lab, stats, cents = cv2.connectedComponentsWithStats(255 - bw, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if not (0.2 * sp * sp <= a <= 0.9 * sp * sp and 0.5 * sp <= w <= 1.3 * sp and 0.25 * sp <= h <= 0.8 * sp):
            continue
        if not 0.55 <= a / (w * h) <= 0.9:
            continue
        pad_x, pad_y = 0.25 * sp, 0.2 * sp
        out.append(Head(float(cents[i][0]), float(cents[i][1]), x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y, 0.6, True))
    return out


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
    if not (1.6 * sp <= c.h <= 3.0 * sp and 0.5 * sp <= c.w <= 1.8 * sp):
        return None
    st = strokes(c)
    if len(st) >= 2 and all(r >= 0.8 * c.h for _, r in st[:2]):
        return "#"
    if len(st) >= 2 and all(0.45 * c.h <= r < 0.82 * c.h for _, r in st[:2]):
        return "n"
    if len(st) == 1 and st[0][1] >= 0.7 * c.h and st[0][0] < 0.5 * c.w and c.w <= 1.5 * sp:
        return "b"
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


@dataclass
class StaffResult:
    staff: Staff
    clef: str = "treble"
    clef_confirmed: bool = False
    key_fifths: Optional[int] = None
    time_sig_seen: bool = False
    items: list[Item] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    marks: list[dict] = field(default_factory=list)  # for debug rendering


def _mark(res, kind, bbox, label="", conf=1.0):
    res.marks.append({"kind": kind, "bbox": tuple(int(v) for v in bbox), "label": label, "conf": float(conf)})


def analyze_staff(staff: Staff, comps: list[Comp], ledgers: list[Ledger], page: int, system: int,
                  holes: Optional[list[Head]] = None) -> StaffResult:
    sp = staff.spacing
    res = StaffResult(staff)
    src = lambda c=None: Source(page, c.bbox if c else None, system, staff.index, c.id if c else None)  # noqa: E731

    def diag(code, sev, msg, c=None):
        res.diagnostics.append(Diagnostic(code, sev, msg, src(c)))

    bars = [c for c in comps if _is_barline(c, staff)]
    rest = sorted((c for c in comps if c not in bars), key=lambda c: c.x0)

    # ---- header: clef, key signature, time signature -------------------------------
    header_end = float(staff.x0)
    clef = next((c for c in rest if c.x0 < staff.x0 + 8 * sp and c.h >= 2.2 * sp and c.w >= 0.8 * sp), None)
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
            heads_cache[c.id] = find_heads(c, sp)
        return heads_cache[c.id]

    def looks_like_time(c):
        return (3.0 * sp <= c.h <= 4.4 * sp and 0.8 * sp <= c.w <= 2.2 * sp and abs(c.cy - staff.lines[2]) <= 0.7 * sp)

    key_accs, last_x1, idx = [], header_end, 0
    while idx < len(tail):
        c = tail[idx]
        if c.x0 - last_x1 > 3.0 * sp:
            break
        kind = classify_accidental(c, sp)
        if kind in ("#", "b"):
            nxt = tail[idx + 1] if idx + 1 < len(tail) else None
            if nxt is not None and not looks_like_time(nxt) and heads_of(nxt) and nxt.x0 - c.x1 <= 1.5 * sp:
                break  # belongs to the first note, not to the key signature
            key_accs.append((kind, c))
            _mark(res, "key", c.bbox, kind)
            last_x1, idx = c.x1, idx + 1
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
    if idx < len(tail):
        s0 = tail[idx]
        if s0.x0 - header_end <= 3.0 * sp and looks_like_time(s0):
            res.time_sig_seen = True  # two touching digits form one component
            _mark(res, "time", s0.bbox, "time signature (digits not read)")
            header_end = s0.x1
            idx += 1
    if idx + 1 < len(tail):
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
    my_holes = [h for h in (holes or []) if staff.top - 5 * sp <= h.cy <= staff.bottom + 5 * sp and h.cx > header_end]
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
        if owner is not None and classify_accidental(owner, sp):
            continue  # the bowl of a flat sign, not a note head
        if owner is not None and owner.h >= 1.5 * sp:
            comp_heads[owner.id].append(hole)
        else:
            standalone.append(hole)
    for c in content:
        heads = comp_heads[c.id]
        if heads:
            for h in heads:
                it = _make_note(staff, c, h, base, ledgers, res, src)
                if it is not None:
                    notes.append(it)
            continue
        if any(hh.x0 - 0.2 * sp <= c.x0 and c.x1 <= hh.x1 + 0.2 * sp and hh.y0 - 0.2 * sp <= c.y0 and c.y1 <= hh.y1 + 0.2 * sp
               for hh in standalone):
            continue  # a fragment of a hollow head's broken ring
        acc = classify_accidental(c, sp)
        if acc:
            accs.append((acc, c))
            continue
        if _is_dot(c, sp):
            dots.append(c)
            continue
        if _is_tie(c, sp):
            ties.append(c)
            continue
        r = _classify_rest(c, staff)
        if r is not None:
            kind, dur = r
            res.items.append(Item("rest", c.cx, c.bbox, 0.7, c.id, duration=dur, notes=[kind]))
            _mark(res, "rest", c.bbox, kind, 0.7)
            continue
        if c.w >= 0.3 * sp and c.h <= 1.3 * sp and c.w >= 1.2 * c.h:
            frags.append(c)  # maybe a piece of a tie/slur that a staff line cut apart
            continue
        if staff.top - 0.2 * sp <= c.cy <= staff.bottom + 0.2 * sp and c.h >= 0.8 * sp:
            diag("UNCLASSIFIED_SYMBOL", "info", f"unrecognised symbol {c.w / sp:.1f}x{c.h / sp:.1f} sp", c)
            _mark(res, "unknown", c.bbox, "?", 0.0)
    for h in standalone:
        it = _make_note(staff, None, h, base, ledgers, res, src)
        if it is not None:
            notes.append(it)

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

    # drop chord duplicates: keep top note, warn
    notes.sort(key=lambda i: i.x)
    merged: list[Item] = []
    for it in notes:
        if merged and abs(merged[-1].x - it.x) <= 0.8 * sp:
            keep = merged[-1] if merged[-1].pos >= it.pos else it
            drop = it if keep is merged[-1] else merged[-1]
            diag("CHORD_OR_POLYPHONY", "warning", "several note heads in one column; the highest is taken as melody", None)
            keep.notes.append(f"dropped lower note pos {drop.pos}")
            merged[-1] = keep
        else:
            merged.append(it)
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


def _is_barline(c: Comp, staff: Staff) -> bool:
    sp = staff.spacing
    if c.w > 1.0 * sp or c.h < 0.8 * staff.height:
        return False
    # a barline spans the whole staff (stems end somewhere inside or outside it)...
    if not (c.y0 <= staff.top + 0.6 * sp and c.y1 >= staff.bottom - 0.6 * sp):
        return False
    # ...and is one straight vertical stroke (a zigzag quarter rest is as tall but not straight)
    return max(longest_run(c.mask[:, x]) for x in range(c.w)) >= 0.9 * c.h


def _dotted(d: Fraction, dots: int) -> Fraction:
    return d * (2 - Fraction(1, 2**dots))


def _make_note(staff: Staff, c: Optional[Comp], h: Head, base: int, ledgers, res: StaffResult, src) -> Optional[Item]:
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
    conf = max(0.0, 1.0 - 2.0 * abs(pos_f - pos))
    hollow = h.hollow
    stem, tip_len, tip_pt = _find_stem(c, h, sp) if c is not None else (None, 0, None)
    flags = _count_flags(c, h, stem, tip_pt, sp) if stem else 0
    if stem is None:
        dur = Fraction(1) if hollow else Fraction(1, 4)
        if not hollow:
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
    """Count flag/beam strokes beside the stem near its tip."""
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
        for k in range(int(2.2 * sp)):
            y = y_tip + step * k
            if not 0 <= y < c.h:
                break
            if m[y, x]:
                run_len += 1
                inrun = True
            else:
                if inrun and run_len >= 0.15 * sp:
                    n_runs += 1
                inrun, run_len = False, 0
        if inrun and run_len >= 0.15 * sp:
            n_runs += 1
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
