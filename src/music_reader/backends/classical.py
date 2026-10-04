"""Classical computer-vision backend: staff geometry, line removal, connected components, rules.

Deterministic, runs locally, no ML. See OMR.md for what it can and cannot read.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..assemble import build_voice, infer_meter
from ..cv import staff as cvstaff
from ..cv import symbols as cvsym
from ..ir import (Diagnostic, KeySignature, Meter, PageInfo, Score, Source, StaffInfo, SystemInfo)
from ..pdfinput import Document
from ..validate import validate
from .base import PageDebug, RecognitionOptions, RecognitionResult

TARGET_SPACING = 16.0  # px; staff spacing we aim for when choosing the render resolution


class ClassicalBackend:
    name = "classical"

    def recognize(self, doc: Document, options: RecognitionOptions) -> RecognitionResult:
        score = Score(metadata={"title": options.title} if options.title else {})
        result = RecognitionResult(score)
        selected = []
        key_candidates: list[tuple[int, int, int]] = []  # (fifths, page, system)
        ambiguous_systems = 0
        time_sig_seen = False
        pages = options.pages or list(range(1, len(doc) + 1))
        for pno in pages:
            if not 1 <= pno <= len(doc):
                score.diagnostics.append(Diagnostic("BAD_PAGE", "error", f"page {pno} does not exist ({len(doc)} pages)"))
                continue
            info, dbg, systems = self._page(doc, pno, options, score)
            score.pages.append(info)
            result.pages.append(dbg)
            for si, group in enumerate(systems):
                if options.staff is None and len(group) > 1:
                    ambiguous_systems += 1
                k = (options.staff or 1) - 1
                if k >= len(group):
                    score.diagnostics.append(Diagnostic("STAFF_NOT_FOUND", "error",
                        f"system has {len(group)} staves, cannot select staff {k + 1}", Source(pno, None, si)))
                    continue
                res = group[k]
                score.diagnostics.extend(res.diagnostics)
                dbg.marks.extend(res.marks)
                time_sig_seen |= res.time_sig_seen
                if res.key_fifths is not None:
                    key_candidates.append((res.key_fifths, pno, si))
                selected.append((res, pno, si))
                for other in group:
                    if other is not res:
                        dbg.marks.extend({**m, "kind": "other-" + m["kind"]} for m in other.marks)
        if ambiguous_systems:
            score.diagnostics.append(Diagnostic("AMBIGUOUS_MELODY_STAFF", "warning",
                f"{ambiguous_systems} system(s) have several staves and no staff was chosen; the first staff is assumed to carry "
                "the melody (use --staff N to choose)"))
        score.selection = {
            "strategy": "explicit" if options.staff else "assumed-first-staff",
            "staff": options.staff or 1,
            "ambiguous": bool(ambiguous_systems) and not options.staff,
            "description": f"staff {options.staff or 1} of each system"
            + ("" if options.staff else " (assumed; several staves present)" if ambiguous_systems else ""),
        }
        # key
        if options.key:
            score.key = KeySignature.parse(options.key)
        elif key_candidates:
            first = key_candidates[0][0]
            for f, pno, si in key_candidates:
                if f != first:
                    score.diagnostics.append(Diagnostic("KEY_CHANGE", "warning",
                        f"key signature differs between systems ({first} vs {f} fifths); the first is used", Source(pno, None, si)))
                    break
            score.key = KeySignature(first, inferred=True)
            score.diagnostics.append(Diagnostic("KEY_MODE", "info", "major/minor cannot be told from the key signature; written as major"))
        score.voices = [build_voice(selected, score.key, score.diagnostics)]
        # meter
        if options.meter:
            score.meter = Meter.parse(options.meter)
        else:
            score.meter = infer_meter(score.voices[0].measures)
            if score.meter is not None:
                what = "a time signature was seen but its digits are not read" if time_sig_seen else "no time signature was found"
                score.diagnostics.append(Diagnostic("METER_INFERRED", "warning",
                    f"{what}; meter {score.meter} inferred from the measure lengths (use --meter to set it)"))
        score.diagnostics.extend(validate(score))
        return result

    # ------------------------------------------------------------------------------
    def _page(self, doc: Document, pno: int, options: RecognitionOptions, score: Score):
        analysis = doc.analyze(pno - 1)
        dpi = options.dpi or 150.0
        gray, dpi = doc.render(pno - 1, dpi)
        bw0 = cvstaff.binarize(gray)
        skew = cvstaff.estimate_skew(bw0)
        if abs(skew) >= 0.15:
            gray = cvstaff.rotate(gray, skew, border=255)
        bw = cvstaff.binarize(gray)
        staves, line_mask = cvstaff.detect_staves(bw)
        if staves and analysis.kind != "image" and not options.dpi:
            sp = float(np.median([s.spacing for s in staves]))
            if sp < 0.7 * TARGET_SPACING or sp > 1.6 * TARGET_SPACING:
                dpi = float(min(600.0, max(72.0, dpi * TARGET_SPACING / sp)))
                gray, dpi = doc.render(pno - 1, dpi)
                if abs(skew) >= 0.15:
                    gray = cvstaff.rotate(gray, skew, border=255)
                bw = cvstaff.binarize(gray)
                staves, line_mask = cvstaff.detect_staves(bw)
        info = PageInfo(pno, analysis.kind, gray.shape[1], gray.shape[0], dpi, music_fonts=analysis.music_fonts,
                        skew_degrees=skew)
        dbg = PageDebug(pno, gray, staves, skew_degrees=skew, dpi=dpi)
        if not staves:
            score.diagnostics.append(Diagnostic("NO_STAVES", "warning", "no five-line staves found on this page", Source(pno)))
            return info, dbg, []
        sp_med = float(np.median([s.spacing for s in staves]))
        if sp_med < 8:
            score.diagnostics.append(Diagnostic("LOW_RESOLUTION", "warning",
                f"staff spacing is only {sp_med:.1f} px; recognition will be unreliable (try a higher --dpi)", Source(pno)))
        groups = cvstaff.group_systems(staves, bw)
        for a, b in zip(staves, staves[1:]):
            if a.system != b.system and (b.top - a.bottom) < 9 * a.spacing:
                score.diagnostics.append(Diagnostic("POSSIBLY_UNGROUPED_STAVES", "warning",
                    "two staves are close together but not joined by a system line; they are treated as separate systems",
                    Source(pno, None, b.system)))
        residual = cvstaff.remove_staff_lines(bw, staves, line_mask)
        residual, ledgers = cvstaff.remove_ledger_lines(residual, staves)
        comps = cvsym.extract_components(residual, sp_med)
        by_staff = _assign(comps, staves)
        holes = cvsym.find_holes(bw, sp_med)
        systems = []
        for si, group in enumerate(groups):
            results = []
            for s in group:
                r = cvsym.analyze_staff(s, by_staff[s.uid], ledgers, pno, si, holes)
                results.append(r)
            systems.append(results)
            info.systems.append(SystemInfo(si, [StaffInfo(s.index, s.lines, s.x0, s.x1, s.spacing, r.clef, r.clef_confirmed)
                                               for s, r in zip(group, results)]))
        dbg.system_boxes = _system_boxes(groups, gray.shape, bw)
        return info, dbg, systems


# Vertical limits of a system crop, in staff spacings beyond the outer staff lines: room for ledger-line notes and
# clefs above, one line of lyrics (plus its descenders) below.
CROP_ABOVE = 3.5
CROP_BELOW = 6.5
CROP_GAP_STAFF = 4.5  # blank rows (in spacings) tolerated between the staff and its lyrics
CROP_GAP_LYRICS = 1.6  # blank rows (in spacings) that end the lyrics block
CROP_ABOVE_MIN = 2.0  # spacings above the top line that are always kept (ledger-line notes)
CROP_LYRICS_END = 5.0  # where, in spacings below the staff, the lyrics normally end
CROP_NEXT_GAP = 1.0  # spacings kept clear above the next staff when looking for the cut
CROP_PAD = 0.5  # margin (in spacings) kept below the last ink row


def _assign(comps, staves) -> dict[int, list]:
    """Give every component to the staff it belongs to (barlines may belong to several)."""
    out = {s.uid: [] for s in staves}
    for c in comps:
        placed = False
        for s in staves:  # thin tall strokes may be barlines spanning several staves
            ov = min(c.y1, s.bottom) - max(c.y0, s.top)
            if ov >= 0.7 * s.height and c.w <= 1.0 * s.spacing and c.x0 >= s.x0 - s.spacing and c.x1 <= s.x1 + 2 * s.spacing:
                out[s.uid].append(c)
                placed = True
        if placed:
            continue
        s = min(staves, key=lambda s: abs(c.cy - (s.top + s.bottom) / 2))
        if s.top - 5 * s.spacing <= c.cy <= s.bottom + 5 * s.spacing and s.x0 - 2 * s.spacing <= c.cx <= s.x1 + 2 * s.spacing:
            out[s.uid].append(c)
    return out


def _ink_bottom(bw, x0: int, x1: int, y_start: int, y_cap: int, sp: float) -> int:
    """Last ink row of the block hanging below y_start (notes, lyrics): stops at a blank gap (wide until the lyrics
    start, narrow after) or at y_cap."""
    rows = (bw[y_start:y_cap, x0:x1] > 0).any(axis=1)
    last, gap = y_start, 0
    for i, ink in enumerate(rows):
        if ink:
            last, gap = y_start + i, 0
        else:
            gap += 1
            limit = CROP_GAP_STAFF if last < y_start + 2.5 * sp else CROP_GAP_LYRICS
            if gap > limit * sp:
                break
    return last


def _ink_top(bw, x0: int, x1: int, y_start: int, y_cap: int, sp: float) -> int:
    """First row of everything attached to the staff whose top line is y_start (clef, notes, stems, ties, all joined
    through the staff lines), at least CROP_ABOVE_MIN spacings above it for ledger-line notes. Prose and headings that
    only stand close above the staff are separate shapes and stay out."""
    y_end = y_start + int(4 * sp) + 1
    region = (bw[y_cap:y_end, x0:x1] > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(region, connectivity=8)
    row = labels[y_start - y_cap] if y_start - y_cap < labels.shape[0] else labels[-1]
    keep = np.unique(row[row > 0])
    attached = np.isin(labels[: y_start - y_cap + 1], keep)
    rows = np.nonzero(attached.any(axis=1))[0]
    first = y_cap + int(rows[0]) if len(rows) else y_start
    return max(y_cap, min(first, y_start - int(CROP_ABOVE_MIN * sp)))


def _system_boxes(groups, shape, bw=None) -> list[tuple[int, int, int, int]]:
    """Crop box for each system: the full staff width, and vertically the staff plus what hangs on it.

    With the binarised page the box follows the ink: it grows up over clef, high notes and ledger lines, and down over
    the lyrics until a blank gap, so descenders are never cut and headings or prose further away stay out. Between two
    systems the cut goes through the middle of the blank gap (or, if the ink touches, through the emptiest row). Without
    the page the box is limited to CROP_ABOVE / CROP_BELOW spacings beyond the staff (and halfway to the neighbour)."""
    h, w = shape
    xs = [(int(max(0, min(s.x0 for s in g) - 3 * g[0].spacing)), int(min(w, max(s.x1 for s in g) + 1.5 * g[0].spacing)))
          for g in groups]
    if bw is None:
        boxes = []
        for i, g in enumerate(groups):
            sp = g[0].spacing
            top, bottom = g[0].top, g[-1].bottom
            y0 = (groups[i - 1][-1].bottom + top) / 2 if i else top - CROP_ABOVE * sp
            y1 = (bottom + groups[i + 1][0].top) / 2 if i + 1 < len(groups) else bottom + CROP_BELOW * sp
            y0, y1 = max(y0, top - CROP_ABOVE * sp), min(y1, bottom + CROP_BELOW * sp)
            boxes.append((xs[i][0], int(max(0, y0)), xs[i][1], int(min(h, y1))))
        return boxes
    n = len(groups)
    y0s, y1s = [0] * n, [0] * n
    sp0 = groups[0][0].spacing
    y0s[0] = max(0, _ink_top(bw, xs[0][0], xs[0][1], int(groups[0][0].top), max(0, int(groups[0][0].top - 8 * sp0)), sp0)
                 - int(CROP_PAD * sp0))
    for i, g in enumerate(groups):
        sp, bottom = g[0].spacing, int(g[-1].bottom)
        x0, x1 = xs[i]
        pad = int(CROP_PAD * sp)
        if i + 1 == n:
            y1s[i] = min(h, _ink_bottom(bw, x0, x1, bottom, min(h, bottom + int(12 * sp)), sp) + pad)
            continue
        next_top = int(groups[i + 1][0].top)
        # Lyrics end about 4.7 spacings below the staff, the next system's clef starts about 1.6 above its staff: look
        # for the blank rows between them, and cut through the run nearest to where the lyrics end.
        lo = min(bottom + int(CROP_LYRICS_END * sp * 0.9), next_top - 2)
        hi = max(lo + 1, next_top - int(CROP_NEXT_GAP * sp))
        ink = (bw[lo:hi, x0:x1] > 0).sum(axis=1)
        blank = np.nonzero(ink == 0)[0]
        if len(blank):
            runs = np.split(blank, np.nonzero(np.diff(blank) > 1)[0] + 1)
            run = min(runs, key=lambda r: abs(lo + int(r[0]) - (bottom + CROP_LYRICS_END * sp)))
            start, end = lo + int(run[0]), lo + int(run[-1]) + 1
            mid = (start + end) // 2
            last = _ink_bottom(bw, x0, x1, bottom, max(bottom + 1, start), sp)
            y1s[i] = min(last + pad, mid)
            # the next box starts at its own clef / high notes, not at the end of the blank run (prose may lie between)
            nsp = groups[i + 1][0].spacing
            y0s[i + 1] = max(end - pad, mid, _ink_top(bw, xs[i + 1][0], xs[i + 1][1], next_top, y1s[i], nsp) - int(CROP_PAD * nsp))
        else:  # the ink touches everywhere: cut through the emptiest row
            y1s[i] = y0s[i + 1] = lo + int(np.argmin(ink))
    return [(xs[i][0], int(max(0, y0s[i])), xs[i][1], int(min(h, y1s[i]))) for i in range(n)]
