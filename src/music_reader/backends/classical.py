"""Classical computer-vision backend: staff geometry, line removal, connected components, rules.

Deterministic, runs locally, no ML. See OMR.md for what it can and cannot read.
"""
from __future__ import annotations

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
        dbg.system_boxes = _system_boxes(groups, gray.shape)
        return info, dbg, systems


# Vertical limits of a system crop, in staff spacings beyond the outer staff lines: room for ledger-line notes and
# clefs above, one line of lyrics (plus its descenders) below.
CROP_ABOVE = 3.5
CROP_BELOW = 6.5


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


def _system_boxes(groups, shape) -> list[tuple[int, int, int, int]]:
    """Crop box for each system: full staff width, from halfway to the system above to halfway to the one below,
    but at most CROP_ABOVE / CROP_BELOW staff spacings beyond the staff, so lyrics under the staff stay with their
    system while headings and prose between two systems do not."""
    h, w = shape
    boxes = []
    for i, g in enumerate(groups):
        sp = g[0].spacing
        top, bottom = g[0].top, g[-1].bottom
        prev_b = groups[i - 1][-1].bottom if i else None
        next_t = groups[i + 1][0].top if i + 1 < len(groups) else None
        y0 = (prev_b + top) / 2 if prev_b is not None else top - CROP_ABOVE * sp
        y1 = (bottom + next_t) / 2 if next_t is not None else bottom + CROP_BELOW * sp
        y0 = max(y0, top - CROP_ABOVE * sp)
        y1 = min(y1, bottom + CROP_BELOW * sp)
        x0 = min(s.x0 for s in g) - 3 * sp
        x1 = max(s.x1 for s in g) + 1.5 * sp
        boxes.append((int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))))
    return boxes
