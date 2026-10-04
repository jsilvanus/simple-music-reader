"""Deterministic unit tests of the CV stages on generated pages (no ML, no network)."""
import numpy as np
import pytest

from music_reader.cv import staff as cvstaff
from music_reader.pdfinput import Document

SCORE = "E4q G4q B4q D5q | C5h A4q rq | E4q F4e G4e A4h"


def page(make_pdf, name="cv", text=SCORE, sp=5.0, dpi=200, **kw):
    doc = Document(make_pdf(name, text, sp=sp, **kw))
    gray, _ = doc.render(0, dpi)
    return gray, cvstaff.binarize(gray)


@pytest.mark.parametrize("sp,dpi", [(4.0, 250), (5.0, 200), (6.5, 150), (8.0, 120)])
def test_staff_detection_is_scale_invariant(make_pdf, sp, dpi):
    _, bw = page(make_pdf, f"scale-{sp}", sp=sp, dpi=dpi)
    staves, _ = cvstaff.detect_staves(bw)
    assert len(staves) == 1
    s = staves[0]
    assert s.spacing == pytest.approx(sp * dpi / 72, rel=0.03)
    assert np.allclose(np.diff(s.lines), s.spacing, atol=1.0)


def test_two_systems_and_grouping(make_pdf, pdfdir):
    from tests.synth import StaffSpec, SystemSpec, parse_measures, render_pdf
    m = parse_measures(SCORE)
    path = pdfdir / "two.pdf"
    render_pdf(path, [[SystemSpec([StaffSpec(m), StaffSpec(m)]), SystemSpec([StaffSpec(m)])]])
    _, bw = (lambda g: (g, cvstaff.binarize(g)))(Document(path).render(0, 200)[0])
    staves, _ = cvstaff.detect_staves(bw)
    systems = cvstaff.group_systems(staves, bw)
    assert [len(g) for g in systems] == [2, 1]
    assert [s.index for s in systems[0]] == [0, 1] and staves[2].system == 1


def test_staff_position_maps_lines_and_spaces(make_pdf):
    _, bw = page(make_pdf, "pos")
    s = cvstaff.detect_staves(bw)[0][0]
    assert s.position(s.bottom) == pytest.approx(0)
    assert s.position(s.top) == pytest.approx(8, abs=0.05)
    assert s.position((s.lines[0] + s.lines[1]) / 2) == pytest.approx(7, abs=0.05)
    assert s.y_of(2) == pytest.approx(s.lines[3], abs=0.5)


def test_line_removal_keeps_symbols_and_clears_lines(make_pdf):
    _, bw = page(make_pdf)
    staves, lm = cvstaff.detect_staves(bw)
    res = cvstaff.remove_staff_lines(bw, staves, lm)
    s = staves[0]
    inside = res[int(s.top) + 2 : int(s.bottom) - 2, s.x0 : s.x1]
    assert res.sum() < bw.sum() * 0.8  # staff lines are gone
    mid = int(s.lines[2])
    # no long horizontal line left on the middle line
    row = res[mid, s.x0 : s.x1] > 0
    longest = max((len(r) for r in "".join("1" if v else "0" for v in row).split("0")), default=0)
    assert longest < 3 * s.spacing
    assert inside.sum() > 0  # note heads survive


def test_deskew_estimate(make_pdf):
    gray, bw = page(make_pdf)
    assert abs(cvstaff.estimate_skew(bw)) < 0.15
    rotated = cvstaff.rotate(gray, 1.2, border=255)
    est = cvstaff.estimate_skew(cvstaff.binarize(rotated))
    assert est == pytest.approx(-1.2, abs=0.2)


def test_blank_page_has_no_staves():
    bw = np.zeros((800, 600), np.uint8)
    assert cvstaff.detect_staves(bw)[0] == []


def test_ledger_lines_detected(make_pdf):
    gray, bw = page(make_pdf, "ledger", "C4q A3q C6q A5q | G5w", sp=5.0)
    staves, lm = cvstaff.detect_staves(bw)
    res = cvstaff.remove_staff_lines(bw, staves, lm)
    _, ledgers = cvstaff.remove_ledger_lines(res, staves)
    positions = sorted({round(staves[0].position(l.y)) for l in ledgers})
    assert positions == [-4, -2, 10, 12]


def test_system_crop_is_limited_to_staff_and_lyrics():
    from types import SimpleNamespace as NS
    from music_reader.backends.classical import _system_boxes

    def staff(top):  # spacing 10 px
        return NS(spacing=10.0, top=top, bottom=top + 40, x0=100, x1=900)

    # two systems 600 px apart: the midpoint box would swallow the prose between them
    boxes = _system_boxes([[staff(100)], [staff(700)]], (1000, 1000))
    assert boxes[0][1] == 100 - 35          # 3.5 spacings above the first staff
    assert boxes[0][3] == 140 + 65          # bottom of staff + 6.5 spacings
    assert boxes[1][1] == 700 - 35          # 3.5 spacings above, not halfway to the system above


def test_short_staff_is_found_next_to_a_full_width_rule():
    import numpy as np
    from music_reader.cv.staff import detect_staves

    bw = np.zeros((400, 1000), dtype=np.uint8)
    bw[40:42, 100:900] = 255                      # heading rule, much wider than the staff
    for k in range(5):
        bw[200 + 12 * k : 202 + 12 * k, 100:350] = 255  # short staff, spacing 12 px
    staves, _ = detect_staves(bw)
    assert len(staves) == 1 and abs(staves[0].spacing - 12) < 0.5
