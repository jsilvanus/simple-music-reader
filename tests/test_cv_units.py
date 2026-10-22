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



def test_system_crop_keeps_a_second_volta_lyric_line():
    import numpy as np
    from types import SimpleNamespace as NS
    from music_reader.backends.classical import _system_boxes

    bw = np.zeros((420, 600), dtype=np.uint8)
    for top in (100, 300):
        for k in range(5):
            bw[top + 10 * k, 100:500] = 255
        bw[top - 14 : top + 41, 110] = 255  # clef-like stroke
    bw[158:164, 150:390] = 255              # first lyric line
    bw[184:192, 190:430] = 255              # alternate/volta lyric line

    def staff(top):
        return NS(spacing=10.0, top=top, bottom=top + 40, x0=100, x1=500)

    first, second = _system_boxes([[staff(100)], [staff(300)]], bw.shape, bw)
    assert first[3] >= 192
    assert second[1] > 192
    assert second[1] <= 300 - 14 + 1


def test_short_staff_is_found_next_to_a_full_width_rule():
    import numpy as np
    from music_reader.cv.staff import detect_staves

    bw = np.zeros((400, 1000), dtype=np.uint8)
    bw[40:42, 100:900] = 255                      # heading rule, much wider than the staff
    for k in range(5):
        bw[200 + 12 * k : 202 + 12 * k, 100:350] = 255  # short staff, spacing 12 px
    staves, _ = detect_staves(bw)
    assert len(staves) == 1 and abs(staves[0].spacing - 12) < 0.5


def test_system_crops_follow_the_ink_in_tight_layouts():
    import numpy as np
    from types import SimpleNamespace as NS
    from music_reader.backends.classical import _system_boxes

    bw = np.zeros((300, 600), dtype=np.uint8)
    for top in (100, 200):
        for k in range(5):
            bw[top + 10 * k, 100:500] = 255          # staff lines, spacing 10
        bw[top - 14 : top + 41, 110] = 255           # clef-like stroke standing on the staff
    bw[158:171, 150:400] = 255                       # lyrics of system 1 ...
    bw[171:179, 200:204] = 255                       # ... with a descender (the "j")
    bw[60:70, 150:400] = 255                         # prose well above system 1, must stay out

    def staff(top):
        return NS(spacing=10.0, top=top, bottom=top + 40, x0=100, x1=500)

    first, second = _system_boxes([[staff(100)], [staff(200)]], bw.shape, bw)
    assert first[3] >= 179                           # the descender is inside the first crop
    assert second[1] > 178                           # ... and not repeated at the top of the second
    assert second[1] <= 200 - 14 + 1                 # the second crop still starts above its clef
    assert first[1] > 70                             # the prose above stays out

def test_classify_accidental_proportions_and_split_merging():
    from music_reader.cv.symbols import Comp, classify_accidental, _try_merge_accidental
    sp = 10.0
    h = 32  # 3.2 sp tall sharp
    w = 8
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 1:3] = True
    mask[:, 5:7] = True
    mask[8:12, :] = True
    mask[20:24, :] = True
    c = Comp(1, 10, 10, 10 + w, 10 + h, int(mask.sum()), mask)
    assert classify_accidental(c, sp) == "#"

    # Non-flat with top-right ink (e.g. digit 4 or 3) should not be classified as flat
    mask_digit = np.zeros((26, 12), dtype=bool)
    mask_digit[:, 1:3] = True
    mask_digit[2:8, 4:10] = True
    mask_digit[14:24, 2:10] = True
    c_digit = Comp(2, 10, 10, 22, 36, int(mask_digit.sum()), mask_digit)
    assert classify_accidental(c_digit, sp) is None

    # Split sharp components merge into #
    left_mask = np.zeros((32, 3), dtype=bool)
    left_mask[:, 1:3] = True
    c_left = Comp(3, 10, 10, 13, 42, int(left_mask.sum()), left_mask)

    right_mask = np.zeros((32, 4), dtype=bool)
    right_mask[:, 1:3] = True
    c_right = Comp(4, 15, 10, 19, 42, int(right_mask.sum()), right_mask)

    res = _try_merge_accidental(c_left, c_right, sp)
    assert res is not None
    assert res[0] == "#"

    # Split flat (stem + bowl fragment)
    stem_mask = np.zeros((27, 3), dtype=bool)
    stem_mask[:, 0:2] = True
    c_stem = Comp(5, 10, 10, 13, 37, int(stem_mask.sum()), stem_mask)

    bowl_mask = np.zeros((12, 7), dtype=bool)
    bowl_mask[:, :] = True
    c_bowl = Comp(6, 14, 25, 21, 37, int(bowl_mask.sum()), bowl_mask)

    res_flat = _try_merge_accidental(c_stem, c_bowl, sp)
    assert res_flat is not None
    assert res_flat[0] == "b"


def test_head_like_orientation_accepts_tilted_heads():
    import cv2
    from music_reader.cv.symbols import _head_like_orientation
    # Create an ellipse tilted at 45 degrees
    region = np.zeros((40, 40), dtype=np.uint8)
    cv2.ellipse(region, (20, 20), (10, 6), 45, 0, 360, 255, -1)
    assert _head_like_orientation(region)



def test_find_holes_accepts_tilted_hollow_head():
    import cv2
    from music_reader.cv.symbols import find_holes
    sp = 10.0
    bw = np.zeros((100, 100), dtype=np.uint8)
    # Ring of a hollow notehead tilted at 35 degrees
    cv2.ellipse(bw, (50, 50), (6, 4), 35, 0, 360, 255, 2)
    holes = find_holes(bw, sp)
    assert len(holes) == 1
    assert abs(holes[0].cx - 50) <= 1.0
    assert abs(holes[0].cy - 50) <= 1.0


def test_flag_hole_rejected_on_stem_with_solid_head():
    from music_reader.cv.staff import Staff
    from music_reader.cv.symbols import analyze_staff, Comp, Head
    sp = 10.0
    staff = Staff([40.0, 50.0, 60.0, 70.0, 80.0], 0, 200, sp, 1.0)
    clef_comp = Comp(1, 10, 40, 25, 80, 600, np.ones((40, 15), dtype=bool))
    # An eighth note stem with a solid head at bottom and a flag loop hole at top
    stem_mask = np.zeros((30, 14), dtype=bool)
    stem_mask[:, 10:13] = True
    stem_mask[20:30, 0:12] = True
    stem_comp = Comp(10, 100, 45, 114, 75, int(stem_mask.sum()), stem_mask)
    solid_head = Head(105.0, 70.0, 100.0, 65.0, 110.0, 75.0, 1.0, False)
    flag_hole = Head(111.0, 48.0, 108.0, 45.0, 114.0, 51.0, 0.6, True)

    import music_reader.cv.symbols as cvsym
    orig_find_heads = cvsym.find_heads
    try:
        cvsym.find_heads = lambda c, sp: [solid_head] if c.id == stem_comp.id else []
        r = analyze_staff(staff, [clef_comp, stem_comp], [], 1, 0, [flag_hole], None, 'top')
        notes = [it for it in r.items if it.kind == 'note']
        assert len(notes) == 1
        assert not notes[0].hollow
    finally:
        cvsym.find_heads = orig_find_heads


def test_split_hole_across_staff_line_is_paired():
    from music_reader.cv.symbols import find_holes
    sp = 10.0
    bw = np.zeros((100, 100), dtype=np.uint8)
    # A black staff line at y=48..50
    bw[48:50, :] = 255
    # A ring of a hollow notehead on line 48..50:
    # Outer ellipse filled with 255, inner ellipse with 0 (white hole)
    # The staff line crosses the inner white hole, cutting it into two halves
    import cv2
    cv2.ellipse(bw, (50, 49), (7, 5), 35, 0, 360, 255, 2)
    # Line still crosses inside:
    bw[48:50, 45:55] = 255
    holes = find_holes(bw, sp)
    assert len(holes) == 1
    assert abs(holes[0].cx - 50) <= 2.0
    assert abs(holes[0].cy - 49) <= 2.5


def test_on_beam_rejects_junction_blob_touching_beam():
    from music_reader.cv.symbols import _on_beam, Head
    beams = np.zeros((100, 100), dtype=np.uint8)
    # A horizontal beam at y=20..26, x=30..80
    beams[20:26, 30:80] = 255
    # A junction blob from stem at x=50, y=25..34 (cy=29.5)
    blob_head = Head(50.0, 29.5, 46.0, 25.0, 54.0, 34.0, 1.0, False)
    assert _on_beam(beams, blob_head)
