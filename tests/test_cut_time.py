import numpy as np

from music_reader.cv.symbols import Comp, _is_cut_time


def component(mask):
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = mask[y0:y1, x0:x1].astype(bool)
    return Comp(1, x0, y0, x1, y1, int(cropped.sum()), cropped)


def test_cut_time_has_a_long_central_vertical_stroke():
    # Simplified connected glyph at 2 staff spacings wide and 4 high.
    mask = np.zeros((20, 10), dtype=bool)
    mask[1:3, 1:9] = True
    mask[17:19, 1:9] = True
    mask[1:19, 1:3] = True
    mask[2:18, 4:6] = True  # the cut-time stroke
    assert _is_cut_time(component(mask), sp=5.0)


def test_common_time_c_glyph_without_cut_stroke_is_not_cut_time():
    mask = np.zeros((20, 10), dtype=bool)
    mask[1:3, 1:9] = True
    mask[17:19, 1:9] = True
    mask[1:19, 1:3] = True
    assert not _is_cut_time(component(mask), sp=5.0)
