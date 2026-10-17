import numpy as np

from music_reader.cv.symbols import Comp, _is_cut_time


def component(mask):
    ys, xs = np.nonzero(mask)
    return Comp(1, int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1,
                int(mask.sum()), mask.astype(bool))


def test_cut_time_has_a_long_central_vertical_stroke():
    # Simplified connected glyph at 2 staff spacings wide and 4 high.
    mask = np.zeros((20, 10), dtype=bool)
    mask[1:3, 1:9] = True
    mask[17:19, 1:9] = True
    mask[1:19, 1:3] = True
    mask[:, 4:6] = True  # the cut-time stroke
    assert _is_cut_time(component(mask), sp=5.0)


def test_common_time_c_glyph_without_cut_stroke_is_not_cut_time():
    mask = np.zeros((20, 10), dtype=bool)
    mask[1:3, 1:9] = True
    mask[17:19, 1:9] = True
    mask[1:19, 1:3] = True
    assert not _is_cut_time(component(mask), sp=5.0)
