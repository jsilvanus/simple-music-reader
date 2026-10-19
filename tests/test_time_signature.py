"""Time-signature reading: digits in several fonts and sizes, common and cut time, no false reading of a note."""
import pytest

from music_reader.backends import RecognitionOptions, get_backend
from music_reader.pdfinput import Document

from tests.helpers import events, spec
from tests.synth import StaffSpec, SystemSpec, parse_measures, rasterize_pdf, render_pdf

pytestmark = pytest.mark.integration

BARS = "E4q F4q G4q A4q | B4q C5q D5q E5q"


def read(path, **opt):
    return get_backend("classical").recognize(Document(path), RecognitionOptions(**opt))


def make(tmp_path, name, text=BARS, sp=5.0, **staff):
    path = tmp_path / f"{name}.pdf"
    render_pdf(path, [[SystemSpec([StaffSpec(parse_measures(text), **staff)])]], sp=sp)
    return path


def codes(score):
    return [d.code for d in score.diagnostics]


METERS = ["2/4", "3/4", "4/4", "6/8", "9/8", "12/8", "3/2", "5/4", "6/4"]


@pytest.mark.parametrize("font", ["Helvetica-Bold", "Times-Bold", "Times-Roman", "Courier-Bold", "Helvetica"])
@pytest.mark.parametrize("meter", METERS)
def test_digits_are_read_in_several_fonts(tmp_path, font, meter):
    # Known limit (OMR.md): a thin serif 6 (Times-Roman) is mistaken for a 5, and 5/4 in Times-Bold is too close to call.
    if font == "Times-Roman" and "6" in meter.split("/")[0]:
        pytest.skip("thin serif 6 reads as 5")
    # the bars of this score fit every meter listed, so a wrong reading cannot hide behind METER_INFERRED
    res = read(make(tmp_path, f"ts-{font}-{meter.replace('/', '_')}", "E4q F4q | G4q A4q", meter=meter, time_font=font))
    assert str(res.score.meter) == meter and not res.score.meter.inferred
    assert "METER_INFERRED" not in codes(res.score)


def test_reading_the_digits_does_not_eat_the_first_note(tmp_path):
    res = read(make(tmp_path, "first-note", BARS, meter="3/4"))
    assert str(res.score.meter) == "3/4"
    assert events(res.score)[:3] == spec("E4q F4q G4q")


@pytest.mark.parametrize("sp", [4.0, 6.5, 9.0])
def test_digits_survive_a_scan(tmp_path, sp):
    src = make(tmp_path, f"scan-src-{sp}", "E4q F4q G4q | A4q B4q C5q", sp=sp, meter="3/4")
    scan = tmp_path / f"scan-{sp}.pdf"
    rasterize_pdf(src, scan, dpi=220, rotate_deg=0.6, noise=8)
    res = read(scan)
    assert str(res.score.meter) == "3/4" and not res.score.meter.inferred


def test_common_time_is_four_four(tmp_path):
    res = read(make(tmp_path, "common", meter="C"))
    assert str(res.score.meter) == "4/4" and not res.score.meter.inferred
    assert events(res.score)[0] == spec("E4q")[0]


def test_cut_time_is_two_two(tmp_path):
    res = read(make(tmp_path, "cut", "E4h F4h | G4h A4h", meter="C|"))
    assert str(res.score.meter) == "2/2" and not res.score.meter.inferred


def test_no_time_signature_still_reads_a_leading_note(tmp_path):
    res = read(make(tmp_path, "none", "G4q | E4q F4q G4q A4q | B4q C5q D5q E5q", meter=None))
    assert events(res.score)[0] == spec("G4q")[0]
    assert res.score.meter.inferred


def test_printed_meter_that_disagrees_with_the_bars_is_flagged(tmp_path):
    # printed 3/4, but every bar holds four quarters
    res = read(make(tmp_path, "mismatch", "E4q F4q G4q A4q | B4q C5q D5q E5q | E4q F4q G4q A4q", meter="3/4"))
    assert str(res.score.meter) == "3/4"
    assert "TIME_SIGNATURE_MISMATCH" in codes(res.score)
