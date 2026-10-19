"""Beam groups of three or four, slanted beams, sixteenth beams and dotted pairs (generated scores)."""
import pytest

from music_reader.backends import RecognitionOptions, get_backend
from music_reader.pdfinput import Document

from tests.helpers import events, spec
from tests.synth import StaffSpec, SystemSpec, parse_measures, rasterize_pdf, render_pdf

pytestmark = pytest.mark.integration


def bars(bar, n=4):
    return " | ".join([bar] * n)


CASES = {
    "flat-four-eighths": (bars("C5e D5e E5e F5e G5h"), dict(beam_group_size=4)),
    "rising-four-eighths": (bars("C4e E4e G4e C5e C5h"), dict(beam_group_size=4, max_slope=0.3)),
    "falling-four-eighths": (bars("C6e G5e E5e C5e C5h"), dict(beam_group_size=4, max_slope=0.3)),
    "three-eighths-stems-down": (bars("E5e D5e B4e re A4h"), dict(beam_group_size=3, max_slope=0.3)),
    "three-eighths-stems-up": (bars("C4e D4e E4e re G4h"), dict(beam_group_size=3, max_slope=0.3)),
    "four-sixteenths": (bars("C5s B4s A4s G4s E4q E4h"), dict(beam_group_size=4, max_slope=0.3)),
    "rising-sixteenths": (bars("E4s G4s B4s D5s E4q E4h"), dict(beam_group_size=4, max_slope=0.3)),
    "eighth-and-two-sixteenths": (bars("C5e B4s A4s G4q E4h"), dict(beam_group_size=3, max_slope=0.3)),
    "dotted-eighth-and-sixteenth": (bars("E4e. F4s G4q E4h"), dict(beam_group_size=2, max_slope=0.3)),
    "sixteenth-eighth-sixteenth": (bars("F4s G4e A4s G4q E4h"), dict(beam_group_size=3, max_slope=0.3)),
}


def read(path):
    return get_backend("classical").recognize(Document(path), RecognitionOptions(meter="4/4"))


@pytest.mark.parametrize("name", CASES)
def test_beamed_groups(tmp_path, name):
    text, kw = CASES[name]
    path = tmp_path / f"{name}.pdf"
    render_pdf(path, [[SystemSpec([StaffSpec(parse_measures(text), **kw)])]], sp=5.0)
    res = read(path)
    assert events(res.score) == spec(text)
    assert not res.score.has_errors()


@pytest.mark.parametrize("name", ["rising-four-eighths", "four-sixteenths", "three-eighths-stems-down"])
def test_beamed_groups_in_a_scan(tmp_path, name):
    text, kw = CASES[name]
    src = tmp_path / f"{name}.pdf"
    render_pdf(src, [[SystemSpec([StaffSpec(parse_measures(text), **kw)])]], sp=5.0)
    scan = tmp_path / f"{name}-scan.pdf"
    rasterize_pdf(src, scan, dpi=200, rotate_deg=0.5, noise=6)
    assert events(read(scan).score) == spec(text)
