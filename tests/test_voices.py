"""Two voices on one staff (soprano stems up, alto stems down): the voice option picks which one is the melody."""
import pytest

from music_reader.backends import RecognitionOptions, get_backend
from music_reader.cli import main
from music_reader.pdfinput import Document

from tests.helpers import events, spec
from tests.synth import StaffSpec, SystemSpec, parse_measures, render_pdf

pytestmark = pytest.mark.integration

# soprano / alto, thirds and wider, same rhythm
TWO = "G4/E4q A4/F4q B4/G4h | C5/A4q D5/B4q E5/C5h | D5/G4q C5/E4q B4/G4h"
SOPRANO = "G4q A4q B4h C5q D5q E5h D5q C5q B4h"
ALTO = "E4q F4q G4h A4q B4q C5h G4q E4q G4h"


@pytest.fixture(scope="module")
def two_voices(tmp_path_factory):
    path = tmp_path_factory.mktemp("voices") / "two.pdf"
    render_pdf(path, [[SystemSpec([StaffSpec(parse_measures(TWO))])]], sp=5.0, lyrics=False)
    return path


def read(path, **opt):
    return get_backend("classical").recognize(Document(path), RecognitionOptions(meter="4/4", **opt))


def test_top_voice_is_the_default_melody(two_voices):
    res = read(two_voices)
    assert events(res.score) == spec(SOPRANO)
    assert [d.code for d in res.score.diagnostics].count("CHORD_OR_POLYPHONY") == 1  # one note, not one per column


def test_bottom_voice_can_be_chosen(two_voices):
    assert events(read(two_voices, voice="bottom").score) == spec(ALTO)


def test_cli_voice_option(two_voices, tmp_path):
    out = tmp_path / "alto.abc"
    assert main([str(two_voices), "--meter", "4/4", "--voice", "bottom", "-o", str(out)]) == 0
    assert out.read_text().count("E") >= 1 and "CHORD_OR_POLYPHONY" in out.read_text()


def test_single_voice_is_untouched(tmp_path):
    path = tmp_path / "one.pdf"
    render_pdf(path, [[SystemSpec([StaffSpec(parse_measures(SOPRANO))])]], sp=5.0)
    res = read(path)
    assert "CHORD_OR_POLYPHONY" not in [d.code for d in res.score.diagnostics]
