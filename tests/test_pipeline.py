"""Multi-page, multi-system, melody selection, crops, debug output, CLI, JSON."""
import json
from fractions import Fraction as F

import cv2
import pytest

from music_reader.backends import RecognitionOptions, get_backend
from music_reader.cli import main
from music_reader.crops import write_crops
from music_reader.debug import write_debug
from music_reader.pdfinput import Document

from tests.helpers import events, measures, parse_abc, spec
from tests.synth import StaffSpec, SystemSpec, parse_measures, render_pdf

pytestmark = pytest.mark.integration

A = "E4q G4q B4q D5q | C5h A4q rq"
B = "E4q F4e G4e A4h | G4w"
C = "D4h F4h | E4w"


@pytest.fixture(scope="module")
def multi(pdfdir):
    """Two pages: page 1 has two systems, page 2 has one."""
    path = pdfdir / "multi.pdf"
    st = lambda t, **k: StaffSpec(parse_measures(t), **k)  # noqa: E731
    render_pdf(path, [[SystemSpec([st(A)]), SystemSpec([st(B)])], [SystemSpec([st(C, final=True)])]])
    return path


def read(path, **opt):
    return get_backend("classical").recognize(Document(path), RecognitionOptions(**opt))


def test_multiple_systems_and_pages_are_read_in_order(multi):
    res = read(multi, meter="4/4")
    assert events(res.score) == spec(A + " " + B + " " + C)
    assert [p.number for p in res.score.pages] == [1, 2]
    assert [len(p.systems) for p in res.score.pages] == [2, 1]
    assert len(res.score.melody.measures) == 6
    # measure indices run on across systems and pages; every event remembers where it came from
    assert [e.source.page for e in res.score.melody.events()][-1] == 2
    assert {e.source.system for e in res.score.melody.events() if e.source.page == 1} == {0, 1}
    assert not res.score.has_errors()


def test_page_selection(multi):
    res = read(multi, pages=[2], meter="4/4")
    assert events(res.score) == spec(C)
    assert [p.number for p in res.score.pages] == [2]


def test_source_information_is_preserved(multi):
    res = read(multi)
    e = next(res.score.melody.events())
    assert e.source.page == 1 and e.source.staff == 0 and e.source.bbox is not None
    x0, y0, x1, y1 = e.source.bbox
    assert x1 > x0 and y1 > y0 and 0 < e.confidence <= 1


@pytest.fixture(scope="module")
def two_staves(pdfdir):
    path = pdfdir / "grand.pdf"
    mel, acc = "E4q G4q B4q D5q | C5w", "C4h E4h | G3w"
    render_pdf(path, [[SystemSpec([StaffSpec(parse_measures(mel)), StaffSpec(parse_measures(acc))])]])
    return path


def test_ambiguous_melody_staff_is_reported_not_hidden(two_staves):
    res = read(two_staves, meter="4/4")
    codes = [d.code for d in res.score.diagnostics]
    assert "AMBIGUOUS_MELODY_STAFF" in codes
    assert res.score.selection["ambiguous"] and res.score.selection["strategy"] == "assumed-first-staff"
    assert events(res.score) == spec("E4q G4q B4q D5q C5w")
    assert "% melody source" in __import__("music_reader.export", fromlist=["export"]).export(res.score, "abc")


def test_user_selects_the_melody_staff(two_staves):
    res = read(two_staves, staff=2, meter="4/4")
    assert "AMBIGUOUS_MELODY_STAFF" not in [d.code for d in res.score.diagnostics]
    assert res.score.selection == {"strategy": "explicit", "staff": 2, "ambiguous": False, "description": "staff 2 of each system"}
    assert events(res.score) == spec("C4h E4h G3w")


def test_missing_staff_is_an_error(two_staves):
    res = read(two_staves, staff=3)
    assert "STAFF_NOT_FOUND" in [d.code for d in res.score.diagnostics] and res.score.has_errors()


def test_crops_cover_every_system(multi, tmp_path):
    doc = Document(multi)
    res = get_backend("classical").recognize(doc, RecognitionOptions())
    manifest = write_crops(res, doc, tmp_path, dpi=300)
    assert [(m["page"], m["system"]) for m in manifest] == [(1, 1), (1, 2), (2, 1)]
    for m in manifest:
        img = cv2.imread(str(tmp_path / m["file"]), cv2.IMREAD_GRAYSCALE)
        assert img is not None and img.shape[1] > img.shape[0] * 3  # a wide strip of one system
        assert m["dpi"] == 300 and 0 <= m["bbox_norm"][0] < m["bbox_norm"][2] <= 1
        assert (img < 128).mean() > 0.01  # has ink
    assert json.loads((tmp_path / "crops.json").read_text()) == manifest
    # a 300 dpi crop of the same system is bigger than the one at recognition resolution
    low = write_crops(res, doc, tmp_path / "low")
    assert manifest[0]["bbox_px"][2] > low[0]["bbox_px"][2] * 1.1 or res.pages[0].dpi >= 300


def test_crop_keeps_lyrics_with_their_system(multi, tmp_path):
    doc = Document(multi)
    res = get_backend("classical").recognize(doc, RecognitionOptions())
    box = res.pages[0].system_boxes[0]
    staff = res.pages[0].staves[0]
    assert box[1] < staff.top and box[3] > staff.bottom + 4 * staff.spacing  # lyric line sits below the staff


def test_debug_output(multi, tmp_path):
    res = get_backend("classical").recognize(Document(multi), RecognitionOptions())
    paths = write_debug(res, tmp_path)
    assert {p.name for p in paths} >= {"page-1.png", "page-2.png", "debug.pdf"}
    assert (tmp_path / "diagnostics.txt").exists()
    img = cv2.imread(str(tmp_path / "page-1.png"))
    assert img.ndim == 3 and (img[:, :, 0] != img[:, :, 2]).any()  # coloured annotations


def test_cli_abc_to_file_and_json(multi, tmp_path, capsys):
    out = tmp_path / "song.abc"
    assert main([str(multi), "-o", str(out), "--meter", "4/4", "--title", "Test"]) == 0
    abc = out.read_text()
    assert "T:Test" in abc and "M:4/4" in abc and "K:C" in abc
    assert [(p, d) for p, d, _ in parse_abc(abc)] == spec(A + " " + B + " " + C)
    assert main([str(multi), "--format", "json", "--meter", "4/4"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["meter"] == {"numerator": 4, "denominator": 4, "inferred": False} and len(data["voices"][0]["measures"]) == 6
    first = data["voices"][0]["measures"][0]["events"][0]
    assert first["pitch"] == "E4" and first["duration"] == "1/4" and first["source"]["page"] == 1


def test_cli_musicxml_by_extension_and_debug_and_crops(multi, tmp_path):
    out = tmp_path / "song.musicxml"
    assert main([str(multi), "-o", str(out), "--meter", "4/4", "--debug", str(tmp_path / "dbg"), "--crops", str(tmp_path / "crops")]) == 0
    assert out.read_text().startswith("<?xml")
    assert (tmp_path / "dbg" / "page-1.png").exists() and (tmp_path / "crops" / "crops.json").exists()


def test_cli_strict_exit_code(multi, capsys):
    assert main([str(multi), "--meter", "3/4", "--strict"]) == 2  # 4/4 music read as 3/4: overfull measures
    assert main([str(multi), "--meter", "3/4"]) == 0
    capsys.readouterr()


def test_image_input(make_pdf, tmp_path):
    gray, _ = Document(make_pdf("imgin", "E4q G4q B4q D5q | C5w")).render(0, 200)
    png = tmp_path / "scan.png"
    cv2.imwrite(str(png), gray)
    res = read(png, meter="4/4")
    assert events(res.score) == spec("E4q G4q B4q D5q C5w") and res.score.pages[0].kind == "image"


def test_too_low_resolution_is_flagged(make_pdf):
    res = read(make_pdf("tiny", "E4q G4q B4q D5q | C5w"), dpi=30)
    codes = [d.code for d in res.score.diagnostics]
    assert "NO_STAVES" in codes or "LOW_RESOLUTION" in codes


def test_bundle_for_downstream_consumers(multi, tmp_path):
    assert main([str(multi), "--meter", "4/4", "--bundle", str(tmp_path / "b"), "-o", str(tmp_path / "x.abc")]) == 0
    b = tmp_path / "b"
    assert {p.name for p in b.iterdir()} == {"melody.abc", "melody.musicxml", "score.json", "crops", "manifest.json"}
    m = json.loads((b / "manifest.json").read_text())
    assert m["source"]["file"] == "multi.pdf" and len(m["source"]["sha256"]) == 64
    assert [(c["page"], c["system"]) for c in m["crops"]] == [(1, 1), (1, 2), (2, 1)]
    assert m["melody"]["measures"] == 6 and m["melody"]["meter"] == "4/4" and m["pages"][0]["kind"] == "vector"
    assert (b / "crops" / m["crops"][0]["file"]).exists()
    assert m["needs_review"] is False
