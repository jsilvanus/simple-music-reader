# simple-music-reader

Extract the **melody** of simple printed church sheet music (hymns, liturgical chants) from a PDF and give it back in a
clean, machine-readable form. It also cuts out the **notation as printed**, one image per system.

```
PDF ──► page analysis ──► render ──► staff detection ──► symbols ──► Music IR ──► ABC / MusicXML / JSON
 │                                         │
 └─ vector or scan?                        └─► per-system images (crops) + debug overlays
```

It was built for [anno-api](https://github.com/jsilvanus/anno-api) (the church year of the Evangelical-Lutheran Church of
Finland) with two goals in mind:

| Goal | What this project delivers |
|---|---|
| a) images of the liturgy notation | `--crops DIR`: one PNG per detected system (lyrics stay with their system), plus `crops.json` with page/system numbers and bounding boxes. Works from the staff geometry alone, so it does not depend on the melody being read correctly. |
| b) notation that can be re-rendered in flexible layouts | `--format abc` / `musicxml` / `json`: the recognised melody as exact pitches and durations, with confidence and source positions. |

`--bundle DIR` writes everything for one piece in one go (`melody.abc`, `melody.musicxml`, `score.json`, `crops/`,
`manifest.json`). The manifest says whether the result `needs_review`, and why.

**Status: first vertical slice.** It reads simple monophonic printed melodies on a treble staff and has been tested on
*generated* scores only (no real church PDF was available when it was written). Read [OMR.md](OMR.md) for what it can
and cannot do, and [ROADMAP.md](ROADMAP.md) for what comes next.

## Install and run

Python 3.10+.

```bash
pip install -e ".[test]"

music-reader hymn.pdf                         # ABC on standard output
music-reader hymn.pdf -o hymn.abc --meter 3/4 --key G
music-reader hymn.pdf -o hymn.musicxml
music-reader hymn.pdf --format json
music-reader hymn.pdf --debug debug/          # annotated pages, debug.pdf, diagnostics.txt
music-reader hymn.pdf --crops crops/ --crop-dpi 300
music-reader hymn.pdf --bundle out/hymn1 --staff 1
```

Useful options: `--staff N` (which staff of each system carries the melody), `--meter`, `--key`, `--pages 1-3`,
`--dpi`, `--strict` (exit status 2 on error-level diagnostics). Time-signature **digits are not read yet**: the meter is
inferred from the measure lengths unless you give `--meter`, and a warning says so.

Warnings go to standard error and, for ABC, into `%` comment lines at the top of the output. Nothing uncertain is
corrected silently.

## Principles

* The Music IR (`src/music_reader/ir.py`) is the core. ABC is just one exporter.
* Exact (rational) durations; semantic pitches (`F#4`), never pixels, plus source page/bbox/staff/confidence.
* Geometry is measured in staff spacings, never in fixed pixel sizes.
* Deterministic classical computer vision, no ML model, no network. Heavy OMR engines can be added behind
  `backends/base.py`.
* Validate musical structure and report problems instead of fixing them.

## Documentation

[ARCHITECTURE.md](ARCHITECTURE.md) · [OMR.md](OMR.md) (engines evaluated, licences, capabilities, failure cases) ·
[DEVELOPMENT.md](DEVELOPMENT.md) · [ROADMAP.md](ROADMAP.md) · [LICENSE](LICENSE)
