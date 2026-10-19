# OMR survey, licences and what this project reads

## 1. Existing open-source OMR evaluated

This was a desk survey (README pages fetched on 2026-09-30), **not** a hands-on evaluation: none of the engines was run
on a church PDF. Items marked *not verified* were not confirmed from a source and must be checked before relying on them.

| Engine | Code licence | Model/weights licence | Maintenance | Interface | Input / output | Notes |
|---|---|---|---|---|---|---|
| [Audiveris](https://github.com/Audiveris/audiveris) | AGPL-3.0 | no ML weights listed in the README | active (development branch, installers for Win/Linux/macOS, releases every 6-12 months per the repository page) | Java application with GUI; batch/CLI use *not verified here* | images; PDF input *not verified here* (believed to be supported, check the handbook); MusicXML 4.0 export, `.omr` project files | The most complete classical OMR. Strong candidate for an external-process backend. |
| [homr](https://github.com/liebharc/homr) | AGPL-3.0 | **not stated** in the README; builds on oemer's segmentation models and the Polyphonic-TrOMR transformer, whose licences were *not verified* | active (many commits, open issues/PRs) | Python; run through `uvx`/Poetry (CPU, CUDA or ROCm) | "camera pictures or PDFs" in; MusicXML out | Neural. Documented limits: pitch and rhythm only, bass/treble clef, no dynamics/articulation/double accidentals. |
| [Oemer](https://github.com/BreezeWhite/oemer) | MIT | **not stated**; trained on CvcMuscima-Distortions and DeepScores-extended (dataset licences *not verified*); checkpoints are downloaded separately | v0.1.7 from Oct 2023, little recent release activity | Python CLI `oemer image`; ONNX Runtime (or TensorFlow) | image in (no PDF); MusicXML out | UNet segmentation + SVM + rules. Minutes per page; printed Western notation. |

Not evaluated: other projects exist (the survey did not look for them beyond these three). Any further candidate should be
added here with the same columns.

### Decision

* **Own backend first** (`backends/classical.py`): the target is *simple, printed, mostly monophonic* hymn melodies, where
  deterministic geometry is enough, runs everywhere, has no model-licence question and gives per-symbol confidence and
  bounding boxes that the debug output and crops need.
* **No third-party OMR code or weights are included or copied.** Audiveris and homr (AGPL-3.0) can later be used as
  *separate processes* behind the backend interface (see ARCHITECTURE.md) once the licensing consequences for how anno-api
  would be distributed or served have been decided. AGPL's network-use clause is the part to read carefully if such a
  process is exposed as a service.
* Oemer is MIT for the code, but the checkpoint and dataset licences are unknown, so using it would also need that check.

## 2. Licences of what *is* used

| Component | Role | Licence (checked in installed package metadata, 2026-09-30) |
|---|---|---|
| music-reader (this repository) | our code | MIT, see `LICENSE` (the owner may change this) |
| pypdfium2 | PDF rendering and object inspection (wraps PDFium) | BSD-3-Clause / Apache-2.0 (also the bundled PDFium binary; see the package for the third-party notices) |
| opencv-python-headless | image processing | Apache-2.0 (the wheel bundles FFmpeg and other libraries under their own licences, see the package) |
| numpy | arrays | BSD-3-Clause and others |
| Pillow | debug PDF writing | MIT-CMU (HPND) |
| reportlab, pytest (test extra only) | generated test scores, test runner | BSD; MIT |

**PyMuPDF was deliberately not used**: it is AGPL-3.0 (or commercial). No model weights, no cloud service and no external
runtime beyond Python are required. The test corpus contains no copyrighted music; real PDFs are never committed.

## 3. Input PDFs: vector or scanned

`pdfinput.analyze` classifies each page:

* `vector`: staff lines, noteheads and barlines are PDF path objects or font glyphs (Sibelius, Finale, MuseScore,
  LilyPond). The same render-then-CV route is used, but at a resolution chosen from the staff size so edges are crisp.
  The PDF objects (exact line positions; music-font glyph names such as SMuFL) could be read directly, which would be both
  more accurate and cheaper; this is *investigated but not implemented* (ROADMAP item 6). What was verified: object
  counts, image coverage and font names are readable with pypdfium2, and on generated PDFs music is drawn as paths.
  What was **not** verified: how real Finnish hymn books are produced.
* `raster`: one image covers the page (a scan). Rendering just resamples that image, so recognition quality is bounded by
  the scan resolution. Deskew and light noise are handled (tested on a rotated, noisy rasterised generated page).
* `image`: a PNG/JPG file was given.

## 4. What the classical backend reads today

Verified by tests on generated scores (sizes 4-8 pt staff spacing, rotated/noisy raster included):

* page layout: multiple pages, multiple systems per page, one or several staves per system, lyrics and titles ignored
* clef: treble (confirmed by height); a bass-clef-sized clef is *guessed* with a warning
* key signature: 0-7 sharps or flats in the header; major is assumed (minor cannot be told)
* notes: whole, half, quarter, eighth, sixteenth; dotted (and double-dotted); ledger lines above and below
* accidentals: sharp, flat, natural, with bar-line scope and key-signature interplay
* rests: whole, half, quarter, eighth
* barlines, double/final barlines; measure construction; pick-up bars
* ties between equal pitches (also when the arc lies on a staff line and is cut in two)
* time signature: stacked digits (2, 3, 4, 5, 6, 7, 8, 9, 12, 15 over 1, 2, 4, 8, 16), common time and cut time, read by
  template matching against digits rendered at run time (no learned model, `cv/digits.py`); a weak reading is not
  used, the meter is then inferred from the bars (or `--meter`). A disagreement between the printed and the inferred
  meter is flagged
* beams: groups of two to four, slanted, single and double (sixteenth) beams; a beam lying on a staff line is not taken
  for a staff line
* triplets (3:2): the "3" above or below three neighbouring notes or rests, with or without a bracket, also when a staff
  line crosses the number; the three durations are two thirds of what is written (`Event.tuplet = (3, 2)`). A level
  bracket stroke is told from a tie by being straight, so a triplet of repeated pitches is not read as tied. Other tuplets
  (5, 6, 2:3 duplets) are not read
* grace notes: a small slashed note with a flag in front of a note; it takes no time, is kept as `Event.graces` of the
  note it leans on (ABC `{G}`, MusicXML `<grace>`, `grace_notes` in `melody.json`), and the bar's length is unchanged. One
  flag per grace note (beamed groups of grace notes, accidentals on a grace note and long appoggiaturas are not
  handled)
* two voices on one staff (stacked heads on a common stem or on two stems): `--voice top|bottom` picks the voice that is
  exported, with one `CHORD_OR_POLYPHONY` warning per staff
* melody staff: with several staves per system the single treble-clef staff is chosen (strategy `treble-clef`);
  otherwise the first staff is assumed and `AMBIGUOUS_MELODY_STAFF` is raised
* `melody.json` (`music-reader.melody/1`): measures with pitch, MIDI number, duration, onset, tie flag and confidence
  per event, meter and key with their source, `needs_review` and the non-info diagnostics. No images, no pixel
  positions. Additions keep the schema number at 1; a breaking change gets a new number
* diagnostics: overfull/incomplete measures, jumps, range, low confidence, ambiguous staff choice, orphan accidentals,
  stemless heads, chords, key changes between systems

## 5. Known failure cases and untested ground

Honest list; most items are untested on real engravings.

* **No real church PDF has been tried.** Real fonts differ from the generator (curved flags, slanted beams, thin or heavy
  stems, wedge-shaped noteheads, treble clefs of other shapes). Expect to retune thresholds (`cv/symbols.py`, all in
  multiples of the staff spacing).
* Beams are verified on generated scores only (flat, slanted, groups of 2-4, double beams, also after a scan
  simulation). Beam groups across barlines and engraving fonts with curved or very thick beams are untested.
* Hollow heads sitting between two lines are found as enclosed white regions because line removal breaks their ring;
  very small or blurred scans (staff spacing below ~8 px, flagged `LOW_RESOLUTION`) will fail.
* Noteheads touching accidentals, dots, or each other (seconds in a chord) merge components; chords are reduced to the
  chosen voice (top by default) with `CHORD_OR_POLYPHONY`. Voice separation works on stacked heads; voices that cross
  or share one head are not told apart.
* A flat sign's bowl and a quarter rest can resemble other symbols; accidentals are classified by vertical strokes only.
* Time-signature digits are read by templates and are the least certain part: a thin serif 6 can read as a 5, and fonts
  far from the bundled ones may fail (the meter then falls back to the bars). Unusual meters (3/2, 6/4, 5/8 ...) are
  not in the accepted list.
* Triplets and grace notes are verified on generated scores at 5-8 pt staff spacing and in a rotated noisy scan; at 4 pt
  the triplet number crossing a staff line and a scanned grace note are missed. Tempo marks, dynamics, fermatas, repeat
  signs, voltas, other tuplets, multi-measure rests,
  clef/key changes inside a system and lyrics are not interpreted. An unrecognised symbol inside the staff is logged as
  `UNCLASSIFIED_SYMBOL` (info) and drawn red in the debug image.
* Text with closed letters (o, e, a) inside the staff area could be taken for hollow heads; they are rejected when
  they are not on a ledger-line position, but lyrics set very close to the staff are a risk.
* Staves not joined by a system line are treated as separate systems (`POSSIBLY_UNGROUPED_STAVES` warns when close).
* Slurs between different pitches are ignored (`SLUR_IGNORED`, info), not exported.
* Minor keys are written as their relative major; the mode is not detected.
* Skew estimation covers ±3°; curved (photographed) pages are out of scope.

## 6. Adding a backend or an output format

See "Extension points" in [ARCHITECTURE.md](ARCHITECTURE.md): implement `recognize()` returning a Music IR `Score` and
register it; or write `export_x(score, title)` and register it. Both are a few lines of wiring.
