# Architecture

```
src/music_reader/
  pdfinput.py      Document (PDF via pypdfium2, or an image file): page analysis + rendering
  ir.py            Music IR: Score > Voice > Measure > Event; Pitch, Meter, KeySignature, Diagnostic, PageInfo
  cv/staff.py      deskew, binarise, staff detection, system grouping, staff-line and ledger-line removal
  cv/symbols.py    components, note heads, stems, flags/beams, accidentals, rests, dots, ties, barlines, header
  assemble.py      recognised items -> IR (pitch from staff position, accidental scope, ties, meter inference)
  backends/        RecognitionBackend interface (+ the classical CV backend)
  validate.py      musical-structure checks on the IR
  export/          abc.py, musicxml.py, json_export.py (+ registry)
  debug.py         annotated page images / debug.pdf
  crops.py         per-system images + crops.json
  bundle.py        everything for one piece, for downstream consumers
  cli.py           music-reader command
tests/synth.py     generator of synthetic scores (reportlab) used by the tests
```

## Data flow

1. **Page analysis** (`pdfinput.analyze`): counts image/path/text objects, lists fonts. A page is `vector` (notation drawn
   as objects), `raster` (one large image covers the page) or `image` (an image file was given). Embedded music fonts
   (Bravura, Emmentaler, Opus, Maestro, ...) are reported.
2. **Render** to grayscale. The first pass uses 150 dpi; if the detected staff spacing is far from 16 px the page is
   rendered again at a resolution that gives about 16 px (vector pages stay sharp at any dpi).
3. **Deskew** by maximising the variance of the horizontal projection (±3°).
4. **Staves**: morphological opening finds long horizontal runs, rows are grouped into line centres, five equally
   spaced lines make a staff (`Staff.spacing` is the unit for everything after this). Staves joined by a vertical line at
   their left edge form a *system*; staves that are close but not joined produce a warning.
5. **Line removal**: staff lines are erased column by column unless ink continues beyond the line (a stem, the cap of a
   hollow head, an arc lying on the line). Ledger lines are found as short thin strokes on the staff's half-step grid
   and erased the same way.
6. **Components** of the remaining ink are assigned to staves (barlines may belong to several).
7. **Per staff** (`analyze_staff`): header (clef, key signature accidentals, time-signature block) then content: note
   heads, stems, flags/beams, accidentals, dots, rests, ties, barlines. Pitch comes from the head's vertical position
   in half-spaces above the bottom line. Duration from head type, stem and flag/beam count and dots.
8. **Melody selection** (`backends/classical.py`): if a system has several staves and the user did not choose one, staff 1
   is taken **and an `AMBIGUOUS_MELODY_STAFF` warning is produced**; `selection` in the Score records how it was chosen.
9. **Assembly** (`assemble.py`): barlines split measures; accidentals persist for the same pitch until the barline; the key
   signature supplies the rest; ties connect same-pitch neighbours; the meter is taken from `--meter` or inferred from the
   modal measure length (flagged `inferred`).
10. **Validation** (`validate.py`): overfull/incomplete measures (a pick-up bar completed by the last bar is accepted), empty
    measures, low confidence, large jumps, unusual range.
11. **Export** from the IR only.

## Serving both anno-api goals

* *Images of the notation (a)*: steps 1-4 are enough. `crops.py` cuts each system from halfway to the previous system to
  halfway to the next one, so lyrics under the staff stay with their system, and can re-render the PDF at a higher
  resolution (`--crop-dpi`). The result does not depend on pitch recognition working.
* *Flexible layouts (b)*: the Music IR is exported as ABC (small, diffable, renderable by abcjs) and MusicXML 4.0
  (readable by MuseScore, Verovio, ...). JSON carries the whole IR including confidences and source boxes, so a UI can
  show the original crop next to the recognised melody and let a human correct it.
* `bundle.py` keeps the two linked by page/system numbers and adds `needs_review` for the consumer.

## Extension points

**Another recognition backend.** Implement `recognize(doc: Document, options: RecognitionOptions) -> RecognitionResult`
(see `backends/base.py`) and call `register_backend(MyBackend())` in `backends/__init__.py`. Return a `Score` built from
the Music IR (`ir.py`); fill `PageDebug` if you want debug overlays and crops. A wrapper around Audiveris or homr would
run the external program, parse its MusicXML into the IR (a MusicXML *reader* is not written yet, see ROADMAP) and
copy diagnostics. A backend that reads vector PDFs directly can use `Document.analyze`.

**Another output format.** Write `def export_x(score: Score, title: str | None = None) -> str` in `export/` and call
`register_exporter("x", export_x, "ext")` in `export/__init__.py`. The CLI picks it up for `--format x` and by file
extension. Exporters must read only the IR.

**More recognition coverage** goes into `cv/symbols.py` (new symbol classes) and `assemble.py` (semantics); keep
geometry relative to `Staff.spacing`.

## Why Python

PDF rendering plus image processing is where Python has the mature, permissively licensed stack (pypdfium2 for PDF
rendering and object inspection, OpenCV and numpy for morphology and components); the equivalent Node.js stack would
need native bindings for both. The CLI is small and installable with pip. The Node-based anno-api consumes the files
(ABC/MusicXML/JSON/PNG) or runs the CLI; nothing needs the two to share a process.
