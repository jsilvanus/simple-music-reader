# Roadmap

Ordered by what most improves *useful melodies from real church PDFs*.

1. **Run on real material.** Collect (locally, uncommitted) 10-20 Finnish hymn/liturgy PDFs of both kinds (vector and
   scan), write expected melodies, record the failures. Everything below is prioritised by what they show. Check the
   embedded fonts of vector files (`PageAnalysis.fonts`).
2. **Read the time signature.** Digit templates or a small classifier for stacked digits and the common-time sign; compare
   with the inferred meter and flag a disagreement.
3. **Slanted beams, beam groups of 3+, sixteenth beams, grace notes, triplets** in `cv/symbols.py` (beams are currently
   read as a count of strokes beside the stem, tested with flat beams only).
4. **Voices in one staff.** Hymn books often print soprano and alto on one staff (stems up/down, chords). Today a chord is
   reduced to its top note with a warning; add voice separation by stem direction and a `--voice` option.
5. **Better melody-staff choice.** Use clef, range, lyric lines under the staff and stem direction instead of "first staff".
6. **Vector-first backend.** For vector PDFs read staff lines, barlines and music-font glyphs straight from the PDF objects
   (exact geometry, no thresholds). `Document.analyze` already lists fonts; needs a glyph-to-SMuFL mapping per font.
7. **External OMR backends** (Audiveris, homr) behind `backends/base.py`, with a **MusicXML reader** into the IR. Decide the
   licensing story first (both are AGPL-3.0, see OMR.md); running them as separate processes is the usual route.
8. **Repeats, voltas, D.C./D.S., fermatas**, key changes and mid-system clef changes; minor-key detection.
9. **Lyrics alignment** (the crops already keep lyrics with the system; OCR of Finnish text and syllable alignment is
   separate work).
10. **Human correction UI** over the JSON (confidence and bounding boxes are already there).
11. **Packaging for anno-api**: a small HTTP/MCP wrapper or an npm script that calls `music-reader --bundle`, and a rule for
    where bundles are stored.
