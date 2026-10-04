# Kirkkokäsikirja I (jpkirja.doc) as input

`anno-api/refs/jpkirja.doc` is a Word file whose notation is stored as embedded WMF pictures. The pictures draw staff
lines, stems, beams and ties as shapes, but every note head, clef, accidental and rest is a text glyph in a font called
`capella`, which is not embedded. Converted as is, those glyphs come out as empty boxes.

## Reproducing the 2. sävelmäsarja crops

1. `python tools/capella-font/build.py` writes `capella.ttf`; install it (for example in `~/.fonts`, then `fc-cache -f`).
2. `soffice --headless --convert-to pdf jpkirja.doc` (needs LibreOffice Writer). The 2. sävelmäsarja is pages 333-346.
3. `music-reader jpkirja.pdf --pages 333-346 --crops crops --crop-dpi 300`

## What the font substitute is based on

Nothing from the original font. The glyph codes and their positions are read from the WMF records, and shapes were drawn
from scratch. Meaning of each code was inferred from position and context:

| Code | Meaning | How it was established |
|---|---|---|
| `A`, `E` | treble clef, bass clef | sits on the G line / F line of each system |
| `Q`, `S`, `R` | flat, sharp, natural | key signature patterns (B-E-A flats, F-C sharps) |
| E5, E4, E3 (0xE5, 0xE4, 0xE3) | filled, hollow with stem, hollow without stem head | stem rectangles next to them; measure lengths add up |
| `K`, `J` | quarter rest, half rest | measure lengths add up in 3/4 and 6/4 examples |
| 0xE6, 0xEA | eighth flag up, down | placed at the top or bottom of a stem |
| `.` | augmentation dot | follows a head or rest |
| `:`, `;` | common time, cut time | at the time-signature position |
| `L` | eighth rest (not confirmed) | rest-like placement on the middle line; no measure arithmetic backs it |
| 0xE1 | unknown | does not occur in the 2. sävelmäsarja pages |

Check rests and accidentals against the printed book before relying on the music.

The texts are © Kirkkohallitus. Do not commit the document, the PDF or the crops.
