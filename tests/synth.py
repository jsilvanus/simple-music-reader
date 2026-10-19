"""Synthetic sheet-music generator for tests (no copyrighted material).

Draws a simplified but realistic engraving into a PDF with reportlab: staff lines, clef, key
signature, time-signature digits, note heads (filled/hollow), stems, flags, beams, dots,
accidentals, ties, rests, barlines, lyrics and a title. All sizes are relative to the staff
spacing ``sp`` so the same score can be generated at several scales.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from reportlab.pdfgen import canvas

DUR = {"w": 1.0, "h": 0.5, "q": 0.25, "e": 0.125, "s": 0.0625}
STEPS = "CDEFGAB"
SHARPS = [(("F", 5)), ("C", 5), ("G", 5), ("D", 5), ("A", 4), ("E", 5), ("B", 4)]
FLATS = [("B", 4), ("E", 5), ("A", 4), ("D", 5), ("G", 4), ("C", 5), ("F", 4)]


@dataclass
class Note:
    pitch: str  # "E4", "F#4" (accidental in the name is only used for pitch; see acc)
    dur: str = "q"
    dots: int = 0
    acc: Optional[str] = None  # printed accidental: "#", "b", "n"
    tie: bool = False  # tied to the next note


@dataclass
class Rest:
    dur: str = "q"


@dataclass
class StaffSpec:
    measures: list[list]
    key_fifths: int = 0
    meter: Optional[str] = "4/4"
    clef: str = "treble"
    beam_group_size: int = 2  # longest run of eighths/sixteenths joined by one beam (2 = pairs, as before)
    max_slope: float = 0.0  # steepest beam (rise per unit of width); 0 draws flat beams, as before
    time_font: str = "Helvetica-Bold"  # font of the time-signature digits
    final: bool = False  # final double barline after the last measure


@dataclass
class SystemSpec:
    staves: list[StaffSpec]


def _diatonic(name: str) -> int:
    return int(name[-1]) * 7 + STEPS.index(name[0])


class _Drawer:
    def __init__(self, c: canvas.Canvas, sp: float, height: float):
        self.c, self.sp, self.H = c, sp, height

    # --- primitives in y-down page coordinates ---------------------------------------
    def Y(self, y):
        return self.H - y

    def line(self, x1, y1, x2, y2, lw):
        self.c.setLineWidth(lw)
        self.c.line(x1, self.Y(y1), x2, self.Y(y2))

    def disc(self, x, y, r):
        self.c.circle(x, self.Y(y), r, stroke=0, fill=1)

    def poly(self, pts, lw, close=False):
        self.c.setLineWidth(lw)
        self.c.setLineJoin(1)
        self.c.setLineCap(1)
        p = self.c.beginPath()
        p.moveTo(pts[0][0], self.Y(pts[0][1]))
        for x, y in pts[1:]:
            p.lineTo(x, self.Y(y))
        if close:
            p.close()
        self.c.drawPath(p, stroke=1, fill=0)

    def fill_poly(self, pts):
        p = self.c.beginPath()
        p.moveTo(pts[0][0], self.Y(pts[0][1]))
        for x, y in pts[1:]:
            p.lineTo(x, self.Y(y))
        p.close()
        self.c.drawPath(p, stroke=0, fill=1)

    def filled_rect(self, x0, y0, x1, y1):
        self.c.rect(x0, self.Y(y1), x1 - x0, y1 - y0, stroke=0, fill=1)

    def ellipse(self, x, y, rx, ry, angle=0, fill=True, lw=0.0):
        c = self.c
        c.saveState()
        c.translate(x, self.Y(y))
        c.rotate(angle)
        if fill:
            c.ellipse(-rx, -ry, rx, ry, stroke=0, fill=1)
        else:
            c.setLineWidth(lw)
            c.ellipse(-rx, -ry, rx, ry, stroke=1, fill=0)
        c.restoreState()


class _StaffLayout:
    def __init__(self, d: _Drawer, top: float, x0: float, x1: float, spec: StaffSpec):
        self.d, self.sp, self.top, self.x0, self.x1, self.spec = d, d.sp, top, x0, x1, spec
        self.bottom = top + 4 * d.sp

    def ypos(self, pos: float) -> float:
        return self.bottom - pos * self.sp / 2

    def pos_of(self, name: str) -> int:
        base = _diatonic("E4") if self.spec.clef == "treble" else _diatonic("G2")
        return _diatonic(name) - base

    # --- drawing ------------------------------------------------------------------
    def staff_lines(self):
        for i in range(5):
            y = self.top + i * self.sp
            self.d.line(self.x0, y, self.x1, y, 0.1 * self.sp)

    def clef(self, x):
        d, sp = self.d, self.sp
        if self.spec.clef == "treble":
            d.line(x, self.ypos(11), x, self.ypos(-2.5), 0.2 * sp)
            d.ellipse(x, self.ypos(2.5), 0.95 * sp, 0.95 * sp, fill=False, lw=0.2 * sp)
            d.ellipse(x - 0.5 * sp, self.ypos(-2.5), 0.4 * sp, 0.4 * sp, fill=False, lw=0.2 * sp)
            return x + 1.3 * sp
        d.disc(x, self.ypos(6), 0.45 * sp)
        d.poly([(x - 0.4 * sp, self.ypos(6.4)), (x + 0.5 * sp, self.ypos(5)), (x - 0.5 * sp, self.ypos(1))], 0.3 * sp)
        return x + 1.3 * sp

    def key_signature(self, x):
        n = self.spec.key_fifths
        if n == 0:
            return x
        seq = SHARPS[:n] if n > 0 else FLATS[:-n]
        for step, octave in seq:
            y = self.ypos(self.pos_of(f"{step}{octave}"))
            if n > 0:
                self.sharp(x, y)
            else:
                self.flat(x, y)
            x += 1.35 * self.sp
        return x

    def time_signature(self, x):
        if not self.spec.meter:
            return x
        d, c, sp = self.d, self.d.c, self.sp
        if self.spec.meter in ("C", "C|"):
            cx, cy, r = x, self.ypos(4), 0.95 * sp
            pts = [(cx + r * math.cos(math.radians(a)), cy + 1.05 * r * math.sin(math.radians(a))) for a in range(40, 321, 10)]
            d.poly(pts, 0.35 * sp)
            if self.spec.meter == "C|":
                d.line(cx, self.ypos(8.3), cx, self.ypos(-0.3), 0.18 * sp)
            return x + 1.6 * sp
        num, den = self.spec.meter.split("/")
        c.setFont(self.spec.time_font, 2.7 * sp)
        c.drawCentredString(x, d.Y(self.ypos(4.05)), num)
        c.drawCentredString(x, d.Y(self.ypos(0.05)), den)
        return x + 1.6 * sp

    def sharp(self, x, y):
        d, sp = self.d, self.sp
        for dx in (-0.22, 0.22):
            d.line(x + dx * sp, y - 1.3 * sp, x + dx * sp, y + 1.3 * sp, 0.12 * sp)
        d.line(x - 0.5 * sp, y + 0.1 * sp, x + 0.5 * sp, y - 0.3 * sp, 0.3 * sp)
        d.line(x - 0.5 * sp, y + 0.75 * sp, x + 0.5 * sp, y + 0.35 * sp, 0.3 * sp)

    def flat(self, x, y):
        d, sp = self.d, self.sp
        d.line(x - 0.3 * sp, y - 2.0 * sp, x - 0.3 * sp, y + 0.6 * sp, 0.13 * sp)
        d.poly([(x - 0.3 * sp, y + 0.6 * sp), (x + 0.3 * sp, y + 0.1 * sp), (x + 0.35 * sp, y - 0.4 * sp), (x - 0.3 * sp, y - 0.2 * sp)], 0.17 * sp)

    def natural(self, x, y):
        d, sp = self.d, self.sp
        d.line(x - 0.25 * sp, y - 1.3 * sp, x - 0.25 * sp, y + 0.5 * sp, 0.12 * sp)
        d.line(x + 0.25 * sp, y - 0.5 * sp, x + 0.25 * sp, y + 1.3 * sp, 0.12 * sp)
        d.line(x - 0.25 * sp, y + 0.45 * sp, x + 0.25 * sp, y + 0.3 * sp, 0.26 * sp)
        d.line(x - 0.25 * sp, y - 0.3 * sp, x + 0.25 * sp, y - 0.45 * sp, 0.26 * sp)

    def rest(self, x, kind):
        d, sp = self.d, self.sp
        if kind == "w":
            y = self.ypos(6)
            d.filled_rect(x - 0.6 * sp, y, x + 0.6 * sp, y + 0.5 * sp)
        elif kind == "h":
            y = self.ypos(4)
            d.filled_rect(x - 0.6 * sp, y - 0.5 * sp, x + 0.6 * sp, y)
        elif kind == "q":
            y = self.ypos(7)
            d.poly([(x - 0.3 * sp, y), (x + 0.4 * sp, y + 0.8 * sp), (x - 0.35 * sp, y + 1.5 * sp), (x + 0.35 * sp, y + 2.3 * sp), (x - 0.2 * sp, y + 3.0 * sp)], 0.3 * sp)
        else:  # eighth
            y = self.ypos(5)
            d.disc(x + 0.35 * sp, y, 0.3 * sp)
            d.line(x + 0.35 * sp, y, x - 0.35 * sp, y + 1.7 * sp, 0.18 * sp)

    # --- events -----------------------------------------------------------------------
    def layout(self, x_start, x_end):
        """Return (items, barline xs). items: dicts with x, kind, spec, ..."""
        sp = self.sp
        ptr = x_start
        items, bars = [], []
        for mi, measure in enumerate(self.spec.measures):
            for it in measure:
                acc = isinstance(it, Note) and it.acc
                xc = ptr + (2.5 if acc else 1.0) * sp
                items.append({"x": xc, "it": it, "m": mi})
                dur = DUR[it.dur]
                extra = 2.0 if dur >= 1 else 1.0 if dur >= 0.5 else 0.0 if dur >= 0.25 else -0.3
                ptr = xc + (1.0 + extra + (0.9 if getattr(it, "dots", 0) else 0.0) + 1.6) * sp
            bars.append(ptr - 0.8 * sp)
        natural = ptr - x_start
        avail = x_end - x_start - 0.5 * sp
        if natural > avail:
            raise ValueError(f"staff too crowded: needs {natural / sp:.1f} sp, has {avail / sp:.1f} sp")
        f = avail / natural
        for it in items:
            it["x"] = x_start + (it["x"] - x_start) * f
        bars = [x_start + (b - x_start) * f for b in bars]
        return items, bars

    def draw_events(self, items, bars):
        d, sp = self.d, self.sp
        # beam groups: consecutive eighth/sixteenth notes of one measure, at most four (``beam_groups`` off: pairs only)
        groups, cur = [], []
        for i, e in enumerate(items):
            ok = _is_beamable(e["it"])
            if ok and cur and (items[cur[-1]]["m"] != e["m"] or len(cur) >= self.spec.beam_group_size):
                groups.append(cur)
                cur = []
            if ok:
                cur.append(i)
            elif cur:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        groups = [g for g in groups if len(g) >= 2]
        beamed = {i for g in groups for i in g}
        for i, e in enumerate(items):
            it, x = e["it"], e["x"]
            if isinstance(it, Rest):
                self.rest(x, it.dur)
                continue
            pos = self.pos_of(it.pitch)
            y = self.ypos(pos)
            if it.acc:
                ax = x - 1.7 * sp
                {"#": self.sharp, "b": self.flat, "n": self.natural}[it.acc](ax, y)
            self.ledgers(x, pos)
            e["pos"], e["y"] = pos, y
            if it.dur in ("w", "h"):
                if it.dur == "w":
                    d.ellipse(x, y, 0.72 * sp, 0.46 * sp, fill=False, lw=0.2 * sp)
                else:
                    d.ellipse(x, y, 0.55 * sp, 0.36 * sp, angle=20, fill=False, lw=0.2 * sp)
            else:
                d.ellipse(x, y, 0.65 * sp, 0.46 * sp, angle=20)
            if it.dots:
                dy = y - 0.5 * sp if pos % 2 == 0 else y
                for k in range(it.dots):
                    d.disc(x + (1.15 + 0.5 * k) * sp, dy, 0.2 * sp)
        # stems / flags / beams
        for i, e in enumerate(items):
            it = e["it"]
            if isinstance(it, Rest) or it.dur == "w" or i in beamed:
                continue
            down = e["pos"] >= 4
            sw = 0.13 * sp
            sx = e["x"] - 0.65 * sp + sw / 2 if down else e["x"] + 0.65 * sp - sw / 2
            tip = e["y"] + (3.5 if down else -3.5) * sp
            d.line(sx, e["y"], sx, tip, sw)
            nflags = {"q": 0, "h": 0, "e": 1, "s": 2}[it.dur]
            for k in range(nflags):
                off = k * 0.75 * sp * (-1 if down else 1)  # further flags sit closer to the head
                ty = tip + off
                d.line(sx, ty, sx + 1.0 * sp, ty + (-1.7 if down else 1.7) * sp, 0.28 * sp)
        for g in groups:
            self._beam_group(items, g)
        # ties
        for i, e in enumerate(items[:-1]):
            it = e["it"]
            if isinstance(it, Note) and it.tie:
                nxt = items[i + 1]
                up = e["pos"] >= 4  # stems down -> tie above
                above = e["pos"] >= 4
                x1, x2 = e["x"] + 0.3 * sp, nxt["x"] - 0.3 * sp
                y0 = e["y"] + (-0.75 if above else 0.75) * sp
                dy = (-1.0 if above else 1.0) * sp
                self.d.c.setLineWidth(0.17 * sp)
                p = self.d.c.beginPath()
                p.moveTo(x1, self.d.Y(y0))
                p.curveTo(x1 + (x2 - x1) * 0.2, self.d.Y(y0 + dy), x2 - (x2 - x1) * 0.2, self.d.Y(y0 + dy), x2, self.d.Y(y0))
                self.d.c.drawPath(p, stroke=1, fill=0)
                del up

    def _beam_group(self, items, g):
        """Stems and beams of one group; the beam follows the contour of the notes (slanted), at most ``max_slope``."""
        d, sp = self.d, self.sp
        sw = 0.13 * sp
        down = sum(items[i]["pos"] for i in g) / len(g) >= 4
        sign = 1 if down else -1
        xs = [items[i]["x"] + (-0.65 * sp + sw / 2 if down else 0.65 * sp - sw / 2) for i in g]
        tips = [items[i]["y"] + sign * 3.5 * sp for i in g]
        slope = (tips[-1] - tips[0]) / (xs[-1] - xs[0])
        slope = max(-self.spec.max_slope, min(self.spec.max_slope, slope))
        # the line through the mean tip with that slope, pushed outwards until every stem is at least 3 spacings long
        mx, my = sum(xs) / len(xs), sum(tips) / len(tips)

        def line_y(x):
            return my + slope * (x - mx)

        shift = max(0.0, max(sign * ((items[i]["y"] + sign * 3.0 * sp) - line_y(x)) for i, x in zip(g, xs)))
        my += sign * shift

        def edge(x, inward):  # y of the beam's outer edge (inward=0) or a line `inward` further towards the heads
            return line_y(x) - sign * inward

        thick, gap = 0.5 * sp, 0.25 * sp
        for i, x in zip(g, xs):
            d.line(x, items[i]["y"], x, line_y(x), sw)
        d.fill_poly([(xs[0] - sw / 2, edge(xs[0] - sw / 2, 0)), (xs[-1] + sw / 2, edge(xs[-1] + sw / 2, 0)),
                     (xs[-1] + sw / 2, edge(xs[-1] + sw / 2, thick)), (xs[0] - sw / 2, edge(xs[0] - sw / 2, thick))])
        # second beam: between neighbouring sixteenths; a lone sixteenth gets a stub towards its neighbour
        is16 = [items[i]["it"].dur == "s" for i in g]
        off = thick + gap
        for k, sixteenth in enumerate(is16):
            if not sixteenth:
                continue
            if k + 1 < len(g) and is16[k + 1]:
                a, b = xs[k] - sw / 2, xs[k + 1] + sw / 2
            elif k > 0 and not is16[k - 1] or (k + 1 == len(g)):
                a, b = xs[k] - 0.9 * sp, xs[k] + sw / 2
            elif k == 0 or not is16[k - 1]:
                a, b = xs[k] - sw / 2, xs[k] + 0.9 * sp
            else:
                continue
            d.fill_poly([(a, edge(a, off)), (b, edge(b, off)), (b, edge(b, off + thick)), (a, edge(a, off + thick))])

    def ledgers(self, x, pos):
        sp = self.sp
        if pos <= -2:
            for p in range(-2, pos - 1, -2):
                y = self.ypos(p)
                self.d.line(x - 1.0 * sp, y, x + 1.0 * sp, y, 0.1 * sp)
        elif pos >= 10:
            for p in range(10, pos + 1, 2):
                y = self.ypos(p)
                self.d.line(x - 1.0 * sp, y, x + 1.0 * sp, y, 0.1 * sp)

    def barlines(self, bars, final):
        sp = self.sp
        for i, b in enumerate(bars):
            last = i == len(bars) - 1
            if last and final:
                self.d.line(b - 0.5 * sp, self.top, b - 0.5 * sp, self.bottom, 0.12 * sp)
                self.d.line(b + 0.1 * sp, self.top, b + 0.1 * sp, self.bottom, 0.45 * sp)
            else:
                self.d.line(b, self.top, b, self.bottom, 0.12 * sp)


def _is_beamable(it) -> bool:
    return isinstance(it, Note) and it.dur in ("e", "s")


def render_pdf(path, pages: list[list[SystemSpec]], sp: float = 5.0, title: str = "Esimerkkivirsi",
               lyrics: bool = True, size=(595.0, 842.0)) -> None:
    """Write a vector PDF. ``pages`` is a list of pages, each a list of systems."""
    W, H = size
    c = canvas.Canvas(str(path), pagesize=size)
    d = _Drawer(c, sp, H)
    for page in pages:
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(W / 2, H - 45, title)
        y = 70 + 3 * sp
        for sysspec in page:
            top = y
            layouts = []
            for k, st in enumerate(sysspec.staves):
                lay = _StaffLayout(d, top, 50.0, W - 50.0, st)
                lay.staff_lines()
                layouts.append(lay)
                top = lay.bottom + 8 * sp
            if len(layouts) > 1:
                d.line(50.0, layouts[0].top, 50.0, layouts[-1].bottom, 0.14 * sp)
            for lay in layouts:
                x = lay.clef(50.0 + 1.2 * sp)
                x = lay.key_signature(x + 0.8 * sp)
                x = lay.time_signature(x + 1.0 * sp)
                items, bars = lay.layout(x + 0.6 * sp, W - 50.0)
                lay.draw_events(items, bars)
                lay.barlines(bars, lay.spec.final)
                if lyrics and lay is layouts[0]:
                    c.setFont("Helvetica", 2.6 * sp)
                    for it in items[:: 2]:
                        c.drawString(it["x"] - sp, H - (lay.bottom + 4.3 * sp), "ooo")
            y = layouts[-1].bottom + 14 * sp
        c.showPage()
    c.save()


def rasterize_pdf(src, dst, dpi=200, rotate_deg=0.0, noise=0.0, seed=0) -> None:
    """Turn a (vector) PDF into a scanned-looking image-only PDF."""
    import cv2
    import pypdfium2 as pdfium
    from PIL import Image

    rng = np.random.default_rng(seed)
    pdf = pdfium.PdfDocument(str(src))
    imgs = []
    for page in pdf:
        arr = np.ascontiguousarray(page.render(scale=dpi / 72, grayscale=True).to_numpy())
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if rotate_deg:
            h, w = arr.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, 1.0)
            arr = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        if noise:
            arr = np.clip(arr.astype(np.float32) + rng.normal(0, noise, arr.shape), 0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(arr))
    imgs[0].save(str(dst), save_all=True, append_images=imgs[1:], resolution=dpi)


def parse_measures(text: str) -> list[list]:
    """Tiny notation: "E4q F4q G4h | C5w". Suffix letters: w h q e s, '.' dot, '~' tie, prefix #/b/n accidental."""
    out = []
    for bar in text.split("|"):
        items = []
        for tok in bar.split():
            if tok[0] == "r":
                items.append(Rest(tok[1]))
                continue
            acc = None
            if tok[0] in "#bn" and len(tok) > 1 and tok[1] in STEPS:
                acc, tok = tok[0], tok[1:]
            pitch, rest = tok[:2], tok[2:]
            tie = rest.endswith("~")
            rest = rest.rstrip("~")
            dots = rest.count(".")
            items.append(Note(pitch, rest.replace(".", ""), dots, acc, tie))
        out.append(items)
    return out
