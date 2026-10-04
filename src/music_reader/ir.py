"""Music IR: the core data model. ABC (or any other format) is derived from this.

Durations are exact ``fractions.Fraction`` values in whole-note units
(1/4 = quarter note). Pitch is semantic (step, octave, alter), never pixels.
Source information (page, bounding box, staff, confidence) is kept on events
so that debugging and later human correction are possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

STEPS = "CDEFGAB"
_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True)
class Pitch:
    step: str  # "C".."B"
    octave: int  # scientific pitch notation, middle C = C4
    alter: int = 0  # -1 flat, 0 natural, +1 sharp

    def __post_init__(self):
        if self.step not in STEPS:
            raise ValueError(f"bad step {self.step!r}")

    @property
    def diatonic(self) -> int:
        return self.octave * 7 + STEPS.index(self.step)

    @property
    def midi(self) -> int:
        return 12 * (self.octave + 1) + _SEMITONE[self.step] + self.alter

    @property
    def name(self) -> str:
        return f"{self.step}{'#' * self.alter if self.alter > 0 else 'b' * -self.alter}{self.octave}"

    @classmethod
    def from_diatonic(cls, n: int, alter: int = 0) -> "Pitch":
        return cls(STEPS[n % 7], n // 7, alter)

    @classmethod
    def parse(cls, text: str) -> "Pitch":
        step, rest = text[0], text[1:]
        alter = 0
        while rest and rest[0] in "#b":
            alter += 1 if rest[0] == "#" else -1
            rest = rest[1:]
        return cls(step, int(rest), alter)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Source:
    """Where an element came from in the input. bbox is (x0, y0, x1, y1) in page pixels."""

    page: int
    bbox: Optional[tuple[int, int, int, int]] = None
    system: Optional[int] = None
    staff: Optional[int] = None
    component: Optional[int] = None


@dataclass
class Event:
    kind: str  # "note" | "rest"
    duration: Fraction
    pitch: Optional[Pitch] = None  # sounding pitch (alter resolved) for notes
    onset: Fraction = Fraction(0)  # position inside the measure
    measure: int = 0  # index into Voice.measures
    voice: str = "1"
    accidental: Optional[str] = None  # accidental printed in front of the note: "#", "b", "n"
    dots: int = 0
    tie_start: bool = False
    tie_stop: bool = False
    confidence: float = 1.0
    source: Optional[Source] = None

    def __post_init__(self):
        if self.kind not in ("note", "rest"):
            raise ValueError(self.kind)
        if self.kind == "note" and self.pitch is None:
            raise ValueError("note needs a pitch")
        if self.duration <= 0:
            raise ValueError("duration must be positive")


@dataclass
class Measure:
    index: int
    events: list[Event] = field(default_factory=list)
    source: Optional[Source] = None

    @property
    def duration(self) -> Fraction:
        return sum((e.duration for e in self.events), Fraction(0))


@dataclass
class Voice:
    id: str
    label: str = ""
    measures: list[Measure] = field(default_factory=list)
    source_staff: Optional[str] = None  # e.g. "system 1 staff 2"

    def events(self):
        for m in self.measures:
            yield from m.events


@dataclass(frozen=True)
class Meter:
    numerator: int
    denominator: int
    inferred: bool = False

    @property
    def measure_duration(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @classmethod
    def parse(cls, text: str) -> "Meter":
        n, d = text.split("/")
        return cls(int(n), int(d))

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"


_MAJOR_TONICS = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F", 0: "C",
    1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}
_MINOR_TONICS = {
    -7: "Ab", -6: "Eb", -5: "Bb", -4: "F", -3: "C", -2: "G", -1: "D", 0: "A",
    1: "E", 2: "B", 3: "F#", 4: "C#", 5: "G#", 6: "D#", 7: "A#",
}
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"


@dataclass(frozen=True)
class KeySignature:
    fifths: int = 0  # >0 sharps, <0 flats
    mode: str = "major"  # "major" | "minor"; recognition cannot tell, so default major
    inferred: bool = False

    @property
    def tonic(self) -> str:
        return (_MAJOR_TONICS if self.mode == "major" else _MINOR_TONICS)[self.fifths]

    def alter_for(self, step: str) -> int:
        """Alteration the key signature applies to a letter."""
        if self.fifths > 0 and step in _SHARP_ORDER[: self.fifths]:
            return 1
        if self.fifths < 0 and step in _FLAT_ORDER[: -self.fifths]:
            return -1
        return 0

    @classmethod
    def parse(cls, text: str) -> "KeySignature":
        """'G', 'Bb', 'Em', 'F#m' (ABC-style tonic names; only major/minor)."""
        minor = text.endswith("m") or text.lower().endswith("min")
        tonic = text.rstrip("m").replace("min", "").strip()
        table = _MINOR_TONICS if minor else _MAJOR_TONICS
        for fifths, name in table.items():
            if name == tonic:
                return cls(fifths, "minor" if minor else "major")
        raise ValueError(f"unknown key {text!r}")


@dataclass
class Diagnostic:
    code: str
    severity: str  # "error" | "warning" | "info"
    message: str
    source: Optional[Source] = None
    measure: Optional[int] = None

    def __str__(self):
        loc = ""
        if self.source is not None:
            loc = f" [page {self.source.page}"
            if self.source.system is not None:
                loc += f", system {self.source.system + 1}"
            if self.source.staff is not None:
                loc += f", staff {self.source.staff + 1}"
            loc += "]"
        if self.measure is not None:
            loc += f" (measure {self.measure + 1})"
        return f"{self.severity.upper()} {self.code}{loc}: {self.message}"


@dataclass
class StaffInfo:
    index: int  # within system
    lines: list[float]
    x0: int
    x1: int
    spacing: float
    clef: str = "treble"
    clef_confirmed: bool = False


@dataclass
class SystemInfo:
    index: int
    staves: list[StaffInfo] = field(default_factory=list)


@dataclass
class PageInfo:
    number: int  # 1-based
    kind: str  # "vector" | "raster" | "image"
    width_px: int
    height_px: int
    dpi: float
    systems: list[SystemInfo] = field(default_factory=list)
    music_fonts: list[str] = field(default_factory=list)
    skew_degrees: float = 0.0


@dataclass
class Score:
    metadata: dict = field(default_factory=dict)
    meter: Optional[Meter] = None
    key: KeySignature = KeySignature()
    pages: list[PageInfo] = field(default_factory=list)
    voices: list[Voice] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    selection: dict = field(default_factory=dict)  # how the melody staff was chosen

    @property
    def melody(self) -> Optional[Voice]:
        return self.voices[0] if self.voices else None

    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)
