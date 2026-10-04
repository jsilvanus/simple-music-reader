"""Output formats. Add one by writing ``def export_x(score, title=None) -> str`` and registering it."""
from __future__ import annotations

from typing import Callable

from ..ir import Score
from .abc import export_abc
from .json_export import export_json
from .musicxml import export_musicxml

EXPORTERS: dict[str, Callable[..., str]] = {}


def register_exporter(name: str, fn: Callable[..., str], extension: str | None = None) -> None:
    EXPORTERS[name] = fn
    EXTENSIONS[name] = extension or name


EXTENSIONS: dict[str, str] = {}
register_exporter("abc", export_abc)
register_exporter("json", export_json)
register_exporter("musicxml", export_musicxml, "musicxml")


def export(score: Score, fmt: str, title: str | None = None) -> str:
    try:
        return EXPORTERS[fmt](score, title=title)
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; available: {sorted(EXPORTERS)}") from None
