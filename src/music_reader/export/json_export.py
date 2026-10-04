"""JSON exporter: the full Music IR including diagnostics and source information."""
from __future__ import annotations

import json
from dataclasses import is_dataclass
from fractions import Fraction

from ..ir import Pitch, Score


def _convert(o):
    if isinstance(o, Fraction):
        return str(o)
    if isinstance(o, Pitch):
        return o.name
    if is_dataclass(o) and not isinstance(o, type):
        return {k: _convert(v) for k, v in vars(o).items()}
    if isinstance(o, dict):
        return {str(k): _convert(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_convert(v) for v in o]
    return o


def export_json(score: Score, title: str | None = None) -> str:
    data = _convert(score)
    if title:
        data["metadata"]["title"] = title
    data["key"]["tonic"] = score.key.tonic
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
