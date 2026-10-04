from __future__ import annotations

from .base import Backend, PageDebug, RecognitionOptions, RecognitionResult
from .classical import ClassicalBackend

BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    BACKENDS[backend.name] = backend


register_backend(ClassicalBackend())


def get_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}") from None
