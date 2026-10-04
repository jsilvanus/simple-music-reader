from __future__ import annotations

from pathlib import Path

import pytest

from tests.synth import SystemSpec, StaffSpec, parse_measures, render_pdf


@pytest.fixture(scope="session")
def pdfdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("pdfs")


@pytest.fixture(scope="session")
def make_pdf(pdfdir):
    """make_pdf(name, 'E4q ... | ...', **staff_kwargs) -> path; one staff, one system."""
    cache = {}

    def make(name, text, sp=5.0, **kw):
        key = (name, text, sp, tuple(sorted(kw.items())))
        if key not in cache:
            path = pdfdir / f"{name}.pdf"
            final = kw.pop("final", True)
            render_pdf(path, [[SystemSpec([StaffSpec(parse_measures(text), final=final, **kw)])]], sp=sp)
            cache[key] = path
        return cache[key]

    return make
