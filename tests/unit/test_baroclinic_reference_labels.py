"""The baroclinic matrix family is labelled with the reference it implements.

Every grid builds the DCMIP-2016 / Ullrich et al. (2014) baroclinic-wave
state (tests/test_cases/baroclinic_wave.py; the cube through the FV3
DCMIP16_BC port), not the Jablonowski & Williamson (2006) / DCMIP 2008
state, and the DCMIP 2008 document numbers its rotated variants 1-3-0 /
2-3-0 (there is no "4-1"/"4-2").  These checks keep the labels honest.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]

LABELLED = [
    "tests/test_cases/baroclinic_wave.py",
    "tests/test_cases/dcmip2008/jablonowski_rotated.py",
    "scripts/matrix/run_atmosphere_test_matrix.py",
    "scripts/matrix/_cube_duo_adapters.py",
    "docs/validation/dycore_validation_catalog.md",
    "README.md",
]


def _norm(text: str) -> str:
    """Dashes -> '-', whitespace runs (incl. newlines) -> one space."""
    text = re.sub("[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"\s+", " ", text)


def test_baroclinic_module_cites_ullrich_2014_not_jw06_as_its_reference():
    import tests.test_cases.baroclinic_wave as bw
    import tests.test_cases.dcmip2008.jablonowski_rotated as rot
    doc = _norm(bw.__doc__)
    assert ("Ullrich, P. A., Melvin, T., Jablonowski, C., & Staniforth, A. "
            "(2014)") in doc
    assert "DCMIP-2016" in doc
    # the rotated module names the state it rotates and the 2008 numbers
    rdoc = _norm(rot.__doc__)
    assert "Ullrich et al. (2014)" in rdoc
    assert "1-3-0" in rdoc and "2-3-0" in rdoc


# stale labels (matched on dash/whitespace-normalised text, so a phrase
# split over two lines or written with an en dash is still caught):
# catalog ids, the nonexistent 2008 "4-1/4-2", and calling this state
# Jablonowski-Williamson / JW06 / J-W.  Historical citations stay allowed.
BAD = re.compile(
    r"JW06-(BW|BS|RS|RB)|§4-[12]\b|DCMIP-? ?2008 4-[12x]\b|4-1 ?/ ?§?4-2"
    r"|rotated JW\b|\bJ-W\b|JW06 (balanced|solution|baroclinic)"
    r"|Jablonowski-? ?Williamson (\(?2006\)? )?baroclinic[- ]wave"
    r"|rotated Jablonowski-? ?Williamson|for JW06\)",
    re.IGNORECASE)

STALE = ["# canonical J-W", "Rotated JW steady-state (α=π/4)",
         "Initialize Jablonowski-Williamson baroclinic wave on MPAS",
         "Jablonowski\u2013Williamson baroclinic wave",
         "Jablonowski-Williamson\n    baroclinic wave",
         '"""DCMIP 2008 4-x: rotated Jablonowski-Williamson tests.',
         "analytic JW06 balanced state", "canonical DCMIP-2008 4-1 / 4-2",
         "Rotated J-W on a lat-lon grid", "DCMIP 2008 §4-1 / §4-2",
         "JW06 baroclinic wave", "rotated J\u2013W", "2008 rotated cases 4-1/4-2",
         "Rotates the standard Jablonowski-Williamson (2006) baroclinic-wave"
         " test"]
HISTORICAL = ["Jablonowski & Williamson 2006, QJRMS 132",
              "(NOT the Jablonowski-Williamson 2006 state the DCMIP 2008 "
              "tests use)",
              "(Related, different test: Jablonowski & Williamson 2006, "
              "QJRMS 132.)"]


def test_no_jw06_ids_or_nonexistent_dcmip2008_sections():
    assert all(BAD.search(_norm(s)) for s in STALE), [
        s for s in STALE if not BAD.search(_norm(s))]
    assert not any(BAD.search(_norm(s)) for s in HISTORICAL)
    hits = []
    for rel in LABELLED:
        text = _norm((ROOT / rel).read_text())
        hits += [f"{rel}: ...{text[max(0, m.start() - 40):m.end() + 20]}..."
                 for m in BAD.finditer(text)]
    assert not hits, "\n".join(hits)
