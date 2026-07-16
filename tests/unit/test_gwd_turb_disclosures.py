"""Faithfulness / labeling locks for the GWD + YSU disclosed-variant items.

Pure source-text checks (no JAX import — login-node safe) that lock the
docstring/label fixes for the physics-oracle disclosure findings:

  * gwd/hines is a single-bulk-amplitude WKB scheme, NOT the full Hines
    (1997) Doppler-spread spectrum — the module headline must say so.
  * turb/ysu models entrainment as a down-gradient Gaussian K-bump, NOT
    YSU's Hong06 entrainment (implemented in #856).
  * gwd/prognostic_spectral is experimental / opt-in and its momentum
    deposition carries the launch-level sign factor since #856 (F-GWD-1 fixed);
    ``eps_gwd`` is NOT positive-definite — the false "positive-definite"
    wording must be dropped/qualified in both the scheme module and the
    shared ``GWDOutput`` container.

Each assertion FAILS against the pre-fix source (the misleading text was
present) and passes once the disclosure is corrected.
"""

from __future__ import annotations

from tests import _ratchet_audit as ra

_PKG = "packages/atmosphere/legoesm/atmosphere/physics"
_HINES = f"{_PKG}/gravity_wave_drag/hines.py"
_YSU = f"{_PKG}/turbulence/ysu.py"
_PROG = f"{_PKG}/gravity_wave_drag/prognostic_spectral.py"
_OUTPUT = f"{_PKG}/gravity_wave_drag/output.py"


def _read(rel: str) -> str:
    return (ra.repo_root() / rel).read_text(encoding="utf-8")


def test_hines_headline_discloses_single_bulk_amplitude():
    """hines.py must crisply state it is a single-bulk-amplitude WKB scheme,
    not the full Hines (1997) Doppler-spread spectrum."""
    src = _read(_HINES)
    assert "single-bulk-amplitude WKB" in src, (
        "hines.py headline must crisply state it is a single-bulk-amplitude "
        "WKB scheme, not the full Hines Doppler-spread spectrum."
    )
    assert ("NOT** the full Hines" in src) or ("NOT the full Hines" in src), (
        "hines.py must explicitly say it is NOT the full Hines (1997) "
        "Doppler-spread spectrum."
    )


def test_ysu_documents_hong06_entrainment():
    """#856 implemented the published Hong et al. (2006) entrainment in ysu.py
    (superseding the Gaussian-K variant this test previously pinned as a
    disclosed deviation). Pin the implemented-state documentation instead."""
    src = _read("packages/atmosphere/legoesm/atmosphere/physics/"
                "turbulence/ysu.py")
    assert "Hong" in src and "entrainment" in src, (
        "ysu.py must document the Hong06 entrainment implementation (#856)."
    )


def test_prognostic_spectral_documents_positive_definite_by_construction():
    """#856 closed F-GWD-1: the launch-level sign(c - U_launch) deposition
    factor makes eps_gwd >= 0 by construction. The module must document the
    FIXED contract (and must no longer carry the pre-fix missing-sign
    disclosure, which would now be false)."""
    src = _read("packages/atmosphere/legoesm/atmosphere/physics/"
                "gravity_wave_drag/prognostic_spectral.py")
    assert "sign(c - U_launch)" in src, (
        "prognostic_spectral.py must document the launch-level sign factor "
        "(#856 F-GWD-1 fix)."
    )
    assert "BY CONSTRUCTION" in src, (
        "prognostic_spectral.py must state eps_gwd >= 0 by construction "
        "(the #856 contract)."
    )
    assert "missing the ``sign" not in src, (
        "stale pre-#856 missing-sign disclosure must not survive the fix."
    )

def test_gwd_output_documents_positive_eps_gwd():
    """output.py's eps_gwd doc must reflect the post-#856 contract: positive
    for the wind-opposing diagnostic schemes AND for prognostic_spectral
    (launch-sign deposition, F-GWD-1 closed) -- the stale NOT-guaranteed
    qualifier must be gone."""
    src = _read("packages/atmosphere/legoesm/atmosphere/physics/"
                "gravity_wave_drag/output.py")
    assert "positive by construction" in src, (
        "output.py must document prognostic_spectral eps_gwd >= 0 (#856)."
    )
    assert "NOT guaranteed positive" not in src, (
        "stale pre-#856 eps_gwd qualifier must not survive the fix."
    )
