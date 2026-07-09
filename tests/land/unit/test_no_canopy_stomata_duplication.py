"""Dedup guard: the stomatal-conductance / photosynthesis roster lives in ONE
neutral module (``legoesm.land.stomata``), not duplicated across ``canopy/``
and ``carbon/``.  Substring-based (mirrors the ocean
``test_no_scheme_duplication`` precedent) so re-introducing a per-subpackage
copy goes red with a message pointing at the offending file.
"""

from __future__ import annotations

import pathlib

_LAND = pathlib.Path(__file__).resolve().parents[3] / "packages" / "land" / "legoesm" / "land"
_STOMATA = _LAND / "stomata.py"
_ENERGY_BALANCE = _LAND / "canopy" / "energy_balance.py"


def test_single_neutral_stomata_module():
    # The one file exists at the neutral land top level (not canopy/ or carbon/).
    assert _STOMATA.is_file(), "legoesm/land/stomata.py must be the single stomata module"
    assert not (_LAND / "carbon" / "stomata.py").exists(), (
        "carbon/stomata.py must not exist — stomata moved to the neutral land/stomata.py")
    assert not (_LAND / "canopy" / "stomatal.py").exists(), (
        "canopy/stomatal.py must not exist — folded into land/stomata.py")
    assert not (_LAND / "stomatal_kernels.py").exists(), (
        "stomatal_kernels.py must not exist — folded into land/stomata.py")
    src = _STOMATA.read_text()
    assert "def ball_berry_gs(" in src and "def medlyn_gs(" in src
    # The H2O:CO2 diffusivity ratio is single-sourced in leaf_biophysics; the
    # stomata module imports it (does not carry its own literal copy).
    assert "DIFFUSIVITY_RATIO_H2O_CO2" in src
    assert "_VPD_FLOOR_KPA = 0.05" in src


def test_diffusivity_ratio_single_source():
    # The H2O:CO2 diffusivity ratio is a shared physical constant: defined in
    # exactly ONE place (leaf_biophysics.py); every consumer imports it.  A
    # module-level ``DIFFUSIVITY_RATIO_H2O_CO2 = <literal>`` anywhere else is a
    # re-derivation and goes red.
    definers = []
    for py in _LAND.rglob("*.py"):
        for line in py.read_text().splitlines():
            stripped = line.lstrip()
            if stripped.startswith("DIFFUSIVITY_RATIO_H2O_CO2 =") and "import" not in line:
                definers.append(str(py.relative_to(_LAND)))
    assert definers == ["leaf_biophysics.py"], (
        f"H2O:CO2 diffusivity ratio must be defined only in leaf_biophysics.py, "
        f"got {definers}")


def test_no_other_module_defines_the_gs_kernels():
    # Exactly one definition of each kernel across the whole land package.
    offenders = []
    for py in _LAND.rglob("*.py"):
        if py == _STOMATA:
            continue
        src = py.read_text()
        if "def ball_berry_gs(" in src or "def medlyn_gs(" in src:
            offenders.append(str(py.relative_to(_LAND)))
    assert not offenders, f"gs kernels re-defined outside land/stomata.py: {offenders}"


def test_canopy_energy_balance_imports_from_neutral_module():
    src = _ENERGY_BALANCE.read_text()
    assert "from legoesm.land.stomata import ball_berry_gs, medlyn_gs" in src, (
        "canopy/energy_balance.py must import the gs kernels from land.stomata")
    # ...and must not re-inline the formula.
    assert "* RH / Cs_safe" not in src and "jnp.sqrt(VPD" not in src, (
        "canopy/energy_balance.py re-inlines the gs formula; call land.stomata instead")
