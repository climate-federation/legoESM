"""Enforcement test for ocean dynamics scheme consolidation (#214).

Verifies that the shared helpers introduced under
``src/legoesm/ocean/dynamics/`` are actually used by the grid-specific
``ocean_pe_*`` and ``barotropic_*`` files instead of being silently
duplicated again the next time someone fixes a bug.

The checks are intentionally narrow: they look for the *pattern* that
was previously copied across files and verify it now lives in the
common module, not for full structural equivalence (which is impossible
because the grid-specific stencils legitimately differ).

If a future change re-introduces a copy of one of the consolidated
patterns into a grid-specific file, this test will fail and point at
the offending file so the duplication can be folded back into
``ocean_tendency_common`` or ``barotropic_common``.
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
DYN = ROOT / "src" / "legoesm" / "ocean" / "dynamics"

# Files that previously held duplicated logic and now should delegate
# to the common modules.
OCEAN_PE_FILES = [
    DYN / "ocean_pe_cdgrid.py",
    DYN / "ocean_pe_latlon_cgrid.py",
    DYN / "ocean_pe_mpas.py",
]
BAROTROPIC_FILES_WITH_FILTER = [
    DYN / "barotropic_latlon_cgrid.py",
    DYN / "barotropic_mpas.py",
]
BAROTROPIC_FILES_WITH_BOTTOM_DRAG = [
    DYN / "barotropic_latlon_cgrid.py",
    DYN / "barotropic_mpas.py",
]


def _read(path: pathlib.Path) -> str:
    return path.read_text()


# ---------------------------------------------------------------------
# Common modules exist
# ---------------------------------------------------------------------

def test_ocean_tendency_common_module_present():
    """``ocean_tendency_common.py`` must exist and export the helpers."""
    mod = DYN / "ocean_tendency_common.py"
    assert mod.is_file(), f"{mod} should exist (introduced by #214)"
    text = _read(mod)
    for name in (
        "iterate_eos_and_pressure_anomaly",
        "apply_sponge_tracer_relaxation",
        "apply_freshwater_virtual_salt_top",
        "implicit_bottom_drag_factor",
    ):
        assert f"def {name}" in text, (
            f"ocean_tendency_common.py is missing helper {name}; the "
            "ocean PE files will silently regress to inline copies."
        )


def test_barotropic_common_module_present():
    """``barotropic_common.py`` must exist and export the helpers."""
    mod = DYN / "barotropic_common.py"
    assert mod.is_file(), f"{mod} should exist (introduced by #214)"
    text = _read(mod)
    for name in ("compute_filter_weights", "bebt_blend", "maxvel_clip"):
        assert f"def {name}" in text, (
            f"barotropic_common.py is missing helper {name}; the "
            "barotropic solvers will silently regress to inline copies."
        )


# ---------------------------------------------------------------------
# Phase 1: baroclinic tendency consolidation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("path", OCEAN_PE_FILES, ids=lambda p: p.name)
def test_ocean_pe_uses_eos_helper(path: pathlib.Path):
    """Each ocean_pe_*.py must call the shared EOS-iteration helper."""
    text = _read(path)
    assert "iterate_eos_and_pressure_anomaly" in text, (
        f"{path.name} no longer routes the EOS / pressure-anomaly "
        "iteration through ocean_tendency_common — please refactor to "
        "call iterate_eos_and_pressure_anomaly instead of duplicating "
        "the 2-pass loop inline."
    )


@pytest.mark.parametrize("path", OCEAN_PE_FILES, ids=lambda p: p.name)
def test_ocean_pe_no_inline_eos_loop(path: pathlib.Path):
    """No grid-specific file may keep an inline 2-pass EOS loop."""
    text = _read(path)
    # The exact pattern previously duplicated across all three files.
    assert "for _ in range(2):" not in text, (
        f"{path.name} contains an inline ``for _ in range(2):`` — use "
        "ocean_tendency_common.iterate_eos_and_pressure_anomaly instead."
    )
    # ``compute_hydrostatic_pressure`` should also be routed through the
    # helper.  None of the three files needs to import it directly after
    # the refactor.
    assert "compute_hydrostatic_pressure" not in text, (
        f"{path.name} directly references compute_hydrostatic_pressure — "
        "route through iterate_eos_and_pressure_anomaly so the EOS / "
        "reference-Jacobian convention stays in one place."
    )


def test_latlon_cgrid_pe_uses_sponge_helper():
    """Lat-lon C-grid PE delegates tracer sponge to the helper."""
    text = _read(DYN / "ocean_pe_latlon_cgrid.py")
    assert "apply_sponge_tracer_relaxation" in text, (
        "ocean_pe_latlon_cgrid.py should reuse "
        "apply_sponge_tracer_relaxation instead of re-implementing the "
        "tracer sponge inline."
    )


def test_mpas_pe_uses_sponge_helper():
    """MPAS PE delegates tracer sponge + virtual-salt to the helpers."""
    text = _read(DYN / "ocean_pe_mpas.py")
    assert "apply_sponge_tracer_relaxation" in text
    assert "apply_freshwater_virtual_salt_top" in text


# ---------------------------------------------------------------------
# Phase 2: barotropic substep consolidation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("path", BAROTROPIC_FILES_WITH_FILTER,
                         ids=lambda p: p.name)
def test_barotropic_uses_filter_helper(path: pathlib.Path):
    """Cosine/box filter weights come from the shared helper."""
    text = _read(path)
    assert "compute_filter_weights" in text, (
        f"{path.name} should call barotropic_common.compute_filter_weights "
        "instead of inlining the cosine bell formula."
    )
    # Reject the previous inline cosine formulation.
    assert "1.0 + jnp.cos(" not in text, (
        f"{path.name} still contains an inline ``1.0 + jnp.cos(...)`` "
        "filter — please use compute_filter_weights."
    )


@pytest.mark.parametrize("path", BAROTROPIC_FILES_WITH_FILTER,
                         ids=lambda p: p.name)
def test_barotropic_uses_bebt_helper(path: pathlib.Path):
    text = _read(path)
    assert "bebt_blend" in text, (
        f"{path.name} should call barotropic_common.bebt_blend for the "
        "semi-implicit eta blending."
    )


@pytest.mark.parametrize("path", BAROTROPIC_FILES_WITH_FILTER,
                         ids=lambda p: p.name)
def test_barotropic_uses_maxvel_helper(path: pathlib.Path):
    text = _read(path)
    # Either the helper is used or MAXVEL is not present at all.
    assert "maxvel_clip" in text or "_maxvel" not in text


@pytest.mark.parametrize("path", BAROTROPIC_FILES_WITH_BOTTOM_DRAG,
                         ids=lambda p: p.name)
def test_barotropic_uses_bottom_drag_helper(path: pathlib.Path):
    text = _read(path)
    assert "implicit_bottom_drag_factor" in text, (
        f"{path.name} should call ocean_tendency_common."
        "implicit_bottom_drag_factor for the per-substep bottom drag."
    )


# ---------------------------------------------------------------------
# Common modules are import-safe
# ---------------------------------------------------------------------

def test_common_modules_import_cleanly():
    """The new common modules must import without grid-specific deps."""
    import importlib

    mod_oc = importlib.import_module(
        "legoesm.ocean.dynamics.ocean_tendency_common")
    mod_bt = importlib.import_module(
        "legoesm.ocean.dynamics.barotropic_common")

    # Public helpers are exposed.
    for name in (
        "iterate_eos_and_pressure_anomaly",
        "apply_sponge_tracer_relaxation",
        "apply_freshwater_virtual_salt_top",
        "implicit_bottom_drag_factor",
    ):
        assert hasattr(mod_oc, name), (
            f"ocean_tendency_common does not expose {name}")
    for name in ("compute_filter_weights", "bebt_blend", "maxvel_clip"):
        assert hasattr(mod_bt, name), (
            f"barotropic_common does not expose {name}")
