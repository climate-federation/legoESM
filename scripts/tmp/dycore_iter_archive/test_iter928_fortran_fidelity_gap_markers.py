"""Iter-928 meta-fidelity sentinel: lock the 8 Fortran-fidelity
gap markers identified in the user's iter-927 audit into the
test suite so future iters that PAPER OVER them (rewriting the
comments away, removing the gap acknowledgement, etc.) are
caught at CI time.

This test does NOT enforce that the gaps are CLOSED — closing
them is the iter-927→iter-93x backlog.  It enforces that the gap
DOCUMENTATION remains in place.  Each marker grep below is a
sentinel string from the existing source; if a future iter
silently removes the marker without corresponding code change,
this test fires and forces an explicit accept/reject decision.

The 8 gaps from user iter-927 audit:

1. Production not using true FV3 time step (RK3 vs FB chain).
2. Pressure/Coriolis balance is wrong operator family.
3. Mass transport is not true d_sw1 by default.
4. div_damp is not FV3 d_sw5 (deferred port comment).
5. boundary_fix is Python-only stabilizer.
6. Real d_sw5 machinery exists only in experimental FB path.
7. Cube-vertex halo handling NOT PORTED in non-duogrid path.
8. FV3FBShallowWaterModel marked experimental (unstable).

Each test below greps for a load-bearing sentinel substring in
the file the user pointed at.  All 8 must be present.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.legoesm_paths import legoesm_source_path


REPO = Path(__file__).resolve().parents[1]
SW_FV3_CDGRID = legoesm_source_path(
    "atmosphere/dynamics/gcm/shallow_water_fv3_cdgrid.py"
)
OPERATORS_CDGRID = legoesm_source_path("core/operators_cdgrid.py")
FV3_SW_CORE = legoesm_source_path("core/fv3_sw_core.py")


def _file_contains(path: Path, needle: str) -> bool:
    text = path.read_text()
    return needle in text


@pytest.mark.parametrize(
    "issue, path, needle",
    [
        # 1. Production not using true FV3 time step.
        # The FV3FBShallowWaterModel docstring/class header acknowledges
        # FV3EdgeShallowWaterModel uses RK3 + fv3_sw_tendencies, NOT
        # the c_sw → p_grad_c → d_sw1/d_sw4/d_sw5/d_sw6 chain.
        (
            "issue1_production_not_true_fv3_timestep",
            SW_FV3_CDGRID,
            "fv3_fb_sw_step",
        ),
        # 2. Pressure/Coriolis balance is wrong operator family.
        # operators_cdgrid.py contains the cell-centre A-L formula
        #   dv_cc = -zeta_abs * u_cc - dB_dy_cc
        # which the user explicitly identified as the wrong family.
        (
            "issue2_pressure_coriolis_wrong_operator_family",
            OPERATORS_CDGRID,
            "dv_cc = -zeta_abs * u_cc - dB_dy_cc",
        ),
        # 3. Mass transport is not true d_sw1 by default.
        # cgrid_mass_flux_divergence is the production default, gated
        # by use_fv3_dsw1_mass_transport opt-in (which is rejected).
        (
            "issue3_mass_transport_not_true_dsw1_by_default",
            OPERATORS_CDGRID,
            "use_fv3_dsw1_mass_transport",
        ),
        # 4. div_damp is not FV3 d_sw5: comment block at
        # operators_cdgrid.py:2025-2031 explicitly defers the
        # holistic d_sw5 port.
        (
            "issue4_divdamp_is_not_fv3_dsw5_deferred_holistic_port",
            OPERATORS_CDGRID,
            "deferred to a\n    # dedicated iter that ports d_sw5 holistically",
        ),
        # 5. boundary_fix is Python-only stabilizer.
        # operators_cdgrid.py contains explicit acknowledgement.
        (
            "issue5_boundary_fix_python_only_stabilizer",
            OPERATORS_CDGRID,
            "Fortran has NO post-tendency smoothing analog",
        ),
        # 6. Real d_sw5 machinery exists only in experimental FB path.
        # d_sw5_corner_divergence's docstring says production
        # fv3_sw_tendencies does NOT call this helper.
        (
            "issue6_real_dsw5_only_in_fb_path",
            FV3_SW_CORE,
            "Production ``fv3_sw_tendencies`` does NOT call this helper",
        ),
        # 7. Cube-vertex halo handling NOT PORTED in non-duogrid path.
        # d2a2c_vect documents the gap explicitly.
        (
            "issue7_cube_vertex_halo_not_ported_non_duogrid",
            FV3_SW_CORE,
            "NOT PORTED in Python's non-duogrid path",
        ),
        # 8. FV3FBShallowWaterModel marked experimental (unstable).
        # The class docstring acknowledges this.
        (
            "issue8_fb_model_experimental_unstable",
            SW_FV3_CDGRID,
            "FV3FBShallowWaterModel",
        ),
        # 9. fv3_sw_tendencies hyperdiff_coeff is silent no-op
        # (signature-only; cdgrid_momentum_tendencies implements it
        # but fv3_sw_tendencies does not).  Iter-1019 added a
        # UserWarning to make this explicit instead of silent.
        (
            "issue9_fv3_sw_tendencies_hyperdiff_silent_noop",
            OPERATORS_CDGRID,
            "is silently ignored on this code path",
        ),
    ],
)
def test_iter928_fortran_fidelity_gap_marker_present(issue, path, needle):
    """Each user-identified gap marker must remain in source.

    If a future iter PAPERS OVER the marker (removes the comment
    documenting the gap) without corresponding code change to
    actually close the gap, this test fires.

    To resolve a fired gate, EITHER:
    1. Close the gap (rewrite the production code to be Fortran-
       faithful for that operator) AND remove this parametric
       entry — i.e., the gap is genuinely closed, no marker needed.
    2. Update the marker substring in this test to match a renamed
       comment that still acknowledges the gap.
    DO NOT silently remove the source comment without removing
    this parametric entry first.
    """
    assert path.exists(), f"{path} does not exist"
    assert _file_contains(path, needle), (
        f"Issue {issue}: marker '{needle[:60]}' not found in {path.name}.  "
        f"Either the gap was closed (remove this parametric entry) or "
        f"the source comment was edited (update the marker substring)."
    )


def test_iter928_marker_count_invariant():
    """The 9-issue audit must remain at exactly 9 entries.  Adding new
    gaps requires conscious choice of marker; removing entries
    requires explicit gap closure.  Locking the count guards against
    accidental drift in either direction.

    Iter-1019 added issue9 (fv3_sw_tendencies hyperdiff silent no-op)
    after Codex audit.  The hyperdiff branch in fv3_sw_tendencies
    is signature-only (the implementation exists in
    cdgrid_momentum_tendencies but is NOT applied in
    fv3_sw_tendencies).  Iter-1019 added a UserWarning so callers
    don't silently get zero damping; the underlying structural gap
    remains.
    """
    expected_count = 9
    # Inspect the parametrize decorator on the parametric test
    # function rather than grepping the source file; that avoids
    # false positives on `issueN_...` references in the module
    # docstring or comments.
    marks = test_iter928_fortran_fidelity_gap_marker_present.pytestmark
    assert len(marks) == 1, (
        f"Expected exactly one @pytest.mark.parametrize on the gap-"
        f"marker test; found {len(marks)}."
    )
    parametrize_args = marks[0].args[1]
    actual_count = len(list(parametrize_args))
    assert actual_count == expected_count, (
        f"iter-928 expects exactly {expected_count} gap markers; "
        f"found {actual_count} parametric entries.  If you added/"
        f"removed a gap, update both the parametric list above AND "
        f"this expected count in lockstep."
    )
