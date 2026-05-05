"""Iter-942 sentinel (REWRITTEN at iter-944b): pin the Fortran-faithful
absence of a `synchronize_corner_scalar(ke_corner)` call inside
`_d_sw_native`.

iter-942 (original): added an UNCONDITIONAL ke_corner scalar sync
between d_sw5 (KE-add) and d_sw6 (KE-gradient), claiming it lifted FB
chain step survival from 63 → 209 steps on non-duogrid C36 W2.

iter-944b (Fortran-fidelity audit): the reference source
`atmos_cubed_sphere-symmetryclean/model/dyn_core.F90:1029-1055` AND
`:1180-1207` show TWO COMMENTED-OUT corner ke sync blocks that map to
iter-942's smoother:

    !if(duogrid) then
    !  ...
    !  call mpp_get_boundary(kee(:,:,:), domain, ...,
    !                        position=CORNER, complete=.true.)
    !  ...
    !endif

Adjacent commentary reads "seems to reduce noise a lot for few timesteps
only" and "is this ok?" — Fortran has explicitly DISABLED these syncs.
iter-942 was Python-only smoothing that improved measured step survival
only by violating the Fortran-fidelity contract.

iter-944b reverted iter-942's sync.  This sentinel pins the absence of
the smoother as a Fortran-fidelity contract.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import inspect


def test_iter942_ke_corner_sync_is_NOT_applied_in_d_sw_native():
    """`_d_sw_native` must NOT contain a
    `synchronize_corner_scalar(ke_corner` call.  Fortran's reference
    has the corresponding ke corner sync explicitly commented out
    (dyn_core.F90:1029-1055, 1180-1207); enabling it in our Python
    port would be Python-only smoothing not present in the reference.
    """
    from legoesm.core import fv3_sw_core

    src = inspect.getsource(fv3_sw_core._d_sw_native)
    bad_patterns = [
        "synchronize_corner_scalar(ke_corner",
        "synchronize_corner_scalar( ke_corner",
    ]
    for pattern in bad_patterns:
        assert pattern not in src, (
            f"`_d_sw_native` contains a forbidden ke_corner sync "
            f"(`{pattern}`).  iter-944b removed iter-942's Python-"
            f"only smoothing because Fortran has the equivalent "
            f"sync commented out at dyn_core.F90:1029-1055 and "
            f":1180-1207.  Re-introducing it violates the Fortran-"
            f"fidelity contract."
        )


def test_iter942_d_sw_native_carries_iter944b_audit_note():
    """`_d_sw_native` must document the iter-944b removal of iter-942's
    sync, including the dyn_core.F90 line numbers from the reference
    source.  Removing the audit comment would lose the "why this is NOT
    here" context and risk a future re-introduction.
    """
    from legoesm.core import fv3_sw_core

    src = inspect.getsource(fv3_sw_core._d_sw_native)
    assert "iter-944b" in src and "iter-942" in src, (
        "`_d_sw_native` must reference iter-944b's revert of iter-942 "
        "(both tags) so future reviewers understand WHY there is no "
        "ke_corner sync."
    )
    assert "1029-1055" in src or "1180-1207" in src, (
        "`_d_sw_native` must cite at least one of the dyn_core.F90 "
        "line ranges where Fortran's ke corner sync is commented "
        "out.  This anchors the Fortran-fidelity claim."
    )
