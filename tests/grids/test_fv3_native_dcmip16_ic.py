"""Direct tests for the DCMIP16_BC face assembler.

WHY THIS FILE EXISTS AT ALL: ``fv3_native_dcmip16_ic`` was left UNTRACKED
while two committed validators (``scripts/validate/fv3_native/
ic_face_map_parity.py`` and ``ic_parity_diagnostics.py``) import it, so a
fresh checkout could not run the very harness that established the 1e-14
IC-parity result. It is staged with these tests.

The oracle cannot be called from here, so these are the identities the
assembler must satisfy for ANY correct implementation, plus a pin on each
of the three "easy to get wrong" behaviours its own docstring names.
"""
import numpy as np
import pytest

from legoesm.core.fv3_native_dcmip16_bc import (
    GFS_CONSTANTS,
    P0_PA,
    temperature,
)
from legoesm.core.fv3_native_dcmip16_ic import dcmip16_bc_face
from legoesm.core.fv3_native_eta import set_eta_analytic
from legoesm.grids.fv3_native_metrics import great_circle_dist as _gcd

KM = 5
C = GFS_CONSTANTS


def _gcdr(p1, p2, r):
    """The validators' wrapper: great_circle_dist on the unit sphere x r."""
    return _gcd(np.asarray(p1, float), np.asarray(p2, float)) * r


def _patch(ni=4, nj=3, lon0=0.30, lat0=0.55, d=0.02, skew=0.0):
    """A small patch of CORNERS and the CENTRES it implies.

    Not a cubed-sphere face -- the assembler is purely local (it only ever
    reads corner pairs and cell centres), so a patch exercises every path
    without dragging in the grid generator.

    ``skew`` shears the lattice so the +j direction is NOT meridional.
    At skew=0 this is a plain lon/lat patch, on which the v-edges run due
    north and ``v`` is legitimately ~0 -- so a skewed patch is required to
    see the e_lon projection at all. That pair is used as a control.
    """
    i = np.arange(ni + 1)[:, None]
    j = np.arange(nj + 1)[None, :]
    lon = lon0 + d * i + skew * d * j + 0.0 * (i + j)
    lat = lat0 + d * j + skew * d * i
    corners = np.stack(np.broadcast_arrays(lon, lat), axis=-1)
    centres = 0.25 * (corners[:-1, :-1] + corners[1:, :-1]
                      + corners[:-1, 1:] + corners[1:, 1:])
    return corners, centres


def _build(**kw):
    corners, centres = _patch()
    ak, bk, _, _ = set_eta_analytic(KM)
    opts = dict(constants=C, do_pert=False, great_circle_dist=_gcdr)
    opts.update(kw)
    return dcmip16_bc_face(corners, centres, ak, bk, KM, **opts), ak, bk


def test_shapes_follow_the_dgrid_staggering():
    out, _, _ = _build()
    ni, nj = 4, 3
    assert out["ps"].shape == (ni, nj)
    assert out["delp"].shape == (ni, nj, KM)
    assert out["pt"].shape == (ni, nj, KM)
    assert out["gz"].shape == (ni, nj, KM + 1)
    assert out["u"].shape == (ni, nj + 1, KM)
    assert out["v"].shape == (ni + 1, nj, KM)


def test_ps_is_exactly_p0_and_delp_closes_the_column():
    """test_cases.F90:6528 -- ps is set to p0, not solved for."""
    out, ak, bk = _build()
    assert np.all(out["ps"] == P0_PA)
    want = (np.diff(ak)[None, None, :]
            + P0_PA * np.diff(bk)[None, None, :])
    assert np.allclose(out["delp"], want, rtol=0.0, atol=0.0)
    assert np.allclose(out["delp"].sum(axis=2), P0_PA - ak[0], rtol=1e-15)


def test_gz_increases_upward_and_starts_at_zero():
    """gz(npz+1) = 0 at the surface (:6577); the march runs upward."""
    out, _, _ = _build()
    assert np.all(out["gz"][:, :, KM] == 0.0)
    assert np.all(np.diff(out["gz"][:, :, ::-1], axis=2) > 0.0)


def test_pt_is_the_hydrostatic_layer_mean_not_the_midpoint_temperature():
    """The FIRST of the three traps the module docstring names.

    ``pt`` is back-derived from the Newton-solved interface heights
    (:6602-6608), NOT evaluated from ``DCMIP16_BC_temperature`` at the
    layer midpoint. The midpoint version looks entirely reasonable, which
    is why this needs a test rather than a comment.
    """
    out, ak, bk = _build()
    gz = out["gz"]
    p = ak + P0_PA * bk                                  # (km+1,)
    rrdgrav = C.grav / C.rdgas
    # p[k] is the TOP interface of layer k; the level below layer km-1 is
    # p0 itself (peln is seeded with log(p0) at :6572).
    p_below = np.concatenate([p[1:KM], [P0_PA]])
    recon = (rrdgrav * (gz[:, :, :KM] - gz[:, :, 1:KM + 1])
             / (np.log(p_below)[None, None, :] - np.log(p[:KM])[None, None, :]))
    assert np.allclose(out["pt"], recon, rtol=1e-13)

    # ...and it is NOT the analytic temperature at the mid-height.
    lat = _patch()[1][..., 1]
    zmid = 0.5 * (gz[:, :, :KM] + gz[:, :, 1:KM + 1])
    tmid = temperature(zmid, lat[:, :, None], C)
    assert not np.allclose(out["pt"], tmid, rtol=1e-6), \
        "pt matches the midpoint temperature -- the Newton march is bypassed"


def test_v_is_the_zonal_wind_projected_onto_the_j_edge():
    """The THIRD trap: BOTH u and v project onto e_lon (:6631, :6679).

    The analytic wind is purely zonal, so ``v`` is non-zero exactly where
    a v-edge is not due north. The control is the unskewed patch, where
    the v-edges ARE meridional and v is correctly ~0 -- so a port that
    (wrongly) projected v onto e_lat would pass on a lon/lat patch and
    only fail on the skewed one. That is why the skew exists.
    """
    ak, bk, _, _ = set_eta_analytic(KM)
    kw = dict(constants=C, do_pert=False, great_circle_dist=_gcdr)

    flat_c, flat_ce = _patch(skew=0.0)
    flat = dcmip16_bc_face(flat_c, flat_ce, ak, bk, KM, **kw)
    assert np.max(np.abs(flat["u"])) > 1.0
    assert np.max(np.abs(flat["v"])) < 1e-10, \
        "v is non-zero on a lon/lat patch, where the +j edge is due north"

    skew_c, skew_ce = _patch(skew=0.6)
    skew = dcmip16_bc_face(skew_c, skew_ce, ak, bk, KM, **kw)
    assert np.max(np.abs(skew["v"])) > 1.0, \
        "v vanished on a SKEWED patch -- it was projected onto e_lat"


def test_the_perturbation_is_localised_and_nonzero():
    """do_pert=True is test_case -13; False is -12 (:6656, :6703)."""
    corners, centres = _patch(lon0=C.ppcenter[0] - 0.02,
                              lat0=C.ppcenter[1] - 0.02)
    ak, bk, _, _ = set_eta_analytic(KM)
    kw = dict(constants=C, great_circle_dist=_gcdr)
    near = dcmip16_bc_face(corners, centres, ak, bk, KM, do_pert=True, **kw)
    base = dcmip16_bc_face(corners, centres, ak, bk, KM, do_pert=False, **kw)
    dnear = np.max(np.abs(near["u"] - base["u"]))
    assert dnear > 1e-2, f"no perturbation at ppcenter (max |du| = {dnear})"

    far_c, far_ce = _patch(lon0=C.ppcenter[0] + 2.5, lat0=-C.ppcenter[1])
    fnear = dcmip16_bc_face(far_c, far_ce, ak, bk, KM, do_pert=True, **kw)
    fbase = dcmip16_bc_face(far_c, far_ce, ak, bk, KM, do_pert=False, **kw)
    dfar = np.max(np.abs(fnear["u"] - fbase["u"]))
    assert dfar < dnear * 1e-3, \
        f"perturbation is not localised: far={dfar} vs near={dnear}"

    # The scalars must not see the perturbation at all (:6656 touches only
    # the wind loops).
    assert np.array_equal(near["pt"], base["pt"])
    assert np.array_equal(near["delp"], base["delp"])


def test_base_state_is_zonally_symmetric():
    """The J&W base state depends on latitude only.

    This is the property that made every zonally-symmetric certificate
    BLIND to face orientation (see fv3_duo_gaps/STATE.md); pinning it here
    documents why a localized feature was needed to find the face map.
    """
    ak, bk, _, _ = set_eta_analytic(KM)
    kw = dict(constants=C, do_pert=False, great_circle_dist=_gcdr)
    a = dcmip16_bc_face(*_patch(lon0=0.30), ak=ak, bk=bk, km=KM, **kw)
    b = dcmip16_bc_face(*_patch(lon0=1.90), ak=ak, bk=bk, km=KM, **kw)
    assert np.allclose(a["pt"], b["pt"], rtol=1e-14)
    assert np.allclose(a["u"], b["u"], rtol=1e-12)


def test_corner_and_centre_extents_must_agree():
    corners, centres = _patch()
    ak, bk, _, _ = set_eta_analytic(KM)
    kw = dict(constants=C, do_pert=False, great_circle_dist=_gcdr)
    with pytest.raises(ValueError, match="one larger"):
        dcmip16_bc_face(corners[:-1], centres, ak, bk, KM, **kw)


def test_ak_bk_length_is_checked():
    corners, centres = _patch()
    ak, bk, _, _ = set_eta_analytic(KM)
    kw = dict(constants=C, do_pert=False, great_circle_dist=_gcdr)
    with pytest.raises(ValueError, match="km"):
        dcmip16_bc_face(corners, centres, ak[:-1], bk[:-1], KM, **kw)


# ---------------------------------------------------------------------- #
# the oracle's constants -- the confound that outranks every other one
# ---------------------------------------------------------------------- #

def test_fv3_pinned_constants_are_the_gfs_set_not_gfdl_and_not_legoesm():
    """FMS ships two constant sets and DEFAULTS to GFDL when neither is
    defined (fmsconstants.F90:67-68). The oracle binary this port is
    scored against is GFS-linked, so every pinned value must match
    ``fms-src/constants/gfs_constants.h`` and NOT legoESM's own.

    ``FV3_KAPPA`` is the one that bites hardest: it enters ``ptop**akap``
    and every ``pk = exp(akap*log(p))``, and the idealised 2/7 differs
    from the oracle's RDGAS/CP_AIR by 7.5e-5 relative -- nine orders
    above the ~1e-14 parity floor.
    """
    from legoesm import constants as legoesm_constants
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_CP_AIR,
        FV3_GRAV,
        FV3_KAPPA,
        FV3_OMEGA,
        FV3_RADIUS_M,
        FV3_RDGAS,
        FV3_RVGAS,
    )

    # gfs_constants.h:33,34,35-36,42,43,47
    assert FV3_RADIUS_M == 6.3712e6      # const-ok: gfs_constants.h:33
    assert FV3_OMEGA == 7.2921e-5        # const-ok: gfs_constants.h:34
    assert FV3_GRAV == 9.80665           # const-ok: gfs_constants.h:35-36
    assert FV3_RDGAS == 287.05           # const-ok: gfs_constants.h:42
    assert FV3_RVGAS == 461.50           # const-ok: gfs_constants.h:43
    assert FV3_CP_AIR == 1004.6          # const-ok: gfs_constants.h:47
    # :53 -- derived, never written out independently
    assert FV3_KAPPA == FV3_RDGAS / FV3_CP_AIR

    # Which of these differ from legoESM's own is MEASURED here, not
    # asserted from memory: an earlier version of this test claimed RDGAS
    # differed and went red, because it does not.
    assert FV3_RDGAS == float(legoesm_constants.R_d), \
        "RDGAS no longer coincides with legoESM R_d -- update the comment"
    for name, pinned, ours in (
            ("cp_air", FV3_CP_AIR, legoesm_constants.c_pd),
            ("radius", FV3_RADIUS_M, legoesm_constants.R_earth),
            ("omega", FV3_OMEGA, legoesm_constants.Omega),
            ("grav", FV3_GRAV, legoesm_constants.g),
            ("rvgas", FV3_RVGAS, legoesm_constants.R_v)):
        assert pinned != float(ours), (
            f"{name}: pinned {pinned} now equals legoESM {float(ours)} -- "
            f"if that is real, say so; until then it means the oracle pin "
            f"was silently replaced by the model's own constant")

    # RDGAS coinciding is the trap: the two SETS still disagree, because
    # cp_air does. Substituting legoESM's pair shifts kappa.
    ours_kappa = float(legoesm_constants.R_d) / float(legoesm_constants.c_pd)
    assert FV3_KAPPA != ours_kappa

    # And the idealised 2/7 is NOT the oracle's kappa either.
    assert FV3_KAPPA != 2.0 / 7.0
    assert abs(FV3_KAPPA - 2.0 / 7.0) / FV3_KAPPA > 1e-5


def test_fv3_kappa_error_would_dwarf_the_parity_floor():
    """Non-vacuity for the test above, in the units that matter.

    Quantifies WHY the constant matters rather than merely asserting it:
    at a typical 1e5 Pa the 2/7 slip moves ``p**kappa`` by ~1e-3
    relative, versus the ~1e-14 quad-geometry floor the IC achieves.
    """
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA

    p = 1.0e5
    rel = abs(p ** FV3_KAPPA - p ** (2.0 / 7.0)) / p ** FV3_KAPPA
    assert rel > 1e-4, f"pk sensitivity to kappa is only {rel}"
