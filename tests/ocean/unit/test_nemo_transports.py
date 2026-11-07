"""Unit test for the NEMO AMOC reference reader's pure core (amoc_core).

``scripts/validate/nemo_transports.amoc_core`` computes the Atlantic MOC
strength from pure arrays (split out from the NetCDF I/O for testability).
Exercise it on a SYNTHETIC analytic overturning with a known streamfunction
peak, the Atlantic longitude mask, and the deep-AABW exclusion.  Pure NumPy.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Load the validator script by path (scripts/ is not an importable package).
_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "nemo_transports", _ROOT / "scripts" / "validate" / "nemo_transports.py")
_nt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nt)


def _synthetic(nz=10, ny=20, nx=8, j_target=None, atlantic=True, deep_cell=False):
    """Build (voe3, e1v, gphiv, glamv, depthv) with an analytic overturning at
    the row nearest 26.5 N: a(z) = +1 (upper half) / -1 (lower half), e1v=1e6,
    nx cells in band -> ψ = -8·cumsum(a) [Sv], peak |ψ-ψ_surf| = 8*nz/2*... ."""
    depthv = np.array([100., 300., 600., 1000., 1500., 2200., 2800.,
                       3500., 4200., 5000.])[:nz]
    lat_axis = np.linspace(-30.0, 60.0, ny)
    gphiv = np.broadcast_to(lat_axis[:, None], (ny, nx)).copy()
    lon = -35.0 if atlantic else 160.0                    # in / out of [-75,15]
    glamv = np.full((ny, nx), lon)
    e1v = np.full((ny, nx), 1.0e6)
    j = int(np.argmin(np.abs(lat_axis - 26.5))) if j_target is None else j_target
    a = np.where(np.arange(nz) < nz // 2, 1.0, -1.0)      # + upper, - lower
    voe3 = np.zeros((nz, ny, nx))
    voe3[:, j, :] = a[:, None]
    if deep_cell:
        # A strong abyssal overturning below 3000 m that must be EXCLUDED.
        voe3[7:, j, :] = 10.0
    return voe3, e1v, gphiv, glamv, depthv, j


def test_amoc_core_analytic_peak():
    voe3, e1v, gphiv, glamv, depthv, j = _synthetic()
    r = _nt.amoc_core(voe3, e1v, gphiv, glamv, depthv, target_lat=26.5)
    # ψ = -cumsum(8e6·a)/1e6 = -8·cumsum(a); a=[+1×5,-1×5] -> cumsum peaks at 5,
    # ψ' = ψ-ψ[0]; peak |ψ'| in upper (depth<3000) = 8*(5-1) = 32 Sv at z=4.
    assert r["amoc_Sv"] == np.float64(32.0) or abs(r["amoc_Sv"] - 32.0) < 1e-9
    assert abs(r["row_lat_deg"] - 26.5) < 5.0
    assert r["j"] == j


def test_amoc_core_atlantic_mask_excludes_pacific():
    """A Pacific-longitude overturning (outside the Atlantic band) -> ~0 AMOC."""
    voe3, e1v, gphiv, glamv, depthv, _ = _synthetic(atlantic=False)
    r = _nt.amoc_core(voe3, e1v, gphiv, glamv, depthv, target_lat=26.5)
    assert r["amoc_Sv"] < 1e-9


def test_amoc_core_excludes_deep_aabw_cell():
    """A huge abyssal (>3000 m) overturning is excluded; AMOC stays the upper
    32 Sv, not the deep cell."""
    voe3, e1v, gphiv, glamv, depthv, _ = _synthetic(deep_cell=True)
    r = _nt.amoc_core(voe3, e1v, gphiv, glamv, depthv, target_lat=26.5)
    assert abs(r["amoc_Sv"] - 32.0) < 1e-9
    assert r["depth_of_max_m"] < 3000.0


# ---------------------------------------------------------------------------
# ACC@Drake (acc_drake_core) — fixed-i meridian section, signed, regularity-checked
# ---------------------------------------------------------------------------
_LON_ACC = np.array([-90., -80., -70., -68., -66., -60., -55., -50.])  # -68 exact
_E2U = 1.0e5     # u-face meridional width [m] (100 km)
_U_COL = 200.0   # depth-integrated zonal transport per U-point [m^2/s]
_BAND = (-65.0, -45.0)   # default Drake band (matches diagnostics_climate)


def _synthetic_acc(nz=5, sign=1.0, extra_lon_col=None, lon_shear_deg_per_row=0.0):
    """(uoe3, e2u, gphiu, glamu, lat_axis) with eastward flow concentrated at the
    Drake meridian (lon = -68, exact in ``_LON_ACC``).  Depth-integrated u at
    that column = ``sign*_U_COL`` for every row; e2u = _E2U everywhere.
    ``lon_shear_deg_per_row`` tilts glamu with the j-index to emulate a
    CURVILINEAR grid where a constant i is no longer a meridian."""
    lat_axis = np.linspace(-80.0, -40.0, 20)
    nx = _LON_ACC.size
    ny = lat_axis.size
    gphiu = np.broadcast_to(lat_axis[:, None], (ny, nx)).copy()
    glamu = np.broadcast_to(_LON_ACC[None, :], (ny, nx)).copy()
    if lon_shear_deg_per_row:
        glamu = glamu + lon_shear_deg_per_row * (np.arange(ny)[:, None] - ny // 2)
    e2u = np.full((ny, nx), _E2U)
    i_drake = int(np.argmin(np.abs(_LON_ACC - (-68.0))))
    uoe3 = np.zeros((nz, ny, nx))
    uoe3[:, :, i_drake] = sign * _U_COL / nz                # Σ_z = sign*_U_COL
    if extra_lon_col is not None:                            # flow off the section
        uoe3[:, :, extra_lon_col] = 5.0 * _U_COL / nz
    return uoe3, e2u, gphiu, glamu, lat_axis


def _n_band_rows(lat_axis, lat_south=_BAND[0], lat_north=_BAND[1]):
    return int(((lat_axis >= lat_south) & (lat_axis <= lat_north)).sum())


def test_acc_core_analytic_transport():
    """Uniform eastward Drake-section flow -> SIGNED transport = n_band·U_col·e2u,
    on the fixed -68 column with zero longitude deviation (regular grid)."""
    uoe3, e2u, gphiu, glamu, lat_axis = _synthetic_acc()
    r = _nt.acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=-68.0)
    expect = _n_band_rows(lat_axis) * _U_COL * _E2U / 1.0e6
    assert r["n_rows"] == _n_band_rows(lat_axis)
    assert abs(r["acc_Sv"] - expect) < 1e-9                  # signed, positive
    assert abs(r["section_lon_deg"] - (-68.0)) < 1e-9
    assert r["max_lon_dev_deg"] < 1e-9                       # regular -> no drift


def test_acc_core_picks_only_section_meridian():
    """Flow at a column OFF the fixed Drake meridian is NOT summed."""
    uoe3, e2u, gphiu, glamu, lat_axis = _synthetic_acc(extra_lon_col=0)  # lon -90
    r = _nt.acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=-68.0)
    expect = _n_band_rows(lat_axis) * _U_COL * _E2U / 1.0e6
    assert abs(r["acc_Sv"] - expect) < 1e-9                  # extra column ignored


def test_acc_core_band_excludes_out_of_band_rows():
    """A narrower band cuts the transport proportionally (fewer section rows)."""
    uoe3, e2u, gphiu, glamu, lat_axis = _synthetic_acc()
    r = _nt.acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=-68.0,
                           lat_south=-60.0, lat_north=-58.0)
    n = _n_band_rows(lat_axis, -60.0, -58.0)
    assert r["n_rows"] == n
    assert abs(r["acc_Sv"] - n * _U_COL * _E2U / 1.0e6) < 1e-9
    assert n < _n_band_rows(lat_axis)                        # genuinely narrower


def test_acc_core_sign_preserved():
    """Sign is PRESERVED (not abs): eastward -> +, westward -> -, equal |.|.
    A reversed U convention would surface as a sign flip, not be hidden."""
    pos = _nt.acc_drake_core(*_synthetic_acc(sign=+1.0)[:4], drake_lon=-68.0)
    neg = _nt.acc_drake_core(*_synthetic_acc(sign=-1.0)[:4], drake_lon=-68.0)
    assert pos["acc_Sv"] > 0.0
    assert neg["acc_Sv"] < 0.0
    assert abs(pos["acc_Sv"] + neg["acc_Sv"]) < 1e-9        # equal magnitude


def test_acc_core_circular_drake_lon():
    """drake_lon given as +292 (== -68 mod 360) selects the same section."""
    args = _synthetic_acc()[:4]
    a = _nt.acc_drake_core(*args, drake_lon=-68.0)
    b = _nt.acc_drake_core(*args, drake_lon=292.0)
    assert a["i_col"] == b["i_col"]
    assert abs(a["acc_Sv"] - b["acc_Sv"]) < 1e-9


# ---------------------------------------------------------------------------
# MHT (mht_core) — global meridional ocean heat transport
# ---------------------------------------------------------------------------
def _synthetic_mht(nz=4, ny=20, nx=8, vrow=0.5, theta=10.0, j_lat=30.0):
    """(voe3, theta_v, e1v, gphiv, j): northward (vrow>0) warm (theta degC) flow
    concentrated at the v-row nearest ``j_lat`` -> poleward heat transport there."""
    lat = np.linspace(-60.0, 60.0, ny)
    gphiv = np.broadcast_to(lat[:, None], (ny, nx)).copy()
    e1v = np.full((ny, nx), 1.0e5)
    j = int(np.argmin(np.abs(lat - j_lat)))
    voe3 = np.zeros((nz, ny, nx))
    voe3[:, j, :] = vrow
    theta_v = np.full((nz, ny, nx), theta)
    return voe3, e1v, gphiv, j, nz, nx, theta, vrow


def test_mht_core_analytic_nh_peak():
    voe3, e1v, gphiv, j, nz, nx, theta, vrow = _synthetic_mht()
    theta_v = np.full_like(voe3, theta)
    r = _nt.mht_core(voe3, theta_v, e1v, gphiv)
    expect = _nt._RHO0 * _nt._CP * (nz * vrow) * theta * (nx * 1.0e5) / 1.0e15
    assert abs(r["nh_peak_PW"] - expect) < 1e-9
    assert r["nh_peak_lat"] > 0.0


def test_mht_core_sign_on_curve():
    """Curve at the flow row: northward warm -> +, southward warm -> -."""
    voe3, e1v, gphiv, j, *_ = _synthetic_mht(vrow=0.5)
    theta_v = np.full_like(voe3, 10.0)
    pos = _nt.mht_core(voe3, theta_v, e1v, gphiv)["mht_PW"][j]
    neg = _nt.mht_core(-voe3, theta_v, e1v, gphiv)["mht_PW"][j]
    assert pos > 0.0 and neg < 0.0
    assert abs(pos + neg) < 1e-12


def test_mht_core_sh_min_captures_southward():
    """Southward warm flow in the SH -> negative MHT captured by sh_min."""
    voe3, e1v, gphiv, j, *_ = _synthetic_mht(vrow=-0.5, j_lat=-40.0)
    theta_v = np.full_like(voe3, 10.0)
    r = _nt.mht_core(voe3, theta_v, e1v, gphiv)
    assert r["sh_min_PW"] < 0.0
    assert r["sh_min_lat"] < 0.0


def test_mht_core_zero_flow_zero():
    voe3, e1v, gphiv, j, *_ = _synthetic_mht(vrow=0.0)
    theta_v = np.full_like(voe3, 10.0)
    r = _nt.mht_core(voe3, theta_v, e1v, gphiv)
    assert abs(r["nh_peak_PW"]) < 1e-12 and abs(r["sh_min_PW"]) < 1e-12


def test_acc_core_rejects_curvilinear_section():
    """On a CURVILINEAR grid (longitude shears with j so a constant i is not a
    meridian) the fixed-i section RAISES instead of silently mis-sampling."""
    import pytest
    uoe3, e2u, gphiu, glamu, _ = _synthetic_acc(lon_shear_deg_per_row=2.0)
    with pytest.raises(ValueError, match="not regular"):
        _nt.acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=-68.0,
                           max_lon_dev_deg=5.0)


# --------------------------------------------------------------------------
# arctic_gateways_core: the NEMO -> legoESM C-grid face shift.
# --------------------------------------------------------------------------
def _synthetic_gateways(nz=2, ny=6, nx=8, v_row=1, v0=0.5, lon=-170.0):
    """All-wet box whose 66N boundary is ONE v-row.

    Latitudes [60,63,67,70,73,76] put the region at rows 2..5, so the only
    boundary is the face between row 1 and row 2 = NEMO v index 1.  A uniform
    northward transport there is analytic: v0 * e1v * nx.
    """
    gphit = np.repeat(np.array([60.0, 63.0, 67.0, 70.0, 73.0, 76.0])[:, None],
                      nx, axis=1)
    glamt = np.full((ny, nx), lon)
    tmask = np.ones((ny, nx))
    uoe3 = np.zeros((nz, ny, nx))
    voe3 = np.zeros((nz, ny, nx))
    voe3[:, v_row, :] = v0 / nz          # splits evenly over levels
    e2u = np.full((ny, nx), 1.0e6)
    e1v = np.full((ny, nx), 1.0e6)
    return uoe3, voe3, e2u, e1v, gphit, glamt, tmask


def test_arctic_gateways_core_analytic_inflow():
    """+0.5 m^2/s northward on the boundary row over 8 faces of 1e6 m = 4 Sv."""
    gw, diag = _nt.arctic_gateways_core(*_synthetic_gateways())
    assert diag["n_v_faces"] == 8
    assert gw["bering_pacific"] == pytest_approx(4.0)
    for name in ("davis_caa", "atlantic_nordic", "siberian_other"):
        assert gw[name] == pytest_approx(0.0)


def test_arctic_gateways_core_sign_is_into_the_arctic():
    """Southward transport on the same face reports NEGATIVE (out)."""
    gw, _ = _nt.arctic_gateways_core(*_synthetic_gateways(v0=-0.5))
    assert gw["bering_pacific"] == pytest_approx(-4.0)


def test_arctic_gateways_core_bins_by_longitude():
    """Half the box in the Atlantic bin lands in atlantic_nordic, not Bering."""
    uoe3, voe3, e2u, e1v, gphit, glamt, tmask = _synthetic_gateways()
    glamt = glamt.copy()
    glamt[:, 4:] = -20.0                       # inside (-45, 70) = atlantic
    gw, _ = _nt.arctic_gateways_core(uoe3, voe3, e2u, e1v, gphit, glamt, tmask)
    assert gw["bering_pacific"] == pytest_approx(2.0)
    assert gw["atlantic_nordic"] == pytest_approx(2.0)


def test_arctic_gateways_core_v_face_shift_is_load_bearing():
    """NON-VACUITY for v: the NEMO->legoESM index shift carries the result.

    Feeding the flow one row AWAY from the boundary must not be collected. If
    the shift were dropped this row would be read as the boundary face and the
    test would report 4 Sv instead of 0.
    """
    gw, diag = _nt.arctic_gateways_core(*_synthetic_gateways(v_row=2))
    assert gw["bering_pacific"] == pytest_approx(0.0)
    # the same numbers WITHOUT the shift do collect it -- that is the proof
    # the shift is what carries the result, not an accident of the fixture
    assert diag["unshifted_Sv"]["bering_pacific"] == pytest_approx(4.0)


def _synthetic_gateways_with_u(nz=2, ny=6, nx=8, lon=-170.0):
    """Region bounded in LONGITUDE too, with a NON-UNIFORM zonal transport.

    Every other fixture here has zero zonal transport and no selected u-faces,
    so none of them pins the u index shift (codex).  A UNIFORM zonal flow will
    not do either: what enters the western boundary leaves the eastern one, so
    the net vanishes whether or not the u-faces were selected at all, and the
    test cannot fail (caught in review).  Making the transport vary with i
    gives a known NON-ZERO net that only appears if the u-faces really are
    selected, at the right indices.

    Wet region: columns 2..5 of rows 2..5, so the zonal boundary faces are our
    u-face 2 (west, inflow) and u-face 6 (east, outflow).
    """
    gphit = np.repeat(np.array([60.0, 63.0, 67.0, 70.0, 73.0, 76.0])[:, None],
                      nx, axis=1)
    glamt = np.full((ny, nx), lon)
    tmask = np.zeros((ny, nx))
    tmask[:, 2:6] = 1.0
    # NONLINEAR in i.  A LINEAR profile is useless here: shifting both
    # boundary samples by one column preserves their DIFFERENCE, so the net is
    # unchanged and the test cannot see the shift (codex caught this).
    i = np.arange(nx).astype(np.float64)
    uoe3 = np.zeros((nz, ny, nx))
    uoe3[:, :, :] = ((0.1 + 0.05 * i * i) / nz)[None, None, :]
    voe3 = np.zeros((nz, ny, nx))
    e2u = np.full((ny, nx), 1.0e6)
    e1v = np.full((ny, nx), 1.0e6)
    return uoe3, voe3, e2u, e1v, gphit, glamt, tmask


def test_arctic_gateways_core_u_faces_carry_a_known_nonzero_net():
    """NON-VACUITY for u: a non-uniform zonal flow gives a known net.

    Our u-face 2 is the western boundary and carries NEMO's u index 1; our
    u-face 6 is the eastern boundary and carries NEMO's index 5.  Positive is
    INTO the region, so inflow at the west counts +, outflow at the east -, over
    the four wet rows 2..5 of width 1e6 m:

        net = 4 * 1e6 * (u_nemo[1] - u_nemo[5])

    If the u-faces were not selected, or were selected at NEMO's own indices
    instead of ours, this number changes.  A uniform flow could not detect
    either.
    """
    uoe3, voe3, e2u, e1v, gphit, glamt, tmask = _synthetic_gateways_with_u()
    gw, diag = _nt.arctic_gateways_core(uoe3, voe3, e2u, e1v, gphit, glamt,
                                        tmask)
    u_col = uoe3.sum(axis=0)[0]                    # depth-summed, per column
    expected = 4.0 * 1.0e6 * (u_col[1] - u_col[5]) / 1.0e6
    assert diag["n_region_cells"] == 16
    assert gw["bering_pacific"] == pytest_approx(expected, tol=1e-9)
    assert abs(expected) > 0.1                     # the test can actually fail
    # and the unshifted counterfactual differs, so the shift is load-bearing
    assert diag["unshifted_Sv"]["bering_pacific"] != pytest_approx(
        expected, tol=1e-9)


def test_arctic_gateways_core_land_fill_values_do_not_poison_the_sum():
    """NaN on LAND must not propagate through the zero-weighted faces.

    section_transport multiplies unselected faces by a zero weight and
    0 * NaN is NaN, so one decoded land fill value would otherwise turn every
    gateway into NaN (codex).  The fixture's cell (0, 0) is made DRY so the
    value is genuinely land -- putting it on a WET cell is a different thing
    entirely and must be refused, which the next test checks.
    """
    uoe3, voe3, e2u, e1v, gphit, glamt, tmask = _synthetic_gateways()
    uoe3 = uoe3.copy()
    tmask = tmask.copy()
    tmask[0, 0] = 0.0                        # genuinely land
    uoe3[:, 0, 0] = np.nan
    gw, diag = _nt.arctic_gateways_core(uoe3, voe3, e2u, e1v, gphit, glamt,
                                        tmask)
    assert diag["n_nonfinite_u"] == 2        # both levels of that one cell
    assert np.isfinite(list(gw.values())).all()
    assert gw["bering_pacific"] == pytest_approx(4.0)


def test_arctic_gateways_core_reports_a_masked_face_on_the_section():
    """A face the section uses but the run masked is counted ZERO and SAID SO.

    Demanding a globally clean field would refuse every real run -- the
    oracle's mesh and run disagree on ~0.06% of upper-ocean shelf faces, two of
    which land on this very section.  So the transport is summed with those
    faces at zero, which is what the run itself did, and the assumption is
    stated rather than hidden behind a tolerance knob.  What must NOT happen is
    silent omission, so the count is surfaced in the diagnostics.
    """
    uoe3, voe3, e2u, e1v, gphit, glamt, tmask = _synthetic_gateways()
    voe3 = voe3.copy()
    voe3[:, 1, 0] = np.nan            # NEMO v index 1 IS the boundary face
    gw, diag = _nt.arctic_gateways_core(uoe3, voe3, e2u, e1v, gphit, glamt,
                                        tmask)
    assert diag["n_masked_section_v"] == 1
    # seven of the eight boundary faces still carry 0.5 m^2/s over 1e6 m
    assert gw["bering_pacific"] == pytest_approx(3.5)


def test_arctic_gateways_core_tolerates_a_missing_value_off_the_section():
    """The same defect away from the section is reported, not fatal."""
    uoe3, voe3, e2u, e1v, gphit, glamt, tmask = _synthetic_gateways()
    uoe3 = uoe3.copy()
    uoe3[:, 4, 4] = np.nan            # interior of the region, no face uses it
    gw, diag = _nt.arctic_gateways_core(uoe3, voe3, e2u, e1v, gphit, glamt,
                                        tmask)
    assert diag["n_nonfinite_u"] == 2
    assert gw["bering_pacific"] == pytest_approx(4.0)


def pytest_approx(x, tol=1e-9):
    import pytest
    return pytest.approx(x, abs=tol)
