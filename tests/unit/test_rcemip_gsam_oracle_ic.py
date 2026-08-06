"""RCEMIP1 initial condition vs the gSAM RCE300 sounding ORACLE.

What the oracle is
------------------
gSAM 1.8.8 ships ``CASES/RCEMIP1/`` with one sounding per SST —
``snd_rcemip_{295,300,305}s6.11.2`` (``snd`` is a byte-identical copy of the
300 K file).  That is the profile SAM itself starts RCEMIP1 from, so it is
ground truth for "what should our RCE300 initial column look like", and it
settles questions the Wing 2018 text was previously being guessed at for.

Two things the oracle establishes that no analytic-constant argument could:

1. gSAM's sounding is **not** the Wing analytic form.  Its tropospheric lapse
   is steeper than ``WING_GAMMA`` and its stratosphere **warms** with height
   where the analytic form caps isothermally.  A warming stratosphere cannot be
   produced by any choice of ``(T_v0, Gamma)`` in a two-piece profile, so the
   faithful RCEMIP1 IC has to read the table — hence ``--sounding`` on
   ``scripts/run/run_rcemip_plane.py``.
2. gSAM's RCE300 surface is **sub**saturated.  A previously-shipped
   ``WING_T_V0 = 295 K`` paired with ``q_sfc = 0.01865`` put the initial column
   at ~140 % RH; the oracle disproves that pairing directly rather than by
   arithmetic.

The baseline
------------
The gSAM tree is EXTERNAL and is not committed.  ``tests/oracle_baselines/
gsam_rcemip300_snd.json`` is a ~15-level distillation with full provenance
(upstream URL, source SHA256, level indices, regeneration command) — the
"tiny tracked numeric baseline" carve-out.  When ``LEGOESM_GSAM_ROOT`` points
at a real gSAM checkout the full-column tests additionally verify the baseline
SHA256 and compare all 74 levels; without it they skip cleanly.

Non-vacuity
-----------
``test_pre_oracle_constants_fail_the_gate`` feeds the pre-oracle constant triple
back in and asserts it *must* fail, so this gate cannot rot into passing
trivially, and the claim "these tests fail for the old constants" is machine-
checked rather than asserted in a commit message.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.forcing.sam_case_forcing import SAMSounding, read_sam_snd
from legoesm.atmosphere.idealized import rcemip_initial_conditions as ic
from legoesm.thermo import saturation_mixing_ratio

BASELINE = (Path(__file__).resolve().parents[1] / "oracle_baselines"
            / "gsam_rcemip300_snd.json")

# The pre-oracle constant triple (T_v0 [K], Gamma [K/m], q_sfc [kg/kg]) that
# shipped before the gSAM sounding was readable.  Kept ONLY as the non-vacuity
# probe for `test_pre_oracle_constants_fail_the_gate`.
_PRE_ORACLE_TRIPLE = (295.0, 0.0067, 0.01865)

# Tolerances for "the analytic Wing form reproduces the oracle".  The analytic
# form is a two-piece fit and cannot represent the oracle's WARMING
# stratosphere, so the T comparison stops at the tropopause and asks only that
# the constants be oracle-anchored rather than from a different case.  The
# surface state and the cold point are gated separately and tightly, because
# the calibration is constrained through both.
_ANALYTIC_T_SFC_TOL_K = 0.75      # surface T, |analytic - oracle|
_ANALYTIC_Q_SFC_RTOL = 0.03       # surface q_v, relative
_ANALYTIC_TROP_T_MAE_K = 3.0      # mean |dT| over z <= 15 km
_ANALYTIC_RH_TOL = 0.10           # surface RH, absolute
# Tropopause cap vs the oracle cold point. Tight BECAUSE the shipped
# calibration is endpoint-constrained through it (measured error 0.000 K); the
# retracted least-squares calibration missed by 5.030 K.
_ANALYTIC_COLD_POINT_TOL_K = 1.0

# Tolerances for the SOUNDING path (read_sam_snd -> sam_case_setup -> IC).
# This is a round trip through OUR interpolation onto model levels, so the only
# error budget is vertical interpolation + the reference-pressure column, not a
# fit.  Correspondingly tight.
_IC_T_TOL_K = 0.5
_IC_Q_RTOL = 0.02
_IC_RH_TOL = 0.03
# The cold point is a KINK, and linear-interpolation error at a kink is first
# order in dz — it measures the vertical grid, not the IC pipeline. Budgeted
# separately at the magnitude actually measured (1.08 K reconstruction
# overshoot at nlev=48, job 9331622) rather than folded into _IC_T_TOL_K, which
# would have loosened every other level in the column by the same amount.
_IC_COLD_POINT_TOL_K = 1.5
# Above the tropopause the model's hydrostatic integral and gSAM's own pressure
# column have been integrated over ~20 km on DIFFERENT grids, so their
# difference accumulates with height: measured monotone 0.006 K at the lowest
# level to 0.611 K at 19.7 km. Budgeted separately so the troposphere keeps the
# tight 0.5 K.
_IC_T_STRAT_TOL_K = 1.0


def _load_baseline() -> dict:
    if not BASELINE.is_file():
        pytest.fail(
            f"missing tracked oracle baseline {BASELINE}. Regenerate with "
            "scripts/data/extract_gsam_rcemip_baseline.py (see the "
            "'provenance.command' field of the committed file).")
    return json.loads(BASELINE.read_text())


def _baseline_sounding(base: dict) -> SAMSounding:
    """The vendored levels as a ``SAMSounding`` — same NamedTuple ``read_sam_snd``
    returns, so the setup path under test is byte-for-byte the production one."""
    z = np.asarray(base["z"], dtype=np.float64)
    return SAMSounding(
        z=z,
        p=np.asarray(base["p"], dtype=np.float64),
        theta=np.asarray(base["theta"], dtype=np.float64),
        q_v=np.asarray(base["q_v"], dtype=np.float64),   # already kg/kg
        u=np.zeros_like(z), v=np.zeros_like(z),
        pres0=float(base["pres0_mb"]),
    )


def _rh_from(theta_K, p_mb, q_v_kgkg):
    """RH on the SAM mixing-ratio convention, via the model's own thermo.

    ``q`` in a SAM ``snd`` is a MIXING RATIO, so the convention-matched ratio is
    ``r / r_sat`` — never a re-derived Tetens/Magnus fit (CLAUDE.md: saturation
    comes from ``legoesm.thermo`` only).
    """
    p_pa = np.asarray(p_mb, dtype=np.float64) * 100.0
    T = np.asarray(theta_K, dtype=np.float64) * (
        p_pa / constants.p_ref) ** constants.kappa
    r_sat = np.asarray(saturation_mixing_ratio(T, p_pa), dtype=np.float64)
    return T, np.asarray(q_v_kgkg, dtype=np.float64) / r_sat


def _analytic_at(z, *, T_v0, Gamma, q_sfc):
    """Analytic Wing profile (T, q_v, p, RH) at heights ``z``, matched-height."""
    z = np.asarray(z, dtype=np.float64)
    T_v = np.asarray(ic.wing2018_virtual_temperature_profile(
        z, T_v0=T_v0, Gamma=Gamma), dtype=np.float64)
    q_v = np.asarray(ic.wing2018_qv_profile(z, q_sfc=q_sfc), dtype=np.float64)
    p = np.asarray(ic.wing2018_pressure_profile(
        z, T_v0=T_v0, Gamma=Gamma), dtype=np.float64)
    T = T_v / (1.0 + (1.0 / constants.epsilon - 1.0) * q_v)
    r_sat = np.asarray(saturation_mixing_ratio(T, p), dtype=np.float64)
    return T, q_v, p, q_v / r_sat


# --------------------------------------------------------------------------
# The tracked baseline itself
# --------------------------------------------------------------------------

def test_baseline_is_wellformed_and_carries_provenance():
    """A vendored oracle baseline with no provenance is unfalsifiable data."""
    base = _load_baseline()
    prov = base["provenance"]
    for key in ("upstream_url", "upstream_member", "source_sha256",
                "source_n_levels", "level_indices", "extraction_script",
                "command", "reader"):
        assert key in prov and prov[key], f"provenance.{key} missing/empty"
    assert len(prov["source_sha256"]) == 64
    assert prov["source_n_levels"] == 74, prov["source_n_levels"]

    n = len(base["z"])
    assert 10 <= n <= 20, f"baseline should stay TINY, got {n} levels"
    for col in ("p", "theta", "q_v", "T", "rh"):
        assert len(base[col]) == n, col
    z = np.asarray(base["z"])
    assert np.all(np.diff(z) > 0.0), "baseline heights must be increasing"
    assert np.all(np.isfinite(np.asarray(base["q_v"])))
    # q is kg/kg here (the reader divides by 1000); a g/kg regression shows up
    # immediately as a ~14 rather than ~0.014 surface value.
    assert 0.005 < base["q_v"][0] < 0.030, base["q_v"][0]

    # The derived columns must be reproducible from the raw oracle columns with
    # the CURRENT thermo — a drift tripwire on the saturation curve.
    T, rh = _rh_from(base["theta"], base["p"], base["q_v"])
    np.testing.assert_allclose(T, base["T"], rtol=1e-10)
    np.testing.assert_allclose(rh, base["rh"], rtol=1e-8)


def test_oracle_surface_is_subsaturated():
    """gSAM's own RCE300 sounding is NOT supersaturated anywhere.

    This is the fact that disproves the ``T_v0=295`` + ``q_sfc=0.01865`` pairing
    directly against SAM rather than by internal-consistency argument.
    """
    base = _load_baseline()
    _, rh = _rh_from(base["theta"], base["p"], base["q_v"])
    assert rh.max() < 1.0, (
        f"oracle max RH {rh.max():.4f} >= 1 — the baseline or the saturation "
        "curve changed; the gSAM RCE300 sounding is subsaturated throughout.")
    assert 0.5 < rh[0] < 0.95, rh[0]


# --------------------------------------------------------------------------
# The analytic fallback vs the oracle
# --------------------------------------------------------------------------

def _analytic_errors(T_v0, Gamma, q_sfc, base):
    z = np.asarray(base["z"], dtype=np.float64)
    T_o = np.asarray(base["T"], dtype=np.float64)
    q_o = np.asarray(base["q_v"], dtype=np.float64)
    rh_o = np.asarray(base["rh"], dtype=np.float64)
    T_a, q_a, _, rh_a = _analytic_at(z, T_v0=T_v0, Gamma=Gamma, q_sfc=q_sfc)
    trop = z <= ic.WING_Z_T
    return {
        "dT_sfc": abs(float(T_a[0] - T_o[0])),
        "dq_sfc_rel": abs(float(q_a[0] - q_o[0]) / float(q_o[0])),
        "dRH_sfc": abs(float(rh_a[0] - rh_o[0])),
        "trop_T_mae": float(np.abs(T_a[trop] - T_o[trop]).mean()),
    }


def test_analytic_cold_point_matches_the_oracle():
    """THE GATE THAT WAS MISSING. The analytic tropopause must reproduce gSAM's
    cold point.

    An intermediate calibration of this module fitted ``(T_v0, Gamma)`` by
    least squares over the troposphere, which minimises the MEAN residual and
    misses both endpoints: it moved the analytic cold point 5.03 K below the
    oracle, and the existing 193-196 K band in
    ``test_build_rcemip1_small_reference`` was widened to 187-192 K to
    accommodate it. Nothing gated the quantity itself. This does.

    The cold point sets OLR, cirrus and CAPE, so it is not a decoration; the
    tolerance is tight because the shipped calibration is endpoint-constrained
    THROUGH it and hits it to 0.000 K.
    """
    base = _load_baseline()
    T_o = np.asarray(base["T"], dtype=np.float64)
    T_cold_o = float(T_o.min())
    T_cap = ic.WING_T_V0 - ic.WING_GAMMA * ic.WING_Z_T   # virtual == actual, q_t~0
    assert abs(T_cap - T_cold_o) <= _ANALYTIC_COLD_POINT_TOL_K, (
        f"analytic tropopause cap {T_cap:.3f} K vs oracle cold point "
        f"{T_cold_o:.3f} K. Fix the CONSTANTS, not this tolerance — a fit that "
        "minimises the tropospheric mean will fail here by ~5 K.")
    # Non-vacuity: the pre-oracle triple happened to hit the cold point (that is
    # what it was chosen for), so guard with the estimator that did NOT.
    T_lstsq = 300.444 - 0.0074034 * ic.WING_Z_T
    assert abs(T_lstsq - T_cold_o) > _ANALYTIC_COLD_POINT_TOL_K, (
        "the retracted least-squares calibration now passes the cold-point "
        "gate — the tolerance has been widened past the defect it catches.")


def test_analytic_fallback_is_anchored_to_the_oracle():
    """The analytic default reproduces the oracle's surface state and
    tropospheric temperature within the stated (loose) tolerances.

    The analytic form CANNOT match the oracle's warming stratosphere, so the
    gate stops at the tropopause and is explicit about it.  Use ``--sounding``
    for a faithful RCEMIP1 column.
    """
    base = _load_baseline()
    e = _analytic_errors(ic.WING_T_V0, ic.WING_GAMMA, ic.WING_Q_SFC_DEFAULT,
                         base)
    # NOTE dq_sfc_rel is ~0 BY CONSTRUCTION for the shipped q_sfc: it is
    # defined as the value that reproduces the sounding's lowest level through
    # this very shape function. It is kept because it is NOT vacuous for the
    # value it exists to reject (the pre-oracle 0.01865 is 31 % off), but it
    # cannot fail for a correctly-derived q_0 — do not read it as evidence
    # that the moisture PROFILE matches.
    assert e["dq_sfc_rel"] <= _ANALYTIC_Q_SFC_RTOL, e
    assert e["dT_sfc"] <= _ANALYTIC_T_SFC_TOL_K, e
    assert e["dRH_sfc"] <= _ANALYTIC_RH_TOL, e
    assert e["trop_T_mae"] <= _ANALYTIC_TROP_T_MAE_K, e


def _first_supersaturated_z(T_v0, Gamma, q_sfc, z_max=30_000.0):
    """Lowest height at which the analytic column reaches RH >= 1 (or None)."""
    z = np.linspace(0.0, z_max, 3001)
    _, _, _, rh = _analytic_at(z, T_v0=T_v0, Gamma=Gamma, q_sfc=q_sfc)
    assert np.isfinite(rh).all(), "non-finite analytic RH"
    hit = np.flatnonzero(rh >= 1.0)
    return (None, rh) if hit.size == 0 else (float(z[hit[0]]), rh)


def test_analytic_fallback_is_not_supersaturated():
    """The analytic column must not ship supersaturated ANYWHERE — the failure
    mode the oracle exposed (RH 1.396 at the surface, saturated to ~10 km).

    Checked on the profile the module actually emits, so it fails for ANY
    (T_v0, Gamma, q_sfc) triple that saturates, not just the known-bad one.

    History worth keeping: under an intermediate least-squares calibration
    (Gamma = 0.0074034) this gate had to be SCOPED to z <= 10 km, because the
    steeper lapse made the column cross RH=1 near the tropopause where Wing's
    moisture shape leaves far more vapour than gSAM carries. A companion test
    pinned that as a documented limitation. The endpoint-constrained
    calibration removed it: the column is subsaturated over the whole 0-30 km
    range, so the scope restriction and the limitation test are both gone. If a
    future recalibration needs the scope back, that is a signal about the
    calibration, not a licence to narrow the gate.
    """
    z_hit, rh = _first_supersaturated_z(
        ic.WING_T_V0, ic.WING_GAMMA, ic.WING_Q_SFC_DEFAULT, z_max=30_000.0)
    assert z_hit is None, (
        f"analytic IC supersaturates at z={z_hit:.0f} m "
        f"(max RH {rh.max():.3f} over 0-30 km)")


def test_pre_oracle_constants_fail_the_gate():
    """NON-VACUITY: the constants that shipped before the oracle was readable
    must FAIL this gate.  Without this, a future edit could relax the tolerances
    until everything passes and the gate would still look green.
    """
    base = _load_baseline()
    T_v0, Gamma, q_sfc = _PRE_ORACLE_TRIPLE
    e = _analytic_errors(T_v0, Gamma, q_sfc, base)
    assert (e["dq_sfc_rel"] > _ANALYTIC_Q_SFC_RTOL
            or e["dT_sfc"] > _ANALYTIC_T_SFC_TOL_K
            or e["dRH_sfc"] > _ANALYTIC_RH_TOL
            or e["trop_T_mae"] > _ANALYTIC_TROP_T_MAE_K), (
        f"pre-oracle constants {_PRE_ORACLE_TRIPLE} now PASS the oracle gate "
        f"({e}) — the tolerances have been widened past the defect they exist "
        "to catch.")
    # and specifically: that triple supersaturated the column from the SURFACE
    # up, which is what made the CRM condense violently on step 1.
    z_hit, rh = _first_supersaturated_z(T_v0, Gamma, q_sfc, z_max=10_000.0)
    assert z_hit is not None and z_hit < 100.0, (
        f"the pre-oracle triple no longer supersaturates near the surface "
        f"(first RH>=1 at {z_hit}) — the non-vacuity probe has gone stale.")
    assert rh[0] > 1.3, rh[0]


# --------------------------------------------------------------------------
# The --sounding path: read_sam_snd -> sam_case_setup -> plane IC
# --------------------------------------------------------------------------

# Model column used by the round-trip tests.  nlev=96 (not 48) is a MEASURED
# requirement, not padding: at nlev=48 the stretched grid reaches dz ~1.8 km by
# the tropopause, and linearly interpolating gSAM's cold point — a sharp V at
# 14.5 km — onto that grid and back overshoots it by 1.08 K (job 9331622),
# above the 0.5 K budget.  The overshoot is a grid-resolution artefact of a
# kink, not a pipeline error: every other level agreed to <0.5 K and the
# surface to 0.003 K.  Refining the column is the honest fix; widening the
# tolerance would have hidden a real IC bug of the same size.
_IC_NLEV = 96
_IC_H = 20_000.0


def _build_ic_from(snd, *, nlev=_IC_NLEV, H=_IC_H, dz_sfc=50.0):
    """Run the SHARED SAM-deck setup path and return (z, T, q_v, RH) columns."""
    import jax.numpy as jnp

    from legoesm.atmosphere.dynamics.crm.sam_case_setup import (
        build_sam_case_height_coord,
        build_sam_case_initial_state,
    )
    from legoesm.atmosphere.forcing.sam_case_forcing import (
        extend_sounding_to_top,
    )
    from legoesm.grids.plane import create_plane_grid

    snd = extend_sounding_to_top(snd, H)
    p_sfc = float(snd.pres0) * 100.0
    hc = build_sam_case_height_coord(
        snd, nlev=nlev, H=H, p_sfc_pa=p_sfc, dz_sfc=dz_sfc,
        dtype=jnp.float64)
    grid = create_plane_grid(nx=4, ny=4, nlev=nlev, dx=4000.0, dy=4000.0,
                             dtype=jnp.float64)
    st = build_sam_case_initial_state(
        snd, grid, hc, n_tracers=3, seed_amp=0.0, dtype=jnp.float64)

    z = np.asarray(hc.z_full, dtype=np.float64)
    # theta = theta_ref + theta'   (the state carries the perturbation)
    theta = (np.asarray(hc.theta_ref, dtype=np.float64)
             + np.asarray(st.theta_prime.data, dtype=np.float64)[0, 0, :])
    # HeightCoordinate stores the Exner reference, not pressure:
    # p = p_ref * exner^(1/kappa).
    p_pa = constants.p_ref * np.asarray(
        hc.exner_ref, dtype=np.float64) ** (1.0 / constants.kappa)
    q_v = np.asarray(st.tracers.data, dtype=np.float64)[0, 0, :, 0]
    T = theta * (p_pa / constants.p_ref) ** constants.kappa
    r_sat = np.asarray(saturation_mixing_ratio(T, p_pa), dtype=np.float64)
    return z, T, q_v, q_v / r_sat


def test_sounding_ic_reproduces_the_oracle_surface_state():
    """ACCEPTANCE: the ``--sounding`` IC lands on the oracle's surface T, q and
    RH.

    Comparison is at MATCHED HEIGHT: the oracle column is evaluated at the
    model's own lowest level rather than at its native 37 m, so the difference
    reported is the IC pipeline's error and not a height offset.
    """
    base = _load_baseline()
    snd = _baseline_sounding(base)
    z_m, T_m, q_m, rh_m = _build_ic_from(snd)

    k = int(np.argmin(z_m))                       # lowest model level
    z_o = np.asarray(base["z"]);
    T_o = np.interp(z_m[k], z_o, np.asarray(base["T"]))
    q_o = np.interp(z_m[k], z_o, np.asarray(base["q_v"]))
    rh_o = np.interp(z_m[k], z_o, np.asarray(base["rh"]))

    assert abs(T_m[k] - T_o) <= _IC_T_TOL_K, (
        f"z={z_m[k]:.1f} m: IC T={T_m[k]:.3f} K vs oracle {T_o:.3f} K")
    assert abs(q_m[k] - q_o) / q_o <= _IC_Q_RTOL, (
        f"z={z_m[k]:.1f} m: IC q={q_m[k]*1e3:.4f} g/kg vs oracle "
        f"{q_o*1e3:.4f} g/kg")
    assert abs(rh_m[k] - rh_o) <= _IC_RH_TOL, (
        f"z={z_m[k]:.1f} m: IC RH={rh_m[k]:.4f} vs oracle {rh_o:.4f}")


def _column_checks(snd, z_o, theta_o, p_o_mb, q_o, label):
    """Compare an IC column to an oracle sounding ON THE MODEL'S OWN LEVELS.

    Deliberately NOT "interpolate the model back onto the oracle grid": doing
    that asks the column to reconstruct gSAM's cold point, which is a sharp V
    at 14.5 km.  Linear interpolation error at a kink is FIRST order in dz, so
    that framing measures the vertical resolution, not the IC pipeline — it is
    the same class of instrument flaw as a metric whose name does not match
    what it computes.  (Measured: 1.08 K at the vertex on nlev=48, and it
    persists at nlev=96 while every other level agrees.)

    Two checks are made instead, and they are different in kind:

    * PLUMBING (tight): the model's theta and q_v must equal the oracle
      interpolated to the model levels.  Near-tautological by construction and
      labelled as such — its job is to catch a g/kg-vs-kg/kg conversion, a
      wrong tracer slot, or a bottom-up/top-down ordering flip, all of which
      would blow it apart.
    * HYDROSTATIC (the real physics check): our temperature comes from
      ``theta_ref * exner_ref``, where exner is integrated hydrostatically on
      the MODEL grid from p_sfc.  gSAM's comes from the file's OWN pressure
      column.  Those are independent paths, so agreeing on T at matched height
      is a genuine test of the reference-state integration and the p_sfc BC.
    """
    z_m, T_m, q_m, rh_m = _build_ic_from(snd)
    order = np.argsort(z_m)
    inside = (z_m >= z_o.min()) & (z_m <= z_o.max())
    assert inside.sum() >= 20, f"{label}: too few model levels inside the snd"

    theta_i = np.interp(z_m, z_o, theta_o)
    q_i = np.interp(z_m, z_o, q_o)
    np.testing.assert_allclose(q_m[inside], q_i[inside], rtol=1e-10,
                               atol=1e-12, err_msg=f"{label}: q_v plumbing")

    # Oracle T on the model levels, from the FILE's pressure column.
    # Interpolate log(p), not p: pressure is near-exponential in z, and linear
    # interpolation across the vendored ~2.5 km stratospheric gaps introduced a
    # ~2 K artefact that was NOT a property of the IC (measured job 9331694,
    # error growing with height and peaking at z=16225 m). Interpolating in log
    # space is the physically correct choice, not a tolerance concession.
    p_i_pa = np.exp(np.interp(
        z_m, z_o, np.log(np.asarray(p_o_mb, dtype=np.float64)))) * 100.0
    T_o_i = theta_i * (p_i_pa / constants.p_ref) ** constants.kappa
    # Budget split by height. Our exner comes from a hydrostatic integral on
    # the MODEL grid; gSAM's p column came from an integral on ITS grid. The
    # difference between two such integrations accumulates UPWARD, and that is
    # exactly what is measured: monotone from 0.006 K at the lowest level to
    # 0.611 K at 19.7 km (job 9331711). Tolerating the accumulation aloft while
    # keeping the troposphere tight reports the physics; a single loose number
    # would have hidden a 0.5 K tropospheric error.
    dT = np.abs(T_m - T_o_i)
    trop = inside & (z_m <= 15_000.0)
    strat = inside & (z_m > 15_000.0)
    assert dT[trop].max() <= _IC_T_TOL_K, (
        f"{label}: TROPOSPHERIC hydrostatic column vs gSAM's own pressure "
        f"differs by {dT[trop].max():.3f} K (mean {dT[trop].mean():.3f} K) at "
        f"z={z_m[trop][int(np.argmax(dT[trop]))]:.0f} m")
    if strat.any():
        assert dT[strat].max() <= _IC_T_STRAT_TOL_K, (
            f"{label}: STRATOSPHERIC hydrostatic column differs by "
            f"{dT[strat].max():.3f} K at "
            f"z={z_m[strat][int(np.argmax(dT[strat]))]:.0f} m")
    assert np.isfinite(rh_m).all()
    return z_m, T_m, q_m, rh_m, order


def test_sounding_ic_matches_the_oracle_through_the_column():
    """The whole vendored column, on the model's own levels."""
    base = _load_baseline()
    _column_checks(
        _baseline_sounding(base),
        np.asarray(base["z"], dtype=np.float64),
        np.asarray(base["theta"], dtype=np.float64),
        np.asarray(base["p"], dtype=np.float64),
        np.asarray(base["q_v"], dtype=np.float64),
        "vendored baseline")


def test_sounding_ic_carries_the_oracle_cold_point():
    """The tropopause cold point survives the IC, to the accuracy the vertical
    grid can carry.

    Called out separately from the column check because a kink is the one
    feature whose reconstruction error is set by ``dz``, not by the pipeline.
    The tolerance is the RESOLUTION budget (measured: 1.08 K reconstruction
    overshoot at nlev=48), so this asserts the tropopause is in the right place
    at roughly the right temperature — not that it is bit-reproduced.
    """
    base = _load_baseline()
    z_m, T_m, *_ = _build_ic_from(_baseline_sounding(base))
    T_o = np.asarray(base["T"], dtype=np.float64)
    z_o = np.asarray(base["z"], dtype=np.float64)
    k_o, k_m = int(np.argmin(T_o)), int(np.argmin(T_m))
    assert abs(z_m[k_m] - z_o[k_o]) <= 1500.0, (
        f"cold point moved: IC {z_m[k_m]:.0f} m vs oracle {z_o[k_o]:.0f} m")
    assert abs(T_m[k_m] - T_o[k_o]) <= _IC_COLD_POINT_TOL_K, (
        f"cold point T: IC {T_m[k_m]:.3f} K vs oracle {T_o[k_o]:.3f} K "
        f"(resolution budget {_IC_COLD_POINT_TOL_K} K at nlev={_IC_NLEV})")


def test_sounding_ic_is_not_supersaturated():
    """The oracle column is subsaturated; the IC built from it must be too."""
    base = _load_baseline()
    _, _, _, rh = _build_ic_from(_baseline_sounding(base))
    assert np.isfinite(rh).all()
    assert rh.max() <= 1.0, f"sounding IC supersaturates: max RH {rh.max():.3f}"


# --------------------------------------------------------------------------
# The CLI surface
# --------------------------------------------------------------------------

def test_run_rcemip_plane_exposes_the_sounding_flag():
    """``--sounding`` round-trips through the driver's own parser, and the
    default stays analytic so no existing RCEMIP result changes meaning.

    NOTE this is only an argparse check. The wiring itself is covered by
    ``test_driver_sounding_path_builds_the_column`` below — a CLI round-trip
    would pass even if the entire ``--sounding`` branch in ``main()`` were
    deleted, which is exactly the vacuity this file has to avoid.
    """
    from scripts.run.run_rcemip_plane import parse_args

    args = parse_args(["--sounding", "/tmp/snd", "--stretched-vertical"])
    assert args.sounding == "/tmp/snd"
    assert args.stretched_vertical is True
    assert parse_args([]).sounding is None
    # --sounding-grd was removed: gSAM's CASES/RCEMIP1/grd is a 1-column list
    # that read_sam_grd rejects, so the flag could never have worked on the
    # file its help text named.
    assert not hasattr(parse_args([]), "sounding_grd")


def test_driver_sounding_path_builds_the_column(tmp_path):
    """THE WIRING TEST: the driver's own sounding path builds the column.

    Exercises ``build_sounding_height_coord`` + ``build_sounding_initial_state``
    — the functions ``main()`` calls — rather than argparse strings, so this
    fails if the ``--sounding`` branch is deleted or unhooked.
    """
    import jax.numpy as jnp

    from legoesm.grids.plane import create_plane_grid
    from scripts.run.run_rcemip_plane import (
        build_sounding_height_coord,
        build_sounding_initial_state,
        parse_args,
    )

    snd_path = tmp_path / "snd"
    _write_synthetic_snd(snd_path, n_lev=60, z_top=25_000.0)
    args = parse_args(["--sounding", str(snd_path), "--stretched-vertical",
                       "--nlev", "64", "--H", "20000",
                       "--theta-noise-amp", "0.0"])
    snd, hc = build_sounding_height_coord(args, ic.WING_P_SFC,
                                          dtype=jnp.float64)
    assert hc.n_levels == 64
    grid = create_plane_grid(nx=4, ny=4, nlev=64, dx=4000.0, dy=4000.0,
                             dtype=jnp.float64)
    st = build_sounding_initial_state(snd, grid, hc, args, 3,
                                      dtype=jnp.float64)

    z_m = np.asarray(hc.z_full, dtype=np.float64)
    q_m = np.asarray(st.tracers.data, dtype=np.float64)[0, 0, :, 0]
    # q_v came from the sounding, in kg/kg, into tracer slot 0. Compared only
    # WITHIN the sounding's range: the model's lowest level sits below the
    # sounding's first level, where np.interp clamps but the production helper
    # extrapolates — a difference in the comparison, not in the code.
    z_snd = np.asarray(snd.z)
    inside = (z_m >= z_snd.min()) & (z_m <= z_snd.max())
    assert inside.sum() >= 50, inside.sum()
    np.testing.assert_allclose(
        q_m[inside], np.interp(z_m[inside], z_snd, np.asarray(snd.q_v)),
        rtol=1e-10, atol=1e-14)
    # The below-sounding level must still be physical, not clamped to zero.
    assert 0.5 * q_m[inside][-1] < q_m[~inside].min() <= 2.0 * q_m[inside][-1]
    # theta' ~ 0 because theta_ref IS the sounding (no acoustic shock at t=0)
    # — WITHIN the sounding's range.
    #
    # PRE-EXISTING SHARED-PATH INCONSISTENCY, found by this test and NOT fixed
    # here: below the sounding's lowest level, build_sam_case_height_coord's
    # theta_ref_fn uses jnp.interp, which CLAMPS, while the IC's
    # interp_sounding_to_levels uses _interp_z, which LINEARLY EXTRAPOLATES
    # (deliberately — sam_case_forcing.py:488 matches forcing.f90's i=2
    # branch). The two therefore disagree at any model level beneath the
    # sounding, leaving a small non-zero theta' there. It affects every
    # SAM-deck case (BOMEX/RICO/DYCOMS/GATE), not just this one, so fixing it
    # belongs in the shared module with its own review. Magnitude is set by the
    # gap and the near-surface theta gradient: 0.06 K for this deliberately
    # steep synthetic deck, ~0.003 K for the real gSAM RCEMIP1 deck (12 m gap,
    # 2.7e-4 K/m).
    theta_p = np.asarray(st.theta_prime.data)[0, 0, :]
    assert np.abs(theta_p[inside]).max() < 1e-8
    assert np.abs(theta_p[~inside]).max() < 0.1
    assert np.isfinite(np.asarray(hc.rho_ref)).all()
    assert np.asarray(hc.rho_ref).min() > 0.0


@pytest.mark.parametrize("argv_extra, expect", [
    (["--nlev", "64"], "requires --stretched-vertical"),          # no stretch
    (["--stretched-vertical", "--nlev", "30"], "too coarse"),     # too coarse
])
def test_driver_sounding_path_refuses_bad_configurations(tmp_path, argv_extra,
                                                         expect):
    """Each guard raises with its OWN message — a SystemExit alone is not
    evidence, since any unrelated crash produces one."""
    import jax.numpy as jnp

    from scripts.run.run_rcemip_plane import (
        build_sounding_height_coord, parse_args)

    snd_path = tmp_path / "snd"
    _write_synthetic_snd(snd_path, n_lev=60, z_top=25_000.0)
    args = parse_args(["--sounding", str(snd_path), "--H", "20000"]
                      + argv_extra)
    with pytest.raises(SystemExit, match=expect):
        build_sounding_height_coord(args, ic.WING_P_SFC, dtype=jnp.float64)


def test_driver_sounding_path_refuses_a_missing_file_and_a_wrong_p_sfc(tmp_path):
    import jax.numpy as jnp

    from scripts.run.run_rcemip_plane import (
        build_sounding_height_coord, parse_args)

    base = ["--stretched-vertical", "--nlev", "64", "--H", "20000"]
    missing = parse_args(["--sounding", str(tmp_path / "nope")] + base)
    with pytest.raises(SystemExit, match="file not found"):
        build_sounding_height_coord(missing, ic.WING_P_SFC, dtype=jnp.float64)

    snd_path = tmp_path / "snd"
    _write_synthetic_snd(snd_path, n_lev=60, z_top=25_000.0)
    args = parse_args(["--sounding", str(snd_path)] + base)
    # A deck whose p_sfc disagrees with the p_sfc the flux/radiation code uses
    # must fail loudly, not silently run two different surface pressures.
    with pytest.raises(SystemExit, match="surface pressure"):
        build_sounding_height_coord(args, ic.WING_P_SFC + 5_000.0,
                                    dtype=jnp.float64)


# --------------------------------------------------------------------------
# The extraction script (every new .py gets a direct test)
# --------------------------------------------------------------------------

def _write_synthetic_snd(path: Path, n_lev: int = 40,
                         z_top: float = 20_000.0) -> None:
    """A minimal but VALID SAM ``snd``: header, one block, 6 columns."""
    z = np.linspace(40.0, z_top, n_lev)
    p = 1014.8 * np.exp(-z / 8000.0)
    theta = 297.0 + 4.0e-3 * z
    q = 14.0 * np.exp(-z / 4000.0)                     # g/kg
    lines = ["\t z[m]\t p[mb]\t tp[K]\t q[g/kg]\t u[m/s]\t v[m/s]",
             f"0\t{n_lev}\t1014.8"]
    lines += [f"\t{z[k]:.4f} {p[k]:.4f} {theta[k]:.4f} {q[k]:.4f} "
              f"0.0000 0.0000" for k in range(n_lev)]
    path.write_text("\n".join(lines) + "\n")


def test_extraction_script_round_trips(tmp_path):
    """``extract_gsam_rcemip_baseline`` distils a sounding into a provenance-
    carrying JSON whose values match the source at the recorded indices."""
    from scripts.data.extract_gsam_rcemip_baseline import main as extract_main

    snd_path = tmp_path / "snd_rcemip_300s6.11.2"
    out = tmp_path / "baseline.json"
    _write_synthetic_snd(snd_path)

    assert extract_main(["--snd", str(snd_path), "--out", str(out),
                         "--n-levels", "12"]) == 0
    got = json.loads(out.read_text())

    assert got["provenance"]["source_sha256"] == hashlib.sha256(
        snd_path.read_bytes()).hexdigest()
    assert got["provenance"]["source_n_levels"] == 40
    idx = got["provenance"]["level_indices"]
    assert 10 <= len(idx) <= 16 and idx == sorted(set(idx))
    assert idx[0] == 0 and idx[-1] == 39      # endpoints always vendored

    src = read_sam_snd(snd_path)
    np.testing.assert_allclose(np.asarray(src.z)[idx], got["z"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(src.q_v)[idx], got["q_v"],
                               rtol=1e-12)
    # derived columns reproduce from the raw ones with the model's own thermo
    T, rh = _rh_from(got["theta"], got["p"], got["q_v"])
    np.testing.assert_allclose(T, got["T"], rtol=1e-10)
    np.testing.assert_allclose(rh, got["rh"], rtol=1e-8)
    # full-column max is over ALL levels, so it must be >= the vendored
    # subset's max and must equal the max of the source column.
    _, src_rh = _rh_from(np.asarray(src.theta), np.asarray(src.p),
                         np.asarray(src.q_v))
    assert got["full_column_summary"]["rh_max"] == pytest.approx(
        float(src_rh.max()), rel=1e-10)
    assert got["full_column_summary"]["rh_max"] >= rh.max() - 1e-12


def test_extraction_level_selection_includes_the_named_extrema():
    """``select_levels`` must include the max-RH and cold-point levels by NAME —
    an index-spread-only subsample can miss exactly the levels the acceptance
    criteria quote."""
    from scripts.data.extract_gsam_rcemip_baseline import select_levels

    n = 74
    z = np.linspace(37.0, 33_000.0, n)
    rh = np.zeros(n); rh[17] = 0.9          # deliberately off the spread grid
    T = np.full(n, 250.0); T[41] = 190.0
    idx = select_levels(z, rh, T, n_target=15)
    assert 17 in idx and 41 in idx, idx
    assert idx[0] == 0 and idx[-1] == n - 1


# --------------------------------------------------------------------------
# Opt-in full-column checks against a real gSAM checkout
# --------------------------------------------------------------------------

def _gsam_snd_path():
    root = os.environ.get("LEGOESM_GSAM_ROOT")
    if not root:
        return None
    cand = Path(root) / "CASES" / "RCEMIP1" / "snd_rcemip_300s6.11.2"
    return cand if cand.is_file() else None


requires_gsam = pytest.mark.skipif(
    _gsam_snd_path() is None,
    reason="set LEGOESM_GSAM_ROOT to a gSAM 1.8.8 checkout for the "
           "full-column oracle checks (the tiny tracked baseline covers the "
           "rest)")


@requires_gsam
def test_vendored_baseline_matches_the_real_gsam_file():
    """The tracked baseline is a faithful subsample of the real file — SHA256
    of the source AND value-by-value at the recorded level indices."""
    base = _load_baseline()
    path = _gsam_snd_path()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == base["provenance"]["source_sha256"], (
        f"{path} sha256 {sha} != baseline provenance "
        f"{base['provenance']['source_sha256']} — the oracle file changed; "
        "regenerate the baseline rather than relaxing this.")
    snd = read_sam_snd(path)
    idx = base["provenance"]["level_indices"]
    np.testing.assert_allclose(np.asarray(snd.z)[idx], base["z"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(snd.theta)[idx], base["theta"],
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(snd.q_v)[idx], base["q_v"],
                               rtol=1e-12)
    # p matters as much as the rest: the vendored T and rh are DERIVED from it,
    # so a corrupted pressure column would otherwise pass unnoticed.
    np.testing.assert_allclose(np.asarray(snd.p)[idx], base["p"], rtol=1e-12)
    # And the header surface pressure, which the driver's --sounding p_sfc
    # guard compares against.
    assert float(snd.pres0) == pytest.approx(base["pres0_mb"], rel=1e-12)


@requires_gsam
def test_full_gsam_column_sounding_ic():
    """The same checks on all 74 native levels rather than the 13 vendored ones
    — catches a subsample that flatters the comparison."""
    snd = read_sam_snd(_gsam_snd_path())
    z_o = np.asarray(snd.z, dtype=np.float64)
    z_m, _, _, rh_m, _ = _column_checks(
        snd, z_o, np.asarray(snd.theta, dtype=np.float64),
        np.asarray(snd.p, dtype=np.float64),
        np.asarray(snd.q_v, dtype=np.float64), "full gSAM column")
    _, rh_o = _rh_from(np.asarray(snd.theta), np.asarray(snd.p),
                       np.asarray(snd.q_v))
    assert rh_m.max() <= 1.0, rh_m.max()
    k = int(np.argmin(z_m))
    assert abs(rh_m[k] - np.interp(z_m[k], z_o, rh_o)) <= _IC_RH_TOL


@requires_gsam
def test_gsam_ships_one_sounding_per_sst():
    """The three per-SST decks exist and DIFFER — the fact that refutes "RCEMIP
    prescribes one profile for all SSTs", which is how ``T_v0=295`` was
    justified."""
    root = Path(os.environ["LEGOESM_GSAM_ROOT"]) / "CASES" / "RCEMIP1"
    q0 = {}
    for sst in (295, 300, 305):
        p = root / f"snd_rcemip_{sst}s6.11.2"
        assert p.is_file(), p
        q0[sst] = float(read_sam_snd(p).q_v[0])
    assert q0[295] < q0[300] < q0[305], q0
    # `snd` is the 300 K case verbatim
    assert (hashlib.sha256((root / "snd").read_bytes()).hexdigest()
            == hashlib.sha256(
                (root / "snd_rcemip_300s6.11.2").read_bytes()).hexdigest())
