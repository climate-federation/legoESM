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

# Tolerances for "the analytic Wing form reproduces the oracle".  These are
# LOOSE on purpose: the analytic form is a two-piece fit and cannot represent
# the oracle's warming stratosphere, so the analytic gate is restricted to the
# troposphere and asks only that the constants be oracle-anchored rather than
# from a different case.  The values are set just wide enough to pass with the
# oracle-derived constants and are ~5x tighter than the pre-oracle error.
_ANALYTIC_T_SFC_TOL_K = 0.75      # surface T, |analytic - oracle|
_ANALYTIC_Q_SFC_RTOL = 0.03       # surface q_v, relative
_ANALYTIC_TROP_T_MAE_K = 3.0      # mean |dT| over z <= 15 km
_ANALYTIC_RH_TOL = 0.10           # surface RH, absolute

# Tolerances for the SOUNDING path (read_sam_snd -> sam_case_setup -> IC).
# This is a round trip through OUR interpolation onto model levels, so the only
# error budget is vertical interpolation + the reference-pressure column, not a
# fit.  Correspondingly tight.
_IC_T_TOL_K = 0.5
_IC_Q_RTOL = 0.02
_IC_RH_TOL = 0.03


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
        pres0=float(ic.WING_P_SFC / 100.0),
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
    assert e["dq_sfc_rel"] <= _ANALYTIC_Q_SFC_RTOL, e
    assert e["dT_sfc"] <= _ANALYTIC_T_SFC_TOL_K, e
    assert e["dRH_sfc"] <= _ANALYTIC_RH_TOL, e
    assert e["trop_T_mae"] <= _ANALYTIC_TROP_T_MAE_K, e


def test_analytic_fallback_is_not_supersaturated():
    """No analytic column may ship supersaturated — the failure mode the oracle
    exposed.  Checked on the profile the module actually emits, so it fails for
    ANY (T_v0, Gamma, q_sfc) triple that saturates, not just the known-bad one.
    """
    z = np.linspace(0.0, 30_000.0, 301)
    _, _, _, rh = _analytic_at(z, T_v0=ic.WING_T_V0, Gamma=ic.WING_GAMMA,
                               q_sfc=ic.WING_Q_SFC_DEFAULT)
    assert np.isfinite(rh).all()
    assert rh.max() <= 1.0, f"analytic IC supersaturates: max RH {rh.max():.3f}"


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
    # and specifically: that triple supersaturated the column.
    z = np.linspace(0.0, 20_000.0, 201)
    _, _, _, rh = _analytic_at(z, T_v0=T_v0, Gamma=Gamma, q_sfc=q_sfc)
    assert rh.max() > 1.0, (
        "the pre-oracle triple no longer supersaturates — the non-vacuity "
        "probe has gone stale.")


# --------------------------------------------------------------------------
# The --sounding path: read_sam_snd -> sam_case_setup -> plane IC
# --------------------------------------------------------------------------

def _build_ic_from(snd, *, nlev=48, H=20_000.0, dz_sfc=50.0):
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


def test_sounding_ic_matches_the_oracle_through_the_column():
    """The whole vendored column, matched height by matched height."""
    base = _load_baseline()
    snd = _baseline_sounding(base)
    z_m, T_m, q_m, _ = _build_ic_from(snd)
    z_o = np.asarray(base["z"], dtype=np.float64)
    keep = (z_o >= z_m.min()) & (z_o <= z_m.max())
    assert keep.sum() >= 8, "too few overlapping levels to be a real check"
    order = np.argsort(z_m)
    T_i = np.interp(z_o[keep], z_m[order], T_m[order])
    q_i = np.interp(z_o[keep], z_m[order], q_m[order])
    np.testing.assert_allclose(T_i, np.asarray(base["T"])[keep],
                               atol=_IC_T_TOL_K)
    q_o = np.asarray(base["q_v"])[keep]
    wet = q_o > 1.0e-4                      # rtol is meaningless in dry air
    np.testing.assert_allclose(q_i[wet], q_o[wet], rtol=5 * _IC_Q_RTOL)


def test_sounding_ic_is_not_supersaturated():
    """The oracle column is subsaturated; the IC built from it must be too."""
    base = _load_baseline()
    _, _, _, rh = _build_ic_from(_baseline_sounding(base))
    assert np.isfinite(rh).all()
    assert rh.max() <= 1.0, f"sounding IC supersaturates: max RH {rh.max():.3f}"


# --------------------------------------------------------------------------
# The CLI surface
# --------------------------------------------------------------------------

def test_run_rcemip_plane_exposes_the_sounding_flags():
    """``--sounding`` / ``--sounding-grd`` round-trip through the driver's own
    parser (a library-only fix that no run script can reach is not a fix)."""
    from scripts.run.run_rcemip_plane import parse_args

    args = parse_args(["--sounding", "/tmp/snd", "--sounding-grd", "/tmp/grd",
                       "--stretched-vertical"])
    assert args.sounding == "/tmp/snd"
    assert args.sounding_grd == "/tmp/grd"
    assert args.stretched_vertical is True
    # default stays analytic — the sounding path is strictly opt-in, so no
    # existing RCEMIP result silently changes meaning.
    assert parse_args([]).sounding is None
    assert parse_args([]).sounding_grd is None


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


@requires_gsam
def test_full_gsam_column_sounding_ic():
    """The same acceptance, but on all 74 native levels rather than the 15
    vendored ones — catches a subsample that flatters the interpolation."""
    snd = read_sam_snd(_gsam_snd_path())
    z_m, T_m, q_m, rh_m = _build_ic_from(snd)
    z_o = np.asarray(snd.z, dtype=np.float64)
    T_o, rh_o = _rh_from(np.asarray(snd.theta), np.asarray(snd.p),
                         np.asarray(snd.q_v))
    keep = (z_o >= z_m.min()) & (z_o <= z_m.max())
    order = np.argsort(z_m)
    T_i = np.interp(z_o[keep], z_m[order], T_m[order])
    np.testing.assert_allclose(T_i, T_o[keep], atol=_IC_T_TOL_K)
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
