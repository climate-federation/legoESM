"""#1455 SG-FS: does the FREE-SURFACE FORM STRESS cancel the depth-ladder's
pressure-gradient error?  The ownership test spec'd in
``southern_term_torque_matched.py``'s docstring, run here.

THE QUESTION.  ``southern_term_torque_matched.py`` (dbd421031) established that
on BIT-IDENTICAL states the only large, one-signed, growing lego-minus-NEMO
momentum-term difference in the closed southern band is the depth-integrated
hydrostatic pressure gradient: band mean -6.13 (day 0) -> -15.85 (day 90)
m3/s2 per u-row on legoESM's analytic 1-D depth ladder, collapsing to
-0.07 -> -0.05 on NEMO's true T-depth ladder.  The circulation deficit that has
to be explained is only -0.61 m3/s2 per row, near-uniform and constant in time.
For the ladder to OWN the deficit, the split-explicit free surface must cancel
~95% of that pressure error.  This probe measures whether it does.

THE EXTRACTION, and why no solver instrumentation is needed.  NEMO's
surface-pressure-gradient trend is a DEPTH-INDEPENDENT acceleration: with
``ln_dynadv_vec=.true.`` (cfgs/DINO/RUN_90D_TWIN/namelist_cfg:321 -- the twin's
OWN namelist; there is no EXPREF/namelist_cfg in this tree) the barotropic
update at dynspg_ts.F90:684 adds ``zu_spg = -g * d(ssh)/dx`` with no thickness
weighting, and the dumped ``utrd_spg`` (dynspg.F90:185-187) is the resulting
depth-uniform 3-D increment.  legoESM's barotropic momentum equation is the
same velocity form: ``U_bar_new = U_bar + dt_s * ( ... - g * d(eta)/dx )``
(barotropic_latlon_cgrid.py:906), reconstructed onto every level as a
depth-uniform increment (barotropic_latlon_cgrid.py:1520).  BOTH facts are
also MEASURED here, not merely cited (control X1).

So the row-integrated free-surface torque is a pure function of sea level,
bathymetry and metrics:

    Phi_fs(j) = -g * sum_i H_u(i,j) * ( eta(i+1,j) - eta(i,j) )        [m3/s2]
              = +g * sum_c eta(c,j) * ( H_u(c,j) - H_u(c-1,j) )

(algebraically identical for a zonally WALLED row; the identity is asserted to
roundoff below).  The second form is the classical free-surface FORM STRESS
``g * integral eta dH/dx``; the first is the stencil both models integrate.

RESULT (all numbers band-mean m3/s2 per u-row, fp64, controls below all pass).

  THE HEADLINE -- eta-only free-surface torque gap on legoESM's own 90-day
  trajectories, SAME depth ladder on both sides (day 0 is exactly 0.000, the
  free control this reading provides):

      arm            d0      d30      d60      d90     PGF gap d0->d90
      arm1_pre     0.000  -11.833  -32.995  -34.364    -6.13 -> -15.86
      arm2_fixes   0.000   -5.207  -18.730  -38.183          (same, the
      arm3_bn2     0.000  -21.050  -26.538  -39.617           matched-state
                                                              series)

  For the free surface to CANCEL a pressure-gradient gap of X it must supply
  the opposite sign.  It does not: the gap is NEGATIVE in 9 of 9 arm-days,
  the SAME sign as the PGF gap, and by day 90 it is 2.2-2.5x LARGER.  The
  free surface is not merely failing to cancel the ladder's pressure error --
  on legoESM's own trajectory it is the larger of the two pressure-torque
  errors in this band.

  WHAT THAT DOES AND DOES NOT SETTLE.  It removes the free-surface form
  stress from the list of candidate cancellers, which is what the spec'd test
  was for.  It does NOT refute ladder ownership, and this probe must not be
  read as doing so: the band's budget is ``sum over ALL terms of
  (lego - NEMO) = -0.61``, the raw ladder error is ~8x NEMO's own spin-up
  torque for these rows, so ~90-95% of it IS being cancelled by something in
  the running model -- the barotropic Coriolis and bottom drag sit in the same
  solver stage (X1's +17.8 remainder) and are not measured here.  Ownership
  of the -0.61 by the depth ladder is UNMEASURED, not refuted.

  The measurement that would settle it, named but NOT run here: a
  ``LEGOESM_NEMO_E3T=gdept_only`` twin arm against the recorded ``off`` arm
  over the SAME window, comparing d(R_lego - R_NEMO)/dt per row.  One
  variable (the T-depth ladder; the thickness ladder and therefore the
  barotropic face depth are untouched).  It must report max|u| in both arms,
  because the true-ladder instability starts inside 20 days.

  Also measured here, from the arms rather than quoted: the deficit's own
  per-interval torque is -0.37 to -1.06 across arms and 30-day intervals,
  i.e. roughly CONSTANT, against a PGF gap that grows 2.6x over the window.

RETRACTION (this probe's own, from its own later revision).  An earlier
revision reported the trajectory gap as +17.7 / -10.8 / -14.7 / -16.1 by
weighting legoESM's side with its OWN column depth while the PGF half came
from the lane's reducer (NEMO's ``e3u_0``).  That mixes two metrics and is a
budget in neither: the +17.7 at day 0 was PURE LADDER WEIGHT, since the arms'
day-0 sea level equals NEMO's to 3e-8 m.  RETRACTED.  Both readings are now
separated: the eta-only gap above, and the ladder-weight bias as a standalone
size that is never summed with anything.  A second retracted framing: calling
the own-ladder weighting "maximally favourable" -- it is legoESM's real
min-rule barotropic face depth (barotropic_latlon_cgrid.py:199), i.e. a
different metric, not a favour.

THIS EXTRACTION'S OWN FLOOR, stated before any number is read (skill Rule 3).
NEMO does not difference ``sshn``: it differences ``zsshp2_e``, the half-step
back-interpolated barotropic sea level of the sub-cycle (dynspg_ts.F90:676-679),
and it uses ``grav`` = 9.80665 where this probe uses ``legoesm.constants.g`` =
9.80616.  Together those are ~5e-4 of a ~600 m3/s2 term, i.e. ~0.3 m3/s2 --
HALF the deficit.  Measured, both folded in: the extracted -g d(eta)/dx scores
slope 0.9995 against NEMO's own dumped surface-pressure trend (control X1).
They largely cancel in a lego-minus-NEMO difference (identical on both sides),
but this probe can never resolve -0.61 directly and nothing below is read at
that scale.

WHICH READING ANSWERS THE QUESTION -- and which does NOT.
  * TRAJECTORY (the headline, and the only reading that can exhibit the
    mechanism).  The hypothesised cancellation is a TRAJECTORY ADJUSTMENT:
    legoESM's own sea level evolving away from NEMO's under its own wrong
    pressure gradient.  So the number that matters is
    ``Phi_fs(eta_lego, H) - Phi_fs(eta_NEMO, H)`` with the SAME H on both
    sides -- an eta-only difference, no ladder confound -- evaluated on
    legoESM's recorded 90-day twin arms.  The band's budget is
    ``sum over ALL terms of (lego - NEMO) = -0.61``, so a PGF gap of X
    requires every OTHER gap to sum to -0.61 - X ~ -X.  The free surface is
    ONE of those other terms (the barotropic Coriolis and bottom drag are in
    the same solver stage and are measured here only as X1's +17.8
    remainder), so this reading can show the free surface is NOT the
    canceller; it cannot show that nothing cancels.
  * MATCHED-STATE (reported, but it cannot answer the question, and that is
    stated rather than hidden).  When both models are handed NEMO's own eta and
    the lane's reducer weights both with NEMO's ``e3u_0``, the free-surface
    torque is IDENTICAL on the two sides -- the spg acceleration is
    depth-independent in both -- so the "net" is the raw PGF gap by
    construction.  ``southern_term_torque_matched.py:88-91`` predicted exactly
    this blind spot in advance.
  * LADDER-WEIGHT BIAS (reported, and deliberately NOT summed with anything).
    Re-weighting NEMO's OWN eta by legoESM's own barotropic face depth
    (``H_ana`` = sum_k e3t_1d*umask, which IS legoESM's min-rule u-face depth
    on the analytic ladder, barotropic_latlon_cgrid.py:199) sizes how much of
    the free-surface torque the ladder moves.  It is NOT added to the PGF gap:
    the PGF gap comes from the lane's reducer, which weights with NEMO's
    ``e3u_0``, so their sum would mix two metrics and be a budget in neither.
    It is also ONE NUMBER SHARED BY BOTH ARMS -- ``nemo_state_bridge.
    effective_vertical_scale_factors`` returns the 1-D thickness ladder in
    BOTH "off" and "gdept_only" (they differ in gdept, not thickness) -- so it
    carries NO information about which arm is right.

CONTROLS (all fatal unless marked)
  C1  fp64 policy set explicitly and gated (``require_fp64``); geometry dtypes
      printed; NaN on the wet mask fatal.
  ID  the direct and by-parts (form-stress) discretizations must agree to
      roundoff on the real eta.  NON-VACUOUS: the two differ whenever a row's
      easternmost u-face is wet, which 35 rows of this grid have (the band's 13
      do not) -- so this verifies the walled assumption instead of assuming it.
  P1  planted BLOCK eta shift over an interior column range: must move Phi_fs
      by exactly -g*d_eta*(H_u(ia-1) - H_u(ib)), hand-computed from two depths.
      Asserted non-trivial.
  P2  REPORTED IDENTITY, NOT A CONTROL: a rigid eta shift moves Phi_fs by
      exactly zero.  ``phi_fs`` reads eta only through its zonal difference, so
      this holds for any mask, any H and any sign convention and CANNOT FAIL.
      Printed because the invariance is worth seeing; it certifies nothing.
  P3  planted FLAT bathymetry (H_u replaced by a constant on the wet faces):
      must reduce to the side-wall terms g*H0*(eta_west - eta_east) exactly,
      i.e. the interior form stress vanishes.
      P1 and P3 write their predictions out from the grid and call nothing that
      ``phi_fs`` calls -- but they DO share ``FACE``, ``H_TRUE`` and the
      constants, so no planted control here can detect a geometry error.  That
      is what X1/X2, which use NEMO's own fields, are for.
  X1  extraction vs NEMO's OWN trend: ``utrd_spg`` must be depth-uniform to
      <1e-9 relative AND its k=0 field must match ``-g d(eta)/dx`` at slope
      0.99..1.01 (this is the gate that would catch a sign or staggering flip).
      Its row integral is reported against Phi_fs but NOT gated -- see below.
  X2  the decomposition must close on the ORACLE's own data: the offline
      Leibniz TOTAL pressure torque (``southern_circulation_budget.phi_pres``,
      an independent third route) must equal NEMO's dumped baroclinic
      ``utrd_hpg`` PLUS this probe's Phi_fs.  Gated PER ROW (a signed band mean
      would let +-40 cancel to zero) against the offline route's OWN
      Leibniz-vs-z-level discretisation spread, also per row.
      WHAT X2 CAN AND CANNOT CERTIFY, stated because it matters: the per-row
      residual is 32-39 m3/s2 and the offline route's own per-row spread is
      32-39 -- they track each other, so the residual IS that route's
      discretisation error.  X2 therefore certifies "hpg + form stress IS the
      total pressure torque" to ~35 m3/s2 per row, which is 57x the deficit.
      It cannot certify anything at the deficit's scale, and nothing below is
      read at that scale.
  1e  the committed PGF-gap series (-6.13 -> -15.85 analytic; -0.07 -> -0.05
      true-depth) must be reproduced from the npz this probe reads.
  T0  each trajectory arm's day-0 sea level must reproduce the NEMO restart it
      was bridged from, else the arm's columns are misaligned.

WHAT X1 CANNOT BE.  ``utrd_spg`` is NOT the surface-pressure gradient alone: at
dynspg_ts.F90:345 NEMO first REMOVES the vertical mean of the baroclinic RHS
and then adds the barotropic solution back (dynspg_ts.F90:940-974), so its row
integral is Phi_fs plus the barotropic Coriolis and bottom-drag torques.  Its
row integral is therefore a bound, never an equality gate.

Diagnosis only: reads recorded artifacts (the two matched-arm npz produced by
``southern_term_torque_matched.py``, the recorded 90-day twin arms, NEMO's
restarts, the mesh) and writes only its own npz.  No model is run and no
production code is touched.

Run (after producing the two arm npz with the recorded matched probe):
  for M in off gdept_only; do
    JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=$M .venv/bin/python \
      scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \
      scripts/validate/ocean_fidelity/dino_1226/southern_term_torque_matched.py \
      --scratch /tmp/dino_stitch --days 0,10,20,30,40,50,60,70,80,90 \
      --out-npz /tmp/matched_$M.npz
  done
  JAX_ENABLE_X64=1 .venv/bin/python \
    scripts/validate/ocean_fidelity/dino_1226/southern_freesurface_form_stress.py \
      --matched-off /tmp/matched_off.npz \
      --matched-gdept /tmp/matched_gdept_only.npz \
      --arm NAME=/path/arm.npz [--arm ...] [--out-npz /tmp/fs.npz]
"""
from __future__ import annotations

import argparse
import os
import sys

import netCDF4 as nc
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402

set_policy(PrecisionPolicy.fp64())

import acceptance_gate_90d as G  # noqa: E402
import southern_circulation_budget as B  # noqa: E402  (recorded reducers/geometry)
from rebuild_nemo_restart import rebuild  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import require_fp64  # noqa: E402

ROWS = list(B.ROWS)
DAYS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
G_ACC = B.G_ACC
RHO0 = B.RHO0

# The committed matched-probe series this probe must reproduce before adding
# anything to it (Rule 1e; southern_term_torque_matched.py RESULT block).
COMMITTED_PGF = {"off": {0: -6.13, 30: -9.46, 60: -12.96, 90: -15.85},
                 "gdept_only": {0: -0.07, 30: -0.07, 60: -0.06, 90: -0.05}}
# The deficit this whole lane is trying to explain (57200616f).
DEFICIT = -0.61
DEFICIT_SPREAD = 0.16
# X2 is a CLOSURE gate on the DECOMPOSITION, not on the deficit.  It is priced
# against the offline pressure route's OWN per-row discretisation spread
# (Leibniz minus z-level), because that spread is the accuracy the third route
# has -- not against the deficit, which it is nowhere near resolving.  A
# wrong-sign or missing form stress leaves ~2|Phi_fs| ~ 1200 m3/s2 per row,
# i.e. ~30x this budget, so the gate is far from vacuous.
X2_SPREAD_FACTOR = 1.5

_yxz = lambda a: np.moveaxis(np.asarray(a, np.float64), 0, -1)  # noqa: E731


# --------------------------------------------------------------- geometry ----
def _column_depths():
    """(H_true, H_analytic) u-column depths [m] and the wet u-face mask.

    ``H_true``  = sum_k e3u_0 * umask     -- NEMO's real 3-D ladder.
    ``H_ana``   = sum_k e3t_1d * umask    -- the analytic 1-D ladder legoESM
                  integrates with when ``LEGOESM_NEMO_E3T`` is "off" or
                  "gdept_only" (nemo_state_bridge.effective_vertical_scale_
                  factors returns the 1-D thickness ladder in BOTH of those
                  modes, so the two arms share this depth).
    """
    with nc.Dataset(f"{B.A.DINO}/RUN_TRAJ/mesh_mask.nc") as mm:
        e3t1d = np.asarray(mm["e3t_1d"][0], np.float64).ravel()
    h_true = np.sum(np.where(B.umask, B.e3u0, 0.0), axis=2)
    h_ana = np.sum(np.where(B.umask, e3t1d[None, None, :], 0.0), axis=2)
    return h_true, h_ana, B.umask[:, :, 0]


H_TRUE, H_ANA, FACE = _column_depths()


# ------------------------------------------------------------- extraction ----
def phi_fs(eta, h_u):
    """Free-surface (surface-pressure-gradient) row torque [m3/s2] per u-row.

        Phi_fs(j) = -g * sum_i H_u * ( eta(i+1) - eta(i) )   over wet u-faces

    SIGN: z up, u>0 eastward, and Phi_fs>0 accelerates the row EASTWARD (it is
    the depth integral of the force -g d(eta)/dx per unit mass, times e1u/e1u).
    """
    eta = np.asarray(eta, np.float64)
    if not np.isfinite(eta[FACE]).all():
        raise SystemExit("FATAL C1: non-finite eta on a wet u-face column")
    d = np.zeros_like(eta)
    d[:, :-1] = eta[:, 1:] - eta[:, :-1]
    return np.sum(np.where(FACE, -G_ACC * h_u * d, 0.0), axis=1)


def phi_fs_byparts(eta, h_u):
    """The SAME torque written as the form stress  +g * sum_c eta * dH/dx.

    H is taken as 0 on land faces, so the wall terms are included automatically
    and this is exact for a zonally walled row.  Used only as the ID control.
    """
    eta = np.asarray(eta, np.float64)
    hw = np.where(FACE, h_u, 0.0)
    dh = np.zeros_like(hw)                       # H(face c) - H(face c-1)
    dh[:, 1:] = hw[:, 1:] - hw[:, :-1]
    dh[:, 0] = hw[:, 0]
    # The exact identity also carries a -g*hw[:, -1]*eta[:, -1] term for a row
    # whose EASTERNMOST u-face is wet.  It is omitted here deliberately: on a
    # zonally walled row that face is dry and the term is identically zero, and
    # the ID control is precisely the check that this holds for every band row.
    return G_ACC * np.sum(eta * dh, axis=1)


def _row_stats(a):
    v = np.asarray(a, np.float64)[ROWS]
    return float(v.mean()), float(v.std()), float(v.min()), float(v.max())


# --------------------------------------------------------------- controls ----
def control_identity(eta):
    """ID: the direct stencil and the form-stress (by-parts) form must agree."""
    for tag, h in (("H_true", H_TRUE), ("H_ana", H_ANA)):
        a = phi_fs(eta, h)
        b = phi_fs_byparts(eta, h)
        d = float(np.max(np.abs((a - b)[ROWS])))
        scale = float(np.max(np.abs(a[ROWS])))
        print(f"  ID  direct vs by-parts form stress ({tag}): band max |diff| "
              f"{d:.3e} m3/s2 on a term of {scale:.1f} (rel {d / scale:.2e})")
        if d > 1e-8 * max(scale, 1.0):
            raise SystemExit(
                "FATAL ID: the direct surface-pressure stencil and the "
                "form-stress integral disagree -- either a band row is not "
                "zonally walled or the extraction is not a form stress")


def control_planted(eta):
    """P1/P2/P3 -- predictions written from the grid, sharing no code with
    ``phi_fs``."""
    base = phi_fs(eta, H_TRUE)
    # P1: shift eta by d_eta on the interior column block [ia, ib].
    ia, ib, d_eta = 12, 30, 0.037
    e2 = np.asarray(eta, np.float64).copy()
    e2[:, ia:ib + 1] += d_eta
    got = phi_fs(e2, H_TRUE) - base
    # only faces ia-1 (east neighbour lifted) and ib (east neighbour not lifted)
    # see a changed difference; everything else cancels.
    pred = np.where(FACE[:, ia - 1], -G_ACC * H_TRUE[:, ia - 1] * d_eta, 0.0) \
        + np.where(FACE[:, ib], +G_ACC * H_TRUE[:, ib] * d_eta, 0.0)
    d = float(np.max(np.abs((got - pred)[ROWS])))
    mag = float(np.max(np.abs(pred[ROWS])))
    print(f"  P1  planted +{d_eta} m eta over columns {ia}..{ib}: predicted row "
          f"shift up to {mag:.3f} m3/s2, measured minus predicted "
          f"{d:.3e}")
    if mag < 1.0:
        raise SystemExit("FATAL P1: the plant is a no-op (vacuous)")
    if d > 1e-9 * max(mag, 1.0):
        raise SystemExit("FATAL P1: the block eta shift does not move the form "
                         "stress by the hand-computed wall-pair amount")
    # P2: rigid shift.  REPORTED IDENTITY, NOT A CONTROL -- phi_fs reads eta
    # only through its zonal difference, so this is exactly zero for any mask,
    # any H and any sign convention, and cannot fail.  Printed because the
    # invariance is worth seeing; it certifies nothing and is not asserted.
    got2 = phi_fs(np.asarray(eta, np.float64) + 0.25, H_TRUE) - base
    d2 = float(np.max(np.abs(got2[ROWS])))
    print(f"  P2  rigid +0.25 m eta shift: band max |change| {d2:.3e} m3/s2 "
          "(an ALGEBRAIC IDENTITY, reported not gated -- it cannot fail)")
    # P3: flat bathymetry -- only the two side walls survive.
    h0 = 3000.0
    flat = np.where(FACE, h0, 0.0)
    got3 = phi_fs(eta, flat)
    e = np.asarray(eta, np.float64)
    pred3 = np.zeros(B.NY)
    for j in ROWS:
        wet = np.flatnonzero(FACE[j])
        if wet.size == 0:
            continue
        # per CONTIGUOUS run of wet faces (an interior island would split the
        # row into several runs; not assumed, handled)
        cuts = np.flatnonzero(np.diff(wet) > 1)
        starts = np.concatenate([[wet[0]], wet[cuts + 1]])
        ends = np.concatenate([wet[cuts], [wet[-1]]])
        if int(ends.max()) + 1 >= e.shape[1]:
            raise SystemExit(
                f"FATAL P3: row {j} has a wet u-face on the last column, so "
                "the row is not zonally walled and the side-wall prediction "
                "has no east-side cell -- ID should already have caught this")
        pred3[j] = G_ACC * h0 * float(np.sum(e[j, starts] - e[j, ends + 1]))
    d3 = float(np.max(np.abs((got3 - pred3)[ROWS])))
    mag3 = float(np.max(np.abs(pred3[ROWS])))
    print(f"  P3  planted FLAT bathymetry H={h0:.0f} m: interior form stress "
          f"must vanish, leaving only the side walls of each wet run "
          f"({mag3:.3f} m3/s2); measured minus predicted {d3:.3e}")
    if mag3 < 1.0:
        raise SystemExit("FATAL P3: the flat-bathymetry prediction is ~0, so "
                         "the control is vacuous")
    if d3 > 1e-9 * max(mag3, 1.0):
        raise SystemExit("FATAL P3: over flat bathymetry the extraction does "
                         "not reduce to the side-wall pressure difference")


def control_oracle_closure(day, eta, raw):
    """X1 + X2 -- the extraction measured against NEMO's OWN data."""
    spg = _yxz(raw["utrd_spg"])
    v = np.where(B.umask, spg, np.nan)
    amp = np.nanmax(np.abs(v), axis=2)
    rng = np.nanmax(v, axis=2) - np.nanmin(v, axis=2)
    col = B.umask.any(axis=2)
    rel = float(np.nanmax(rng[col] / np.maximum(amp[col], 1e-30)))
    d = np.zeros_like(eta)
    d[:, :-1] = eta[:, 1:] - eta[:, :-1]
    pred_k0 = -G_ACC * d / B.e1u
    f = FACE
    slope = float((spg[:, :, 0][f] * pred_k0[f]).sum() / (pred_k0[f] ** 2).sum())
    corr = float(np.corrcoef(spg[:, :, 0][f], pred_k0[f])[0, 1])
    fs = phi_fs(eta, H_TRUE)
    spg_row = B._row_int_trend(spg)
    hpg_row = B._row_int_trend(_yxz(raw["utrd_hpg"]))
    pres = B.phi_pres(_yxz(raw["tn"]), _yxz(raw["sn"]), eta)
    pres_lev = B.phi_pres_levelform(_yxz(raw["tn"]), _yxz(raw["sn"]), eta)
    x2 = pres - (hpg_row + fs)
    print(f"  X1  day {day:2d}: utrd_spg depth-uniformity {rel:.2e} (want "
          f"<1e-9); its k=0 field vs -g d(eta)/dx corr {corr:.5f} slope "
          f"{slope:.5f}; row integral {spg_row[ROWS].mean():8.3f} vs Phi_fs "
          f"{fs[ROWS].mean():8.3f}  (remainder "
          f"{(spg_row - fs)[ROWS].mean():+.3f} = barotropic Coriolis+drag, "
          "reported not gated)")
    if rel > 1e-9:
        raise SystemExit("FATAL X1: utrd_spg is NOT depth-uniform, so the "
                         "free-surface torque is not the depth-independent "
                         "acceleration this extraction assumes")
    # THE gate that catches a sign or staggering flip in phi_fs: a flipped
    # stencil scores slope ~ -1 here and would otherwise only be caught by
    # luck downstream.
    if not (0.99 < slope < 1.01):
        raise SystemExit(
            f"FATAL X1: the extracted -g d(eta)/dx has slope {slope:.5f} "
            "against NEMO's own surface-pressure trend -- a sign, staggering "
            "or metric error in phi_fs")
    # PER-ROW, not a signed band mean: +-40 that averages to zero must not pass.
    x2_row = float(np.max(np.abs(x2[ROWS])))
    spread_row = float(np.max(np.abs((pres - pres_lev)[ROWS])))
    print(f"  X2  day {day:2d}: offline TOTAL pressure torque {pres[ROWS].mean():8.3f}"
          f"  vs  NEMO utrd_hpg {hpg_row[ROWS].mean():8.3f} + Phi_fs "
          f"{fs[ROWS].mean():8.3f} = {(hpg_row + fs)[ROWS].mean():8.3f};   "
          f"residual band mean {x2[ROWS].mean():+7.3f}, MAX PER ROW "
          f"{x2_row:7.3f}  vs the offline route's own max per-row "
          f"discretisation spread {spread_row:7.3f}")
    return x2_row, spread_row


# -------------------------------------------------------------------- main ----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matched-off", required=True,
                    help="npz from southern_term_torque_matched.py, E3T=off")
    ap.add_argument("--matched-gdept", required=True,
                    help="npz from southern_term_torque_matched.py, "
                         "E3T=gdept_only")
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME=/path/arm.npz -- a recorded 90-day legoESM twin "
                         "arm, for the TRAJECTORY reading (repeatable)")
    ap.add_argument("--out-npz", default=None)
    args = ap.parse_args(argv)

    print("=" * 108)
    print("C1 PRECISION / GEOMETRY")
    print("=" * 108)
    require_fp64(context="southern_freesurface_form_stress")
    print(f"  precision policy control dtype = {get_policy().control}")
    for nm, a in (("e1u", B.e1u), ("e3u_0", B.e3u0), ("umask", B.umask),
                  ("H_true", H_TRUE), ("H_ana", H_ANA)):
        print(f"  {nm:8s} {np.asarray(a).dtype}  shape {np.asarray(a).shape}")
    # BAND rows only -- the whole analysis lives there, and the domain-wide
    # statistic is a different (larger-spread) number.
    wet = np.zeros_like(FACE)
    wet[ROWS] = FACE[ROWS]
    dH = (H_ANA - H_TRUE)[wet]
    rel = np.abs(dH / H_TRUE[wet])
    print(f"  column depth over the BAND: NEMO true "
          f"{H_TRUE[wet].min():.1f}..{H_TRUE[wet].max():.1f} m;  analytic "
          f"ladder deeper by {dH.min():.3g}..{dH.max():.1f} m (mean "
          f"{dH.mean():.1f}); max relative difference {rel.max():.4f} "
          f"(the largest absolute and the largest relative occur at different "
          f"columns)")
    if np.abs(dH).max() < 1.0:
        raise SystemExit("FATAL: the two ladders give the same column depth, so "
                         "the ladder-only reading below would perturb a zero")

    nemo = {}
    for day in DAYS:
        kt = G.KT_RESTART + day * G.STEPS_PER_DAY
        nemo[day] = rebuild(f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart*.nc",
                            ["sshn", "utrd_spg", "utrd_hpg", "tn", "sn"])

    eta0 = np.asarray(nemo[0]["sshn"], np.float64).squeeze()
    print("\n" + "=" * 108)
    print("CONTROLS ON THE EXTRACTION (identity, planted, oracle closure)")
    print("=" * 108)
    control_identity(eta0)
    control_planted(eta0)
    x2s = []
    for day in (0, 30, 60, 90):
        eta = np.asarray(nemo[day]["sshn"], np.float64).squeeze()
        x2_row, spread_row = control_oracle_closure(day, eta, nemo[day])
        x2s.append(x2_row / max(spread_row, 1e-30))
    print(f"  X2 verdict: worst per-row residual over days 0/30/60/90 = "
          f"{max(x2s):.2f}x the offline route's own per-row discretisation "
          f"spread (gate {X2_SPREAD_FACTOR:.1f}x).  A missing or wrong-sign "
          "form stress would leave ~30x.  NOTE the floor this implies: the "
          "closure is certified to ~35 m3/s2 per row = 57x the -0.61 deficit, "
          "so NOTHING below is read at the deficit's scale.")
    if max(x2s) > X2_SPREAD_FACTOR:
        raise SystemExit(
            "FATAL X2: baroclinic hpg + free-surface form stress does NOT "
            "reproduce the total pressure torque on NEMO's own data -- the "
            "decomposition is wrong and no arm below may be read")

    # ---- Rule 1e: reproduce the committed PGF-gap series from the npz -------
    print("\n" + "=" * 108)
    print("Rule 1e -- the committed PGF-gap series, reproduced from the arm npz")
    print("=" * 108)
    arms = {"off": np.load(args.matched_off),
            "gdept_only": np.load(args.matched_gdept)}
    gaps = {}
    for name, z in arms.items():
        gaps[name] = {}
        for day in DAYS:
            k = f"lego_PGFONLY_u_d{day}"
            if k not in z.files:
                raise SystemExit(f"FATAL: {name} npz lacks {k}")
            gaps[name][day] = (np.asarray(z[k], np.float64)
                               - np.asarray(z[f"nemo_hpg_d{day}"], np.float64))
        line = "  ".join(f"d{d}={float(gaps[name][d][ROWS].mean()):+.2f}"
                         for d in (0, 30, 60, 90))
        print(f"  {name:11s} {line}")
        for d, want in COMMITTED_PGF[name].items():
            got = float(gaps[name][d][ROWS].mean())
            if abs(got - want) > 0.05:
                raise SystemExit(
                    f"FATAL 1e: {name} day {d} PGF gap {got:+.3f} does not "
                    f"reproduce the committed {want:+.3f}")
    print("  reproduced (tol 0.05 m3/s2)")

    # ---- the ownership table ------------------------------------------------
    fs_true = {d: phi_fs(np.asarray(nemo[d]["sshn"], np.float64).squeeze(),
                         H_TRUE) for d in DAYS}
    fs_ana = {d: phi_fs(np.asarray(nemo[d]["sshn"], np.float64).squeeze(),
                        H_ANA) for d in DAYS}

    print("\n" + "=" * 108)
    print("MATCHED-STATE readings -- NEITHER can exhibit the mechanism under "
          "test; printed for completeness")
    print("=" * 108)
    print("  (i) SHARED-ETA IDENTITY, not a measurement.  Both models are "
          "handed NEMO's eta and the lane's")
    print("      reducer weights both with NEMO's e3u_0; the spg acceleration "
          "is depth-independent in both")
    print("      models, so the free-surface torque is the SAME number on the "
          "two sides and the 'net' is the")
    print("      raw PGF gap BY CONSTRUCTION.  It refutes nothing and is "
          "restated here only so it is not")
    print("      mistaken for evidence later.")
    print(f"  {'arm':11s}{'day':>5s}{'PGF gap = NET':>15s}{'NET/def':>9s}"
          f"{'row min':>10s}{'row max':>10s}{'row sd':>9s}")
    for name in ("off", "gdept_only"):
        for day in DAYS:
            mu, sd, lo, hi = _row_stats(gaps[name][day])
            print(f"  {name:11s}{day:5d}{mu:15.3f}{mu / DEFICIT:9.1f}"
                  f"{lo:10.3f}{hi:10.3f}{sd:9.3f}")
    print("\n  (ii) LADDER-WEIGHT BIAS, a standalone size -- deliberately NOT "
          "added to the PGF gap (that would")
    print("       mix legoESM's face depth with the reducer's e3u_0 and be a "
          "budget in neither metric), and")
    print("       IDENTICAL in the two arms, so it says nothing about which "
          "arm is right.")
    print(f"  {'day':>5s}{'Phi_fs(H_true)':>16s}{'Phi_fs(H_lego)':>16s}"
          f"{'difference':>12s}{'x deficit':>11s}")
    for day in DAYS:
        a = float(fs_true[day][ROWS].mean())
        b = float(fs_ana[day][ROWS].mean())
        print(f"  {day:5d}{a:16.3f}{b:16.3f}{b - a:12.3f}"
              f"{(b - a) / DEFICIT:11.1f}")

    print(f"\n  the deficit to explain: {DEFICIT:+.2f} m3/s2 per row, "
          f"near-uniform (+/-{DEFICIT_SPREAD:.2f}, i.e. row sd/|mean| ~ "
          f"{DEFICIT_SPREAD / abs(DEFICIT):.2f}); its own per-interval value "
          "is MEASURED in the trajectory table below.")

    # ---- time shape: per-interval, both quantities, both labelled --------
    print("\n  TIME SHAPE -- per 10-day interval.  The deficit's own measured "
          "per-interval value is in the")
    print("  trajectory table below; the two quantities here are the PGF gap "
          "and the ladder-weight bias.")
    print(f"  {'quantity':26s}" + "".join(f"{f'{a}-{b}':>9s}"
                                          for a, b in zip(DAYS[:-1], DAYS[1:])))
    for name in ("off", "gdept_only"):
        vals = [0.5 * (float(gaps[name][a][ROWS].mean())
                       + float(gaps[name][b][ROWS].mean()))
                for a, b in zip(DAYS[:-1], DAYS[1:])]
        print(f"  {'PGF gap / ' + name:26s}" + "".join(f"{v:9.2f}" for v in vals))
    vals = [0.5 * (float((fs_ana[a] - fs_true[a])[ROWS].mean())
                   + float((fs_ana[b] - fs_true[b])[ROWS].mean()))
            for a, b in zip(DAYS[:-1], DAYS[1:])]
    print(f"  {'ladder-weight bias':26s}" + "".join(f"{v:9.2f}" for v in vals))
    print("  NOTE both shapes, do not quote one: the PGF gap GROWS 2.6x over "
          "the window while the ladder-weight")
    print("  bias is nearly flat.  The deficit is constant, so the PGF gap's "
          "shape is a strike against ladder")
    print("  ownership only if the canceller's efficiency is assumed constant "
          "-- an equilibrating free surface")
    print("  would not be.  Reported as a shape comparison, NOT as a "
          "refutation.")

    traj = {}
    if not args.arm:
        raise SystemExit(
            "FATAL: at least one --arm NAME=/path/arm.npz is required.  The "
            "matched-state readings above CANNOT answer the ownership "
            "question (southern_term_torque_matched.py:88-91 says so in "
            "advance); only the trajectory reading can.")
    print("\n" + "=" * 108)
    print("TRAJECTORY reading -- THE HEADLINE.  Does legoESM's own sea level "
          "adjust in the CANCELLING direction?")
    print("  'eta-only gap' = Phi_fs(eta_lego, H_true) - Phi_fs(eta_NEMO, "
          "H_true): the SAME depth ladder on both")
    print("     sides, so the only thing that moves is the free surface.  "
          "That is the mechanism under test.")
    print("  For the free surface to cancel a PGF gap of X it must supply "
          "-X (OPPOSITE sign).  The matched-state")
    print("     PGF gap on the analytic ladder is printed alongside for the "
          "sign comparison (it is measured on")
    print("     NEMO's states, not on the arm's, so only its SIGN and ORDER "
          "are read here).")
    print("=" * 108)
    print(f"  {'arm':16s}{'day':>5s}{'NEMO':>10s}{'lego':>10s}"
          f"{'eta-only gap':>14s}{'row sd':>9s}{'PGF gap':>10s}"
          f"{'cancels?':>10s}{'deficit torque':>16s}")
    eta_n = {d: np.asarray(nemo[d]["sshn"], np.float64).squeeze()
             for d in (0, 30, 60, 90)}
    r_nemo = {d: B.row_circulation(_yxz(rebuild(
        f"{G.RUN_90D_TWIN}/DINO_{G.KT_RESTART + d * G.STEPS_PER_DAY:08d}"
        "_restart*.nc", ["un"])["un"])) for d in (0, 30, 60, 90)}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        z = np.load(path)
        traj[name] = {}
        # T0 -- the arm's day-0 sea level IS the NEMO restart it was bridged
        # from (fp32 snapshot quantum).  Without this the whole trajectory
        # column could be a silent column/row misalignment.
        d0 = float(np.max(np.abs(
            np.asarray(z["eta3d_day0"], np.float64) - eta_n[0])[FACE]))
        print(f"  T0 {name}: max|arm eta(day 0) - NEMO ssh| on wet u-face "
              f"columns = {d0:.3e} m (arm eta is stored fp32; quantum ~1e-7 "
              f"of |eta|~1, which is ~0.02 m3/s2 on Phi_fs)")
        if d0 > 1.0e-6:
            raise SystemExit(
                f"FATAL T0: arm {name} day-0 sea level does not reproduce "
                "the NEMO restart it was bridged from -- the trajectory "
                "column is misaligned and may not be read")
        for day in (0, 30, 60, 90):
            el = np.asarray(z[f"eta3d_day{day}"], np.float64)
            if el.shape != H_TRUE.shape:
                raise SystemExit(f"FATAL: arm {name} eta shape {el.shape} "
                                 f"!= {H_TRUE.shape}")
            fl = phi_fs(el, H_TRUE)          # SAME ladder as the NEMO side
            g_eta = fl - fs_true[day]
            traj[name][day] = g_eta
            pgf = float(gaps["off"][day][ROWS].mean())
            mu = float(g_eta[ROWS].mean())
            sd = float(g_eta[ROWS].std())
            if day == 0:
                verdict, def_s = "--", "     --"
            else:
                verdict = "YES" if mu * pgf < 0 else "NO (adds)"
                prev = day - 30
                dl = (B.row_circulation(G.load_candidate(path, day)["u"])
                      - B.row_circulation(G.load_candidate(path, prev)["u"]))
                dn = r_nemo[day] - r_nemo[prev]
                def_s = f"{float(((dl - dn) / (30.0 * B.SEC_PER_DAY))[ROWS].mean()):+.3f}"
            print(f"  {name:16s}{day:5d}{float(fs_true[day][ROWS].mean()):10.3f}"
                  f"{float(fl[ROWS].mean()):10.3f}{mu:14.3f}{sd:9.3f}"
                  f"{pgf:10.3f}{verdict:>10s}{def_s:>16s}")
    print("  'cancels?' = does the eta-only free-surface gap OPPOSE the PGF "
          "gap (the sign cancellation needs)?")
    print("  'deficit torque' = d(R_lego - R_NEMO)/dt over the PRECEDING 30 "
          "days, same reducer -- the quantity")
    print("     the -0.61 summarises, measured here rather than quoted.")

    if args.out_npz:
        out = {"rows": np.array(ROWS), "days": np.array(DAYS),
               "H_true": H_TRUE, "H_ana": H_ANA}
        for d in DAYS:
            out[f"fs_true_d{d}"] = fs_true[d]
            out[f"fs_ana_d{d}"] = fs_ana[d]
            for n in gaps:
                out[f"pgfgap_{n}_d{d}"] = gaps[n][d]
        for n, dd in traj.items():
            for d, v in dd.items():
                out[f"traj_etaonly_{n}_d{d}"] = v
        np.savez_compressed(args.out_npz, **out)
        print(f"\n[artifact] -> {args.out_npz}")


if __name__ == "__main__":
    main()
