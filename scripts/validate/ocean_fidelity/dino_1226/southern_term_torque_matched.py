"""#1455 SG-T: which momentum term carries the southern gyre's missing torque?
lego's OWN per-term operators vs NEMO's dumped trends, on BIT-IDENTICAL states,
at all ten 10-day NEMO restarts of the 90-day twin window.

RESULT.  Band-mean lego-minus-NEMO row torque [m3/s2], rows 1..13, four arms
that differ in ONE thing: which of NEMO's two vertical ladders the twin is
bridged onto (LEGOESM_NEMO_E3T).

  pressure gradient       day 0    day 30    day 60    day 90
    off        (both ladders analytic)   -6.13   -9.46  -12.96  -15.85
    e3t_only   (true THICKNESS only)     -6.12   -9.45  -12.95  -15.85
    gdept_only (true T-DEPTH only)       -0.07   -0.07   -0.06   -0.05
    both                                 -0.07   -0.07   -0.06   -0.05

  every other term, all arms (unchanged by the ladder):
    vorticity (f+zeta)xu   -0.04 .. -0.52   (erratic, no trend; see DEBT below)
    lateral friction       -0.00 .. +0.02
    vertical advection     +0.000 .. +0.003
    KE gradient            +0.000 .. +0.002
    wind (vs NEMO zdf)     -0.09 .. -0.12   (steady; the drag asymmetry below)

THE FINDING.  Every momentum OPERATOR is at or below this instrument's own
resolution.  The one large, one-signed, monotonically growing difference is the
depth-integrated hydrostatic pressure gradient, and the four-arm ladder A/B
isolates its cause to ONE variable: **NEMO's T-DEPTH ladder (gdept)**.
Swapping in NEMO's true THICKNESS ladder alone changes nothing (-6.12 vs
-6.13); swapping in NEMO's true T-DEPTH ladder alone collapses the gap 88x.
The #1226/#1455 twin arms run with ``LEGOESM_NEMO_E3T`` UNSET, i.e. BOTH
analytic 1-D ladders, so they integrate the right pressure-gradient operator
over density referenced to the wrong depths.

RETRACTIONS, both mine, both from this probe's own later measurements:
  1. The first run of this probe inherited ``LEGOESM_NEMO_E3T`` silently and
     was heading for "the pressure-gradient OPERATOR is the owner".  On the
     true ladder that operator agrees to -0.05, matching the recorded
     ``fidelity_bar_gate`` row (corr 1.0, ratio 1.000000018, itself measured at
     ``--e3t-mode both``).  RETRACTED; a fail-closed gate
     (``require_explicit_e3t_mode``) now makes the mode impossible to inherit.
  2. "The vertical THICKNESS ladder owns it" -- REFUTED by the ``e3t_only``
     arm above, which moves the gap by 0.005.  It is the T-DEPTH ladder.
     (Physically consistent: this card's density input is ``eos_depth=
     "geometric"`` and NEMO's ``hpg_sco`` correction differences ``gdept_z0``
     directly, so a wrong T-depth biases the depth-integrated pressure while a
     wrong thickness largely cancels in the vertical integral.  PLAUSIBLE
     mechanism; the ARM is the measurement.)

The day-0 twin gate could not have caught either: it compares T/S/u/v VALUES
and is blind to the geometry holding them (oracle-fidelity skill Rule 2, that
exact row).  Nor could an ablation (Rule 4): a ladder is not a subsystem either
model can switch off.

WHAT IS NOT PROVEN.  The band's missing torque is -0.61 m3/s2 per row
(57200616f).  On the ladder the arms ran, the pressure-gradient error is -6.1
to -15.9 -- 10-26x LARGER -- and structured across rows (peaking mid-band)
where the deficit is near-uniform.  For it to OWN the deficit the
split-explicit barotropic / free-surface adjustment must cancel ~95% of it and
leave a near-uniform residue.  That happens in the ONE stage this budget cannot
see: legoESM's barotropic solve has no per-term diagnostic slot, and this probe
hands legoESM NEMO's own sea surface, so no adjustment occurs here at all.
Ownership is therefore UNMEASURED, not merely plausible.  What IS measured: no
momentum operator carries a near-uniform -0.61 on any ladder.

  CHEAPEST DECISIVE TEST (not run here, spec'd): the free-surface form stress
  is a pure function of sea level and bathymetry -- no solver instrumentation
  needed.  Compute it explicitly each step, pull it out of the accumulation
  probe's lumped remainder, and run both ladder arms.  If the combined
  pressure + free-surface torque collapses from -6 to ~-0.6 in the wrong-ladder
  arm, ownership is established; if it stays at -6, it is refuted.

WHY THE LADDER WAS NOT SIMPLY SWITCHED: legoESM was recorded as unstable when
integrated from a NEMO restart on the true ladder (max|u| 0.66 -> 2.2 m/s over
20 days), and that was the stated reason the default is "off".
RETRACTED 2026-08-21 (#1455): the instability did not reproduce. Four 90-day
arms from the day-180 restart, differing only in the ladder, all ran stable to
day 90 at 0.633-0.635 m/s peak speed, the two end arms confirmed under fp64. The
citation above is also stale -- ``nemo_state_bridge.py`` no longer records the
instability as the reason. A non-reproduction is not a refutation, so the
observation stands unexplained; but this probe's framing may no longer assume
the switch is blocked, and "that instability and this torque error share a root
cause" is now a hypothesis with one of its two legs missing.

WHAT THIS PROBE DOES
--------------------
For each of the ten NEMO restarts spanning the 90-day twin (days 0,10,...,90):
bridge it through the RECORDED twin harness (``kamm_twin_90d._build_twin_state``
with ``--bridge-before``, so the leap-frog before level is NEMO's own), run the
recorded day-0 gate (state bit-identical to that restart), evaluate legoESM's
``MomentumTendencyDiagnostics`` once, row-integrate every term with the
recorded reducer (``southern_circulation_budget._row_int_trend``), and compare
against NEMO's own ``utrd_*`` trends from the SAME restart.

Because both sides are instantaneous, on the same state, at the same ten times,
SAMPLING ERROR LARGELY CANCELS IN THE DIFFERENCE -- which is why this probe,
not the accumulated one (``southern_term_torque_accum.py``), carries the
attribution.  Its blind spot is the converse: it says nothing about a term that
is right on NEMO's states and wrong on lego's own trajectory, and (see above)
it structurally cannot exhibit the free-surface compensation.

TIME LEVEL (Rule 1d) -- AND THE RETRACTION IT FORCED.
NEMO writes a restart AFTER the leap-frog index rotation, so the ``utrd_*``
dumped in restart kt were evaluated one 2700 s step EARLIER than the ``un``/
``tn`` that restart labels "now": they belong to its ``ub``/``tb`` BEFORE
level, which is what the twin harness leaves on ``state.*_before``.
An earlier revision argued the two sides were aligned from a control (P2) that
measured how far each lego term moves in one model step and got 71 m3/s2 for
the vorticity term, concluding a misalignment would show O(10) while only 0.04
was seen.  THAT NUMBER IS RETRACTED: 71 was legoESM's own start-up transient
off a freshly bridged state, not the physical rate of change, so P2
mis-calibrated its own noise floor in exactly the direction that made a
misaligned comparison look aligned.  It is replaced by a TIME-LEVEL A/B that
prices the misalignment directly with this probe's own operators on both arms
(no proxy): the same model call on the restart's now level and on its before
level.  Measured (e3t=both): the vorticity term moves 0.027 (day 0) to 0.484
(day 90) between levels, the pressure term 0.069 to 0.046, the rest ~0.
Every table row is therefore labelled READABLE or DEBT against twice its own
A/B spread, and:

  * the VORTICITY row is DEBT at every day -- its -0.04..-0.52 differences are
    inside this instrument's time-level uncertainty and may NOT be read as an
    operator result.  "The vorticity operator is clean" is NOT supported here;
    the supported statement is that it carries no near-uniform -0.61.
  * the PRESSURE row is DEBT on the true ladder (-0.05 against an A/B of 0.05,
    i.e. indistinguishable from exact) and READABLE by ~300x on the arms'
    ladder (-15.85 against the same 0.05).  That is the attribution.

TERM MAPPING (lego -> NEMO), and what does NOT map.
  The card ``nemo_dino_kamm_mlf`` runs ``vorticity_scheme="een_total"``, so the
  PLANETARY Coriolis is INSIDE lego's vorticity flux (ocean_pe_latlon_cgrid.py
  :4129-4135 excludes the separate f-add for een_total) and ``vortcor_u``
  carries (f+zeta) x u.  NEMO dumps the planetary and relative pieces by
  calling the SAME EEN operator twice (dynvor.F90:143-174), and that operator
  is linear in the vorticity handed to it (dynvor.F90:786-812) with vorticity
  masking off (namelist_cfg:333), so ``pvo + rvo`` is exactly the full term.
  ``KE_PGF_u`` is -dKE/dx - dp'/dx/rho0 with dp' the BAROCLINIC pressure only;
  g*d(eta) lives in the barotropic solve, exactly as NEMO's ``hpg_sco``
  (MY_SRC/dynhpg.F90:305-419, ``ln_hpg_sco=.true.``, namelist_cfg:343) carries
  only the ``rhd`` integral plus the s-coordinate ``mi(rhd)*di[g*gdept]``
  correction and leaves g*d(eta) to ``dyn_spg_ts``.  NEMO's KE gradient is the
  HOLLINGSWORTH form (namelist_cfg:322 ``nn_dynkeg=1``, dynkeg.F90:128-153) and
  the card selects the same, so that piece is like-for-like.  So:

    G1 VOR    lego vortcor_u                      == NEMO pvo + rvo     EXACT
    G2 ADVPG  lego KE_PGF_u + vertadv_u + Dterm_u == NEMO keg+zad+hpg   EXACT
    G3 LDF    lego Ah_lap+Bh_bilap+Cs_smag+Cl_leith == NEMO ldf         SEE BELOW
    G4 WIND   lego WIND (see below)              ~  NEMO zdf (recovered)
    --        NEMO spg and NEMO atf have NO lego counterpart in
              ``MomentumTendencyDiagnostics`` (the barotropic solve and the
              Asselin filter are step-level stages).  Reported as ABSENT, never
              as zero.  This is the gap that blocks the ownership claim above.

  KNOWN MAPPING DEBT, stated not hidden:
   * G3: NEMO evaluates lateral friction on the BEFORE velocity
     (dynldf_lev.F90:55, "applied on pu(Kbb)"); this probe evaluates every term
     at one level per arm, so the G3 row inherits whichever arm it is read on.
     The time-level A/B above bounds the error at <=0.004.  The companion
     accumulation probe does take LDF from the Nbb pass.
   * G4 is a 340-to-1 CANCELLATION on NEMO's side: its recovered vertical term
     is a difference of two numbers near 5900 leaving 17.4, so one part in 1e4
     in either half is 0.6 -- the whole deficit.  This row can never resolve
     anything at the campaign's target scale.  Informational only, never a
     ranking row; its steady -0.10 is ASSERTED to be the wind-vs-drag asymmetry
     (lego's bottom drag sits in the implicit tridiagonal,
     ``zdf_drag_in_matrix``, and is not a diagnostics member) and is NOT
     separately computed here.
   * NEMO'S OWN TRENDS DO NOT CLOSE.  Summing all nine dumped trends with the
     vertical term repaired leaves ~17.9 m3/s2 per row unaccounted against the
     one-step circulation change -- 29x the deficit.  NEMO applies a barotropic
     reconciliation near the end of each step (stpmlf.F90:561) with no trend
     slot, which is a CANDIDATE owner, not a finding.  Per the fidelity skill's
     Rule 5 this means no ABSOLUTE reading of NEMO's trend sum is licensed;
     the per-term DIFFERENCES this probe reports are unaffected, because the
     unclosed piece is on NEMO's side of every arm equally.

  The G2 bundle is split EXACTLY, without re-deriving numerics: the same model
  call is evaluated on a state whose velocities are zero and whose T/S/eta are
  untouched, so dKE/dx vanishes identically and the field is the pure pressure
  gradient (NEMO's hpg is velocity-independent too).  Verified by reading the
  path: ``p_prime_filled``/``rho_prime`` derive from eta/T/S only, and the
  Adcroft / smc03 / nemo_sco corrections read rho_prime, h_partial and
  r3t=ssh/ht_0 -- none velocity-dependent.  That split is what shows the KE
  gradient matches to 0.002 and the whole G2 difference is pressure.

A COVERAGE HOLE IN ``MomentumTendencyDiagnostics``, MEASURED NOT ASSUMED.
On this card the per-term fields do NOT sum to the model's own ``total_u``:
the miss is 1.92e-5 m/s2 against a 4.85e-5 total (40%).  It is confined to
level k=0 EXACTLY (max|residual| at k>0 is 0.0), and its row integral
reproduces the analytic DINO wind torque (``phi_wind``) to 2.1-3.7e-4
relative -- i.e. it IS the surface wind-stress deposit
(``_bc_external_surface_forcing``), which enters ``du_dt`` but is captured by
no diagnostic field (``phys_u`` is identically 0 here: the deposit is not a
``physics_fn`` contribution).  This probe therefore carries WIND as an
IDENTIFIED term, ``total_u - sum(components)``, gated by BOTH facts.  The
identification is exact in the k>0 arm and good to 3.7e-4 relative (~0.2
m3/s2 absolute at band values) in the analytic-wind arm, so WIND is identified
to ~0.2 m3/s2 and no finer.  The other diagnostics-zero terms are measured and
printed, not assumed: Dterm/Bh_bilap/Cs_smag/Cl_leith/botdrag/Av_vert/phys/
sponge are all identically 0 on this card.

CONTROLS (all fatal unless marked)
  C1 fp64 policy + printed dtypes; LEGOESM_NEMO_E3T gated explicit and printed;
     the ladder A/B's own baseline printed and asserted non-trivial (the two
     ladders must differ by >1% somewhere, else the A/B perturbs a zero).
  C2 non-finite anywhere on the wet mask -> raise.
  C3 day-0 gate: the bridged state must equal the NEMO restart it came from
     (the recorded ``verify_day0_matches_restart``), at EVERY one of the ten
     restarts.
  C4 REPORTING ONLY, and labelled as such: "components + WIND == total_u" is an
     algebraic identity (WIND is DEFINED as the difference) and cannot fail.
     The real check is the W gate's k>0 arm.
  C5 NEMO coverage: the three waived trends (tau/bfr/bfri) must be identically
     zero (enforced).
  W  the surface residual is the identified wind deposit -- exactly zero at
     k>0 AND equal to the analytic wind torque -- else raise.
  P1 planted: +1e-7 m/s2 into KE_PGF_u, pushed through the FULL downstream path
     (reducer -> term dict -> group mapping -> table), must shift group G2 by
     exactly the hand-computed row integral (425.6..843.9 m3/s2), must leak
     <1e-12 into any other group, and must not be a no-op.  All three asserted.
  A/B the time-level arm described above; every row labelled READABLE or DEBT.
  STITCH the stitched restarts must contain no non-finite value anywhere --
     ``rebuild`` NaN-fills unwritten regions and skips variables missing from a
     tile, so a dropped rank file would otherwise become 0 degC / 0 PSU water
     that every downstream finite check passes.

Diagnosis only: reads recorded artifacts + runs the model; writes only its npz
(and, for the 10-day restarts, a validated stitched single-file copy in
--scratch, since the recorded twin harness reads one file and those restarts
ship as per-rank tiles; nothing about the physics path changes).

Run the four arms -- the single variable is the ladder:
  for M in off e3t_only gdept_only both; do
    JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=$M .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \\
      scripts/validate/ocean_fidelity/dino_1226/southern_term_torque_matched.py \\
      --scratch /tmp/dino_stitch --out-npz /tmp/sweep_$M.npz
  done
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_explicit_e3t_mode,
)

set_policy(PrecisionPolicy.fp64())
# MANDATORY, fails closed.  Unset, the restart bridge silently substitutes
# NEMO's ANALYTIC 1-D thickness ladder (``e3t_1d``) for the real ``e3t_0``,
# which differs by up to 12.9% below k=25 -- and the day-0 twin gate CANNOT
# see it (it compares T/S/u/v VALUES, not the geometry holding them; skill
# Rule 2's documented blind spot).  A depth-integrated pressure gradient on a
# 12.9%-wrong deep ladder is a systematic, time-growing error that looks
# exactly like an operator defect.  That default has already ruined four
# measurements in this campaign; it very nearly ruined this one.
E3T_MODE = require_explicit_e3t_mode(context='southern_term_torque_matched')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import netCDF4 as nc  # noqa: E402

import southern_circulation_budget as B  # noqa: E402  (recorded reducers/geometry)
import acceptance_gate_90d as G  # noqa: E402
from kamm_twin_90d import (  # noqa: E402  (the recorded twin harness)
    DT, _build_twin_state, seasonal_t0_seconds,
)
from legoesm.ocean.experiments.dino import (  # noqa: E402
    apply_dino_lat_lon_surface_forcing,
)
from rebuild_nemo_restart import rebuild  # noqa: E402

RDT = 2.0 * DT
ROWS = list(B.ROWS)

# lego u.data is (n_lat, n_lon+1, nlev); NEMO u[i] == lego u[:, i+1] (the twin
# gate's own slice, kamm_twin_90d.verify_day0_matches_restart).
_USLICE = np.s_[:, 1:, :]

LEGO_GROUPS = {
    "G1 VOR   (f+zeta)xu": ("vortcor_u",),
    "G2 ADVPG KE+pgf+zad": ("KE_PGF_u", "vertadv_u", "Dterm_u"),
    "G3 LDF   lateral":    ("Ah_lap_u", "Bh_bilap_u", "Cs_smag_u", "Cl_leith_u"),
    "G4 WIND  surface":    ("WIND_u", "phys_u", "botdrag_u", "Av_vert_u",
                            "sponge_u"),
}
NEMO_GROUPS = {
    "G1 VOR   (f+zeta)xu": ("pvo", "rvo"),
    "G2 ADVPG KE+pgf+zad": ("keg", "zad", "hpg"),
    "G3 LDF   lateral":    ("ldf",),
    "G4 WIND  surface":    ("__zdf_true__",),
}


def row_int(field_nemo_shaped):
    """sum_i e1u * sum_k e3u_0 * X  [m3/s2] per u-row -- the recorded reducer."""
    return B._row_int_trend(field_nemo_shaped)


def lego_terms(diag, *, gate=False):
    """Per-term row torques [m3/s2] from MomentumTendencyDiagnostics, plus the
    IDENTIFIED wind term (see the module docstring's coverage-hole note).

    ``gate=True`` runs the two facts that identify the residual as the wind
    stress: it is confined to k=0, and its row integral reproduces the
    analytic DINO wind torque.  Fatal if either fails."""
    out = {}
    for f in diag._fields:
        if not f.endswith("_u"):
            continue
        a = np.asarray(diag_field(diag, f), np.float64)
        if not np.isfinite(a[B.umask]).all():
            raise SystemExit(f"FATAL C2: non-finite in lego diagnostic {f}")
        out[f] = row_int(a)
    # WIND: total - components, in FULL 3-D (not row space), so the k=0
    # confinement can be tested before the reduction collapses it.
    full = {f: np.asarray(diag_field(diag, f), np.float64)
            for f in diag._fields if f.endswith("_u")}
    resid = full["total_u"] - sum(v for k, v in full.items() if k != "total_u")
    out["WIND_u"] = row_int(resid)
    if gate:
        deep = float(np.max(np.abs(resid[:, :, 1:])))
        print(f"  identified-WIND gate: max|residual| at k>0 = {deep:.3e} "
              f"(must be 0)")
        if deep != 0.0:
            raise SystemExit(
                "FATAL C4a: the total-minus-components residual is NOT confined "
                "to the surface level, so it is not the wind deposit -- it is an "
                "unidentified bucket and no row of the table may be read")
        ana = B.phi_wind()
        rr = np.abs(out["WIND_u"] - ana)[list(B.ROWS)] / np.maximum(
            np.abs(ana)[list(B.ROWS)], 1e-30)
        print(f"  identified-WIND gate: row integral vs analytic phi_wind, "
              f"band max rel = {float(rr.max()):.3e} "
              f"(e3u_0-vs-live-thickness level)")
        if float(rr.max()) > 5.0e-3:
            raise SystemExit(
                "FATAL C4b: the surface residual does not reproduce the analytic "
                "wind torque -- it is not identified and must not be tabulated")
    return out


def diag_field(diag, name):
    return np.asarray(getattr(diag, name).data)[_USLICE]


def nemo_terms(kt):
    """Row torques [m3/s2] from NEMO's dumped trends at restart step ``kt``."""
    names = [f"utrd_{t}" for t in B.NEMO_TRENDS + B.NEMO_TRENDS_WAIVED] + ["un"]
    raw = rebuild(f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart*.nc", names)
    yxz = lambda a: np.moveaxis(np.asarray(a, np.float64), 0, -1)  # noqa: E731
    for t in B.NEMO_TRENDS_WAIVED:                       # C5: the waiver, enforced
        mx = float(np.max(np.abs(np.asarray(raw[f"utrd_{t}"], np.float64))))
        if mx != 0.0:
            raise SystemExit(
                f"FATAL C5: utrd_{t} is non-zero ({mx:.3e}) at kt={kt} -- it was "
                "WAIVED as identically zero and is not in the mapping")
    tr = {}
    for t in B.NEMO_TRENDS:
        a = yxz(raw[f"utrd_{t}"])
        if not np.isfinite(a[B.umask]).all():
            raise SystemExit(f"FATAL C2: non-finite in NEMO utrd_{t}")
        tr[t] = row_int(a)
    un = yxz(raw["un"])
    dt2 = B._zdf_is_barotropic_subtraction(yxz(raw["utrd_zdf"]), un, kt)
    tr["__zdf_true__"] = tr["zdf"] + B.row_circulation(un) / dt2
    return tr


def group_sum(terms, keys):
    if not keys:
        return np.zeros(B.NY, np.float64)
    return sum(terms[k] for k in keys)



# ---------------------------------------------------------------- multi-day --
# The 10-day NEMO restarts in RUN_90D_TWIN are per-rank TILES; the recorded
# twin harness (`kamm_twin_90d._build_twin_state`) reads a SINGLE file.  Rather
# than duplicate the bridge/config/model construction (which is the recorded
# instrument -- Rule 1e says reuse it, do not rebuild it), stitch each restart
# into one scratch file with `rebuild` and hand THAT to the recorded harness.
# Nothing about the physics path changes; only the file layout.
_STITCH_VARS_3D = ("tn", "sn", "un", "vn", "tb", "sb", "ub", "vb")
_STITCH_VARS_2D = ("sshn", "sshb")


def stitch_restart(kt, scratch):
    """Write a single-file copy of the kt restart; return its path."""
    out = os.path.join(scratch, f"DINO_{kt:08d}_restart.nc")
    if os.path.exists(out):
        return out
    raw = rebuild(f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart*.nc",
                  list(_STITCH_VARS_3D) + list(_STITCH_VARS_2D))
    missing = [v for v in _STITCH_VARS_3D + _STITCH_VARS_2D if v not in raw]
    if missing:
        raise SystemExit(f"FATAL: restart kt={kt} lacks {missing} -- the "
                         "before-level bridge cannot be seeded")
    # ``rebuild`` fills UNWRITTEN regions with NaN and SKIPS a variable that is
    # absent from a tile, and the `missing` test above only requires a variable
    # to appear in ONE tile.  A dropped or truncated rank file would therefore
    # leave a NaN subdomain that the nan_to_num below turns into T=0 / S=0
    # fresh water -- a plausible-looking number that every downstream finite
    # check passes, and the exact failure this repo has already shipped once
    # (39032 wet cells at 0 degC).  The day-0 gate cannot catch it either: it
    # compares the bridged state against THIS SAME stitched file (self-
    # referential w.r.t. stitching) and only at k=0.  So: no NaN may survive
    # anywhere in a stitched field.
    for v in _STITCH_VARS_3D + _STITCH_VARS_2D:
        n_nan = int(np.count_nonzero(~np.isfinite(raw[v])))
        if n_nan:
            raise SystemExit(
                f"FATAL: stitched restart kt={kt} field {v!r} has {n_nan} "
                "non-finite entries -- rebuild() did not cover the global "
                "domain (a missing/truncated rank tile).  Refusing to write a "
                "file whose gaps would become 0 degC / 0 PSU water.")
    ds = nc.Dataset(out, "w")
    nz, ny, nx = raw["tn"].shape
    ds.createDimension("z", nz); ds.createDimension("y", ny)
    ds.createDimension("x", nx); ds.createDimension("t", 1)
    for v in _STITCH_VARS_3D:
        a = np.nan_to_num(raw[v], nan=0.0)
        var = ds.createVariable(v, "f8", ("t", "z", "y", "x"))
        var[0] = a
    for v in _STITCH_VARS_2D:
        a = np.nan_to_num(raw[v], nan=0.0)
        var = ds.createVariable(v, "f8", ("t", "y", "x"))
        var[0] = a
    ds.close()
    return out


def restart_path(kt, scratch):
    """Single-file restart for step kt (day 0 already ships as one file)."""
    single = f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart.nc"
    if os.path.exists(single):
        return G.RUN_90D_TWIN, os.path.basename(single)
    return scratch, os.path.basename(stitch_restart(kt, scratch))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    ap.add_argument("--days", default="0,10,20,30,40,50,60,70,80,90",
                    help="NEMO restart days to evaluate (matched states)")
    ap.add_argument("--scratch", default="/tmp/dino_stitch")
    ap.add_argument("--out-npz", default=None)
    args = ap.parse_args(argv)
    days = [int(x) for x in args.days.split(",") if x != ""]
    os.makedirs(args.scratch, exist_ok=True)

    print("=" * 112)
    print("C1 PRECISION")
    print("=" * 112)
    print(f"  policy control dtype = {get_policy().control}")
    print(f"  LEGOESM_NEMO_E3T = {E3T_MODE!r}  ('both' = NEMO's real "
          f"3-D thickness/depth ladder; 'off' = the analytic 1-D one)")
    # The A/B's own baseline: PRINT how far apart the two ladders actually are
    # before any result rests on switching between them ("a control that
    # perturbs a zero is not a control").  Measured on the mesh the twin is
    # bridged from, not quoted from a comment.
    _e1d = np.asarray(B.A.mm["e3t_1d"][0]).ravel().astype(np.float64)
    _e3d = np.asarray(B.A.mm["e3t_0"][0]).squeeze()
    _e3d = np.moveaxis(_e3d, 0, -1) if _e3d.shape[0] == _e1d.size else _e3d
    _tm = np.asarray(B.tmask) > 0.5
    _wetk = [k for k in range(_e3d.shape[-1]) if _tm[:, :, k].any()]
    _mean = np.array([_e3d[:, :, k][_tm[:, :, k]].mean() for k in _wetk])
    _rel = (_mean - _e1d[_wetk]) / _e1d[_wetk]
    print(f"  vertical-ladder A/B baseline: analytic e3t_1d vs NEMO's e3t_0 "
          f"over {len(_wetk)} wet levels -- max |rel| thickness difference "
          f"{np.abs(_rel).max():.4f} at k={_wetk[int(np.argmax(np.abs(_rel)))]}"
          f"; first level exceeding 1%: k="
          f"{next((k for k, r in zip(_wetk, _rel) if abs(r) > 0.01), None)}"
          f"; TOTAL wet column depth {_e1d[_wetk].sum():.1f} m vs "
          f"{_mean.sum():.1f} m (the two ladders REDISTRIBUTE thickness in the "
          "deep column, they do not change its depth)")
    if np.abs(_rel).max() < 0.01:
        raise SystemExit(
            "FATAL: the two vertical ladders differ by <1% everywhere, so the "
            "LEGOESM_NEMO_E3T A/B would be perturbing a near-zero and could "
            "not support any conclusion -- check the mesh_mask")
    for nm, a in (("e1u", B.e1u), ("e3u0", B.e3u0), ("umask", B.umask)):
        print(f"  {nm:8s} {np.asarray(a).dtype}  shape {np.asarray(a).shape}")

    rows = ROWS
    per_day = {}
    for day in days:
        kt = G.KT_RESTART + day * G.STEPS_PER_DAY
        rdir, rfile = restart_path(kt, args.scratch)
        print("\n" + "#" * 112)
        print(f"# DAY {day}   NEMO restart kt={kt}   ({rdir}/{rfile})")
        print("#" * 112)

        br, cfg, mc, model, forcing, sf, st = _build_twin_state(
            args.recipe, f"{B.A.DINO}/RUN_TRAJ", rdir,
            bridge_before=True, restart_file=rfile)
        if np.asarray(st.u.data).dtype != np.float64:
            raise SystemExit("FATAL C1: model state is not float64")
        if day == days[0]:
            print(f"  state dtypes u/T/eta = "
                  f"{np.asarray(st.u.data).dtype}/"
                  f"{np.asarray(st.T.data).dtype}/"
                  f"{np.asarray(st.eta.data).dtype}")
            print(f"  outer_integrator={getattr(mc, 'outer_integrator', '?')}  "
                  f"vorticity_scheme={cfg.vorticity_scheme}  "
                  f"coriolis_scheme={cfg.coriolis_scheme}  "
                  f"surface_tendency_placement={cfg.surface_tendency_placement}")
        placement = getattr(cfg, "surface_tendency_placement", "applied_now")
        # #1455 SEASONAL CLOCK: this probe re-bridges NEMO's OWN restart for
        # each matched day, so the forcing must be evaluated at that restart's
        # day of year. Passing a bare ``DT`` would run every day's terms at
        # seasonal day 0.03 against a state from day 180+ -- the antiphase
        # corrected in 1c03f8311/076217667. Read per day, because the restart
        # changes per day.
        t0_sec = seasonal_t0_seconds(f"{rdir}/{rfile}")

        def terms_at(state, t_seconds, *, gate=False):
            """lego per-term row torques, mirroring the twin loop's forcing
            application (kamm_twin_90d.run_twin) exactly."""
            if placement == "leapfrog_rhs":
                s2, _r = apply_dino_lat_lon_surface_forcing(
                    state, forcing, br.z_coord, cfg, DT, t_seconds=t_seconds,
                    return_rate=True)
            else:
                s2 = apply_dino_lat_lon_surface_forcing(
                    state, forcing, br.z_coord, cfg, DT, t_seconds=t_seconds)
            _t, diag = model.tendencies_with_diagnostics(
                s2, surface_forcing=sf, sponge=None, dt=RDT)
            return lego_terms(diag, gate=gate), diag

        L, diag0 = terms_at(st, t0_sec + DT, gate=True)

        # C4 -- THE HONEST VERSION.  ``WIND_u`` is DEFINED as
        # ``total_u - sum(components)``, so "sum(terms incl. WIND) == total_u"
        # is an algebraic identity and CANNOT FAIL: it measures nothing.  (An
        # earlier revision printed its 3e-16 as evidence the budget closes;
        # that was vacuous and is retracted.)  What CAN fail, and is what
        # actually licenses treating WIND as an identified term rather than a
        # bucket, is that the components account for the total EVERYWHERE
        # EXCEPT the surface level -- checked in full 3-D by the W gate above.
        # Re-stated here as the numbers the table rests on.
        tot = row_int(np.asarray(diag0.total_u.data)[_USLICE])
        comp = sum(v for k, v in L.items()
                   if k not in ("total_u", "WIND_u", "PGFONLY_u", "KEGRAD_u")
                   and not k.startswith("BEFORE_") and k != "__ab__")
        gap = float(np.max(np.abs((comp - tot)[rows])))
        print(f"  C4 sum(components, WIND EXCLUDED) vs total_u: band max gap "
              f"{gap:.3f} m3/s2 = the surface wind deposit alone "
              f"(identity check with WIND included is vacuous by construction "
              f"and is NOT reported)")

        if day == days[0]:
            # P1 PLANTED CONTROL -- injected into the DIAGNOSTIC FIELD and
            # then pushed through the WHOLE downstream path (row reducer ->
            # term dict -> LEGO_GROUPS mapping -> group table), so it tests
            # the mapping and grouping, not just the linearity of np.sum.  An
            # earlier revision compared row_int(x+c) - row_int(x) against
            # row_int(c), which is true for any array by construction and
            # exercised none of that machinery; retracted.
            # PREDICTION, all three parts asserted:
            #   (a) group G2 (the group KE_PGF_u belongs to) shifts by exactly
            #       the hand-computed row integral of the plant;
            #   (b) groups G1/G3/G4 are BIT-unchanged;
            #   (c) the plant is not a no-op (its own row integral is O(1e2)).
            plant = 1.0e-7
            _kp = np.array(np.asarray(diag0.KE_PGF_u.data))
            _kp[:, 1:, :] = np.where(B.umask, _kp[:, 1:, :] + plant,
                                     _kp[:, 1:, :])
            diagP = diag0._replace(
                KE_PGF_u=diag0.KE_PGF_u.replace(data=jnp.asarray(_kp)),
                total_u=diag0.total_u.replace(
                    data=jnp.asarray(np.asarray(diag0.total_u.data)
                                     + (np.asarray(_kp)
                                        - np.asarray(diag0.KE_PGF_u.data)))))
            LP = lego_terms(diagP)
            pred = row_int(np.where(B.umask, plant, 0.0))
            g2 = "G2 ADVPG KE+pgf+zad"
            shift = (group_sum(LP, LEGO_GROUPS[g2])
                     - group_sum(L, LEGO_GROUPS[g2]))
            d = float(np.max(np.abs((shift - pred)[rows])))
            untouched = {g: float(np.max(np.abs(
                (group_sum(LP, LEGO_GROUPS[g]) - group_sum(L, LEGO_GROUPS[g]))
                [rows]))) for g in LEGO_GROUPS if g != g2}
            print(f"  P1 planted +{plant:.0e} m/s2 into KE_PGF_u, pushed "
                  f"through the full group mapping: predicted G2 row shift "
                  f"{pred[rows].min():.3f}..{pred[rows].max():.3f}; measured "
                  f"minus predicted band max {d:.3e}; other groups moved "
                  + ", ".join(f"{g.split()[0]}={v:.1e}"
                              for g, v in untouched.items()))
            if float(np.min(np.abs(pred[rows]))) < 1e-3:
                raise SystemExit("FATAL P1: the plant is a no-op (vacuous)")
            if d > 1e-9:
                raise SystemExit("FATAL P1: the planted torque does not land in "
                                 "its own group at the predicted value")
            if max(untouched.values()) > 1e-12:
                raise SystemExit("FATAL P1: the planted torque LEAKED into "
                                 f"another group: {untouched}")


        # --- TIME-LEVEL A/B (Rule 1d), priced with OUR OWN operators -------
        # NEMO writes a restart AFTER the leap-frog index rotation, so the
        # trends dumped in restart kt were evaluated one step EARLIER than the
        # ``un``/``tn`` that restart labels "now" -- they belong to its
        # ``ub``/``tb`` (before) level.  Evaluating legoESM at the now level
        # therefore compares across one 2700 s step.
        #
        # This arm prices that misalignment DIRECTLY: the identical model call
        # on the state whose now-fields are the restart's BEFORE fields (the
        # same substitution ``_leapfrog_step`` makes to build its Nbb pass).
        # No proxy operator, no inference.  The A/B spread is the instrument's
        # OWN uncertainty on every row of the table below, and a row whose
        # lego-NEMO difference is smaller than its own A/B spread is DEBT, not
        # agreement.
        #
        # It supersedes the earlier P2 control, which argued alignment from
        # "the vorticity term moves 71 m3/s2 per step but we see 0.04".  That
        # 71 was legoESM's own start-up transient off a freshly bridged state,
        # not the physical rate of change, so P2 mis-calibrated its own noise
        # floor in the direction that made a misaligned comparison look
        # aligned.  P2's number is RETRACTED; this arm replaces it.
        N = nemo_terms(kt)
        st_bb = st._replace(u=st.u_before, v=st.v_before, T=st.T_before,
                            S=st.S_before, eta=st.eta_before)
        L_bb, _dbb = terms_at(st_bb, t0_sec + DT)
        print("  TIME-LEVEL A/B -- same operators on the restart's NOW vs its "
              "BEFORE level (NEMO's trends belong to BEFORE):")
        print(f"    {'term':14s}{'at NOW':>12s}{'at BEFORE':>12s}"
              f"{'|A/B|':>10s}{'lego-NEMO':>12s}{'verdict':>10s}")
        _pairs = (("vortcor_u", ("pvo", "rvo")), ("KE_PGF_u", ("keg", "hpg")),
                  ("vertadv_u", ("zad",)), ("Ah_lap_u", ("ldf",)),
                  ("WIND_u", ("__zdf_true__",)))
        for k, nk in _pairs:
            a = float(L[k][rows].mean())
            b = float(L_bb[k][rows].mean())
            ab = abs(a - b)
            dn = a - float(group_sum(N, nk)[rows].mean())
            verdict = "READABLE" if abs(dn) > 2.0 * ab else "DEBT"
            print(f"    {k:14s}{a:12.3f}{b:12.3f}{ab:10.3f}{dn:12.3f}"
                  f"{verdict:>10s}")
        print("    'DEBT' = the lego-NEMO difference is inside this "
              "instrument's own time-level uncertainty and may NOT be read as "
              "an operator result.")
        L["__ab__"] = np.array([0.0])   # marker; per-term A/B kept in L_bb
        for k in list(L_bb):
            L[f"BEFORE_{k}"] = L_bb[k]

        per_day[day] = (L, N)

        print(f"\n  GROUP TABLE, day {day} [m3/s2 per u-row, band mean unless "
              "stated]")
        print(f"  {'group':24s}{'lego':>11s}{'NEMO':>11s}{'diff':>10s}"
              f"{'diff min':>10s}{'diff max':>10s}{'uniformity':>12s}")
        for gname, lkeys in LEGO_GROUPS.items():
            lv = group_sum(L, lkeys)
            nv = group_sum(N, NEMO_GROUPS[gname])
            d = lv - nv
            mu = float(np.mean(d[rows]))
            sd = float(np.std(d[rows]))
            print(f"  {gname:24s}{lv[rows].mean():11.3f}{nv[rows].mean():11.3f}"
                  f"{mu:10.3f}{d[rows].min():10.3f}{d[rows].max():10.3f}"
                  f"{(sd / abs(mu) if abs(mu) > 1e-12 else np.inf):12.2f}")
        print(f"  {'NEMO spg (no lego side)':24s}{'--':>11s}"
              f"{N['spg'][rows].mean():11.3f}")
        print(f"  {'NEMO atf (no lego side)':24s}{'--':>11s}"
              f"{N['atf'][rows].mean():11.3f}")
        # --- EXACT split of lego's KE_PGF bundle -------------------------
        # lego bundles the kinetic-energy gradient with the baroclinic
        # pressure gradient in ONE field (ocean_pe_latlon_cgrid.py:4062,
        # KE_PGF_u = -dKE/dx - dp'/dx/rho0); NEMO dumps them separately
        # (keg, hpg).  The bundle is split WITHOUT re-deriving any numerics:
        # evaluate the SAME model call on a state whose velocities are zero
        # and whose T/S/eta are untouched.  dKE/dx is then identically 0 and
        # the field is the pure pressure gradient; the difference from the
        # full evaluation is lego's KE gradient.  NEMO's hpg does not depend
        # on velocity either, so the two sides are the same quantity.
        z = jnp.zeros_like(st.u.data)
        zv = jnp.zeros_like(st.v.data)
        st0 = st._replace(u=st.u.replace(data=z), v=st.v.replace(data=zv),
                          u_before=st.u_before.replace(data=z),
                          v_before=st.v_before.replace(data=zv))
        L0, _d0 = terms_at(st0, t0_sec + DT)
        pgf_only = L0["KE_PGF_u"]
        keg_lego = L["KE_PGF_u"] - pgf_only
        print("  EXACT KE/PGF SPLIT (lego evaluated at u=v=0; T/S/eta "
              "untouched):")
        print(f"    lego PGF only {pgf_only[rows].mean():10.3f}   vs NEMO hpg "
              f"{N['hpg'][rows].mean():10.3f}   diff "
              f"{(pgf_only - N['hpg'])[rows].mean():9.3f}"
              f"   [min {(pgf_only - N['hpg'])[rows].min():.3f}, max "
              f"{(pgf_only - N['hpg'])[rows].max():.3f}]")
        print(f"    lego KE grad  {keg_lego[rows].mean():10.3f}   vs NEMO keg "
              f"{N['keg'][rows].mean():10.3f}   diff "
              f"{(keg_lego - N['keg'])[rows].mean():9.3f}"
              f"   [min {(keg_lego - N['keg'])[rows].min():.3f}, max "
              f"{(keg_lego - N['keg'])[rows].max():.3f}]")
        L["PGFONLY_u"] = pgf_only
        L["KEGRAD_u"] = keg_lego
        print("  SPLIT of the pressure/advection group:")
        print(f"    lego KE_PGF_u {L['KE_PGF_u'][rows].mean():10.3f}   vs NEMO "
              f"keg+hpg {(N['keg'] + N['hpg'])[rows].mean():10.3f}   diff "
              f"{(L['KE_PGF_u'] - N['keg'] - N['hpg'])[rows].mean():9.3f}")
        print(f"    lego vertadv_u{L['vertadv_u'][rows].mean():10.3f}   vs NEMO "
              f"zad     {N['zad'][rows].mean():10.3f}   diff "
              f"{(L['vertadv_u'] - N['zad'])[rows].mean():9.3f}")
        print(f"    NEMO keg alone{N['keg'][rows].mean():10.3f}   NEMO hpg "
              f"alone {N['hpg'][rows].mean():10.3f}")
        jax.clear_caches()

    # ------------------------------------------------------------- SUMMARY --
    print("\n" + "=" * 112)
    print("SUMMARY -- lego minus NEMO row torque [m3/s2], band mean per day "
          "(matched states, zero trajectory divergence)")
    print("=" * 112)
    gnames = list(LEGO_GROUPS)
    print(f"  {'day':>4s}" + "".join(f"{g.split()[0]:>12s}" for g in gnames)
          + f"{'sum':>12s}")
    for day in days:
        L, N = per_day[day]
        ds = [float(np.mean((group_sum(L, LEGO_GROUPS[g])
                             - group_sum(N, NEMO_GROUPS[g]))[rows]))
              for g in gnames]
        print(f"  {day:4d}" + "".join(f"{x:12.3f}" for x in ds)
              + f"{sum(ds):12.3f}")

    if args.out_npz:
        out = {"rows": np.array(rows), "days": np.array(days)}
        for day in days:
            L, N = per_day[day]
            for k, v in L.items():
                out[f"lego_{k}_d{day}"] = v
            for k, v in N.items():
                out[f"nemo_{k}_d{day}"] = v
        np.savez_compressed(args.out_npz, **out)
        print(f"\n[artifact] -> {args.out_npz}")


if __name__ == "__main__":
    main()
