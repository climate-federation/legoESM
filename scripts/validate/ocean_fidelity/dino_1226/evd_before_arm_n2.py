#!/usr/bin/env python
"""#1226 y20 spike, STEP 1: print legoESM's OWN two EVD trigger-arm N^2 at the
one interface where the EVD region stops short, beside NEMO's rn2/rn2b.

The finding this continues (commit 7938a590d): NEMO's enhanced-diffusion (EVD,
rn_evd=100 m2/s) region in column (j=158, i=42) reaches legoESM interface 9
(NEMO 0-based w-level 10); legoESM's stops at interface 8 and leaves 2.2e-02
there.  Cell 10 keeps its before-level warm anomaly and ends 5.71e-02 K warmer
than NEMO -- the surviving per-step tracer spike, a closed column
redistribution (0.07%).

At that interface NEMO's own dumps give

    rn2  (now, ts(...,Nnn)) = -7.191933e-12      threshold -1.e-12  ->  7x
    rn2b (before, Nbb)      = -4.066547e-09      threshold -1.e-12  ->  4000x

so NEMO fires through the Nbb arm of ``MIN(rn2,rn2b) <= -1.e-12``
(zdfevd.F90:92-94 / :118-120), with both arms built by ``bn2`` on the SAME Nnn
geometry (MY_SRC/stpmlf.F90:200-201).  The DINO card already selects the
two-armed trigger, so the question is only what N^2 legoESM's before arm
actually receives.

THIS PROBE MEASURES, IT DOES NOT FIX.  Three blocks, each one variable:

  (A) production one-step, hooks on the LIVE trigger: both arms' N^2 at the
      spike interfaces, the two threshold comparisons, the resulting K, and
      the K the implicit solve finally gets.  Continuity control first.
  (B) OFFLINE bracket, no model run: legoESM's OWN bridged before-level
      T/S/eta pushed through (i) NEMO's bn2 formula
      (``compute_buoyancy_frequency_nemo_bn2``, an exact transcription of
      eosbn2.F90:1453-1462) and (ii) the adiabatic parcel-displacement N^2 the
      card's ``convection_n2_mode='adiabatic'`` selects.  Same inputs, two
      formulas -> separates "wrong input tracers" from "wrong N^2 formula".
  (C) INJECTION bracket (``--inject-rn2b``): replace ONLY the before arm's
      N^2 with NEMO's own rn2b dump and re-run the identical step.  If the
      spike collapses, the before-arm N^2 owns it and the collapse ratio is
      its measured share.

CONTROLS: fp64 (``run_fp64.py``), ``LEGOESM_NEMO_E3T=both``, day-0 gate inside
``build_replay_ic``, realized card values printed (Rule 10), FORCING printed,
``jax.clear_caches()`` between arms, hook-fired assertions (a hook that never
fires is the failure mode ``spike_kv_column.py`` documents), and the NEMO dump
time levels taken from ``ocean.fidelity.time_levels`` rather than assumed.

RESULT (2026-08-18, fp64, LEGOESM_NEMO_E3T=both, wind ON, day-0 gate 0.000e+00,
both hooks fired 2x, continuity control 5.710106e-02 @(158,42,10) EXACT)
------------------------------------------------------------------------------
The four numbers at legoESM interface 9 / NEMO w-level 10, threshold -1.e-12:

    NEMO rn2  (now,    Nnn)     -7.191933e-12      FIRES
    NEMO rn2b (before, Nbb)     -4.066547e-09      FIRES
    legoESM live arm 0 (now)    -7.522575e-13      does NOT fire
    legoESM live arm 1 (before) +5.988313e-09      does NOT fire -- OPPOSITE SIGN

Arm identity is proved by the T values the hook printed, not annotated:
arm 0 carries T[9]=14.518496 (= Nnn), arm 1 carries T[9]=14.506905 (= Nbb).
BOTH arms receive the SAME jacobian 0.9999622846894718, i.e. NEMO's Nnn
geometry -- the "two arms, different geometry" trap does NOT apply here.

BRACKET (b), OFFLINE, same inputs / two formulas -- the INPUTS ARE EXONERATED:
legoESM's OWN bridged before-level T/S through NEMO's bn2 transcription
reproduces rn2b to ALL PRINTED DIGITS at every interface 4..13, including
-4.066547e-09 at interface 9; the now arm likewise reproduces rn2 exactly.
The SAME inputs through the card's adiabatic parcel-displacement N^2 give
+5.988313e-09 at interface 9.  So the divergence is entirely the N^2 FORMULA.

BRACKET (a), INJECTION, one variable -- the BEFORE-ARM N^2 OWNS THE SPIKE:
replacing ONLY arm 1's N^2 with NEMO's rn2b dump turns interface 9's K from
1e-05 to 1.e+02 and collapses the spike

    (158,42,10)  base 5.710106e-02  ->  inject -2.837890e-06   ratio 4.970e-05
    (158,42, 9)  base -8.447637e-03 ->  inject -2.941119e-06

i.e. 99.995% of the spike.  The injection is surgical, not a reshuffle: the
next five |dT| cells are BIT-IDENTICAL between the two runs -- e.g. (56,2,0)
-3.905812e-02 in both -- so the argmax moving to (56,2,0) exposes a
pre-existing second mode rather than creating one.

DOMAIN-WIDE (same run, all 322294 wet interior interfaces):
  alignment scan  lego bn2(before) vs NEMO rn2b: best shift (0,0,0),
                  median|rel| = 6.661e-16  (3 x fp64 eps -- the index map
                  "lego interface j <-> NEMO 0-based w-level j+1" is proved,
                  not assumed, and the transcription is exact everywhere).
  trigger census  MIN(now,before) <= -1e-12 population, over this probe's
                  window (legoESM interfaces 1..34 == NEMO w-levels 2..35)
                  NEMO                       60845
                  legoESM n2_mode=nemo_bn2   60845   missed 0, spurious 0
                  legoESM n2_mode=adiabatic  60797   missed 48, spurious 0
                  i.e. the exact transcription reproduces NEMO's EVD
                  population BIT-EXACTLY; the shipped adiabatic trigger
                  misses 48 interfaces (0.0149%), always one-directional,
                  and one of those 48 is this spike.
                  The FULL window (interfaces 0..34) and the five-decade
                  threshold sweep are RUN by the sibling probe
                  evd_n2_mode_fix_measure.py -- 70389/70389 at -1e-12 and
                  0 missed / 0 spurious at every threshold.  Do not quote
                  those figures from here; this probe measures one window
                  at one threshold.

EOS DEPTH (2026-08-19 audit).  The card sets eos_depth="geometric"
(dino.py:1045), but that field reaches the PGF and GM/Redi paths only: the EVD
trigger's density comes from k_profiles._enhanced_diffusion_K, which calls
_compute_rho with NO eos_depth, i.e. the "insitu" default.  So this bracket's
original bare call was FAITHFUL to the path it brackets, and feeding the card's
value would have made it diverge.  The effective convention is now asserted
from the live source (the run aborts if that call ever starts passing the
kwarg) and passed EXPLICITLY.  MEASURED sensitivity: switching the offline
bracket to "geometric" changes the before-arm N^2 field by at most 5.8868e-12
s^-2 and moves the spike interface from +5.988313e-09 to +5.988314e-09 -- no
sign change, no trigger change, no headline number moved.  Under the shipped
n2_mode="nemo_bn2" it cannot matter at all: bn2 reads T/S and the geometric
gdept/gdepw ladders and consults no density helper.

THE STATEMENT-LEVEL DIFF
------------------------
  NEMO      src/OCE/TRA/eosbn2.F90:1459-1466 (bn2_t): depth-weighted local
            alpha/beta at each cell's gdept, interpolated to the w-point by
            the geometric zrw weight, differenced linearly in T and S.
            Called for BOTH arms at MY_SRC/stpmlf.F90:200-201 with Nnn
            geometry; consumed by zdfevd.F90:93 (avt) and :119 (avm).
  legoESM   convection/enhanced_diffusion.py:132/143 -- the card selects
            n2_mode="adiabatic", so the trigger takes
            compute_buoyancy_frequency_adiabatic (both parcels displaced to
            the upper cell's pressure, full nonlinear in-situ density
            difference).  The exact transcription is RIGHT THERE at :147/161
            as n2_mode="nemo_bn2" and is unselected.
            Card line: experiments/dino.py:1125.

REFUTES A CARD COMMENT (experiments/dino.py:1113-1123), which justifies
keeping "adiabatic" as "numerically EQUIVALENT to the exact bn2 (corr 1.0000,
maxdiff ~9e-7 s^-2 -> ~0.1% of marginal interfaces flip)".  Correlation cannot
see this: the whole EVD population in this mixed layer sits at |N^2| ~ 1e-10
..1e-12, five orders below the 1e-5 interior that dominates the correlation,
and the trigger is a SIGN test at -1e-12.  Every EVD interface here IS one of
the "0.1% marginal" ones, and at interface 9 the two formulas disagree in
SIGN.

The zdfevd alignment table (11 rows, form / threshold / branch, all MATCH) is
confirmed by this probe and is NOT the defect -- it covered the COMPARISON,
never the N^2 the comparison consumes.

DUAL ADVERSARIAL REVIEW (2026-08-18; codex CLI unavailable on this account, so
two independent subagent reviewers -- an instrument/diff reviewer and a
mechanism/physics reviewer).  BOTH: the finding SURVIVES.

  instrument reviewer -- no index/alignment/vacuous-control defect that could
  flip the sign conclusion.  It independently confirmed from source that both
  arms share one jacobian and one eta inside _enhanced_diffusion_K, that the
  now arm always runs before the before arm (so the call-parity injection
  targets the right one), that avt/avm being dumped on 35 levels is NEMO's own
  convention and not an off-by-one, and that Block B may legitimately use the
  step-entry eta because the DINO surface forcing never touches eta.  THREE
  defects raised, ALL FIXED HERE and both arms re-run with every number
  reproduced identically:
    #1 the hook-fired control was a floor (">= 2"); now an EXACT pair, because
       a third K-profile pass would move the injection onto the wrong arm.
    #2 the cross-run pairing could silently difference a stale scratch file;
       each saved field now carries a probe-mtime + realized-trigger stamp and
       the pairing REFUSES to run across a mismatch.
    #3 grid dims were hardcoded; now read from the run's own ocean.output via
       the sibling probes' _read_dims (prints jpi=56 jpj=203 jpk=36 nn_hls=2,
       i.e. the hardcoded values were right, but a re-dump at another
       resolution would have reshaped silently).

  mechanism reviewer -- the sign disagreement is physically coherent and the
  card comment's reasoning is unsound.  Its arithmetic: with the DINO S-EOS
  beta is depth- and T-independent (b0/rho0 = 7.461e-4 /psu), so the sign of
  alpha*dT - beta*dS flips exactly where dT/dS crosses beta/alpha = 3.593; at
  this interface beta/alpha is 3.55-3.64, i.e. WITHIN 1-2% OF THE KNIFE EDGE.
  (RATIO LABEL CORRECTED 2026-08-19: this line read "beta/alpha crosses
  dS/dT", which inverts the crossing condition -- dS/dT is the reciprocal,
  0.2785.  The NUMBER was always right.  Independent review reproduced the
  column from the raw restart and measured dT/dS = 3.588 against the knife
  edge beta/alpha = 3.5906, agreeing to four digits.)
  A thermohaline-compensated front (warmer AND saltier below) is precisely
  where a percent-level change in alpha flips the static-stability sign.
  It also confirmed that the two paths' DIVISORS differ -- NEMO divides by the
  true geometric e3w, the adiabatic path by the midpoint reconstruction
  0.5*(dz_ref*J)[k] + 0.5*(dz_ref*J)[k+1] -- but that both are strictly
  positive, so the divisor changes MAGNITUDE ONLY and cannot be the sign
  cause.  (That divisor is the same D-e3w debt recorded at 7938a590d; still
  its own change, still not this one.)

  OPEN, labelled PLAUSIBLE (the mechanism reviewer's one real caveat): the
  injection bracket proves the before-arm N^2 owns the spike, but it does not
  separate "the adiabatic formula's single-reference-pressure design
  structurally drops the lower parcel's thermobaric term" from "the p_cell
  reconstruction feeding only the adiabatic path is itself off".  Both live on
  the SAME side of the DIFF, so ownership is unaffected; only the sub-mechanism
  is open.  Discriminating measurement, if it is ever wanted: re-evaluate the
  adiabatic path with each parcel at its OWN cell depth and see whether that
  alone recovers NEMO's sign.
  PARTIAL ANSWER ALREADY IN HAND (independent review 2026-08-19, reproducing
  this column from the raw restart to 0.15%): evaluating alpha at the LOWER
  cell's depth gives -1.576e-08 -- NEMO's SIGN but 3.9x its magnitude.  So
  "each parcel at its own depth" will NOT by itself recover NEMO's value; only
  bn2's zrw weight, which lands alpha at gdepw EXACTLY, does.  Recorded so
  nobody spends a run rediscovering it.

KNOWN PRE-EXISTING FAILURE, blocks (A) and (C) only (recorded 2026-08-19; NOT
introduced by the eos_depth work, verified by re-running with that change
stashed -- identical exit 1 and identical message).  Since commit 5a9be32ba
flipped the card default to ``convection_n2_mode="nemo_bn2"``, the live trigger
no longer calls ``compute_buoyancy_frequency_adiabatic`` at all, so block (A)'s
N^2 hook fires ZERO times and its call-parity arm selection aborts with::

    *** hook count is not the expected one now/before pair: flag=2 n2=0

Blocks (B) and the domain-wide census/alignment scan run to completion and are
unaffected -- every number in the RESULT block above comes from those and is
reproduced on the current build.  Block (A)'s recorded numbers were taken when
the card still selected "adiabatic"; to re-run them, pass that arm explicitly
rather than relying on the card default.  Fixing (A)'s arm selection for a
bn2 card is a separate change and is NOT attempted here.

Run:
CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  python scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/evd_before_arm_n2.py [--inject-rn2b]
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCRATCH = os.environ.get("TMPDIR", "/tmp")

DT = 2700.0
J, I = 158, 42
IFACE = 9                      # legoESM interface 9 == NEMO 0-based w-level 10
KLO, KHI = 4, 13               # interface window printed
SEQDUMP = os.environ.get(
    "DINO_NEMO_RUN_SEQDUMP",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_Y20_1R")
EVD_THR = -1.0e-12             # zdfevd.F90:93 literal
# Grid dims are READ from the run's own ocean.output (review defect #3), never
# hardcoded -- a re-dump at a different resolution whose size still divides
# evenly would otherwise reshape silently and wrongly.  Same helper the
# sibling bn2/EVD probes use.
_DIMS: dict = {}


def dims():
    if not _DIMS:
        from bn2_alpha_compare import _read_dims       # probe-only sibling
        jpi, jpj, jpk, hls = _read_dims(SEQDUMP)
        _DIMS.update(jpi=jpi, jpj=jpj, jpk=jpk, hls=hls,
                     ni=jpi - 2 * hls, nj=jpj - 2 * hls)
        print(f"  [dims] jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls} -> "
              f"interior ({_DIMS['nj']},{_DIMS['ni']})")
    return _DIMS


def _load_interior(path, ni=None, nj=None):
    """(nlev,nj,ni) interior stream dump -> (j,i,nlev)."""
    d = dims()
    ni = d["ni"] if ni is None else ni
    nj = d["nj"] if nj is None else nj
    a = np.fromfile(path, dtype="<f8")
    nlev = a.size // (ni * nj)
    if nlev * ni * nj != a.size:
        raise SystemExit(f"{path}: size {a.size} not a multiple of {ni*nj}")
    return np.moveaxis(a.reshape(nlev, nj, ni), 0, -1)


def _load_full(path, jpi=None, jpj=None, hls=None):
    """(nlev,jpj,jpi) haloed stream dump -> (j,i,nlev) interior."""
    d = dims()
    jpi = d["jpi"] if jpi is None else jpi
    jpj = d["jpj"] if jpj is None else jpj
    hls = d["hls"] if hls is None else hls
    a = np.fromfile(path, dtype="<f8")
    nlev = a.size // (jpi * jpj)
    if nlev * jpi * jpj != a.size:
        raise SystemExit(f"{path}: size {a.size} not a multiple of {jpi*jpj}")
    return np.moveaxis(a.reshape(nlev, jpj, jpi)[:, hls:-hls, hls:-hls], 0, -1)


def nemo_column():
    """NEMO's own rn2 / rn2b / post-EVD avt on the spike column."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    for d in ("tke_dump_rn2.bin", "tke_dump_rn2b.bin"):
        print(f"  [registry] {d:20s} time level = {time_level_for_dump(d)!r}")
    rn2 = _load_interior(f"{SEQDUMP}/tke_dump_rn2.bin")
    rn2b = _load_interior(f"{SEQDUMP}/tke_dump_rn2b.bin")
    avt = _load_full(f"{SEQDUMP}/dump_avt.bin")
    for nm, a in (("rn2", rn2), ("rn2b", rn2b), ("avt", avt)):
        if not np.isfinite(a).all():
            raise SystemExit(f"*** {nm} non-finite -- FATAL")
    return rn2, rn2b, avt


def main() -> int:
    inject = "--inject-rn2b" in sys.argv
    import jax
    import multistep_replay as mr
    import legoesm.ocean.physics.convection.enhanced_diffusion as ED
    from legoesm.core.precision import get_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
        dino_step_surface_forcing,
    )

    if not mr.have_step1_artifacts():
        print("SKIP: oracle artifacts not present")
        return 0
    mr.provenance("evd_before_arm_n2")
    print(f"PRECISION control dtype = {get_policy().control}")
    print(f"LEGOESM_NEMO_E3T = {os.environ.get('LEGOESM_NEMO_E3T')!r}")
    print(f"MODE = {'INJECT NEMO rn2b into the BEFORE arm' if inject else 'baseline'}")

    print("\n--- NEMO's own arrays on the spike column -------------------")
    rn2, rn2b, avt_nemo = nemo_column()
    print(f"  shapes rn2={rn2.shape} rn2b={rn2b.shape} avt={avt_nemo.shape}")
    print("   lego_if | NEMO k0 |      rn2      |     rn2b      |  avt(post-EVD)")
    for k in range(KLO, KHI + 1):
        k0 = k + 1
        print(f"    {k:6d} | {k0:7d} | {rn2[J,I,k0]: .6e} | {rn2b[J,I,k0]: .6e} "
              f"| {avt_nemo[J,I,k0]: .6e}")

    jax.clear_caches()
    g, br, cfg, st0 = mr.build_replay_ic()
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    ed = mc.physics.convection.enhanced_diffusion
    print("\n--- realized card (instantiated, Rule 10) -------------------")
    print(f"  convection.scheme         = {mc.physics.convection.scheme!r}")
    print(f"  n2_mode                   = {ed.n2_mode!r}")
    print(f"  n2_threshold              = {ed.n2_threshold!r}")
    print(f"  two_level_trigger         = {ed.two_level_trigger}")
    print(f"  evd_n2_time_level         = {ed.evd_n2_time_level!r}")
    print(f"  smooth_transition         = {ed.smooth_transition}")
    print(f"  K_conv / K_bg             = {ed.K_conv} / {ed.K_bg}")
    print(f"  vmix.scheme               = {mc.physics.vertical_mixing.scheme!r}")
    print(f"  vmix_background_mode      = "
          f"{getattr(mc.physics.vertical_mixing,'vmix_background_mode',None)!r}")
    print(f"  tke.n2_mode               = {mc.physics.vertical_mixing.tke.n2_mode!r}")
    print(f"  tke.n2_before_advection   = "
          f"{mc.physics.vertical_mixing.tke.n2_before_advection}")
    print(f"  tke.tke_n2_time_level     = "
          f"{mc.physics.vertical_mixing.tke.tke_n2_time_level!r}")
    print(f"  surface_tendency_placement= "
          f"{getattr(cfg,'surface_tendency_placement',None)!r}")

    # ================= (B) OFFLINE bracket: same inputs, two formulas ========
    print("\n--- (B) OFFLINE: legoESM's OWN before-level T/S, two N2 formulas ---")
    from legoesm.ocean.eos import (
        compute_buoyancy_frequency_nemo_bn2, nemo_bn2_live_ladders,
        compute_hydrostatic_pressure, maybe_partial_h_actual, make_eos_fn,
        compute_buoyancy_frequency_adiabatic,
    )
    from legoesm.ocean.vertical import compute_ocean_jacobian
    cc = mc.physics.constants
    eos_fn = make_eos_fn(eos=mc.eos, eos_linear=mc.eos_linear)
    T_nn = np.asarray(st0.T.data); S_nn = np.asarray(st0.S.data)
    T_bb = np.asarray(st0.T_before.data); S_bb = np.asarray(st0.S_before.data)
    eta_nn = st0.eta.data
    print(f"  dtypes T_nn={T_nn.dtype} T_bb={T_bb.dtype} "
          f"eta={np.asarray(eta_nn).dtype} dz_ref={np.asarray(br.z_coord.dz_ref).dtype}")
    print(f"  column T_nn[9,10] = {T_nn[J,I,9]:.6f} {T_nn[J,I,10]:.6f}   "
          f"S_nn = {S_nn[J,I,9]:.6f} {S_nn[J,I,10]:.6f}")
    print(f"  column T_bb[9,10] = {T_bb[J,I,9]:.6f} {T_bb[J,I,10]:.6f}   "
          f"S_bb = {S_bb[J,I,9]:.6f} {S_bb[J,I,10]:.6f}")
    tdep, wdep = nemo_bn2_live_ladders(br.z_coord, eta_nn, st0.H_bathy.data)
    n2_bn2_bb = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        T_bb, S_bb, tdep, wdep, g=cc.g))
    n2_bn2_nn = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        T_nn, S_nn, tdep, wdep, g=cc.g))
    Jz = compute_ocean_jacobian(eta_nn, st0.H_bathy.data, br.z_coord)

    # Re-use the model's own helper rather than re-deriving the rho/p chain.
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        _compute_rho, _enhanced_diffusion_K,            # probe-only private
    )

    # ---- EOS DEPTH: realized, PROVED from source, never assumed (Rule 10) --
    # The card sets eos_depth (dino.py:1045) but that field reaches the PGF and
    # GM/Redi paths ONLY.  The EVD trigger's density comes from
    # _enhanced_diffusion_K, which calls _compute_rho(...) with NO eos_depth
    # kwarg -- the "insitu" default -- so passing the card's value here would
    # make this offline bracket DIVERGE from the path it is bracketing.  The
    # effective value is asserted from the live source and then passed
    # EXPLICITLY, so the convention is visible instead of inherited silently.
    # TWO guards (the first guard is never the only guard): the CALLER could
    # start passing the kwarg, or the CALLEE's default could change under it.
    import inspect
    from legoesm.ocean.eos import compute_ocean_rho as _cor
    if "eos_depth" in inspect.getsource(_enhanced_diffusion_K):
        raise SystemExit(
            "*** _enhanced_diffusion_K now mentions eos_depth -- this offline "
            "bracket is stale; re-derive the depth convention from the source")
    _EVD_EOS_DEPTH = inspect.signature(_cor).parameters["eos_depth"].default
    if _EVD_EOS_DEPTH not in ("insitu", "geometric"):
        raise SystemExit(f"*** unexpected compute_ocean_rho eos_depth default "
                         f"{_EVD_EOS_DEPTH!r}")
    print(f"  [eos depth] card cfg.eos_depth="
          f"{getattr(cfg, 'eos_depth', '<absent>')!r}  model mc.eos_depth="
          f"{getattr(mc, 'eos_depth', '<absent>')!r}  BUT the EVD trigger's "
          f"density path passes none -> effective {_EVD_EOS_DEPTH!r} "
          f"(READ from compute_ocean_rho's signature default; the caller "
          f"is separately asserted not to override it)")

    def adiabatic_n2(Ta, Sa, eos_depth=_EVD_EOS_DEPTH):
        """Parcel-displacement N^2.  ``eos_depth`` defaults to the convention
        the MODEL's EVD path actually uses; any other value is a labelled
        SENSITIVITY arm and is NOT what the model runs."""
        efn = eos_fn
        if eos_depth == "geometric":
            # make_eos_fn's rho0 feeds ONLY the "nemo_eos80" branch; this card
            # runs "nemo_seos", whose depth recovery uses the rho0 baked into
            # NemoSEOSConfig.  So p = rho0*g*gdept cancels because both are
            # 1026.0, NOT because the kwarg was threaded -- asserted below.
            efn = make_eos_fn(eos=mc.eos, eos_linear=mc.eos_linear,
                              rho0=cc.rho_0)
        stx = st0._replace(T=st0.T.replace(data=Ta), S=st0.S.replace(data=Sa))
        rho = _compute_rho(stx, br.z_coord, Jz, eos_fn=efn,
                           eos_depth=eos_depth,
                           rho0=(cc.rho_0 if eos_depth == "geometric" else None))
        p_cell = compute_hydrostatic_pressure(
            rho, eta_nn, br.z_coord.dz_ref, Jz, cc.rho_0,
            h_actual=maybe_partial_h_actual(stx, br.z_coord))
        return np.asarray(compute_buoyancy_frequency_adiabatic(
            Ta, Sa, p_cell, br.z_coord.dz_ref,
            np.where(np.asarray(Jz) <= 0.0, 1.0, np.asarray(Jz)),
            eos_fn=efn, rho_ref=cc.rho_0, g=cc.g))

    n2_ad_bb = adiabatic_n2(T_bb, S_bb)
    n2_ad_nn = adiabatic_n2(T_nn, S_nn)
    # SENSITIVITY arm, labelled: what the card's own eos_depth WOULD give at
    # the spike interface if the EVD path honoured it.  Non-vacuity checked --
    # an arm that changes nothing is not evidence that the choice is harmless.
    from legoesm.ocean.eos import NemoSEOSConfig as _NSC
    if mc.eos == "nemo_seos":
        _seos_rho0 = float(getattr(mc.eos_nemo_seos, "rho0", None)
                           if getattr(mc, "eos_nemo_seos", None) is not None
                           else _NSC().rho0)
        if abs(_seos_rho0 - float(cc.rho_0)) > 1e-9:
            raise SystemExit(
                f"*** eos_depth='geometric' arm is INVALID: NemoSEOSConfig.rho0"
                f"={_seos_rho0} != constants rho_0={cc.rho_0}, so the "
                f"p=rho0*g*gdept factor does NOT cancel and this arm would "
                f"measure a depth stretch, not the depth convention")
    _n2_geo_bb = adiabatic_n2(T_bb, S_bb, eos_depth="geometric")
    _fin = np.isfinite(n2_ad_bb) & np.isfinite(_n2_geo_bb)
    _dm = float(np.abs(n2_ad_bb - _n2_geo_bb)[_fin].max())
    _sc = float(np.abs(n2_ad_bb)[_fin].max())
    # RELATIVE bar: a 1-ULP move would pass "> 0" while proving nothing.
    if not _dm > 1e-12 * _sc:
        raise SystemExit(
            f"*** the eos_depth='geometric' sensitivity arm moved the field by "
            f"only {_dm:.3e} (scale {_sc:.3e}) -- at or below roundoff, so it "
            f"did not meaningfully take and its agreement is vacuous")
    print(f"  [eos depth] SENSITIVITY adiabatic insitu vs geometric: field DIFF "
          f"max={_dm:.4e} s^-2; at the spike interface "
          f"({J},{I},{IFACE}) {n2_ad_bb[J,I,IFACE]: .6e} -> "
          f"{_n2_geo_bb[J,I,IFACE]: .6e}  "
          f"(fires={bool(_n2_geo_bb[J,I,IFACE] <= EVD_THR)}, "
          f"insitu fires={bool(n2_ad_bb[J,I,IFACE] <= EVD_THR)})")
    print("\n   lego_if | NEMO rn2b     | lego bn2(bb)  | lego adia(bb) ||"
          " NEMO rn2      | lego bn2(nn)  | lego adia(nn)")
    for k in range(KLO, KHI + 1):
        k0 = k + 1
        print(f"    {k:6d} | {rn2b[J,I,k0]: .6e} | {n2_bn2_bb[J,I,k]: .6e} | "
              f"{n2_ad_bb[J,I,k]: .6e} || {rn2[J,I,k0]: .6e} | "
              f"{n2_bn2_nn[J,I,k]: .6e} | {n2_ad_nn[J,I,k]: .6e}")
    print(f"\n  FIRES (N2 <= {EVD_THR:g}) at interface {IFACE}:  "
          f"NEMO rn2b={rn2b[J,I,IFACE+1] <= EVD_THR}  "
          f"lego bn2(bb)={bool(n2_bn2_bb[J,I,IFACE] <= EVD_THR)}  "
          f"lego adia(bb)={bool(n2_ad_bb[J,I,IFACE] <= EVD_THR)}  ||  "
          f"NEMO rn2={rn2[J,I,IFACE+1] <= EVD_THR}  "
          f"lego bn2(nn)={bool(n2_bn2_nn[J,I,IFACE] <= EVD_THR)}  "
          f"lego adia(nn)={bool(n2_ad_nn[J,I,IFACE] <= EVD_THR)}")

    # ---- mandatory alignment scan + domain-wide flip census (skill Rule:
    # a wrong index/halo offset can fake a residual; and the size of the
    # formula DIFF is a measurement, not an impression).
    tm = np.asarray(g.tmask) > 0.5
    wif = tm[..., :-1] & tm[..., 1:]                    # interior wet interface
    lv = slice(1, 35)                                    # lego iface 1..34
    Wm = wif[..., lv]
    nemo_b = rn2b[..., 2:36]
    nemo_n = rn2[..., 2:36]
    best = None
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            for dk in (-1, 0, 1):
                L = np.roll(n2_bn2_bb[..., lv], (dj, di, dk), axis=(0, 1, 2))
                m = Wm & (np.abs(nemo_b) > 0)
                e = float(np.median(np.abs(L[m] / nemo_b[m] - 1.0)))
                if best is None or e < best[0]:
                    best = (e, dj, di, dk)
    print(f"\n  [align scan] lego bn2(bb) vs NEMO rn2b: best shift="
          f"{best[1:]} median|rel|={best[0]:.3e}   (expect (0,0,0) and ~0)")
    m = Wm
    fire_nemo = (np.minimum(nemo_n, nemo_b) <= EVD_THR) & m
    fire_bn2 = (np.minimum(n2_bn2_nn[..., lv], n2_bn2_bb[..., lv]) <= EVD_THR) & m
    fire_ad = (np.minimum(n2_ad_nn[..., lv], n2_ad_bb[..., lv]) <= EVD_THR) & m
    nw = int(m.sum())
    for lab, f in (("lego bn2  MIN(now,before)", fire_bn2),
                   ("lego adia MIN(now,before)", fire_ad)):
        miss = int((fire_nemo & ~f).sum()); extra = int((f & ~fire_nemo).sum())
        print(f"  [census] {lab}: n={int(f.sum()):6d} vs NEMO "
              f"{int(fire_nemo.sum()):6d}  missed={miss:5d}  spurious={extra:5d}"
              f"  XOR={miss+extra:5d} of {nw} wet interfaces "
              f"({100*(miss+extra)/nw:.4f}%)")

    # ================= (A) LIVE trigger, production step =====================
    print("\n--- (A) LIVE trigger inside the production step ------------------")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    _wind = bool(getattr(cfg, "wind_through_step", False))
    sf = dino_step_surface_forcing(forcing) if _wind else None
    print(f"  FORCING wind_through_step={_wind}"
          + (f"  tau_x range=[{float(np.asarray(sf.tau_x).min()):.4f},"
             f"{float(np.asarray(sf.tau_x).max()):.4f}]" if _wind else ""))

    n_flag = {"n": 0}
    n_n2 = {"n": 0}
    real_flag = ED.convective_K_A_flag
    real_ad = ED.compute_buoyancy_frequency_adiabatic
    rn2b_inject = None
    if inject:
        import jax.numpy as jnp
        # NEMO rn2b, aligned to legoESM interfaces: lego iface j <-> NEMO k0 j+1
        rn2b_inject = jnp.asarray(rn2b[..., 1:1 + (T_nn.shape[-1] - 1)])

    def n2_spy(T, S, p_cell, dz, jac, **kw):
        out = real_ad(T, S, p_cell, dz, jac, **kw)
        idx = n_n2["n"]
        n_n2["n"] += 1
        jax.debug.print(
            "   [N2 call {i}] T[9],T[10]={a} {b}   N2[{lo}:{hi}]={n}",
            i=idx, a=T[J, I, 9], b=T[J, I, 10], lo=KLO, hi=KHI + 1,
            n=out[J, I, KLO:KHI + 1])
        if inject and idx % 2 == 1:      # arm 1 == the BEFORE arm
            jax.debug.print("   [INJECT] before-arm N2 replaced by NEMO rn2b")
            return rn2b_inject
        return out

    def flag_spy(rho, dz_ref, jacobian, c, **kw):
        out = real_flag(rho, dz_ref, jacobian, c, **kw)
        idx = n_flag["n"]
        n_flag["n"] += 1
        jax.debug.print(
            "   [arm {i}] jac[j,i]={j}  K[{lo}:{hi}]={k}",
            i=idx, j=jacobian[J, I], lo=KLO, hi=KHI + 1,
            k=out[0][J, I, KLO:KHI + 1])
        return out

    ED.convective_K_A_flag = flag_spy
    ED.compute_buoyancy_frequency_adiabatic = n2_spy
    try:
        if getattr(cfg, "surface_tendency_placement", None) == "leapfrog_rhs":
            st, rate = apply_dino_lat_lon_surface_forcing(
                st0, forcing, br.z_coord, cfg, DT, t_seconds=DT,
                return_rate=True)
        else:
            st = apply_dino_lat_lon_surface_forcing(
                st0, forcing, br.z_coord, cfg, DT, t_seconds=DT)
            rate = None
        st1 = model.step(st, DT, surface_forcing=sf, external_tracer_rate=rate)
        st1.T.data.block_until_ready()
    finally:
        ED.convective_K_A_flag = real_flag
        ED.compute_buoyancy_frequency_adiabatic = real_ad

    # EXACT count, not a floor (review defect #1): the injection arm selects
    # the BEFORE arm by call parity, so a third K-profile pass would silently
    # move the injection onto the wrong arm while a ">= 2" guard still passed.
    if n_flag["n"] != 2 or n_n2["n"] != 2:
        raise SystemExit(f"*** hook count is not the expected one now/before "
                         f"pair: flag={n_flag['n']} n2={n_n2['n']} -- the "
                         f"call-parity arm selection is no longer valid")
    print(f"  [control] convective_K_A_flag fired {n_flag['n']}x, "
          f"adiabatic-N2 {n_n2['n']}x  PASS")

    ns = mr.nemo_now_state_at(mr.IC_STEP + 1)
    tmask3 = np.asarray(g.tmask) > 0.5
    d = np.asarray(st1.T.data) - ns.T
    _abs = np.abs(d)
    a = np.where(tmask3, _abs, -np.inf)
    idx = np.unravel_index(int(np.argmax(a)), a.shape)
    print(f"\nCONTINUITY CONTROL: dT max={float(_abs[idx]):.6e} "
          f"@{tuple(int(v) for v in idx)}  (baseline expects 5.710106e-02 "
          f"@ (158,42,10))")
    print(f"  SPIKE at (158,42,10) = {float(d[J,I,10]):.6e}   "
          f"cell 9 = {float(d[J,I,9]):.6e}")
    # Rule "refute with STATES": an injection that removes one mode can UNCOVER
    # a second one at a different cell.  Rank the top cells in BOTH arms and
    # save the field so the two runs can be differenced directly, instead of
    # reading a moved argmax as a new defect.
    flat = np.where(tmask3, _abs, -np.inf).ravel()
    top = np.argsort(flat)[::-1][:6]
    print("  top |dT| cells (wet):")
    for t in top:
        c = np.unravel_index(int(t), _abs.shape)
        print(f"    {tuple(int(v) for v in c)}  {float(_abs[c]):.6e}")
    # A cross-run pairing is only valid between two runs of the SAME probe on
    # the SAME card (review defect #2): stamp each saved field with the probe
    # file's own mtime + the realized trigger config, and REFUSE to pair
    # across a mismatch rather than silently differencing a stale file.
    stamp = (f"{os.path.getmtime(os.path.abspath(__file__)):.0f}|"
             f"{ed.n2_mode}|{ed.n2_threshold}|{ed.evd_n2_time_level}|"
             f"{ed.two_level_trigger}|{ed.K_conv}|"
             f"{getattr(cfg,'surface_tendency_placement',None)}")
    out = os.path.join(_SCRATCH,
                       f"_probe_evd_dT_{'inject' if inject else 'base'}.npz")
    np.savez(out, dT=d, stamp=np.array(stamp))
    print(f"  dT field saved -> {out}  stamp={stamp}")
    other = os.path.join(_SCRATCH,
                         f"_probe_evd_dT_{'base' if inject else 'inject'}.npz")
    if os.path.isfile(other):
        _z = np.load(other)
        if str(_z["stamp"]) != stamp:
            print(f"  PAIRING SKIPPED: {other} was written by a different "
                  f"probe/card ({str(_z['stamp'])!r}) -- re-run both arms.")
            return 0
        d0 = _z["dT"]
        base, inj = (d0, d) if inject else (d, d0)
        print(f"  PAIRED: spike (158,42,10) base={float(base[J,I,10]):.6e} "
              f"-> inject={float(inj[J,I,10]):.6e}  "
              f"collapse ratio={abs(float(inj[J,I,10]))/abs(float(base[J,I,10])):.3e}")
        for c in ((56, 2, 0),):
            print(f"  PAIRED: cell {c} base={float(base[c]):.6e} "
                  f"-> inject={float(inj[c]):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
