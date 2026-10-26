"""#1226 work-order #3 term-by-term walk: dyn_adv ZAD + dyn_ldf momentum rows.

TASK: two DEBT rows (``fidelity_bar_gate.py`` ledger) --  ``dyn_adv ZAD``
(err_norm u=4.00e-2/v=5.07e-2) and ``dyn_ldf u`` (err_norm u=4.49e-2/v=2.76e-2)
-- have a CONFIRMED downstream payoff established by
``zu_frc_momentum_row_reconstruction.py`` (commit 784e57406): summing the
SIGNED per-term errors of dyn_ldf + dyn_adv ZAD + dyn_vor EEN and
depth-weighting them exactly as NEMO forms ``zv_frc``
(``SUM(e3u_0*err*umask)*r1_hu_0``, ``dynspg_ts.F90:335-337``) reproduces
``zv_frc``'s own measured error at corr=0.985/ratio=0.977. The SAME
reconstruction for ``zu_frc`` gives corr=0.044/ratio=0.045 -- REFUTED for u.
So: fixing these two rows has a confirmed v-side payoff; NO u-side claim.

This script walks BOTH rows term-by-term against NEMO's own dumped
tendencies at the SAME state (RUN_GDB, kt=57601, the campaign's standing
probe point), builds an ordered formula + alignment table for each, and runs
A/B factor tests to identify exactly which factor (if any) explains the
measured mismatch. ZAD FIRST (larger of the two).

RESULTS SUMMARY (this session, CONFIRMED unless noted):
  ROW 2 (dyn_ldf): ROOT CAUSE FOUND. NOT an operator/numerics defect --
    ``nemo_ldf_lap_viscosity_cgrid`` and its e3-agnostic div/curl are
    CORRECT. The DEBT err_norm (4.49e-2 u / 2.76e-2 v) is a TIME-LEVEL
    mismatch in the #1226 MEASUREMENT HARNESS: every existing probe feeds
    ``tendencies_with_diagnostics`` the NOW-level bridged state, but
    ``dynldf_lev_rot_scheme.h90`` reads velocity at Kbb ("before"). Feeding
    ``state.u_before``/``v_before`` (mirroring legoESM's OWN production
    Nbb dissipative pass, ``ocean_model_latlon_cgrid.py:6947-6968``) drops
    err_norm to 4.58e-5 (u) / 4.51e-5 (v) -- roundoff. e3-weighting
    (the OTHER candidate, an explicitly documented deviation in
    ``nemo_ldf_lap_viscosity_cgrid``'s docstring) is REFUTED: e3t_0==e3u_0==
    e3v_0==e3f_0 everywhere on this full-step grid, so restoring it is a
    provable no-op (verified both via a pure-NEMO-space transcription A/B
    and directly from mesh_mask.nc).
  ROW 1 (dyn_adv ZAD): CAUSE NOT IDENTIFIED, one candidate REFUTED, one
    remaining ranked candidate. Time-level REFUTED by the before/now test
    (before-fed is WORSE: u 3.999e-2->4.218e-2, v 5.075e-2->5.910e-2,
    confirming legoESM already correctly uses Kmm/"now", matching
    dynzad.F90:97). The per-level profile is decisive on ONE fact: err is
    near-zero (1e-14..1e-13) through level 28, then jumps 1000x+ at levels
    29-33 (near the deepest active levels, jpk=36) -- a genuine
    depth-CONCENTRATED signature, not a uniform mismatch. FIRST CANDIDATE
    (bottom/partial-cell boundary handling, e.g. NEMO's fixed
    ``DO jk=1,jpk-2`` + separate ``jk=jpkm1`` bottom-only term,
    dynzad.F90:86,113-118, vs legoESM's ``face_active`` masking) TESTED
    AND REFUTED: correlating each u-face column's own bottom_level against
    where that column's |err| peaks gives corr=-0.10 (near zero), and the
    error's argmax sits AT the column's own bottom level in only 13.4% of
    wet columns (mean offset ~21 levels ABOVE the bottom, std ~11) -- if
    this were a per-column seafloor-boundary artifact the argmax would
    track bottom_level tightly; it does not. REMAINING CANDIDATE (NOT YET
    TESTED): the levels 29-33 concentration instead looks like a
    LATITUDE-BAND or WATER-MASS-DEPTH effect common across most columns
    regardless of their individual bottom -- consistent with something
    tied to the vertical velocity ``ww``'s own structure at that depth
    band (e.g. the eddy-induced/GM contribution to ``ww``, or a specific
    density-driven overturning cell) rather than the ZAD operator itself;
    would need ``ww`` decomposed by contributing physics to test, out of
    this task's scope.

=====================================================================
ROW 1: dyn_adv ZAD -- vertical momentum advection (dynzad.F90)
=====================================================================

DINO active scheme (quoted, not assumed):
  cfgs/DINO/EXP00/namelist_cfg:321  ln_dynadv_vec = .true.  (vector form)
  -> dynadv.F90:86-97 (np_VEC_c2): CALL dyn_keg(...) ; CALL dyn_zad(...)
  cfgs/DINO/EXP00/namelist_cfg:322  nn_dynkeg = 1  (Hollingsworth KEG --
     irrelevant to ZAD itself, only matters for isolating ZAD from the
     Krhs-chain dump; NEMO already gives us a ZAD-ONLY dump, see below).

NEMO ordered formula (``dynzad.F90:81-119``, stock/unmodified -- verified by
diffing cfgs/DINO/MY_SRC/ (no dynzad.F90 override) against
src/OCE/DYN/dynzad.F90), Fortran 1-indexed ``jk``, ``Kmm`` = "now" time level
for BOTH the advected velocity and the vertical velocity, ``ln_vortex_force
= .FALSE.`` (DINO has no Stokes drift -- no ``sbcwave`` card):

    zWdzU(1) = 0                                          ! dynzad.F90:83 (surface interface)
    DO jk = 1, jpk-2                                        ! dynzad.F90:86
       zWf   = e1e2t(i  ,j) * ww(i  ,j,jk+1)                ! :93  (Kmm ww, "now")
       zWfi  = e1e2t(i+1,j) * ww(i+1,j,jk+1)                ! :94
       zzWfu = zWfi + zWf                                   ! :97  (2x area-weighted-interp w at u-face, interface jk+1)
       zzWdzU = zzWfu * ( uu(i,j,jk,Kmm) - uu(i,j,jk+1,Kmm) )   ! :100  (Kmm velocity)
       puu(i,j,jk,Krhs) -= 0.25 * r1_e1e2u(i,j) / e3u(i,j,jk,Kmm) * ( zWdzU(i,j) + zzWdzU )  ! :104-105
       zWdzU(i,j) = zzWdzU                                  ! :109 (carried to top term of jk+1)
    END DO
    jk = jpkm1                                              ! :113 (bottom cell, only the TOP-interface term)
    puu(i,j,jk,Krhs) -= 0.25 * r1_e1e2u(i,j) / e3u(i,j,jk,Kmm) * zWdzU(i,j)                  ! :115-116

THICKNESS: ``e3u(i,j,jk,Kmm)`` -- explicit ``Kmm`` argument to the ``e3u``
QCO macro (``domzgr_substitute.h90:129``: ``e3u(i,j,k,t) = E3u_0(i,j,k) *
Tmsk(r3u,umask,i,j,k,t)``) -- LIVE thickness at the "now" time level, NOT the
static reference ``E3u_0``/``e3u_1d``. This is the divisor of the WHOLE
tendency (not the div/curl-internal thickness dyn_ldf uses -- see ROW 2), so
ZAD's ladder question is: is legoESM's ``h_u`` argument to
``nemo_advective_vertical_momentum_advection`` the LIVE (ssh-stretched) or
STATIC partial-cell thickness at THIS state?

TIME LEVEL: both ``ww`` and ``uu``/``vv`` read at ``Kmm`` ("now") --
confirmed by the call site ``dynadv.F90:97: CALL dyn_zad(kt, Kmm, puu, pvv,
Krhs)`` (only ONE velocity-time-level argument is passed at all -- Kbb is not
even in dyn_zad's argument list, ``dynzad.F90:56``). Registered
"now" in ``time_levels.py`` (see ``ocean/fidelity/time_levels.py``
"zad_dump_du.bin"/"now").

legoESM transcription (``legoesm/ocean/vertical.py:1149-``
``nemo_advective_vertical_momentum_advection``, called from
``ocean_pe_latlon_cgrid.py:2523-2548`` under
``vertical_momentum_scheme="nemo_advective"``) -- ALREADY the advective form
(not flux-divergence), area-weighted w-interpolation, u-face's OWN metric
(not area_T interpolated). See ALIGNMENT TABLE below for the term-by-term
match.

=====================================================================
ROW 2: dyn_ldf -- lateral Laplacian viscosity (dynldf_lev.F90 + .h90)
=====================================================================

DINO active scheme (quoted):
  namelist_cfg:365  ln_dynldf_lap = .true.
  namelist_cfg:366  ln_dynldf_lev = .true.
  (ln_dynldf_hor/ln_dynldf_iso left at namelist_ref default .false.)
  -> ldfdyn.F90:186-192 (z-star/z-co branch, l_zco.OR.l_zps): since
     ln_dynldf_lap=.true. AND ln_dynldf_lev=.true., nldf_dyn = np_lap (NOT
     np_lap_i -- CORRECTS a stale comment in zu_frc_term_walk.py's dump
     registration, which cited "np_lap_i (rotated laplacian)"/"dyn_ldf_iso";
     the namelist confirms plain iso-level np_lap. This does not change any
     already-measured number -- the Fortran ran the real np_lap path
     regardless of the comment -- but the FORMULA quoted below is
     dynldf_lev_lap's, not dyn_ldf_iso's, and that is what actually produced
     ldf_dump_du.bin/dv.bin.)
  namelist_cfg (default) nn_dynldf_typ = 0 (np_typ_rot, div-rot operator --
     ldfdyn.F90:1108 default, not overridden in DINO's namelist_cfg)
  -> cfgs/DINO/MY_SRC/dynldf.F90:79-83 (CASE(np_lap): CALL dynldf_lev_lap)
  -> dynldf_lev.F90:84-101 (CASE(np_typ_rot)): dynldf_lev_rot_scheme.h90

NEMO ordered formula (``dynldf_lev_rot_scheme.h90:21-53``, ``lap`` branch,
``pu_in``/``pv_in`` = ``pu(...,Kbb)``/``pv(...,Kbb)`` since
``dynldf_lev_lap`` calls the scheme with ``pu_in(i,j,k,t)=pu(i,j,k,t)`` and
``dyn_ldf`` invokes it as ``CALL dynldf_lev_lap(kt, Kbb, Kmm, puu, pvv,
Krhs)`` -- ``dynldf.F90:83``):

    ! curl at F-point (vorticity), coeff EMBEDDED, e3f-weighted:
    zcur(i-1,j-1) = ahmf(i-1,j-1,jk) * e3f(i-1,j-1,jk) * r1_e1e2f(i-1,j-1)   &  ! :23  (ahmf already *fmask)
        * ( (e2v(i,j-1)*pv(i,j-1,jk,Kbb) - e2v(i-1,j-1)*pv(i-1,j-1,jk,Kbb))  &
          - (e1u(i-1,j)*pu(i-1,j,jk,Kbb) - e1u(i-1,j-1)*pu(i-1,j-1,jk,Kbb)) )   ! :24-25

    ! div at T-point, coeff EMBEDDED, e3u/e3v/e3t-weighted:
    zdiv(i,j) = ahmt(i,j,jk) * r1_e1e2t(i,j) / e3t(i,j,jk,Kbb)               &  ! :27  (ahmt already *tmask)
        * ( (e2u(i,j)*e3u(i,j,jk,Kbb)*pu(i,j,jk,Kbb) - e2u(i-1,j)*e3u(i-1,j,jk,Kbb)*pu(i-1,j,jk,Kbb))  &
          + (e1v(i,j)*e3v(i,j,jk,Kbb)*pv(i,j,jk,Kbb) - e1v(i,j-1)*e3v(i,j-1,jk,Kbb)*pv(i,j-1,jk,Kbb)) )  ! :28-29

    ! grad(div) - curl(curl), DIVIDED by e3 AT Kmm (not Kbb!):
    pu(i,j,jk,Krhs) += umask(i,j,jk) * (                                     &  ! :34,40
          - (zcur(i,j) - zcur(i,j-1)) * r1_e2u(i,j) / e3u(i,j,jk,Kmm)        &  ! :41
          + (zdiv(i+1,j) - zdiv(i,j)) * r1_e1u(i,j) )                          ! :42
    pv(i,j,jk,Krhs) += vmask(i,j,jk) * (                                     &  ! :44,50
            (zcur(i,j) - zcur(i-1,j)) * r1_e1v(i,j) / e3v(i,j,jk,Kmm)        &  ! :51
          + (zdiv(i,j+1) - zdiv(i,j)) * r1_e2v(i,j) )                          ! :52

THICKNESS -- THREE DIFFERENT ANSWERS IN ONE TERM (the ladder family's 6th
confirmed site in this campaign):
  - zcur (curl): e3f(i,j,jk) -- this macro (domzgr_substitute.h90:129) takes
    NO explicit time argument at all; ``r3f`` (the QCO ssh/h0 ratio at
    F-points) is declared ``r3f(jpi,jpj)`` WITHOUT a time dimension
    (dom_oce.F90:194, comment "mid-time-level ratio at f-point") and is
    (re)computed from ``ssh(:,:,Nnn)`` ("now") at every call site
    (cfgs/DINO/MY_SRC/stpmlf.F90:218,303: ``CALL dom_qco_r3c(ssh(:,:,Nnn),
    ..., r3f(:,:))``). So e3f is EFFECTIVELY ALWAYS "now"/Kmm-valued --
    there is no Kbb/Kmm CHOICE at F-points, only one array, and it holds the
    "now" ssh-stretched (LIVE, not static) thickness.
  - zdiv (divergence): e3u/e3v/e3t all read at Kbb ("before") -- explicit
    ``jk,Kbb`` 4th argument, LIVE (QCO-stretched), matching the velocity
    time level used in the SAME line (pu(...,Kbb), pv(...,Kbb)).
  - outer grad(div)/curl(curl) divisor: e3u/e3v at Kmm ("now") -- a THIRD,
    DIFFERENT time level from the div/curl-forming step, on the SAME row.

TIME LEVEL (velocities): Kbb ("before") for pu_in/pv_in (curl and div
inputs) -- ``dyn_ldf`` call ``dynldf.F90:83: CALL dynldf_lev_lap(kt, Kbb,
Kmm, puu, pvv, Krhs)``, and the scheme macro binds ``pu_in(i,j,k,t) =
pu(i,j,k,t)`` with ``t=Kbb`` supplied by ``#define pu_in(i,j,k,t)
pu(i,j,k,t)`` + the CALLER passing ``Kbb`` explicitly nowhere in the .h90 --
actually the .h90 hardcodes ``pu_in(...,jk,Kbb)`` literally at lines 24-25,
28-29 (not a passed-through argument), so this is unambiguous: Laplacian
diffusion acts on the BEFORE-level velocity (leapfrog-with-Asselin-filter
convention), matching NEMO's general MLF diffusive-term convention.
Registered "now" in the existing ``zu_frc_term_walk.py``/
``zu_frc_momentum_row_reconstruction.py`` dump table -- that registration is
for the DUMPED TENDENCY (a Krhs increment, itself timeless/an RHS
contribution, "now" meaning "valid for this step"), not a claim about the
INPUT velocity time level; no conflict with the Kbb finding above.

legoESM transcription (``nemo_ldf_lap_viscosity_cgrid``,
``latlon_cgrid_operators.py:1299-1397``): grad(ahmt.div) - curl(ahmf.curl)
structure MATCHES; ahmt/ahmf coefficient MATCHES (verified below); but
``divergence_cgrid``/``curl_vertex_cgrid`` (the shared 2-D operators reused
here) carry **NO e3 weighting at all** -- confirmed by reading their bodies
(operators_latlon_cgrid.py:846-976, 1039-1268): denominators are cell/vertex
AREA only, never a thickness array. This is an EXPLICITLY DOCUMENTED
deviation in the function's own docstring (:1335-1340).

**A/B RESULT (CONFIRMED, this session): e3-weighting is REFUTED as the
cause.** On DINO's RUN_GDB grid, ``e3t_0 == e3u_0 == e3v_0 == e3f_0``
EVERYWHERE (verified directly from mesh_mask.nc: per-level std ~1e-14,
machine-zero -- this is ``full_step`` topography, no partial cells at all,
confirmed by ``bridge_nemo_to_legoesm_topo(..., full_step=True)`` used
throughout this campaign). With e3 spatially uniform at every level k, the
e3-weighting inside ``zdiv`` (build with ``e3u``/``e3v``, divide by ``e3t``)
and inside ``zcur``/its consuming divisor (``e3f`` build, ``e3u``/``e3v``
divide) is a PURE PER-LEVEL SCALAR that cancels algebraically -- restoring
it is a mathematical no-op ON THIS GRID. Measured via a PURE-NEMO-SPACE
transcription (this script, see below) validated against ``ldf_dump_du.bin``
to err_norm 4.47e-5 (roundoff): variant (A) e3-weighted and variant (B)
e3=1 give IDENTICAL output to 15 digits.

**ACTUAL ROOT CAUSE (CONFIRMED, this session): a TIME-LEVEL mismatch in the
COMPARISON HARNESS, not a defect in legoESM's operator or production
physics.** ``dynldf_lev_rot_scheme.h90:24-25,28-29`` reads the velocity at
Kbb ("before") -- confirmed above from the Fortran. legoESM's PRODUCTION
leap-frog integrator (``ocean_model_latlon_cgrid.py:6947-6968``, the "1b.
DISSIPATIVE Nbb pass") ALREADY does this correctly: it builds
``nbb = state._replace(u=state.u_before, v=state.v_before, ...)`` and calls
``_step_impl(nbb, ...)`` specifically so ``dyn_ldf``-equivalent lateral
viscosity is evaluated on the BEFORE-level velocity, matching NEMO
("evaluate the explicit lateral diffusion forward-in-time on the BEFORE
level (dyn_ldf/tra_ldf(Kbb) ... NOT leap-frog-centred (a centred diffusion
is unconditionally unstable)"). The measurement harness this campaign's
prior scripts use (``model.tendencies_with_diagnostics(state, ...)``) is a
SIMPLER, single-state diagnostic wrapper with NO before/now split -- it
always evaluates ``Ah_lap_u``/``Ah_lap_v`` on whatever single ``state`` is
passed, which every prior #1226 probe populated with the NOW/Kmm-level
bridged restart fields (``bridge_nemo_to_legoesm_topo`` -> ``un``/``vn``).

Measured (this session, ``model.tendencies_with_diagnostics``, legoESM's
REAL production ``nemo_ldf_lap_viscosity_cgrid``, not a numpy transcription):
    fed with NOW  (br.state.u/v, the standard #1226 probe convention):
        u err_norm=4.4948e-02   v err_norm=2.7573e-02   (== the DEBT ledger numbers)
    fed with BEFORE (state.u_before/v_before, matching dynldf_lev_rot_scheme's Kbb):
        u err_norm=4.577e-05    v err_norm=4.505e-05    (roundoff -- MATCHES)
1000x improvement, landing at the same roundoff floor the pure-NEMO-space
transcription self-check reaches. This is NOT a hypothesis -- it is measured
against legoESM's actual production dyn_ldf operator, run through the public
``tendencies_with_diagnostics`` API, with only the INPUT STATE'S velocity
time level changed.

**Implication for the fidelity_bar_gate.py ledger's "dyn_ldf u/v DEBT"
entry**: the row is NOT a production-physics bug requiring a code fix. It IS
a real gap in the MEASUREMENT METHOD -- every #1226 probe comparing
``Ah_lap_u``/``Ah_lap_v`` (or any dyn_ldf-derived quantity) against
``ldf_dump_du.bin``/``dv.bin`` while feeding the single "now" bridged state
is comparing NEMO's Kbb-evaluated term against legoESM's Kmm-evaluated one
-- an apples-to-oranges comparison, not a wiring/numerics defect. The
correct fix is in the FIDELITY HARNESS (build the diagnostic call with
``state.u_before``/``v_before`` substituted for this ONE term, mirroring
what ``_step_impl``'s Nbb pass already does in production), not in
``nemo_ldf_lap_viscosity_cgrid`` or ``divergence_cgrid``/``curl_vertex_cgrid``
themselves -- those are already correct. This finding does NOT change
``zu_frc_momentum_row_reconstruction.py``'s v-side reconstruction result
(that used the SAME now-state err_norm this session reproduces exactly,
4.49e-2/2.76e-2, so its corr=0.985/ratio=0.977 finding is unaffected by this
discovery) -- but it DOES mean "fixing dyn_ldf" for that v-payoff means
fixing the HARNESS'S state selection for this term, not the operator.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/dyn_zad_ldf_walk.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import netCDF4 as nc
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config

# Reuse wholesale (not re-derived): the established loaders + DT.
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    DT, _load_full_3d,
)
# Lane selector (#1455): RUN_DIR/RESTART are lane-dependent, NOT
# zu_frc_term_walk's own hardcoded RUN_GDB/year-5 constants.
from scripts.validate.ocean_fidelity.dino_1226 import dump_lane

RUN_DIR = dump_lane.RUN_DIR
RESTART = dump_lane.RESTART

register_dump("ldf_dump_du.bin", "now",
              "dynldf.F90:83 dynldf_lev_lap Krhs increment (np_lap, NOT "
              "np_lap_i/dyn_ldf_iso -- corrects a stale comment in "
              "zu_frc_term_walk.py; see this module's docstring ROW 2).")
register_dump("ldf_dump_dv.bin", "now", "same as ldf_dump_du")
register_dump("zad_dump_du.bin", "now", "dynadv.F90:97 dyn_zad Krhs increment (stock dynzad.F90:86-119).")
register_dump("zad_dump_dv.bin", "now", "same as zad_dump_du")
register_dump("keg_dump_du.bin", "now", "dynadv.F90:89,92 dyn_keg (Hollingsworth, nn_dynkeg=1) Krhs snapshot before dyn_zad.")
register_dump("keg_dump_dv.bin", "now", "same as keg_dump_du")

for _name in ("ldf_dump_du.bin", "ldf_dump_dv.bin", "zad_dump_du.bin",
              "zad_dump_dv.bin", "keg_dump_du.bin", "keg_dump_dv.bin"):
    time_level_for_dump(_name)


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a):
    return np.asarray(a)[1:, :]


def _err_norm(lego3, nemo3, mask2d):
    """RMS-normalized error, per-level, RMS(nemo) in the denominator (the
    established #1226 convention -- same transform both sides, oracle-
    fidelity Rule "same metric")."""
    n_lat_c = min(lego3.shape[0], nemo3.shape[0], mask2d.shape[0])
    n_lon_c = min(lego3.shape[1], nemo3.shape[1], mask2d.shape[1])
    n_lev_c = min(lego3.shape[2], nemo3.shape[2])
    lo = np.asarray(lego3)[:n_lat_c, :n_lon_c, :n_lev_c]
    ne = np.asarray(nemo3)[:n_lat_c, :n_lon_c, :n_lev_c]
    m = mask2d[:n_lat_c, :n_lon_c]
    err = lo - ne
    err_by_level = np.array([
        float(np.sqrt(np.nanmean(err[..., k][m] ** 2))) for k in range(n_lev_c)
    ])
    rms_by_level = np.array([
        float(np.sqrt(np.nanmean(ne[..., k][m] ** 2))) for k in range(n_lev_c)
    ])
    tot_err_norm = (float(np.sqrt(np.mean(err_by_level ** 2)))
                    / float(np.sqrt(np.mean(rms_by_level ** 2))))
    max_abs = float(np.nanmax(np.abs(err)))
    near_zero_frac = float(np.mean(np.abs(ne) < 1e-30))
    return err, err_by_level, rms_by_level, tot_err_norm, max_abs, near_zero_frac


def _corr_ratio(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2:
        return float("nan"), float("nan")
    corr = float(np.corrcoef(a, b)[0, 1])
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    ratio = rms_a / rms_b if rms_b > 0 else float("nan")
    return corr, ratio


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="dyn_zad_ldf_walk")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both' for this walk)")
    print(dump_lane.banner())
    assert e3t_mode == "both", "run with LEGOESM_NEMO_E3T=both (task rule)"

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.lateral_viscosity_operator={dcfg.lateral_viscosity_operator!r}  "
          f"vertical_momentum_scheme={dcfg.vertical_momentum_scheme!r}")
    assert dcfg.lateral_viscosity_operator == "nemo_div_curl"
    assert dcfg.vertical_momentum_scheme == "nemo_advective"

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br_before = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, br.state, context="dyn_zad_ldf_walk")
    print("dtype check: u", br.state.u.data.dtype, "z_coord.h_partial", br.z_coord.h_partial.dtype)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)

    with jax.disable_jit():
        _tend, diag = model.tendencies_with_diagnostics(br.state, surface_forcing=None, dt=DT)
        # ROW 2 decisive check (see docstring): re-evaluate the SAME
        # production diagnostic call with u/v/T/S swapped for the
        # BEFORE-level restart fields (mirroring
        # ocean_model_latlon_cgrid.py:6959-6961's "nbb = state._replace(
        # u=state.u_before, ...)" Nbb dissipative pass) -- isolates whether
        # dyn_ldf's mismatch is a TIME-LEVEL harness artifact vs a real
        # operator defect, using legoESM's REAL nemo_ldf_lap_viscosity_cgrid
        # (not a numpy transcription).
        st_before_fed = br_before._replace(
            u=br_before.u_before, v=br_before.v_before,
            T=br_before.T_before, S=br_before.S_before)
        _tend_bef, diag_bef = model.tendencies_with_diagnostics(
            st_before_fed, surface_forcing=None, dt=DT)

    ah_lap_u_3d = np.asarray(diag.Ah_lap_u.data)
    ah_lap_v_3d = np.asarray(diag.Ah_lap_v.data)
    vertadv_u_3d = np.asarray(diag.vertadv_u.data)
    vertadv_v_3d = np.asarray(diag.vertadv_v.data)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    nemo_ldf_du = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_ldf_dv = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_dv = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_keg_du = _load_full_3d(os.path.join(RUN_DIR, "keg_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_keg_dv = _load_full_3d(os.path.join(RUN_DIR, "keg_dump_dv.bin"), jpi, jpj, jpkm1, hls)

    ah_lap_u_f = _u_to_nemo(ah_lap_u_3d)
    ah_lap_v_f = _v_to_nemo(ah_lap_v_3d)
    vertadv_u_f = _u_to_nemo(vertadv_u_3d)
    vertadv_v_f = _v_to_nemo(vertadv_v_3d)

    # --- self-check 1: EEN reproduces its own trusted baseline (instrument
    # validation, per task rule) -- reuse the already-established vortcor_u
    # check from zu_frc_term_walk.py implicitly via the corr numbers already
    # on record (err_norm u=1.67e-3, v=2.45e-3, established, not re-derived
    # here -- this script's job is ZAD+ldf only).
    print("\n" + "=" * 78)
    print("SELF-CHECK 1: dyn_keg closure -- keg_dump + zad_dump should equal the")
    print("stp_dump_03_dynadv running total (both from the SAME Fortran dump")
    print("mechanism, dynadv.F90:90-105) -- confirms the ZAD-only dump is what")
    print("it claims to be, independent of legoESM entirely.")
    print("=" * 78)
    stp03_u_path = os.path.join(RUN_DIR, "stp_dump_03_dynadv_du.bin")
    stp03_v_path = os.path.join(RUN_DIR, "stp_dump_03_dynadv_dv.bin")
    if os.path.exists(stp03_u_path):
        stp03_u = _load_full_3d(stp03_u_path, jpi, jpj, jpkm1, hls)
        stp03_v = _load_full_3d(stp03_v_path, jpi, jpj, jpkm1, hls)
        recon_u = nemo_keg_du + nemo_zad_du
        recon_v = nemo_keg_dv + nemo_zad_dv
        max_diff_u = float(np.nanmax(np.abs(recon_u - stp03_u)))
        max_diff_v = float(np.nanmax(np.abs(recon_v - stp03_v)))
        print(f"  max|keg+zad - stp_dump_03_dynadv| u={max_diff_u:.3e}  v={max_diff_v:.3e}"
              "  (want ~0, roundoff only)")
    else:
        print("  stp_dump_03_dynadv_du.bin not present in RUN_DIR -- skip (non-fatal)")

    print("\n" + "=" * 78)
    print("SELF-CHECK 2: near-zero fraction + max|diff| sanity on the raw dumps")
    print("=" * 78)
    for name, arr, mask2 in (("ldf_du", nemo_ldf_du, umask2), ("zad_du", nemo_zad_du, umask2)):
        wet = arr[:mask2.shape[0], :mask2.shape[1], :][mask2]
        print(f"  {name}: wet-cell RMS={float(np.sqrt(np.nanmean(wet**2))):.4e}  "
              f"near-zero-frac(|x|<1e-30)={float(np.mean(np.abs(wet) < 1e-30)):.4f}  "
              f"any-nan={bool(np.isnan(wet).any())}")

    # =====================================================================
    # ROW 1: dyn_adv ZAD -- baseline err_norm (established, re-measured here
    # for a self-contained report) + per-level profile.
    # =====================================================================
    print("\n" + "=" * 78)
    print("ROW 1: dyn_adv ZAD -- baseline err_norm + per-level profile")
    print("=" * 78)
    err_u, ebl_u, rbl_u, tot_u, maxabs_u, nzf_u = _err_norm(vertadv_u_f, nemo_zad_du, umask2)
    err_v, ebl_v, rbl_v, tot_v, maxabs_v, nzf_v = _err_norm(vertadv_v_f, nemo_zad_dv, vmask2)
    print(f"  u: err_norm={tot_u:.4e}  max|diff|={maxabs_u:.4e}  near_zero_frac={nzf_u:.4f}")
    print(f"  v: err_norm={tot_v:.4e}  max|diff|={maxabs_v:.4e}  near_zero_frac={nzf_v:.4f}")
    print(f"  u err_by_level : {np.array2string(ebl_u, precision=3, max_line_width=200)}")
    print(f"  u rms(nemo)_by_level: {np.array2string(rbl_u, precision=3, max_line_width=200)}")
    print(f"  v err_by_level : {np.array2string(ebl_v, precision=3, max_line_width=200)}")
    print(f"  v rms(nemo)_by_level: {np.array2string(rbl_v, precision=3, max_line_width=200)}")
    deep_frac_u = float(np.mean(ebl_u[-10:]) / (np.mean(ebl_u[:10]) + 1e-30))
    deep_frac_v = float(np.mean(ebl_v[-10:]) / (np.mean(ebl_v[:10]) + 1e-30))
    print(f"  ladder discriminator (mean err[last 10 lev] / mean err[first 10 lev]): "
          f"u={deep_frac_u:.3f}  v={deep_frac_v:.3f}  "
          "(>>1 concentrated-at-depth => suspect the e3(Kmm) divisor ladder; "
          "~O(1)/flat => refutes it for THIS row)")

    print("\n" + "=" * 78)
    print("ROW 1 BOTTOM-LEVEL CANDIDATE TEST: does the deep-concentrated error")
    print("track EACH COLUMN'S OWN bottom_level (a per-column seafloor-boundary")
    print("artifact) or a fixed depth band common across columns (something else)?")
    print("=" * 78)
    bottom_level_t = np.asarray(br.z_coord.bottom_level)
    bl_u_face = np.minimum(bottom_level_t, np.roll(bottom_level_t, -1, axis=1))
    n_lat_b = min(err_u.shape[0], bl_u_face.shape[0])
    n_lon_b = min(err_u.shape[1], bl_u_face.shape[1])
    err_u_b = err_u[:n_lat_b, :n_lon_b, :]
    m_b = umask2[:n_lat_b, :n_lon_b]
    bl_u_b = bl_u_face[:n_lat_b, :n_lon_b]
    col_max_err = np.full((n_lat_b, n_lon_b), np.nan)
    argmax_k = np.full((n_lat_b, n_lon_b), -1)
    for jj in range(n_lat_b):
        for ii in range(n_lon_b):
            if m_b[jj, ii]:
                col_max_err[jj, ii] = float(np.max(np.abs(err_u_b[jj, ii, :])))
                argmax_k[jj, ii] = int(np.argmax(np.abs(err_u_b[jj, ii, :])))
    valid = m_b & np.isfinite(col_max_err)
    corr_bl = float(np.corrcoef(col_max_err[valid], bl_u_b[valid].astype(float))[0, 1])
    valid2 = m_b & (argmax_k >= 0)
    offset = argmax_k[valid2] - bl_u_b[valid2]
    frac_at_bottom = float(np.mean(offset == 0))
    frac_near_bottom = float(np.mean(np.abs(offset) <= 1))
    print(f"  corr(per-column max|err|, that column's bottom_level) = {corr_bl:.4f}")
    print(f"  argmax(|err|) - bottom_level: mean={float(np.mean(offset)):.2f}  "
          f"std={float(np.std(offset)):.2f}")
    print(f"  fraction of columns where the error peak IS the column's own "
          f"bottom level: {frac_at_bottom:.4f}  (within 1 level: {frac_near_bottom:.4f})")
    print("  (near-zero corr + low peak-at-bottom fraction => REFUTES a per-column")
    print("  seafloor-boundary-handling cause; a fixed-depth-band or ww-structure")
    print("  cause is more consistent -- see REPORT for the ranked candidate list.)")

    print("\n" + "=" * 78)
    print("ROW 1 TIME-LEVEL CHECK: rule out (by measurement, not assumption) the")
    print("SAME before/now artifact found for ROW 2 -- dynzad.F90:97 reads Kmm")
    print("('now') for BOTH uu/vv and ww (confirmed above from the source), so")
    print("feeding BEFORE should NOT improve the match.")
    print("=" * 78)
    vertadv_u_bef_f = _u_to_nemo(np.asarray(diag_bef.vertadv_u.data))
    vertadv_v_bef_f = _v_to_nemo(np.asarray(diag_bef.vertadv_v.data))
    _, _, _, tot_u_bef, _, _ = _err_norm(vertadv_u_bef_f, nemo_zad_du, umask2)
    _, _, _, tot_v_bef, _, _ = _err_norm(vertadv_v_bef_f, nemo_zad_dv, vmask2)
    print(f"  fed with NOW    : u err_norm={tot_u:.4e}  v err_norm={tot_v:.4e}")
    print(f"  fed with BEFORE : u err_norm={tot_u_bef:.4e}  v err_norm={tot_v_bef:.4e}")
    print("  (BEFORE not better than NOW => REFUTES a time-level explanation for")
    print("  ZAD, confirming dynzad.F90's own Kmm read is what legoESM already")
    print("  uses; the mismatch here is a genuine, separate, unresolved cause --")
    print("  see the ladder discriminator above and REPORT below.)")

    # --- A/B for ZAD's thickness divisor: legoESM's h_u/h_v (the caller-
    # supplied ``h_u``/``h_v`` argument to
    # ``nemo_advective_vertical_momentum_advection``, ocean_pe_latlon_cgrid.py
    # :2544/2548) vs NEMO's own live e3u(Kmm)/e3v(Kmm). A full independent
    # numpy reconstruction of ZAD would ALSO need NEMO's diagnostic ``ww``
    # (not in the restart -- ``ww`` is recomputed from continuity every step,
    # not a prognostic field NEMO checkpoints), so an exact pure-NEMO-space
    # A/B on this row (mirroring ROW 2's) is NOT built here -- report the one
    # thing directly measurable without it: whether legoESM's OWN h_u/h_v
    # divisor is the static reference or the live ssh-stretched thickness at
    # THIS restart's ssh (a nonzero ssh anomaly makes the two differ).
    print("\n" + "=" * 78)
    print("ROW 1 A/B: is legoESM's h_u/h_v divisor LIVE or STATIC at this state?")
    print("=" * 78)
    ssh_now = np.asarray(br.state.eta.data) if hasattr(br.state, "eta") else None
    if ssh_now is not None:
        print(f"  ssh(now) at this restart: mean={float(np.mean(ssh_now)):.4e} m  "
              f"max|ssh|={float(np.max(np.abs(ssh_now))):.4e} m")
        print("  If max|ssh| is O(1 m) against ~4000 m mean depth, the "
              "static-vs-live e3u distinction is a <<0.1% effect for THIS "
              "row's divisor (bounded, not a candidate that can explain a "
              "4-5e-2 err_norm) -- reported as a bound, not a full A/B "
              "(building the exact live/static swap needs the caller's "
              "internal h_u_old, which is not on the public diagnostics "
              "surface and packages/ is READ-ONLY for this task).")
    else:
        print("  br.state has no ssh field -- cannot bound this factor here.")

    # =====================================================================
    # ROW 2: dyn_ldf -- baseline + e3-weighted A/B reconstruction
    # =====================================================================
    print("\n" + "=" * 78)
    print("ROW 2: dyn_ldf -- baseline err_norm + per-level profile")
    print("=" * 78)
    err_u2, ebl_u2, rbl_u2, tot_u2, maxabs_u2, nzf_u2 = _err_norm(ah_lap_u_f, nemo_ldf_du, umask2)
    err_v2, ebl_v2, rbl_v2, tot_v2, maxabs_v2, nzf_v2 = _err_norm(ah_lap_v_f, nemo_ldf_dv, vmask2)
    print(f"  u: err_norm={tot_u2:.4e}  max|diff|={maxabs_u2:.4e}  near_zero_frac={nzf_u2:.4f}")
    print(f"  v: err_norm={tot_v2:.4e}  max|diff|={maxabs_v2:.4e}  near_zero_frac={nzf_v2:.4f}")
    print(f"  u err_by_level : {np.array2string(ebl_u2, precision=3, max_line_width=200)}")
    print(f"  v err_by_level : {np.array2string(ebl_v2, precision=3, max_line_width=200)}")
    deep_frac_u2 = float(np.mean(ebl_u2[-10:]) / (np.mean(ebl_u2[:10]) + 1e-30))
    deep_frac_v2 = float(np.mean(ebl_v2[-10:]) / (np.mean(ebl_v2[:10]) + 1e-30))
    print(f"  ladder discriminator: u={deep_frac_u2:.3f}  v={deep_frac_v2:.3f}")

    print("\n" + "=" * 78)
    print("ROW 2 DECISIVE CHECK: legoESM's REAL production nemo_ldf_lap_viscosity_cgrid")
    print("(via tendencies_with_diagnostics), fed with the NOW state (standard #1226")
    print("probe convention) vs the BEFORE state (state.u_before/v_before/T_before/")
    print("S_before substituted -- mirrors ocean_model_latlon_cgrid.py:6959-6961's")
    print("production Nbb dissipative pass EXACTLY).")
    print("=" * 78)
    ah_lap_u_bef_f = _u_to_nemo(np.asarray(diag_bef.Ah_lap_u.data))
    ah_lap_v_bef_f = _v_to_nemo(np.asarray(diag_bef.Ah_lap_v.data))
    _, _, _, tot_u2_bef, maxabs_u2_bef, _ = _err_norm(ah_lap_u_bef_f, nemo_ldf_du, umask2)
    _, _, _, tot_v2_bef, maxabs_v2_bef, _ = _err_norm(ah_lap_v_bef_f, nemo_ldf_dv, vmask2)
    print(f"  fed with NOW    (br.state.u/v):      u err_norm={tot_u2:.4e}  v err_norm={tot_v2:.4e}")
    print(f"  fed with BEFORE (state.u_before/v_before): u err_norm={tot_u2_bef:.4e}  "
          f"v err_norm={tot_v2_bef:.4e}")
    print(f"  max|diff| before-fed: u={maxabs_u2_bef:.4e}  v={maxabs_v2_bef:.4e}")
    print("  (a ~1000x drop here CONFIRMS the time-level-mismatch root cause -- see")
    print("  docstring for the full argument; this uses legoESM's REAL operator,")
    print("  not a probe transcription.)")

    print("\n" + "=" * 78)
    print("ROW 2 A/B: PURE-NEMO-SPACE transcription of dynldf_lev_rot_scheme.h90")
    print("(:21-52), operating on NEMO's OWN ub/vb + mesh_mask e1/e2/e3 arrays --")
    print("decoupled entirely from legoESM's operators. Variant (A) = exact")
    print("formula (e3-weighted, ahmt/ahmf embedded); variant (B) = same code")
    print("with every e3 factor set to 1 (isolates the e3-weighting factor alone).")
    print("=" * 78)

    # NEMO-native-grid metrics + before-level (Kbb) velocities -- everything
    # in NEMO's own (y=jj, x=ji) 0-indexed numpy space, T-cell-anchored
    # storage (u/v at the T-cell's own (ji,jj) east/north face, matching
    # restart-file convention: numpy index [j,i] <-> Fortran (ji,jj)=(i+1,j+1)).
    with nc.Dataset(os.path.join(RUN_DIR, "mesh_mask.nc")) as ds:
        def _m2(name):
            return np.asarray(ds.variables[name][0], dtype=np.float64)   # (jpj,jpi)
        def _m3(name):
            return np.asarray(ds.variables[name][0], dtype=np.float64).transpose(1, 2, 0)  # (jpk,jpj,jpi)->(jpj,jpi,jpk)
        e1t, e2t = _m2("e1t"), _m2("e2t")
        e1u, e2u = _m2("e1u"), _m2("e2u")
        e1v, e2v = _m2("e1v"), _m2("e2v")
        e1f, e2f = _m2("e1f"), _m2("e2f")
        e3t0, e3u0, e3v0, e3f0 = _m3("e3t_0"), _m3("e3u_0"), _m3("e3v_0"), _m3("e3f_0")
        umask3, vmask3, fmask3 = _m3("umask") > 0.5, _m3("vmask") > 0.5, _m3("fmask") > 0.5

    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    ub, vb = before.u, before.v   # (jpj, jpi, jpk), NEMO-native storage

    # ahmt/ahmf: static, depth-uniform, MATCHES legoESM (verified above) --
    # reuse the SAME nn_ahm_ijk_t=20 formula directly from e1t/e2t/e1f/e2f
    # (ldfc1d_c2d.F90:138-139), independent of legoESM's grid object.
    rn_Uv = 0.27  # namelist_cfg:369
    half_UM = 0.5 * rn_Uv
    ahmt2d = half_UM * np.maximum(e1t, e2t)
    ahmf2d = half_UM * np.maximum(e1f, e2f)

    def _p1(a, axis):
        """a[..., idx+1] brought to position idx (Fortran's (ji+1)/(jj+1)
        neighbor); non-periodic axis=0 (jj, N/S wall) uses edge padding
        (never read -- excluded by the interior mask below), periodic
        axis=1 (ji, DINO's zonally-periodic channel) uses roll."""
        if axis == 1:
            return np.roll(a, -1, axis=1)
        return np.concatenate([a[1:], a[-1:]], axis=0)

    def _m1(a, axis):
        """a[..., idx-1] brought to position idx."""
        if axis == 1:
            return np.roll(a, 1, axis=1)
        return np.concatenate([a[:1], a[:-1]], axis=0)

    def _ldf_lap_transcription(e3t, e3u, e3v, e3f):
        """dynldf_lev_rot_scheme.h90:21-52, vectorized over (jj,ji,jk).

        Re-derivation from the Fortran (loop index (ji,jj), OUTPUT stored at
        zcur(ji-1,jj-1)): let numpy [j,i] be the STORAGE location, so the
        loop's (ji,jj) = (i+1,j+1) in 0-indexed terms. Then:
          zcur[j,i] = ahmf[j,i]*e3f[j,i]/(e1f*e2f)[j,i] * (
                        (e2v[j,i+1]*vb[j,i+1] - e2v[j,i]*vb[j,i])
                      - (e1u[j+1,i]*ub[j+1,i] - e1u[j,i]*ub[j,i]) )
        (:23-25, expanding ji-1->i, jj-1->j, ji->i+1, jj->j+1). zdiv(ji,jj)
        stores directly at [j,i] (no index shift):
          zdiv[j,i] = ahmt[j,i]/(e1t*e2t)[j,i]/e3t[j,i] * (
                        (e2u*e3u*ub)[j,i] - (e2u*e3u*ub)[j,i-1]
                      + (e1v*e3v*vb)[j,i] - (e1v*e3v*vb)[j-1,i] )   (:27-29)
        pu(ji,jj) at native [j,i] (no shift), needs zcur[j,i]-zcur[j-1,i] and
        zdiv[j,i+1]-zdiv[j,i] (:34,40-42); pv similarly (:44,50-52)."""
        ahmf = ahmf2d[..., None]
        ahmt = ahmt2d[..., None]

        e2v_vb = e2v[..., None] * vb
        e1u_ub = e1u[..., None] * ub
        zcur = (ahmf * e3f * (1.0 / (e1f * e2f))[..., None]
                * ((_p1(e2v_vb, 1) - e2v_vb)
                   - (_p1(e1u_ub, 0) - e1u_ub)))
        zcur = zcur * fmask3   # ahmf already carries the surface fmask in NEMO

        e2u_e3u_u = e2u[..., None] * e3u * ub
        e1v_e3v_v = e1v[..., None] * e3v * vb
        zdiv = (ahmt * (1.0 / (e1t * e2t))[..., None] / np.where(e3t > 0, e3t, 1.0)
                * ((e2u_e3u_u - _m1(e2u_e3u_u, 1))
                   + (e1v_e3v_v - _m1(e1v_e3v_v, 0))))

        du = (-(zcur - _m1(zcur, 0)) * (1.0 / e2u)[..., None] / np.where(e3u > 0, e3u, 1.0)
              + (_p1(zdiv, 1) - zdiv) * (1.0 / e1u)[..., None])
        du = du * umask3

        dv = ((zcur - _m1(zcur, 1)) * (1.0 / e1v)[..., None] / np.where(e3v > 0, e3v, 1.0)
              + (_p1(zdiv, 0) - zdiv) * (1.0 / e2v)[..., None])
        dv = dv * vmask3
        return du, dv

    du_A, dv_A = _ldf_lap_transcription(e3t0, e3u0, e3v0, e3f0)
    ones = np.ones_like(e3t0)
    du_B, dv_B = _ldf_lap_transcription(ones, ones, ones, ones)

    # Interior-only comparison mask (exclude the jj=0/-1 wall rows the J-roll
    # wraps spuriously, and the DINO channel's own land mask).
    interior = np.ones((umask3.shape[0], umask3.shape[1]), dtype=bool)
    interior[0, :] = False
    interior[-1, :] = False
    u_interior_mask = umask3[..., 0].astype(bool) & interior
    v_interior_mask = vmask3[..., 0].astype(bool) & interior

    def _cmp(label, du, dv):
        err3u, ebl, rbl, tot, mx, nzf = _err_norm(du, nemo_ldf_du, u_interior_mask)
        err3v, eblv, rblv, totv, mxv, nzfv = _err_norm(dv, nemo_ldf_dv, v_interior_mask)
        print(f"  {label}  u: err_norm={tot:.4e} max|diff|={mx:.4e}   "
              f"v: err_norm={totv:.4e} max|diff|={mxv:.4e}")
        return tot, totv

    print("  [pure-NEMO-space self-check: variant (A) vs ldf_dump_du/dv.bin]")
    totA_u, totA_v = _cmp("(A) e3-weighted (exact transcription):", du_A, dv_A)
    print("  [variant (B): e3 -> 1 everywhere, isolates the e3-weighting factor]")
    totB_u, totB_v = _cmp("(B) e3=1 (no thickness weighting):      ", du_B, dv_B)
    print(f"  Delta from removing e3-weighting: u {totA_u:.4e} -> {totB_u:.4e}   "
          f"v {totA_v:.4e} -> {totB_v:.4e}")
    print("  (variant A is a pure-NEMO-space reimplementation, NOT legoESM's "
          "own operator -- if (A) itself does not reach ~0 against "
          "ldf_dump_du.bin, the transcription/index-convention has a bug and "
          "the A/B is not yet trustworthy; report both numbers, do not "
          "over-claim.)")

    print("\n" + "=" * 78)
    print("DONE -- see terminal output above for the report's numeric inputs.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
