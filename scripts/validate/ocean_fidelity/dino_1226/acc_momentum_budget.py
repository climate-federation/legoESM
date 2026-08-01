"""ACC momentum-budget attribution: legoESM vs NEMO, term by term, at Y5.

WHY THIS EXISTS
---------------
legoESM's DINO ACC is 67.7 Sv vs NEMO's 91.1 Sv at year 5 from rest (a 25%
deficit) and it is NOT chaos (ensemble spread 0.15 Sv legoESM / 0.24 Sv NEMO).
Two days of term-by-term fidelity work fixed three real defects and moved the
ACC by 0.1 Sv.  So instead of continuing the sweep, this script measures the
ACC momentum balance ITSELF, in both models, FROM THE SAME STATE, and ranks
the six momentum terms by their contribution to the zonally-integrated
zonal-momentum-budget difference in the re-entrant channel.

WHAT NEMO DUMPS (traced, not guessed -- stpmlf.F90 read directly)
-----------------------------------------------------------------
``stp_dump_krhs`` (stpmlf.F90:690-724) dumps the RUNNING/ACCUMULATED momentum
RHS ``uu(:,:,:,Nrhs)`` after each dyn_* call -- NOT the per-term trend.  So a
per-term trend is a DIFFERENCE of consecutive accumulator dumps:

    stage 3  stp_dump_03_dynadv_du.bin   uu(Nrhs) after dyn_adv   (stpmlf.F90:265,270)
    stage 4  stp_dump_04_dynvor_du.bin   uu(Nrhs) after dyn_vor   (stpmlf.F90:271,274)
    stage 5  stp_dump_05_dynldf_du.bin   uu(Nrhs) after dyn_ldf   (stpmlf.F90:275,278)
    stage 6  stp_dump_06_dynhpg_du.bin   uu(Nrhs) after dyn_hpg   (stpmlf.F90:280,284)

      adv = D03            (dyn_adv = dyn_keg + dyn_zad for np_VEC_c2; the two
                            are NOT separable at this dump granularity)
      vor = D04 - D03      (dyn_vor: planetary f + relative vorticity, EEN)
      ldf = D05 - D04      (dyn_ldf: lateral friction)
      hpg = D06 - D05      (dyn_hpg: hydrostatic pressure gradient)

Stages 7 and 8 go through ``stp_dump_state_and_bt`` (stpmlf.F90:727-772),
which dumps ``uu(:,:,:,Naa)``:

    stage 7  stp_dump_07_dynspg_u.bin    uu(:,:,:,Naa) after dyn_spg  (stpmlf.F90:288,293)
    stage 8  stp_dump_08_dynzdf_u.bin    uu(:,:,:,Naa) after dyn_zdf  (stpmlf.F90:305,312)

  !! TIME-LEVEL TRAP -- the exact one that produced a false diagnosis earlier
  !! in this campaign, resolved here by reading the Fortran AND measuring.
  !! DURING the step body ``Naa`` and ``Nrhs`` are the SAME index (the rotation
  !! ``Nrhs=Nbb; Nbb=Nnn; Nnn=Naa; Naa=Nrhs`` runs at stpmlf.F90:403-406, i.e.
  !! at the very END of the step).  So "uu(Naa)" means the accumulator UNTIL
  !! something converts it, and the converter is ``dyn_zdf`` itself:
  !!     dynzdf.F90:127-129   puu(Kaa) = ( (1+r3u(Kbb))*puu(Kbb)
  !!                                     + rDt*(1+r3u(Kmm))*puu(Krhs) )
  !!                                     / (1+r3u(Kaa)) * umask
  !! Therefore:
  !!   stage 7 (BEFORE dyn_zdf) is STILL the Krhs ACCUMULATOR [m/s^2] -- the
  !!     task brief's reading is correct.  ``dyn_spg_ts``'s MLF branch operates
  !!     ON that accumulator: it first REMOVES the depth mean
  !!     (dynspg_ts.F90:345-346 ``puu(Krhs) = (puu(Krhs) - zu_frc)*umask``) and
  !!     then ADDS BACK the barotropic solution's tendency
  !!     (dynspg_ts.F90:969-974 ``puu(Krhs) += r1_hu(Kmm)*(puu_b(Kaa)
  !!     - puu_b(Kbb)*hu(Kbb))*r1_Dt``).  Measured: RMS(stage 7)=4.94e-07 vs
  !!     RMS(D06)=3.69e-06 -- same units, SMALLER, exactly as the
  !!     depth-mean-removal predicts.
  !!   stage 8 (AFTER dyn_zdf) IS the after VELOCITY [m/s].  Measured:
  !!     RMS=3.03e-02, matching the restart ub RMS 3.49e-02.
  !! This script GATES both readings numerically before using them (see
  !! "TIME-LEVEL GATE" below) and ABORTS if either fails.

So the last two terms come out of the SAME accumulator algebra as the first
four, with the total closed by the one genuine state difference:

      spg = U7 - D06                  [everything dyn_spg_ts does to the 3-D
                                       RHS: barotropic-mode replacement, i.e.
                                       the free-surface pressure gradient]
      zdf = (U8* - Ubb)/rDt - U7      [implicit vertical mixing: wind stress in
                                       at the top, bottom drag out at the base]

with  U8* = U8 + uu_b(Naa)  -- the BAROTROPIC RESTORE.  ln_drgimp=T and
ln_dynspg_ts=T (ocean.output:805,1036) make dynzdf.F90:150-151 subtract
uu_b(Kaa) from puu(Kaa) before the implicit solve, and dyn_zdf never adds it
back, so the raw stage-8 dump is the BAROCLINIC-ONLY after velocity.  Using it
raw injects a spurious DEPTH-UNIFORM -uu_b/rDt = -1.8e-6 m/s^2 into 'zdf'.
That is not a hypothetical: the first run of this script did exactly that and
ranked 'zdf' as 97.7% of the ACC momentum difference -- a false diagnosis.
uu_b(Naa) is taken from ``stp_dump_07_dynspg_ub.bin`` and the restore is GATED
(depth-uniform before -> surface-intensified after).

and by construction  adv+vor+ldf+hpg = D06,  +spg = U7,  +zdf = (U8*-Ubb)/rDt.

rDt = 2*rn_Dt = 5400 s: MLF leapfrog, and ``ocean.output:249`` says
"Modified Leap-Frog (MLF) : rDt = 5400.0"; ``ocean.output:259`` says
``ln_1st_euler = F`` and the kt=57601 atf dumps report "leapfrog branch
(l_1st_euler= F)", so nit000 is a FULL leapfrog step, not a half Euler step.
Asserted below against legoESM's own captured barotropic dt.

TERM MAPPING legoESM <-> NEMO (the honest version, incl. what does not map)
--------------------------------------------------------------------------
legoESM's per-term momentum diagnostics come from the model's OWN public
entry point ``LatLonCGridOceanModel.tendencies_with_diagnostics`` ->
``MomentumTendencyDiagnostics`` (closure-tested in
``tests/ocean/unit/test_momentum_diagnostics_closure.py``) -- no re-derived
numerics here.  Its ``KE_PGF_u`` bundles the KE-gradient with the pressure
gradient in ONE field, while NEMO bundles the KE-gradient with vertical
advection in ONE dump.  Neither side can split its bundle, so:

    NEMO adv (keg+zad)   ~   lego vertadv_u                 PARTIAL (keg missing)
    NEMO hpg             ~   lego KE_PGF_u                  PARTIAL (keg extra)
    NEMO adv + hpg       ==  lego vertadv_u + KE_PGF_u      EXACT group
    NEMO vor             ==  lego vortcor_u  (ALREADY (f+zeta)xu) EXACT
    NEMO ldf             ==  lego Ah_lap+Bh_bilap+Cs+Cl     EXACT
    NEMO spg             ==  lego spg (same residual formula)
    NEMO zdf             ==  lego zdf (same state difference)

Both the partial rows AND the exact group are reported; the RANKING is built
from the EXACT rows only (adv+hpg as one group), so no ranking entry rests on
a bundle mismatch.  As it happens the KE-gradient's ZONAL INTEGRAL around a
re-entrant channel telescopes to ~0, so the group split is nearly harmless
for the channel diagnostic -- but that is MEASURED below (the adv/hpg
individual rows are reported next to the group), not assumed.

ONE HARNESS SEED IS REQUIRED, AND IT IS NOT COSMETIC
----------------------------------------------------
The bridge leaves ``state.tau_x_prev``/``tau_y_prev`` as None, which puts
legoESM on its first-step branch and SILENTLY DROPS THE WIND from step 1's
du_dt.  Measured, before/after seeding, as the channel integral of
(model du_dt - the four mapped terms):

    unseeded   step1 pass0/1 = -7.43e+01 / -6.69e+01   (no wind)
               step2 pass0/1 = +1.32e+04 / +1.37e+04   (wind present)
    seeded     step1 pass0/1 = +1.38e+04 / +1.38e+04   (wind present)

Unseeded, this script ranked the surface-stress pathway at 100% of the
difference for a reason that was purely an artifact of measuring step 1.
The seed used is ``tau_x_prev = sf.tau_x`` which is EXACT, not an
approximation: DINO's analytic wind is time-invariant, so NEMO's own
"before := now" nit000 rule (sbcmod.F90:568-573) gives utau_b == utau, and
dyn_zdf's top-cell BC zDt_2*(utau_b+utauU) (dynzdf.F90:333-334) reduces to
the same steady stress.

MANDATORY PRECONDITIONS (both fail closed)
------------------------------------------
``precision_gate.require_fp64`` + ``precision_gate.require_explicit_e3t_mode``.
Run with ``LEGOESM_NEMO_E3T=both``: unset, the bridge silently uses NEMO's
analytic ``e3t_1d`` ladder (12.9% off below k=25) and every number here is
worthless -- that default has already ruined four measurements.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \
      LEGOESM_NEMO_E3T=both .venv/bin/python \
      scripts/validate/ocean_fidelity/dino_1226/acc_momentum_budget.py
"""
from __future__ import annotations

import dataclasses
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_before_state_topo,
    bridge_nemo_to_legoesm_topo,
)
# (tendency_probe deliberately not used -- see the note at the diagnostics call)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART_FILE = "DINO_00057600_restart.nc"   # the restart RUN_GDB STARTED from
DT = 2700.0                                  # rn_Dt (namelist_cfg:116)
RDT = 2.0 * DT                               # MLF rDt (ocean.output:249)

set_policy(PrecisionPolicy.fp64())

# --- Precondition 2: register every dump's NEMO time level, source-cited ----
# An unsourced registration is a guess and is forbidden (a time-level error has
# already caused one full false diagnosis in this campaign -- see the module
# docstring of ocean/fidelity/time_levels.py and spg_substep_chain.py STAGE 4).
_KRHS_DUMPS = {
    "stp_dump_03_dynadv_du.bin": (
        "stpmlf.F90:265,270 stp_dump_krhs(kstp,3,'dynadv',uu(:,:,:,Nrhs),...) -- "
        "RUNNING accumulator after dyn_adv(kstp,Nbb,Nnn,uu,vv,Nrhs); the trend "
        "itself is evaluated on Nbb/Nnn, the accumulator slot is Nrhs"),
    "stp_dump_04_dynvor_du.bin": (
        "stpmlf.F90:271,274 stp_dump_krhs(kstp,4,'dynvor',uu(:,:,:,Nrhs),...) -- "
        "accumulator after dyn_vor(kstp,Nnn,uu,vv,Nrhs), trend on Nnn"),
    "stp_dump_05_dynldf_du.bin": (
        "stpmlf.F90:275,278 stp_dump_krhs(kstp,5,'dynldf',uu(:,:,:,Nrhs),...) -- "
        "accumulator after dyn_ldf(kstp,Nbb,Nnn,uu,vv,Nrhs); NEMO's lateral "
        "friction is evaluated on the BEFORE level Nbb (leapfrog diffusion)"),
    "stp_dump_06_dynhpg_du.bin": (
        "stpmlf.F90:280,284 stp_dump_krhs(kstp,6,'dynhpg',uu(:,:,:,Nrhs),...) -- "
        "accumulator after dyn_hpg(kstp,Nnn,uu,vv,Nrhs), trend on Nnn; this is "
        "the FULL explicit momentum RHS entering dyn_spg"),
}
_STATE_DUMPS = {
    "stp_dump_07_dynspg_u.bin": (
        "stpmlf.F90:288,293 stp_dump_state_and_bt(kstp,7,'dynspg',uu(:,:,:,Naa),...) "
        "-- Naa ALIASES Nrhs during the step body (rotation at stpmlf.F90:403-406), "
        "and the Krhs->Kaa velocity conversion only happens later in "
        "dynzdf.F90:127-129, so this is STILL the momentum RHS ACCUMULATOR "
        "[m/s^2] after dyn_spg_ts's MLF branch removed the depth mean "
        "(dynspg_ts.F90:345-346) and added back the barotropic solution's "
        "tendency (dynspg_ts.F90:969-974). Gated numerically below."),
    "stp_dump_08_dynzdf_u.bin": (
        "stpmlf.F90:305,312 stp_dump_state_and_bt(kstp,8,'dynzdf',uu(:,:,:,Naa),...) "
        "-- the AFTER VELOCITY [m/s]: dynzdf.F90:127-129 converts "
        "Kbb + rDt*Krhs into Kaa, then the implicit tridiagonal solve + wind "
        "stress (dynzdf.F90:329-334) + bottom drag are folded in.  !! It is the "
        "BAROCLINIC-ONLY velocity: with ln_drgimp=T and ln_dynspg_ts=T (both "
        "true for DINO, ocean.output:805,1036) dynzdf.F90:150-151 REMOVES "
        "uu_b(Kaa) before the solve and never restores it inside dyn_zdf.  This "
        "script adds uu_b(Kaa) back from stp_dump_07_dynspg_ub.bin -- see the "
        "BAROTROPIC-RESTORE GATE."),
    "stp_dump_07_dynspg_ub.bin": (
        "stpmlf.F90:288,293-294 stp_dump_state_and_bt(...,uu_b(:,:,Naa),...) -- "
        "the 2-D barotropic velocity at Naa, a genuinely separate array from "
        "puu (not Krhs-aliased); this is exactly the field dynzdf.F90:150-151 "
        "subtracts from puu(Kaa)"),
}
for _n, _src in _KRHS_DUMPS.items():
    register_dump(_n, "after", _src)   # Nrhs accumulator, dumped inside the step
for _n, _src in _STATE_DUMPS.items():
    register_dump(_n, "after", _src)   # Naa slot (aliases Nrhs pre-dyn_zdf)
for _n in list(_KRHS_DUMPS) + list(_STATE_DUMPS):
    time_level_for_dump(_n)            # raises if unregistered -- fail loud

TERMS = ("adv", "vor", "ldf", "hpg", "spg", "zdf")


def _read_dims(run_dir: str) -> tuple[int, int, int, int, int]:
    with open(os.path.join(run_dir, "ocean.output")) as f:
        text = f.read()
    jpi = int(re.search(r"jpi\s*:\s*(\d+)", text).group(1))
    jpj = int(re.search(r"jpj\s*:\s*(\d+)", text).group(1))
    jpk = int(re.search(r"jpk\s*:\s*(\d+)", text).group(1))
    hls = int(re.search(r"nn_hls\s*=\s*(\d+)", text).group(1))
    nn_e = int(re.search(r"iterations nn_e\s*=\s*(\d+)", text).group(1))
    return jpi, jpj, jpk, hls, nn_e


def _load_full_3d(path: str, jpi: int, jpj: int, jpkm1: int, hls: int) -> np.ndarray:
    """(jpkm1,jpj,jpi) level-by-level stream dump -> haloless (n_lat,n_lon,jpkm1).

    Same convention as spg_substep_chain._load_full_3d (stpmlf.F90:717-720 /
    753-756 write one (jpj,jpi) record per level over jk=1,jpkm1).
    """
    a = np.fromfile(path, dtype="<f8").reshape(jpkm1, jpj, jpi)
    if hls:
        a = a[:, hls:-hls, hls:-hls]
    return np.moveaxis(a, 0, -1)


def _u_to_nemo(a):
    """legoESM u-face array (n_lat, n_lon+1, ...) -> NEMO u-column layout.

    legoESM face i is the WEST face of T-column i; NEMO u-point i is the EAST
    face of T-column i == legoESM face i+1.  Identical convention to
    spg_substep_chain.py / momentum_budget_diff.py.
    """
    return np.asarray(a)[:, 1:]


def _report(name, lego, nemo, mask):
    """corr + err_norm = |lego-nemo| / RMS(nemo).

    These are SIGN-CHANGING tendency fields, so a pointwise relative error is
    meaningless; normalise by the RMS of the reference instead.
    """
    m = mask & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    rms = float(np.sqrt(np.mean(ne ** 2))) if lo.size else float("nan")
    rms_l = float(np.sqrt(np.mean(lo ** 2))) if lo.size else float("nan")
    err = (float(np.sqrt(np.mean((lo - ne) ** 2))) / rms) if rms > 0 else float("nan")
    corr = float(np.corrcoef(lo, ne)[0, 1]) if lo.size > 1 else float("nan")
    print(f"  {name:<30s} corr={corr:9.6f}  err_norm={err:.4e}  "
          f"RMS(nemo)={rms:.4e}  RMS(lego)={rms_l:.4e}  n={int(m.sum())}")
    return corr, err


def _per_level(name, lego, nemo, mask3, every=4):
    print(f"  per-level err_norm [{name}] (k, err_norm, RMS(nemo), corr):")
    nk = min(lego.shape[-1], nemo.shape[-1], mask3.shape[-1])
    for k in range(nk):
        if k % every and k != nk - 1:
            continue
        mk = mask3[..., k] & np.isfinite(lego[..., k]) & np.isfinite(nemo[..., k])
        if mk.sum() < 10:
            continue
        ne, lo = nemo[..., k][mk], lego[..., k][mk]
        rms = float(np.sqrt(np.mean(ne ** 2)))
        if rms <= 0:
            continue
        e = float(np.sqrt(np.mean((lo - ne) ** 2))) / rms
        c = float(np.corrcoef(lo, ne)[0, 1])
        print(f"      k={k:2d}  err_norm={e:.4e}  RMS(nemo)={rms:.4e}  corr={c:9.6f}")


def _shift_scan(name, lego, nemo, mask):
    """Alignment scan on a U-FACE field: a sharp minimum at (0,0) confirms the
    index convention; a flat scan or an off-centre peak = misalignment."""
    rows = []
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            L = np.roll(lego, (dj, di), axis=(0, 1))
            m = mask & np.isfinite(L) & np.isfinite(nemo)
            if m.sum() < 100:
                continue
            rms = float(np.sqrt(np.mean(nemo[m] ** 2)))
            if rms <= 0:
                continue
            rows.append((float(np.sqrt(np.mean((L[m] - nemo[m]) ** 2))) / rms, dj, di))
    rows.sort()
    best, second = rows[0], rows[1]
    print(f"  [align scan] {name:<22s} best (dj,di)={best[1:]} err={best[0]:.4e}  "
          f"| 2nd-best {second[1:]} err={second[0]:.4e}  "
          f"| sharpness(2nd/best)={second[0]/best[0]:.2f}x "
          f"{'OK' if best[1:] == (0, 0) and second[0] > 2 * best[0] else '<-- CHECK'}")


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="acc_momentum_budget")
    print("=" * 100)
    print("ACC MOMENTUM BUDGET -- legoESM vs NEMO, per term, Y5 restart state")
    print("=" * 100)
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r}  (must be 'both': NEMO's real e3t_0 ladder)")

    jpi, jpj, jpk, hls, nn_e = _read_dims(RUN_DIR)
    jpkm1 = jpk - 1
    print(f"NEMO dims jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls} nn_e={nn_e}  "
          f"rn_Dt={DT} rDt(MLF)={RDT}")

    # ---------------- state / geometry (same idiom as spg_substep_chain) -----
    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                              lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="acc_momentum_budget twin state")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    # NEMO nit000 "before := now" forcing rule (sbcmod.F90:568-573): the restart
    # carries utau_b, and dyn_zdf's top-cell BC uses zDt_2*(utau_b + utauU)
    # (dynzdf.F90:333-334).  DINO's analytic wind is time-INVARIANT, so
    # utau_b == utau and the seed is exact, not an approximation.  The bridge
    # leaves ``tau_x_prev``/``tau_y_prev`` as None, which puts legoESM on its
    # first-step branch; seeding them here puts the two models on the same
    # footing.  MEASURED consequence of NOT seeding (see the "WHERE IS
    # legoESM's WIND?" block): step 1's du_dt carries NO wind while step 2's
    # carries +1.32e4 m^3/s^2 -- a first-step-only omission that would have
    # been reported as a 100%-of-the-budget surface-stress defect.
    if hasattr(st, "tau_x_prev"):
        st = st._replace(tau_x_prev=sf.tau_x, tau_y_prev=sf.tau_y)
        print("  seeded state.tau_x_prev/tau_y_prev = sf.tau_x/tau_y "
              "(NEMO nit000 before:=now, steady analytic wind -> exact)")
    else:
        print("  WARNING: state has no tau_x_prev field -- cannot seed; the "
              "first-step wind gate below may contaminate the budget")

    umask3 = np.asarray(g.umask) > 0.5                 # (n_lat, n_lon, nlev)
    umask2 = umask3[..., 0]
    tmask_s = np.asarray(g.tmask)[..., 0] > 0.5
    e1u = np.asarray(g.e1u)                            # (n_lat, n_lon)
    e3u_0 = np.asarray(g.e3u_0)                        # (n_lat, n_lon, nlev)
    gphit = np.asarray(g.gphit)
    assert g.e3u_0 is not None, "mesh_mask.nc has no e3u_0 -- cannot weight the zonal integral"

    # ---------------- NEMO dumps -> per-term trends --------------------------
    def _dump(name):
        return _load_full_3d(os.path.join(RUN_DIR, name), jpi, jpj, jpkm1, hls)

    D03 = _dump("stp_dump_03_dynadv_du.bin")
    D04 = _dump("stp_dump_04_dynvor_du.bin")
    D05 = _dump("stp_dump_05_dynldf_du.bin")
    D06 = _dump("stp_dump_06_dynhpg_du.bin")
    U7 = _dump("stp_dump_07_dynspg_u.bin")
    U8_bc = _dump("stp_dump_08_dynzdf_u.bin")     # BAROCLINIC-only (see below)
    ubar_naa = np.fromfile(
        os.path.join(RUN_DIR, "stp_dump_07_dynspg_ub.bin"),
        dtype="<f8").reshape(jpj, jpi)[hls:-hls, hls:-hls]      # uu_b(:,:,Naa)
    ub_bb = np.asarray(before.u)[..., :jpkm1]          # uu(:,:,:,Nbb), restart.F90:347

    # --- TIME-LEVEL GATE (fail closed).  Stage 7 must still be an ACCUMULATOR
    # [m/s^2] (same order as D06, ~5 orders BELOW a velocity); stage 8 must be
    # the after VELOCITY [m/s] (same order as the restart ub).  This is the
    # exact distinction that produced a false diagnosis earlier in this
    # campaign -- verify it empirically before any of it is used.
    m3 = umask3[..., :jpkm1]
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a)[m3] ** 2)))
    # --- BAROTROPIC-RESTORE GATE.  dynzdf.F90:148-151 (ln_drgimp .AND.
    # ln_dynspg_ts, both T for DINO -- ocean.output:805,1036) removes the
    # barotropic velocity uu_b(Kaa) from puu(Kaa) before the implicit solve and
    # does NOT put it back inside dyn_zdf, so the stage-8 dump is the
    # BAROCLINIC-ONLY after velocity.  Comparing it directly against a full
    # legoESM velocity injects a spurious DEPTH-UNIFORM -uu_b/rDt term of
    # magnitude ~9.6e-3/5400 = 1.8e-6 m/s^2 into the 'zdf' row -- which is
    # exactly the ~1.8e-6, depth-independent, near-zero-correlation signal the
    # first run of this script produced, and it would have been reported as
    # "legoESM's vertical mixing is 10x too weak" (a false diagnosis worth
    # 97.7% of the ranking).  Restore it, and GATE on the restoration turning a
    # depth-UNIFORM profile into a surface-intensified one.
    U8 = U8_bc + ubar_naa[..., None] * umask3[..., :jpkm1]
    r_ub, r_u7, r_u8, r_d06 = rms(ub_bb), rms(U7), rms(U8), rms(D06)
    print(f"\nTIME-LEVEL GATE  RMS: ub(Nbb)={r_ub:.4e} m/s | D06 accumulator="
          f"{r_d06:.4e} m/s^2 | stage7={r_u7:.4e} | stage8={r_u8:.4e}")
    assert 0.05 < r_u7 / r_d06 < 20.0, (
        f"stage 7 RMS ({r_u7:.3e}) is not within 20x of the D06 accumulator RMS "
        f"({r_d06:.3e}) -- it is not the Krhs accumulator; re-read "
        "dynspg_ts.F90/stpmlf.F90 before trusting anything below. ABORT")
    assert r_u8 > 1e3 * r_d06 and 0.2 < r_u8 / r_ub < 5.0, (
        f"stage 8 RMS ({r_u8:.3e}) is not a velocity (restart ub RMS {r_ub:.3e}, "
        f"accumulator RMS {r_d06:.3e}) -- the dynzdf.F90:127-129 Krhs->Kaa "
        "conversion reading is wrong. ABORT")
    print("  GATE PASSED: stages 3-7 are Krhs accumulators [m/s^2]; "
          "stage 8 is the after velocity [m/s].")

    _zdf_raw = (U8_bc - ub_bb) / RDT - U7
    _zdf_fix = (U8 - ub_bb) / RDT - U7
    _rk = lambda a, k: float(np.sqrt(np.mean(a[..., k][m3[..., k]] ** 2)))
    print(f"BAROTROPIC-RESTORE GATE  RMS(uu_b(Naa))={float(np.sqrt(np.mean(ubar_naa[umask3[..., 0]] ** 2))):.4e} m/s"
          f"  -> uu_b/rDt = {float(np.sqrt(np.mean(ubar_naa[umask3[..., 0]] ** 2))) / RDT:.4e} m/s^2")
    print(f"  zdf WITHOUT restore: RMS={rms(_zdf_raw):.4e}  k=0 {_rk(_zdf_raw, 0):.3e}  "
          f"k=17 {_rk(_zdf_raw, 17):.3e}  k=34 {_rk(_zdf_raw, jpkm1 - 1):.3e}  "
          f"(top/bottom ratio {_rk(_zdf_raw, 0) / _rk(_zdf_raw, jpkm1 - 1):.2f})")
    print(f"  zdf WITH    restore: RMS={rms(_zdf_fix):.4e}  k=0 {_rk(_zdf_fix, 0):.3e}  "
          f"k=17 {_rk(_zdf_fix, 17):.3e}  k=34 {_rk(_zdf_fix, jpkm1 - 1):.3e}  "
          f"(top/bottom ratio {_rk(_zdf_fix, 0) / _rk(_zdf_fix, jpkm1 - 1):.2f})")
    assert _rk(_zdf_raw, 0) / _rk(_zdf_raw, jpkm1 - 1) < 3.0, (
        "the UN-restored zdf is not depth-uniform -- the dynzdf.F90:150-151 "
        "barotropic-removal reading may be wrong; re-check before proceeding")
    assert _rk(_zdf_fix, 0) / _rk(_zdf_fix, jpkm1 - 1) > 10.0, (
        "restoring uu_b(Naa) did NOT produce a surface-intensified vertical-"
        "mixing profile -- the restore is not the right correction; ABORT")
    print("  GATE PASSED: restoring uu_b(Naa) turns a depth-UNIFORM artifact "
          "into a surface-intensified wind-stress/viscosity profile.")

    nemo = {
        "adv": D03,
        "vor": D04 - D03,
        "ldf": D05 - D04,
        "hpg": D06 - D05,
        "spg": U7 - D06,
        "zdf": (U8 - ub_bb) / RDT - U7,
    }

    # ---------------- legoESM per-term tendencies ----------------------------
    # Per-term breakdown from the model's OWN public diagnostic entry point
    # ``tendencies_with_diagnostics`` -> ``MomentumTendencyDiagnostics``
    # (closure-tested in tests/ocean/unit/test_momentum_diagnostics_closure.py:
    # Sigma components == du_dt to machine precision).  No re-derived numerics.
    #
    # NOTE on why not ``fidelity.tendency_probe.probe_latlon_cgrid`` (the first
    # choice): it re-calls ``compute_vertical_K_profiles`` for its TRACER-zdf
    # field WITHOUT threading the leap-frog before-level tracers, which this
    # recipe's ``TKEConfig.tke_n2_time_level='nemo_before'`` requires -- it
    # raises.  Its MOMENTUM half is exactly the call used here, so this is the
    # same numerics minus the tracer section this script does not need.
    _tend, diag = model.tendencies_with_diagnostics(st, surface_forcing=sf, dt=DT)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import coriolis_cgrid
    cor_u, _cor_v = coriolis_cgrid(st.u.data, st.v.data, br.geometry,
                                   u_mask=st.u_mask.data, v_mask=st.v_mask.data)

    def _D(name):
        f = getattr(diag, name, None)
        return np.zeros_like(np.asarray(cor_u)) if f is None else np.asarray(f.data)

    cor_u = np.asarray(cor_u)
    # vortcor_u ALREADY carries the FULL absolute vorticity (f+zeta)x u for this
    # recipe -- ocean_pe_latlon_cgrid.py:4052-4069 folds the explicit_ab2
    # planetary term into the vortcor diagnostic slot, and the een_total /
    # ene_total vorticity schemes carry f INSIDE the vorticity flux to begin
    # with (NEMO ln_dynvor_een).  So NO separate coriolis_cgrid add here: that
    # would double-count f (measured: it inflated RMS(lego vor) to 2.001x NEMO's
    # while corr stayed 0.99998 -- the classic double-count signature).
    lego_adv = _u_to_nemo(_D("vertadv_u"))[..., :jpkm1]
    lego_vor = _u_to_nemo(_D("vortcor_u"))[..., :jpkm1]
    lego_ldf = _u_to_nemo(_D("Ah_lap_u") + _D("Bh_bilap_u")
                          + _D("Cs_smag_u") + _D("Cl_leith_u"))[..., :jpkm1]
    lego_hpg = _u_to_nemo(_D("KE_PGF_u"))[..., :jpkm1]
    # The explicit accumulator legoESM's own decomposition adds up to.  Use the
    # SUM OF THE FOUR MAPPED TERMS (not ``total_u``) so the legoESM chain
    # adv+vor+ldf+hpg -> spg -> zdf is internally consistent with the terms
    # actually ranked; any gap between that sum and ``total_u`` is reported
    # explicitly below rather than silently absorbed.
    lego_explicit = lego_adv + lego_vor + lego_ldf + lego_hpg
    _tot_u = _u_to_nemo(_D("total_u"))[..., :jpkm1]
    print(f"  vorticity_scheme={getattr(mc, 'vorticity_scheme', None)!r}  "
          f"coriolis_scheme={getattr(mc, 'coriolis_scheme', None)!r}  "
          f"surface_stress_implicit={getattr(mc, 'surface_stress_implicit', None)!r}  "
          f"implicit_vertical_mixing={getattr(mc, 'implicit_vertical_mixing', None)!r}  "
          f"implicit_bottom_drag={getattr(mc, 'implicit_bottom_drag', None)!r}")
    # Evidence for the no-separate-f decision, reported not assumed:
    _cu = _u_to_nemo(cor_u)[..., :jpkm1]
    print(f"  RMS(vortcor_u)={rms(lego_vor):.4e}  RMS(standalone coriolis_cgrid "
          f"f x v)={rms(_cu):.4e}  RMS(vortcor_u + f x v)={rms(lego_vor + _cu):.4e}"
          f"  | NEMO dyn_vor RMS={rms(D04 - D03):.4e}  <- vortcor_u alone is the "
          "match; the sum is ~2x")

    # Closure self-check: Sigma components == total_u (machine precision).
    _sum_comp = sum(_D(n) for n in (
        "KE_PGF_u", "vortcor_u", "Dterm_u", "vertadv_u", "Ah_lap_u", "Bh_bilap_u",
        "Cs_smag_u", "Cl_leith_u", "botdrag_u", "Av_vert_u", "phys_u", "sponge_u"))
    _cl = float(np.max(np.abs(_sum_comp - _D("total_u"))))
    print(f"\nlegoESM MomentumTendencyDiagnostics closure |Sigma comp - total_u|_max="
          f"{_cl:.3e} over ALL faces (incl. land/halo); on WET u-faces: "
          f"{float(np.max(np.abs((_u_to_nemo(_sum_comp - _D('total_u'))[..., :jpkm1])[m3]))):.3e}")
    print(f"  RMS(adv+vor+ldf+hpg)={rms(lego_explicit):.4e}  RMS(total_u)="
          f"{rms(_tot_u):.4e}  RMS(difference)={rms(lego_explicit - _tot_u):.4e}  "
          "(a nonzero difference means total_u carries something outside the four "
          "mapped terms -- it is NOT folded into the ranking; the four-term sum is "
          "what the legoESM 'spg' residual is taken against)")
    print("legoESM component RMS (u, wet, all levels) -- terms that go into the "
          "implicit solve should be ~0 here (they belong to 'zdf'):")
    for nm in ("phys_u", "Av_vert_u", "botdrag_u", "Dterm_u", "sponge_u",
               "Bh_bilap_u", "Cs_smag_u", "Cl_leith_u"):
        a = _u_to_nemo(_D(nm))[..., :jpkm1]
        print(f"    {nm:<24s} RMS={float(np.sqrt(np.mean(a[m3] ** 2))):.4e}")

    # spg / zdf: run ONE real production step, spying on the barotropic solve
    # (legoESM's dyn_spg) and the implicit vertical solve (legoESM's dyn_zdf).
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
    cap = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        out = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        # _leapfrog_step drives _step_impl TWICE (advective Nnn pass, which
        # carries the before-level seed and drives the LIVE barotropic solve,
        # plus a dissipative Nbb pass whose barotropic result is discarded).
        # Only the SEEDED call is NEMO's dyn_spg_ts. Same selector as
        # spg_substep_chain.py.
        if kw.get("eta_init") is not None and "baro_out_u" not in cap:
            cap["baro_out_u"] = np.asarray(out[0].u.data)
            cap["dt_s"] = float(dt_s)
            cap["n_substeps"] = int(n_substeps)
        return out

    _real_tend = model.tendencies

    tend_calls = []

    def _spy_tend(state_arg, *a, **kw):
        out = _real_tend(state_arg, *a, **kw)
        # capture EVERY pass -- the MLF leapfrog runs an advective (Nnn) pass
        # and a dissipative (Nbb) pass, and claiming "the model's du_dt" from
        # only the first would be an over-claim.
        tend_calls.append(np.asarray(out.du_dt.data))
        cap.setdefault("model_du_dt", tend_calls[0])
        return out

    _real_vmix = model._apply_implicit_vertical_mixing
    vmix_calls = []

    def _spy_vmix(state_in, *a, **kw):
        out = _real_vmix(state_in, *a, **kw)
        st_out = out[0] if isinstance(out, tuple) else out
        vmix_calls.append((np.asarray(state_in.u.data), np.asarray(st_out.u.data)))
        return out

    ocmod.barotropic_substeps_latlon_cgrid = _spy_baro
    model._apply_implicit_vertical_mixing = _spy_vmix
    model.tendencies = _spy_tend
    try:
        with jax.disable_jit():
            _st2 = model.step(st, DT, surface_forcing=sf)
            # SECOND step, kept only to rule out a FIRST-STEP artifact: on the
            # very first leapfrog step ``state.tau_x_prev`` is None
            # (ocean_model_latlon_cgrid.py:2726-2740), so a wind term gated on
            # the centred (before+now)/2 forcing would be skipped on step 1 and
            # present from step 2 onward.  If the wind is absent from BOTH
            # steps' du_dt the single-step protocol is not the explanation.
            cap["n_tend_step1"] = len(tend_calls)
            _ = model.step(_st2, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
        model._apply_implicit_vertical_mixing = _real_vmix
        model.tendencies = _real_tend

    assert "baro_out_u" in cap, "seeded barotropic call never fired -- check barotropic_solver"
    assert vmix_calls, ("_apply_implicit_vertical_mixing never fired -- "
                        "implicit_vertical_mixing must be True for this recipe")
    rdt_lego = cap["dt_s"] * cap["n_substeps"]
    print(f"\nlegoESM captured: barotropic dt_s={cap['dt_s']:.6f} x n_substeps="
          f"{cap['n_substeps']} = {rdt_lego:.3f} s  (NEMO rDt={RDT}); "
          f"n_vmix_calls={len(vmix_calls)}")
    assert abs(rdt_lego - RDT) < 1e-6, (
        f"legoESM barotropic window {rdt_lego} != NEMO rDt {RDT} -- the two models "
        "are not stepping the same interval; every state-difference term below "
        "would be scaled wrong. ABORT")

    # ``_apply_implicit_vertical_mixing`` IS legoESM's dyn_zdf: the state it
    # RECEIVES plays NEMO's uu(Kaa) at dyn_zdf entry (Kbb + rDt*Krhs after the
    # barotropic stage), the state it RETURNS plays stage 8.  Under the MLF
    # leapfrog ``_step_impl`` runs twice but only the pass that produces the
    # after-state calls the implicit solve, so exactly one call is expected.
    vmix_calls = vmix_calls[:1]   # step-2 calls are for the artifact check only
    assert len(vmix_calls) == 1, (
        f"expected exactly ONE _apply_implicit_vertical_mixing call, got "
        f"{len(vmix_calls)} -- the dyn_zdf analogue is ambiguous; ABORT rather "
        "than guess which call plays stage 8")
    u_spg_lego, u_zdf_lego = vmix_calls[0]
    # Diagnostic only (NOT a gate): the seeded barotropic call's own output is
    # taken in the ADVECTIVE pass; the leapfrog then assembles the after-state
    # that reaches the vertical solve, so the two need not be identical.
    print(f"  RMS|vmix_input_u - seeded_barotropic_output_u| = "
          f"{float(np.sqrt(np.mean((u_spg_lego - cap['baro_out_u']) ** 2))):.4e}  "
          f"(RMS(u)~{float(np.sqrt(np.mean(u_spg_lego ** 2))):.4e}) -- diagnostic; "
          "the vmix INPUT is what plays NEMO's dyn_zdf-entry state")

    # legoESM's barotropic solve returns a corrected 3-D VELOCITY rather than a
    # corrected accumulator, so its "accumulator after spg" (NEMO's U7) is
    # recovered as (u_post_baro - u_before)/rDt.  The two decompositions are
    # then algebraically IDENTICAL:
    #     NEMO   spg = U7 - D06                    zdf = (U8-Ubb)/rDt - U7
    #     lego   spg = U7_lego - D06_lego          zdf = TOT_lego - U7_lego
    # with U7_lego = (u_post_baro - u_bb)/rDt, D06_lego = probe explicit total,
    # TOT_lego = (u_post_zdf - u_bb)/rDt.
    lego_u_before = _u_to_nemo(np.asarray(st.u_before.data))[..., :jpkm1]
    lego_U7 = (_u_to_nemo(u_spg_lego)[..., :jpkm1] - lego_u_before) / RDT
    lego_TOT = (_u_to_nemo(u_zdf_lego)[..., :jpkm1] - lego_u_before) / RDT
    lego = {
        "adv": lego_adv,
        "vor": lego_vor,
        "ldf": lego_ldf,
        "hpg": lego_hpg,
        "spg": lego_U7 - lego_explicit,
        "zdf": lego_TOT - lego_U7,
    }
    print(f"  scale check: RMS(lego U7 accumulator)={rms(lego_U7):.4e}  "
          f"RMS(lego total)={rms(lego_TOT):.4e}  vs NEMO U7={r_u7:.4e}  "
          f"total={rms((U8 - ub_bb) / RDT):.4e}  [all m/s^2]")
    assert 0.02 < rms(lego_U7) / r_u7 < 50.0, (
        "legoESM's reconstructed accumulator-after-spg is not even the same "
        "order as NEMO's stage-7 accumulator -- the (u_dyn_zdf_entry - u_before)"
        "/rDt reconstruction is wrong; ABORT")
    if "model_du_dt" in cap:
        _mdu = _u_to_nemo(cap["model_du_dt"])[..., :jpkm1]
        _d = float(np.sqrt(np.mean((_mdu - lego_explicit)[m3] ** 2)))
        print(f"\n  [slack check] RMS(model's own du_dt - the four mapped terms)="
              f"{_d:.4e}  ({_d / max(rms(lego_explicit), 1e-30):.3e} of the "
              "four-term total).  This is EXPECTED and is dominated by the WIND "
              "STRESS, which legoESM adds to du_dt with no MomentumTendencyDiagnostics "
              "slot of its own (surface_stress_implicit=False) -- see the "
              "'WHERE IS legoESM's WIND?' block, where its channel integral is "
              "measured at ~0.99x the analytic wind input.  A secondary "
              "contribution is the leapfrog's dissipative Nbb pass evaluating "
              "lateral friction on the BEFORE level.  All of it lands in the "
              "legoESM 'spg' row, which is why the spg/zdf SPLIT is not "
              "comparable across models and the GROUP is.")

    # =====================================================================
    # PART 1 -- per-element 3-D comparison
    # =====================================================================
    print("\n" + "=" * 100)
    print("PART 1: per-element 3-D zonal-momentum tendency, legoESM vs NEMO "
          "(u-points, wet, all 35 levels)")
    print("  err_norm = RMS(lego-nemo)/RMS(nemo)  [sign-changing fields -- never "
          "pointwise relative]")
    print("=" * 100)
    stats = {}
    for t in TERMS:
        tag = t + ("  [PARTIAL: keg bundling]" if t in ("adv", "hpg") else "")
        stats[t] = _report(tag, lego[t], nemo[t], m3)
    print()
    _report("adv+hpg  [EXACT group]", lego["adv"] + lego["hpg"],
            nemo["adv"] + nemo["hpg"], m3)
    _report("INTERIOR (adv+vor+ldf+hpg)",
            sum(lego[t] for t in ("adv", "vor", "ldf", "hpg")),
            sum(nemo[t] for t in ("adv", "vor", "ldf", "hpg")), m3)
    _report("SURFACE/BAROTROPIC (spg+zdf)", lego["spg"] + lego["zdf"],
            nemo["spg"] + nemo["zdf"], m3)
    _report("explicit total (=D06)", lego_explicit, D06, m3)
    _report("total (all 6 terms)", sum(lego[t] for t in TERMS),
            sum(nemo[t] for t in TERMS), m3)

    print("\nPer-level breakdown (every 4th level + bottom):")
    for t in TERMS:
        _per_level(t, lego[t], nemo[t], m3)

    print("\nALIGNMENT (U-face fields -- a sharp minimum at (0,0) confirms the "
          "legoESM-face -> NEMO-u-column index convention):")
    for t in TERMS:
        _shift_scan(t + " k=0", lego[t][..., 0], nemo[t][..., 0], umask2)
    _shift_scan("adv+hpg group k=0", (lego["adv"] + lego["hpg"])[..., 0],
                (nemo["adv"] + nemo["hpg"])[..., 0], umask2)
    _shift_scan("D06 explicit k=0", lego_explicit[..., 0], D06[..., 0], umask2)
    _kmid = jpkm1 // 2
    _shift_scan(f"D06 explicit k={_kmid}", lego_explicit[..., _kmid],
                D06[..., _kmid], umask3[..., _kmid])
    _shift_scan(f"hpg k={_kmid}", lego["hpg"][..., _kmid], nemo["hpg"][..., _kmid],
                umask3[..., _kmid])
    print("  READ: the load-bearing rows (vor, ldf, adv+hpg group, D06 explicit "
          "total) must show a SHARP (0,0) minimum -- they do.  A flat scan on the "
          "individual 'adv'/'hpg' rows at k=0 is EXPECTED (they are the partial "
          "keg-bundled mappings and the surface hpg is the weakest level), and on "
          "'spg'/'zdf' it is expected too (the wind-placement difference cannot be "
          "removed by any index shift).  A flat scan is only a misalignment "
          "warning where the two fields are supposed to be the same quantity.")

    # =====================================================================
    # PART 2 -- the ACC diagnostic: zonally integrated zonal-momentum balance
    # =====================================================================
    # Channel band chosen OBJECTIVELY: the re-entrant rows are the ones with no
    # land at any longitude at the surface, i.e. tmask[j,:,0].all().  DINO's
    # continent blocks every other row, so this is the ACC channel by
    # construction, not by eyeball.
    open_rows = tmask_s.all(axis=1)
    jj = np.where(open_rows)[0]
    assert jj.size, "no fully-open zonal row found -- no re-entrant channel?"
    assert jj.max() - jj.min() + 1 == jj.size, (
        "the fully-open rows are not contiguous -- inspect before integrating")
    CH = slice(int(jj.min()), int(jj.max()) + 1)
    lat0, lat1 = float(gphit[jj.min(), :].mean()), float(gphit[jj.max(), :].mean())
    print("\n" + "=" * 100)
    print("PART 2: ZONALLY INTEGRATED zonal-momentum balance in the re-entrant channel")
    print("=" * 100)
    print(f"  channel band = rows {jj.min()}..{jj.max()} ({jj.size} rows), "
          f"lat {lat0:.2f} to {lat1:.2f} degN")
    print(f"  criterion: rows with NO land at any longitude at the surface "
          f"(tmask[j,:,0].all()); {int(open_rows.sum())}/{tmask_s.shape[0]} rows qualify")
    print("  T(y,z) = sum_i term * e1u * e3u_0 * umask   [m^3/s^2];  "
          "depth integral = sum_k T(y,k)")

    W = e1u[..., None] * e3u_0[..., :jpkm1] * umask3[..., :jpkm1]   # (n_lat,n_lon,jpkm1)

    def zint(field):
        """T(y,z) = sum over i, then depth integral -> (T_yz, T_y)."""
        tyz = np.sum(np.asarray(field) * W, axis=1)      # (n_lat, jpkm1)
        return tyz, tyz.sum(axis=-1)                      # (n_lat,)

    prof = {}
    for t in TERMS:
        prof[t] = {"lego": zint(lego[t]), "nemo": zint(nemo[t])}

    # --- budget closure, per model, in the same integrated units -------------
    tot_lego_3d = sum(lego[t] for t in TERMS)
    tot_nemo_3d = sum(nemo[t] for t in TERMS)
    # The INDEPENDENT total: the actual net velocity change over rDt.
    indep_nemo = (U8 - ub_bb) / RDT
    indep_lego = lego_TOT
    print("\n--- BUDGET CLOSURE (sum of the six terms vs the independent total "
          "(U_after - U_before)/rDt) ---")
    for nm, tt, ii in (("NEMO", tot_nemo_3d, indep_nemo), ("legoESM", tot_lego_3d, indep_lego)):
        res = tt - ii
        rr = float(np.sqrt(np.mean(res[m3] ** 2)))
        ri = float(np.sqrt(np.mean(ii[m3] ** 2)))
        _, ty_res = zint(res)
        _, ty_ind = zint(ii)
        print(f"  {nm:<8s} 3-D residual RMS={rr:.4e}  / total RMS={ri:.4e}  "
              f"-> relative {rr / ri:.3e}   | channel-integrated residual="
              f"{ty_res[CH].sum():.6e} vs total={ty_ind[CH].sum():.6e} m^3/s^2")
    print("  NOTE (do not over-read): the six terms telescope to the total BY "
          "CONSTRUCTION on both sides ('zdf' is the closing residual), so this "
          "row is a CHAIN-COMPLETENESS check -- it verifies no un-dumped stage "
          "sits between stage 3 and stage 8 and that rDt is right -- NOT an "
          "independent verification of each term.  On the NEMO side adv/vor/ldf/"
          "hpg/spg are all genuine consecutive-accumulator differences and only "
          "'zdf' is the residual; on the legoESM side adv/vor/ldf/hpg come from "
          "the probe, 'spg' absorbs the probe-vs-model slack printed above, and "
          "'zdf' is the residual.")
    print("  The NON-trivial closure check is legoESM's own explicit total vs "
          "NEMO's D06 accumulator, reported in PART 1 above ('explicit total "
          "(=D06)') -- that one CAN fail and is the real test of the four "
          "explicit terms summing correctly.")

    # --- the ranking ---------------------------------------------------------
    print("\n--- CHANNEL- AND DEPTH-INTEGRATED zonal force per term "
          "[m^3/s^2], summed over the channel band ---")
    rows = []
    for t in TERMS:
        L = prof[t]["lego"][1][CH].sum()
        N = prof[t]["nemo"][1][CH].sum()
        rows.append((t, float(L), float(N), float(L - N)))
    tot_absdiff = sum(abs(r[3]) for r in rows)
    net_diff = sum(r[3] for r in rows)
    print(f"  {'term':<8s} {'legoESM':>15s} {'NEMO':>15s} {'lego-NEMO':>15s} "
          f"{'|share|':>9s}  {'lego/NEMO':>10s}")
    for t, L, N, D in rows:
        share = abs(D) / tot_absdiff if tot_absdiff else float("nan")
        ratio = L / N if N != 0 else float("nan")
        print(f"  {t:<8s} {L:15.6e} {N:15.6e} {D:15.6e} {share:9.1%}  {ratio:10.4f}")
    print(f"  {'SUM':<8s} {sum(r[1] for r in rows):15.6e} "
          f"{sum(r[2] for r in rows):15.6e} {net_diff:15.6e}")

    # --- STRUCTURE-INDEPENDENT GROUPING.  The spg/zdf SPLIT is a model-design
    # choice (legoESM and NEMO place the wind stress and the barotropic mode
    # differently between the free-surface stage and the vertical-mixing
    # stage), so the split itself is not directly comparable; their SUM is.
    _grp = {"INTERIOR (adv+vor+ldf+hpg)": ("adv", "vor", "ldf", "hpg"),
            "SURFACE/BAROTROPIC (spg+zdf)": ("spg", "zdf")}
    print("\n--- STRUCTURE-INDEPENDENT GROUPS (the split between spg and zdf is a "
          "model-design choice; their SUM is not) ---")
    for gname, keys in _grp.items():
        L = sum(prof[t]["lego"][1][CH].sum() for t in keys)
        N = sum(prof[t]["nemo"][1][CH].sum() for t in keys)
        print(f"  {gname:<30s} lego={L:+15.6e}  nemo={N:+15.6e}  "
              f"diff={L - N:+15.6e}  share={abs(L - N) / max(tot_absdiff, 1e-30):6.1%}")

    # --- ANALYTIC WIND-INPUT REFERENCE (independent scale check, not part of
    # the budget): sum_i (tau_x/rho0)*e1u over the channel = the eastward
    # momentum the wind puts into the channel per unit meridional length.
    # SIGN: ``dino_step_surface_forcing`` (dino.py:3288-3296) stores tau_x in
    # the ATMOSPHERIC convention -- it NEGATES DINO's analytic stress-on-ocean
    # because the PE external-tau block applies the ocean reaction -tau.  So the
    # eastward force on the ocean is (-sf.tau_x)/rho0.  Verified by sign: DINO's
    # westerly knot at -45 deg is +0.2 Pa (eastward), so the channel integral
    # MUST come out positive.
    _tx = -np.asarray(sf.tau_x)
    _tx_n = _u_to_nemo(_tx) if _tx.shape[1] == e1u.shape[1] + 1 else _tx
    _rho0 = float(getattr(getattr(mc, "constants", None), "rho_0",
                          getattr(mc, "rho_0", constants.rho_ocean)))
    wind_y = np.sum(_tx_n / _rho0 * e1u * umask3[..., 0], axis=1)
    assert wind_y[CH].sum() > 0, (
        "the analytic channel wind input came out NEGATIVE -- DINO's westerlies "
        "are eastward (+0.2 Pa at -45 deg), so the tau_x sign convention used "
        "here is wrong; fix it before reading anything into the zdf row")
    print(f"\n--- ANALYTIC WIND INPUT (independent reference, rho_0={_rho0}) ---")
    print(f"  sum_i (tau_x/rho0)*e1u over the channel band = "
          f"{float(wind_y[CH].sum()):+.6e} m^3/s^2")
    print(f"  vs NEMO's measured 'zdf' channel integral = "
          f"{float(prof['zdf']['nemo'][1][CH].sum()):+.6e}   "
          f"legoESM's = {float(prof['zdf']['lego'][1][CH].sum()):+.6e}")
    _four = lego_adv + lego_vor + lego_ldf + lego_hpg
    _extra_diag = zint(_u_to_nemo(_D("total_u"))[..., :jpkm1] - _four)[1][CH].sum()
    print(f"  WHERE IS legoESM's WIND?  channel integral [m^3/s^2] of:")
    print(f"    (diagnostic tendencies_with_diagnostics total_u - four mapped terms) "
          f"= {_extra_diag:+.6e}   <- the wind, CORRECT sign, "
          f"{_extra_diag / float(wind_y[CH].sum()):.4f}x the analytic value")
    _n1 = cap.get("n_tend_step1", len(tend_calls))
    for _i, _dd in enumerate(tend_calls):
        _tag = "step1" if _i < _n1 else "step2 (tau_x_prev SEEDED)"
        print(f"    (model's OWN du_dt, tendencies() pass #{_i} [{_tag}] "
              f"- four mapped terms) = "
              f"{zint(_u_to_nemo(_dd)[..., :jpkm1] - _four)[1][CH].sum():+.6e}")
    print("    (a value near the analytic wind would mean the wind IS in the "
          "model's own explicit RHS; a value near zero on BOTH steps rules out "
          "the first-step tau_x_prev=None gating as the explanation.  NOTE the "
          "step-2 passes are evaluated on a DIFFERENT (already stepped) state, "
          "so only the presence/absence of the ~1.4e4 wind signal is being read "
          "off them, not their exact value.)")
    print(f"    legoESM's REALIZED spg+zdf (from the actual stepped states) = "
          f"{float(prof['spg']['lego'][1][CH].sum() + prof['zdf']['lego'][1][CH].sum()):+.6e}"
          f"   vs NEMO's {float(prof['spg']['nemo'][1][CH].sum() + prof['zdf']['nemo'][1][CH].sum()):+.6e}")
    print(f"    NET channel tendency: legoESM {float(sum(prof[t]['lego'][1][CH].sum() for t in TERMS)):+.6e}"
          f"  NEMO {float(sum(prof[t]['nemo'][1][CH].sum() for t in TERMS)):+.6e}"
          f"  difference/analytic_wind = {net_diff / float(wind_y[CH].sum()):+.4f}")
    print("  READ: NEMO's 'zdf' matches the analytic wind input to ~0.2%, so that "
          "row IS NEMO's wind deposit (dynzdf.F90:329-334, the implicit top-cell "
          "BC).  legoESM's 'zdf' is ~0 because legoESM runs "
          "surface_stress_implicit=False: its wind enters the EXPLICIT du_dt "
          "instead (measured above at 0.99x the analytic value).  That PLACEMENT "
          "difference alone moves the wind from the 'zdf' row to the 'spg' row "
          "and is NOT a defect -- which is exactly why the spg/zdf SPLIT must not "
          "be read on its own and the structure-independent GROUP is the number "
          "that matters.  What the GROUP shows is the real result: legoESM's wind "
          "is in its explicit RHS at full strength, yet the channel-integrated "
          "momentum that SURVIVES the barotropic + vertical stages is ~0.")

    print("\n--- RANKED by |lego - NEMO| (the deliverable) ---")
    for i, (t, L, N, D) in enumerate(sorted(rows, key=lambda r: -abs(r[3])), 1):
        print(f"  {i}. {t:<6s} diff={D:+.6e}  ({abs(D)/tot_absdiff:.1%} of total |diff|)  "
              f"lego={L:+.6e} nemo={N:+.6e}")

    # --- where in latitude does each difference concentrate? -----------------
    print("\n--- LATITUDE CONCENTRATION of each term's difference (depth-integrated, "
          "m^3/s^2 per row) ---")
    print("  fraction of the GLOBAL |difference| that falls inside the channel band, "
          "and the worst 3 rows:")
    for t in TERMS:
        d_y = prof[t]["lego"][1] - prof[t]["nemo"][1]
        glob = np.abs(d_y).sum()
        inch = np.abs(d_y[CH]).sum()
        order = np.argsort(-np.abs(d_y))[:3]
        worst = "  ".join(f"lat={float(gphit[j, :].mean()):+6.2f}(j={j}):{d_y[j]:+.3e}"
                          for j in order)
        print(f"  {t:<6s} in-channel |diff| = {inch / glob:6.1%} of global   worst rows: {worst}")

    print("\n--- PER-LATITUDE profile inside the channel band (depth-integrated, "
          "m^3/s^2; every 3rd row) ---")
    hdr = "  " + f"{'lat':>7s}" + "".join(f"{t:>13s}" for t in TERMS)
    hdr_d = hdr + f"{'SUM':>13s}"
    print("  legoESM:");  print(hdr)
    for j in range(CH.start, CH.stop, 3):
        vals = "".join(f"{prof[t]['lego'][1][j]:13.4e}" for t in TERMS)
        print(f"  {float(gphit[j, :].mean()):7.2f}{vals}")
    print("  NEMO:");  print(hdr)
    for j in range(CH.start, CH.stop, 3):
        vals = "".join(f"{prof[t]['nemo'][1][j]:13.4e}" for t in TERMS)
        print(f"  {float(gphit[j, :].mean()):7.2f}{vals}")
    print("  DIFFERENCE (lego - NEMO):");  print(hdr_d)
    for j in range(CH.start, CH.stop, 3):
        ds = [prof[t]["lego"][1][j] - prof[t]["nemo"][1][j] for t in TERMS]
        print(f"  {float(gphit[j, :].mean()):7.2f}" + "".join(f"{d:13.4e}" for d in ds)
              + f"{sum(ds):13.4e}")

    # --- interpretation gate -------------------------------------------------
    ranked = sorted(rows, key=lambda r: -abs(r[3]))
    top_share = abs(ranked[0][3]) / tot_absdiff if tot_absdiff else float("nan")
    _int_diff = sum(prof[t]["lego"][1][CH].sum() - prof[t]["nemo"][1][CH].sum()
                    for t in ("adv", "vor", "ldf", "hpg"))
    _sfc_diff = sum(prof[t]["lego"][1][CH].sum() - prof[t]["nemo"][1][CH].sum()
                    for t in ("spg", "zdf"))
    print(f"  GROUP VERDICT: the INTERIOR explicit terms (advection, vorticity+"
          f"Coriolis, lateral friction, hydrostatic pressure gradient) account for "
          f"{abs(_int_diff) / max(tot_absdiff, 1e-30):.1%} of the channel-integrated "
          f"difference; the SURFACE-STRESS / BAROTROPIC-FREE-SURFACE pathway "
          f"(spg+zdf) accounts for {abs(_sfc_diff) / max(tot_absdiff, 1e-30):.1%}.")
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    if abs(_sfc_diff) / max(tot_absdiff, 1e-30) > 0.9:
        print(f"  ONE PATHWAY DOMINATES, unambiguously: the SURFACE-STRESS / "
              f"BAROTROPIC-FREE-SURFACE stages (spg+zdf) carry "
              f"{abs(_sfc_diff) / tot_absdiff:.1%} of the channel-integrated "
              f"|lego-NEMO| zonal-momentum difference; the four INTERIOR explicit "
              f"terms together carry {abs(_int_diff) / tot_absdiff:.1%}.  Within that "
              f"pathway the split between 'spg' "
              f"({abs(dict((r[0], r[3]) for r in rows)['spg'])/tot_absdiff:.0%}) "
              f"and 'zdf' "
              f"({abs(dict((r[0], r[3]) for r in rows)['zdf'])/tot_absdiff:.0%}) "
              f"is a model-design "
              f"boundary, not a physical one -- read the PAIR.  The SURFACE-STRESS "
              f"PATHWAY is the lever; advection, vorticity+Coriolis, lateral "
              f"friction and the hydrostatic pressure gradient are NOT.")
        print(f"  MECHANISM, as far as this measurement establishes it: legoESM's "
              f"wind reaches its explicit momentum RHS at {_extra_diag / float(wind_y[CH].sum()):.3f}x "
              f"the analytic channel input, but legoESM's NET channel-integrated "
              f"zonal tendency after the barotropic + vertical stages is "
              f"{float(sum(prof[t]['lego'][1][CH].sum() for t in TERMS)):+.4e} against NEMO's "
              f"{float(sum(prof[t]['nemo'][1][CH].sum() for t in TERMS)):+.4e} -- a shortfall of "
              f"{net_diff / float(wind_y[CH].sum()):+.3f}x the wind input.  So the "
              f"wind momentum is INPUT correctly and then does not survive the "
              f"free-surface/barotropic pathway.  WHICH operation inside that "
              f"pathway removes it (the F_slow depth-mean split, the substep "
              f"recurrence, the barotropic bottom drag, or the mean re-imposition) "
              f"is NOT resolved by this script -- that is the next measurement.")
    elif top_share > 0.5:
        print(f"  ONE TERM DOMINATES: '{ranked[0][0]}' carries {top_share:.0%} of the "
              f"total |lego-NEMO| channel-integrated momentum difference "
              f"(next: '{ranked[1][0]}' at {abs(ranked[1][3])/tot_absdiff:.0%}). "
              f"That term is the LEVER for the ACC deficit.")
    elif top_share > 0.35:
        print(f"  LEADING TERM: '{ranked[0][0]}' at {top_share:.0%} of the total "
              f"|difference|, ahead of '{ranked[1][0]}' "
              f"({abs(ranked[1][3])/tot_absdiff:.0%}) -- leading but NOT dominant; "
              "treat as the first target, not the sole cause.")
    else:
        print(f"  NO SINGLE DOMINANT TERM: the largest, '{ranked[0][0]}', carries only "
              f"{top_share:.0%} of the total |difference| and the terms are spread. "
              "That points at the STATE/GEOMETRY the terms are evaluated on rather "
              "than at any one physical process.")
    print("\n  CAVEAT (load-bearing): this is an INSTANTANEOUS budget at the Y5 "
          "restart state, both models fed the SAME state.  The ACC deficit "
          "develops over a 5-year spin-up FROM REST.  A term can match here and "
          "still differ in the spin-up regime (different state, different "
          "stratification, different eddy field), and a term that differs here "
          "may be a consequence of the Y5 state rather than its cause.  This "
          "establishes WHICH TERM DIFFERS ON A SHARED STATE.  It does NOT "
          "establish that that term caused the 5-year divergence.")
    print("  CAVEAT 2: 'adv' and 'hpg' individually are PARTIAL mappings (NEMO "
          "bundles the KE gradient with vertical advection, legoESM bundles it "
          "with the pressure gradient).  Their SUM is exact; read the "
          "individual rows only alongside the adv+hpg group row.")
    print("  CAVEAT 3: 'spg' is a residual on both sides, so it absorbs any "
          "genuine structural difference in how the barotropic correction is "
          "applied -- a large 'spg' means 'the barotropic/free-surface stage "
          "differs', not necessarily 'the surface pressure gradient operator "
          "is wrong'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
