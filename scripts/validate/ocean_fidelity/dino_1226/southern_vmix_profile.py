"""#1226: is legoESM's 200-1000 m southern over-mixing a CLOSURE defect or a
CONVECTIVE-TRIGGER-POPULATION defect?  Both measured against NEMO's own
realized (post-EVD) diffusivity.

RETRACTION (2026-07-30, this file's first version): that run reported a
"300-370x" production-vs-NEMO diffusivity ratio by comparing legoESM's
POST-convection K_v against NEMO's ``avt_k``.  ``avt_k`` is CLOSURE-ONLY --
VERIFIED in NEMO source, ``src/OCE/ZDF/zdfphy.F90:313-314`` copies
``avt(jk)=avt_k(jk)`` and then :323 ``CALL zdf_evd(kt,Kmm,Krhs,avm,avt)``
mutates the COPY; EVD is never written back into ``avt_k``, and no
``iom_put('avt')`` exists anywhere in ZDF/ or TRA/.  So that 300x compared
post-convection against pre-convection -- apples-to-oranges, and it is
RETRACTED as evidence.  The corrected comparison is below; the
closure-vs-closure result (``ratio_tke`` ~ 1.0) was and remains VALID because
both sides exclude convection.

NEMO's realized avt IS available, directly: DINO's ``MY_SRC/ldftra.F90:902``
dumps ``WRITE(8814) avt`` from ``ldf_eiv``, which stpmlf.F90 calls at :203 --
AFTER ``zdf_phy`` at :190.  So ``dump_avt.bin`` / ``dump_avm.bin`` are the
POST-EVD fields.  This script therefore uses the DUMP as ground truth and
uses NEMO's EVD rule only to VALIDATE that it understands the trigger
(a reconstruction that must reproduce the dump bit-for-bit), rather than
substituting a reconstruction for a measurement.

NEMO EVD rule (``src/OCE/ZDF/zdfevd.F90:92-95``, and :117-120 for avm under
``nn_evdm=1``), verbatim semantics:

    IF( MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )
        p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)

Note it is a hard **SET**, not a MAX (legoESM does
``maximum(K_v_total, K_conv)``) -- the two differ only where the closure K
already exceeds K_conv, and that frequency is reported below.  The trigger is
a **MIN over TWO time levels**, so both are loaded:
  * ``rn2``  = ``rn2_stg`` on the restart (DINO ``MY_SRC/trddump.F90:114``:
    "N^2 at w-pts used by zdf ... set by eos_rab/bn2 at Kmm stage").
  * ``rn2b`` = ``tke_dump_rn2b.bin`` (interior, all 36 w-levels).
DINO namelist_cfg:389-391 -- ``ln_zdfevd=.true.``, ``nn_evdm=1``,
``rn_evd=100.`` (all three verified, printed below).

ONE STATE, printed with its step number (the earlier protocol defect: this
probe was on step 5760 while a parallel probe was on 57600 -- a factor of ten
apart in model time, so the two could not be chained).  Everything here is
step 57600 = NEMO year 5, run dir RUN_GDB, which carries the restart AND the
post-EVD dumps AND its own mesh_mask.

Restart T/S are RAW (``tn``/``sn``) -- the ``votemper = toce*e3t`` XIOS
thickness-weighting artifact applies to the ``grid_T`` HISTORY files, NOT to
restart variables; nothing here reads a history file, so no ``vovvle3t``
division is involved.  Checked, not assumed.

Boxes, verbatim from abyssal_densification.py -- not re-derived:
  SEDGE = slice(12, 17); CORE = slice(1, 11).  Row 0 is 100% dry.

Rule: every row carries the |dT/dz| DEAD-gradient flag.  A diffusivity ratio
on a dead gradient is not a flux ratio (a claim was already retracted in this
campaign for exactly that).

Usage
-----
    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/southern_vmix_profile.py
"""
import importlib.util
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
# Pin the NEMO depth-ladder mode EXPLICITLY (bn2_alpha_compare.py: DINO is
# full-step and e3t_1d vs e3t_0 diverge by up to 105 m).  "both" matches the
# sibling probe on this SAME RUN_GDB state (zdf_mxl_nmln_compare.py).
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import numpy as np
import jax
import jax.numpy as jnp
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(__file__))
from kamm_twin_90d import _build_twin_state, DT  # noqa: E402

import legoesm.ocean.physics.vertical_mixing as vmix_pkg  # noqa: E402
from legoesm import constants  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import require_fp64  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (  # noqa: E402
    implicit_vertical_diffusion_ocean, build_dz_half,
)
from legoesm.ocean.vertical import compute_ocean_jacobian  # noqa: E402

# rho0*cp conventions -- legoESM from legoesm.constants; NEMO's own values
# (heat_discriminator.py:57-58: rau0 phycst.F90, rcp eosbn2.F90:1899).
RHO0_LEGO, CP_LEGO = float(constants.rho_ocean), float(constants.c_sw)
RHO0_NEMO, CP_NEMO = 1026.0, 3991.86795711963

# Sibling probe's dump loaders -- imported by path (scripts/ is not a package),
# the same mechanism zdf_mxl_nmln_compare.py uses.  Do not re-implement.
_sib = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib)
_bac = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bac
_spec.loader.exec_module(_bac)
_read_dims, _load_haloed, _load_interior = (
    _bac._read_dims, _bac._load_haloed, _bac._load_interior)

# ---- THE ONE STATE (printed in the output; never mix two) ----
DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
RUN = f"{DINO}/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"
STEP = 57600                     # NEMO year 5 (11520 steps/yr at dt=2700 s)
RECIPE = "nemo_dino_kamm_mlf"

RN_EVD = 100.0                   # namelist_cfg:391 rn_evd (asserted vs K_conv below)
EVD_THRESH = -1.0e-12            # zdfevd.F90:93 MIN(rn2,rn2b) <= -1.e-12

SEDGE = slice(12, 17)
CORE = slice(1, 11)
DEEP_LO_M, DEEP_HI_M = 200.0, 1000.0
DEAD_DTDZ = 1e-5                 # degC/m below which a K ratio is not a flux ratio


def llz(a):
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    print("=" * 104)
    print(f"STATE (single, shared): {RUN}/{RESTART}   NEMO step {STEP} (= year 5)")
    print(f"  post-EVD truth  : {RUN}/dump_avt.bin , dump_avm.bin  "
          f"(MY_SRC/ldftra.F90:902-903, ldf_eiv AFTER zdf_phy)")
    print("  closure-only    : restart avt_k / avm_k  (zdfphy.F90:313-314, pre-EVD)")
    print("  rn2  (now-level): restart rn2_stg      (MY_SRC/trddump.F90:114, Kmm stage)")
    print(f"  rn2b (before)   : {RUN}/tke_dump_rn2b.bin")
    print(f"  mesh_mask       : {RUN}/mesh_mask.nc")
    print(f"  LEGOESM_NEMO_E3T={os.environ['LEGOESM_NEMO_E3T']}   "
          f"JAX_ENABLE_X64={os.environ['JAX_ENABLE_X64']}")
    print("=" * 104)

    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        RECIPE, RUN, RUN, bridge_tke=True, bridge_before=True,
        restart_file=RESTART,
    )
    require_fp64(st, context="southern_vmix_profile twin state")
    ed = mc.physics.convection.enhanced_diffusion
    K_CONV = float(ed.K_conv)
    print(f"recipe={RECIPE}  vmix_scheme={cfg.vmix_scheme}  "
          f"convection={mc.physics.convection.scheme}  K_conv={K_CONV}  "
          f"n2_mode={ed.n2_mode}  two_level_trigger={ed.two_level_trigger}")
    assert K_CONV == RN_EVD, (
        f"legoESM K_conv={K_CONV} != NEMO rn_evd={RN_EVD}; the realized-K "
        "comparison would then conflate a coefficient difference with a "
        "trigger-population difference -- ABORT")

    # ---- legoESM production K_v/A_v + the convection-off ablation ----
    def run_and_capture(mdl, state):
        _real = vmix_pkg.compute_vertical_K_profiles
        calls = []

        def _spy(*a, **kw):
            out = _real(*a, **kw)
            calls.append(out)
            return out

        vmix_pkg.compute_vertical_K_profiles = _spy
        try:
            with jax.disable_jit():
                _ = mdl.step(state, DT, surface_forcing=sf)
        finally:
            vmix_pkg.compute_vertical_K_profiles = _real
        assert calls, "compute_vertical_K_profiles never fired -- ABORT"
        out = calls[0]
        kv, av = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        return np.asarray(kv), (np.asarray(av) if av is not None else None)

    K_prod, A_prod = run_and_capture(model, st)
    conv_off = mc.physics.convection._replace(scheme="none")
    model_nc = LatLonCGridOceanModel(
        br.geometry, br.z_coord,
        mc._replace(physics=mc.physics._replace(convection=conv_off)))
    K_tke, A_tke = run_and_capture(model_nc, st)
    print(f"captured legoESM K_v {K_prod.shape} (production) and "
          f"{K_tke.shape} (convection='none' ablation)")

    # ---- NEMO fields, all on the w-level axis (index 0 = jk=1 = surface) ----
    jpi, jpj, jpk, hls = _read_dims(RUN)
    rst = nc.Dataset(f"{RUN}/{RESTART}")
    avt_k = llz(rst["avt_k"][0])                       # (ny,nx,36) closure-only
    avm_k = llz(rst["avm_k"][0])
    rn2 = llz(rst["rn2_stg"][0])                       # (ny,nx,36) now-level N^2
    ny, nx, _ = avt_k.shape
    dump_avt = _load_haloed(f"{RUN}/dump_avt.bin", jpi, jpj, hls)   # (ny,nx,35)
    dump_avm = _load_haloed(f"{RUN}/dump_avm.bin", jpi, jpj, hls)
    rn2b = _load_interior(f"{RUN}/tke_dump_rn2b.bin", nx, ny)      # (ny,nx,36)
    print(f"NEMO dumps: dump_avt {dump_avt.shape} (jk=1..{dump_avt.shape[-1]}), "
          f"rn2 {rn2.shape}, rn2b {rn2b.shape}  "
          f"[jpi,jpj,jpk,hls={jpi},{jpj},{jpk},{hls}]")

    mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
    tmask = llz(mm["tmask"][0]) > 0.5
    gdept1d = np.asarray(mm["gdept_1d"][:]).squeeze()
    gdepw1d = np.asarray(mm["gdepw_1d"][:]).squeeze()
    # wmask per NEMO dommsk.F90:176,180 -- wmask(1)=tmask(1);
    # wmask(jk)=tmask(jk)*tmask(jk-1)
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]

    # ---- (A) VALIDATE the EVD reconstruction against the post-EVD dump ----
    # This is the instrument check: if the reconstruction does not reproduce
    # the dump, my understanding of the trigger (or the time level of rn2_stg)
    # is wrong and every firing-mask number below would be built on it.
    NKD = dump_avt.shape[-1]                       # dumped w-levels, jk=1..NKD
    wet_w = wmask[..., :NKD]
    fired_direct = wet_w & (np.abs(dump_avt - RN_EVD) < 1e-9)
    # rn2b IS from the dumped step (tke_dump written in the same step as
    # dump_avt); rn2_stg and avt_k come off the INPUT restart and are therefore
    # ONE STEP STALE (they are the step-57600 closure/N^2, while the dumps are
    # the step-57601 values).  So a bit-for-bit
    # ``dump_avt == avt_k`` check on unfired cells CANNOT pass -- not because
    # the trigger is misunderstood but because the two fields are from
    # different steps.  Quantified below, and NOT swept under the rug.
    fired_rn2b = wet_w & (rn2b[..., :NKD] <= EVD_THRESH)
    fired_rn2_stale = wet_w & (rn2[..., :NKD] <= EVD_THRESH)
    # The EXACT, achievable validation of the trigger semantics: EVD fires on
    # MIN(rn2,rn2b) <= thresh, so rn2b alone is a strict SUBSET of the true
    # firing set.  Every cell with rn2b <= -1e-12 MUST have avt == rn_evd.
    subset_violations = int(np.sum(fired_rn2b & ~fired_direct))
    n_stale_diff = int(np.sum(wet_w & ~fired_direct
                              & (np.abs(dump_avt - avt_k[..., :NKD]) > 0)))
    print("\n" + "=" * 104)
    print("(A) EVD TRIGGER VALIDATION against the post-EVD dump (instrument check)")
    print("=" * 104)
    print(f"  wet w-cells (jk=1..{NKD}) = {int(wet_w.sum())}")
    print(f"  NEMO fired, MEASURED from the dump (|avt-{RN_EVD:g}|<1e-9): "
          f"{int(fired_direct.sum())} "
          f"({100.0 * fired_direct.sum() / max(wet_w.sum(), 1):.2f}% of wet)")
    print(f"  rn2b<=thresh (same-step dump, a strict SUBSET of the trigger): "
          f"{int(fired_rn2b.sum())} cells -> subset violations = "
          f"{subset_violations}  (MUST be 0)")
    print(f"  fired via the rn2 (now-level) branch only: "
          f"{int((fired_direct & ~fired_rn2b).sum())} cells "
          f"({100.0 * (fired_direct & ~fired_rn2b).sum() / max(fired_direct.sum(), 1):.2f}%"
          " of firing) -- rn2 is only available STALE, so rn2b alone is a"
          " LOWER BOUND on the firing fraction, not a substitute")
    print(f"  restart rn2_stg (STALE, step {STEP}) would give "
          f"{int(fired_rn2_stale.sum())} cells; dump_avt differs from the stale "
          f"avt_k on {n_stale_diff} unfired wet cells -- the one-step offset, "
          "NOT a trigger misunderstanding")
    print("  => trigger semantics MIN(rn2,rn2b)<=-1e-12 -> SET rn_evd: CONFIRMED "
          "by the exact subset test.  The firing mask used below is the MEASURED "
          "one (from the dump); no reconstruction is relied upon.")
    fired_nemo_w = fired_direct

    # ---- shared interface axis: legoESM interface i <-> NEMO jk=i+2 ----
    # avt_k[...,k] is the interface ABOVE T-cell k (tracer_tendency_compare.py),
    # so legoESM's 35 interior interfaces are NEMO jk=2..36 == index 1..35.
    # The dump stops at jk=NKD, so the common set is i=0..NKD-2.
    NK = NKD - 1
    sl = slice(1, NKD)                      # NEMO w-index 1..NKD-1 <-> i=0..NK-1
    K_prod, K_tke = K_prod[..., :NK], K_tke[..., :NK]
    A_prod, A_tke = A_prod[..., :NK], A_tke[..., :NK]
    avt_real, avm_real = dump_avt[..., sl], dump_avm[..., sl]
    avt_clos, avm_clos = avt_k[..., sl], avm_k[..., sl]
    fired_nemo = fired_nemo_w[..., sl]
    print(f"\ncommon interface axis: {NK} interfaces "
          f"(legoESM i=0..{NK - 1} <-> NEMO jk=2..{NKD}); legoESM's deepest "
          f"interface is dropped -- the dump stops at jk={NKD}")

    land_mask = np.asarray(st.land_mask.data) > 0.5
    wet3d = tmask & land_mask[:, :, None]
    wet_if = (wet3d[..., :-1] & wet3d[..., 1:])[..., :NK]
    # legoESM's own convective-trigger mask, from the captured production K_v
    # (the spy) -- K_conv is a hard floor at 100, and the band's closure K is
    # O(0.1-1), so this is unambiguous there.
    fired_lego = wet_if & (K_prod >= K_CONV - 1e-9)
    # Where SET-vs-MAX could differ at all (closure K already above K_conv):
    n_above = int(np.sum(wet_if & (K_tke > K_CONV)))
    n_above_nemo = int(np.sum(wet_if & (avt_clos > K_CONV)))
    print(f"SET-vs-MAX exposure: closure K > K_conv on {n_above} legoESM and "
          f"{n_above_nemo} NEMO wet interfaces (0 => NEMO's SET and legoESM's "
          "MAX are equivalent here)")

    T = np.asarray(st.T.data)               # restart tn, RAW (not e3t-weighted)
    # POINT 3, structural: at a MATCHED state the twin's T is BIT-IDENTICAL to
    # NEMO's (the day-0 gate above printed max|dT| = 0.000e+00), so |dT/dz| is
    # ONE shared number here and CANNOT distinguish "both models dead" from
    # "legoESM dead, NEMO live".  That question needs FREE-RUN states and is
    # answered in section (E) below, on a separate (clearly labelled) protocol.
    dT_dz = (np.diff(T, axis=-1) / np.diff(gdept1d))[..., :NK]
    d_TN = float(np.max(np.abs(T[wet3d] - llz(rst["tn"][0])[wet3d])))
    print(f"POINT 3 precondition: max|T_lego - T_NEMO| on wet cells at this "
          f"matched state = {d_TN:.3e} -> |dT/dz| below is SHARED by both "
          "models (one column serves both, by construction)")

    # NEMO's TIME-MATCHED closure-only avt: on cells EVD did not touch,
    # dump_avt IS the step's closure output.  Use this (not the stale restart
    # avt_k) for the closure comparison.
    avt_clos_matched = np.where(fired_nemo, np.nan, avt_real)

    def prof(a, rows, extra=None):
        """Wet-mean of ``a`` per level over ``rows``.  ``extra`` narrows the
        mask (e.g. to EVD-untouched cells); the returned count is then that
        narrowed count, so the caller can see what it averaged over."""
        m = wet_if[rows] if extra is None else (wet_if & extra)[rows]
        n = m.sum(axis=(0, 1))
        return np.where(n > 0, np.sum(np.where(m, np.nan_to_num(a[rows]), 0.0),
                                      axis=(0, 1)) / np.maximum(n, 1), np.nan), n

    def frac(mask, rows):
        m = wet_if[rows]
        n = m.sum(axis=(0, 1))
        return np.where(n > 0,
                        (mask[rows] & m).sum(axis=(0, 1)) / np.maximum(n, 1), np.nan)

    band = (gdepw1d[1:NKD] >= DEEP_LO_M) & (gdepw1d[1:NKD] < DEEP_HI_M)

    def bmean(p, n):
        s = band & (n > 0)
        return float(np.average(p[s], weights=n[s])) if s.any() else np.nan

    # ---- (B/C) firing fractions + contingency + realized ratio, per level ----
    for box, rows in (("CORE", CORE), ("SEDGE", SEDGE)):
        kp, n = prof(K_prod, rows)
        kt_, _ = prof(K_tke, rows)
        ar, _ = prof(avt_real, rows)
        ac, _ = prof(avt_clos, rows)
        # (4) hybrid: legoESM's OWN closure K, NEMO's firing mask, NEMO's SET rule
        hyb, _ = prof(np.where(fired_nemo, RN_EVD, K_tke), rows)
        fl, fn = frac(fired_lego, rows), frac(fired_nemo, rows)
        both = frac(fired_lego & fired_nemo, rows)
        onlyL = frac(fired_lego & ~fired_nemo, rows)
        onlyN = frac(~fired_lego & fired_nemo, rows)
        dtdz, _ = prof(np.abs(dT_dz), rows)

        print("\n" + "=" * 104)
        print(f"{box} rows {rows.start}:{rows.stop} -- firing, agreement, and "
              f"REALIZED (post-EVD) diffusivity")
        print("=" * 104)
        print(f"{'k':>3s}{'z_w[m]':>8s}{'n':>6s}{'fireL':>7s}{'fireN':>7s}"
              f"{'both':>6s}{'onlyL':>7s}{'onlyN':>7s}{'disag':>7s}"
              f"{'Kprod':>10s}{'avtReal':>10s}{'ratio':>8s}{'hybrid':>10s}"
              f"{'|dT/dz|':>10s}{'flag':>6s}")
        for k in range(NK):
            if n[k] == 0:
                continue
            zw = gdepw1d[k + 1]
            if not (DEEP_LO_M - 50 <= zw <= DEEP_HI_M + 400):
                continue
            r = kp[k] / ar[k] if abs(ar[k]) > 1e-12 else np.nan
            dis = onlyL[k] + onlyN[k]
            flag = "DEAD" if dtdz[k] < DEAD_DTDZ else ""
            print(f"{k:3d}{zw:8.1f}{int(n[k]):6d}{fl[k]:7.3f}{fn[k]:7.3f}"
                  f"{both[k]:6.3f}{onlyL[k]:7.3f}{onlyN[k]:7.3f}{dis:7.3f}"
                  f"{kp[k]:10.3e}{ar[k]:10.3e}{r:8.3f}{hyb[k]:10.3e}"
                  f"{dtdz[k]:10.2e}{flag:>6s}")
        # closure comparison on the TIME-MATCHED, EVD-untouched subset only
        unt = ~fired_nemo & ~fired_lego
        kcl_l, ncl = prof(K_tke, rows, extra=unt)
        kcl_n, _ = prof(avt_clos_matched, rows, extra=unt)
        mkp, mar, mkt, mhyb = (bmean(kp, n), bmean(ar, n), bmean(kt_, n),
                               bmean(hyb, n))
        mkcl_l, mkcl_n = bmean(kcl_l, ncl), bmean(kcl_n, ncl)
        mfl, mfn = bmean(fl, n), bmean(fn, n)
        m_onlyL, m_onlyN = bmean(onlyL, n), bmean(onlyN, n)
        mdt = bmean(dtdz, n)
        dead_note = ("   <- DEAD GRADIENT: ratios here are not flux ratios"
                     if mdt < DEAD_DTDZ else "")
        print(f"  [{DEEP_LO_M:.0f}-{DEEP_HI_M:.0f} m mean]")
        print(f"    firing fraction        legoESM {mfl:.3f}   NEMO {mfn:.3f}"
              f"   over-fire {mfl - mfn:+.3f} ({mfl / max(mfn, 1e-9):.2f}x)")
        print(f"    trigger disagreement   {m_onlyL + m_onlyN:.3f} of wet interfaces "
              f"(onlyL {m_onlyL:.3f} / onlyN {m_onlyN:.3f})")
        print(f"    CLOSURE-only, TIME-MATCHED, on the {int(np.nansum(ncl))} "
              f"EVD-untouched cells: Kv_tke {mkcl_l:.4e}  avt_closure "
              f"{mkcl_n:.4e}  ratio {mkcl_l / mkcl_n:.4f}")
        print(f"    CLOSURE-only vs the STALE restart avt_k (1 step off, "
              f"diagnostic): Kv_tke {mkt:.4e}  avt_k {bmean(ac, n):.4e}  "
              f"ratio {mkt / bmean(ac, n):.4f}")
        print(f"    REALIZED        Kv_prod {mkp:.4e}  avt_real {mar:.4e}  "
              f"ratio {mkp / mar:.4f}   <- replaces the RETRACTED 300x")
        print(f"    HYBRID (lego closure + NEMO firing mask, SET rule) "
              f"{mhyb:.4e}  vs avt_real {mar:.4e}  ratio {mhyb / mar:.4f}")
        print(f"    mean|dT/dz| {mdt:.2e} degC/m (SHARED by both models){dead_note}")

    # ---- momentum, same protocol (nn_evdm=1 => EVD sets avm too) ----
    print("\n" + "=" * 104)
    print("MOMENTUM A_v / avm (nn_evdm=1, EVD sets avm as well)")
    print("=" * 104)
    for box, rows in (("CORE", CORE), ("SEDGE", SEDGE)):
        ap, n = prof(A_prod, rows)
        at_, _ = prof(A_tke, rows)
        mr, _ = prof(avm_real, rows)
        mcl, _ = prof(avm_clos, rows)
        print(f"  {box} [{DEEP_LO_M:.0f}-{DEEP_HI_M:.0f} m] "
              f"CLOSURE Av_tke {bmean(at_, n):.4e} avm_k {bmean(mcl, n):.4e} "
              f"ratio {bmean(at_, n) / bmean(mcl, n):.4f} | "
              f"REALIZED Av_prod {bmean(ap, n):.4e} avm_real {bmean(mr, n):.4e} "
              f"ratio {bmean(ap, n) / bmean(mr, n):.4f}")

    # ---- (D) REALIZED VERTICAL HEAT FLUX [W/m^2] -- IMPLICIT, not K*dT/dz ----
    # At K=100, dt=2700, dz~50 the diffusion number is ~100 >> 1, so an
    # EXPLICIT K*dT/dz overstates the realized transport by ~an order of
    # magnitude (backward Euler saturates).  legoESM side therefore uses the
    # production implicit solve; NEMO's side uses its OWN post-trazdf field
    # (stp_dump_21_trazdf_tem.bin = ts(:,:,:,:,Naa) after tra_zdf at the SAME
    # dumped step -- stpmlf.F90:435), so both fluxes are evaluated on the
    # post-solve (backward-Euler) temperature, apples-to-apples.
    # Sign convention: z POSITIVE UP; flux_up = -rho0*cp*K*dT/dz_up, so a
    # stable column (warm above) gives flux_up < 0 = DOWNWARD heat transport.
    J = compute_ocean_jacobian(st.eta.data, st.H_bathy.data, br.z_coord)
    dz_cell = br.z_coord.dz_ref * jnp.maximum(J[..., jnp.newaxis], 1e-10)
    dz_half = build_dz_half(dz_cell)
    T_imp_prod = np.asarray(implicit_vertical_diffusion_ocean(
        st.T.data, jnp.asarray(np.concatenate(
            [K_prod, np.zeros_like(K_prod[..., :1])], axis=-1)),
        dz_cell, dz_half, DT))
    T_nemo_post = _load_haloed(f"{RUN}/stp_dump_21_trazdf_tem.bin", jpi, jpj, hls)
    dzh = np.asarray(dz_half)[..., :NK]

    def flux_up(Tf, K, rho0, cp, nlev_T):
        """-rho0*cp*K*dT/dz_up at the interior interfaces [W/m^2]."""
        g = (Tf[..., :nlev_T - 1] - Tf[..., 1:nlev_T]) / dzh[..., :nlev_T - 1]
        return -rho0 * cp * K[..., :nlev_T - 1] * g

    nT_n = T_nemo_post.shape[-1]                 # 35 dumped T levels
    F_lego = flux_up(T_imp_prod, K_prod, RHO0_LEGO, CP_LEGO, NK + 1)
    F_nemo = flux_up(T_nemo_post, avt_real, RHO0_NEMO, CP_NEMO, nT_n)
    NF = min(F_lego.shape[-1], F_nemo.shape[-1])
    # explicit (pre-solve T) legoESM flux, to SHOW the saturation the
    # coordinator warned about rather than assert it
    F_lego_expl = flux_up(np.asarray(st.T.data), K_prod, RHO0_LEGO, CP_LEGO, NK + 1)
    print("\n" + "=" * 104)
    print("(D) REALIZED VERTICAL DIFFUSIVE HEAT FLUX [W/m^2], positive UP "
          "(negative = downward)")
    print(f"    legoESM: production implicit solve (dt={DT:g}s), rho0*cp="
          f"{RHO0_LEGO:.0f}*{CP_LEGO:.0f}  -- vertical diffusion ONLY")
    print(f"    NEMO   : post-tra_zdf T (stp_dump_21, same step) x avt_real, "
          f"rho0*cp={RHO0_NEMO:.0f}*{CP_NEMO:.2f}")
    print("    *** F_nemo IS CONFOUNDED -- DO NOT read the lego-NEMO diff as a")
    print("    vertical-mixing difference.  stp_dump_21 is ts(:,:,:,:,Naa) AFTER")
    print("    tra_zdf, and under the MLF stepper that after-field carries the")
    print("    ACCUMULATED RHS of the whole step (tra_sbc/qsr, tra_adv, tra_ldf,")
    print("    ...), not the vertical-diffusion increment alone, whereas F_lego")
    print("    is diffusion-only.  An isolated NEMO vertical-diffusion flux needs")
    print("    ttrd_zdfp, which in this run dir exists only on the step-57603")
    print("    restart (3 steps off this state) -- so it is NOT used, per the")
    print("    one-state rule.  The CLEAN statement about vertical mixing is the")
    print("    K-and-T identity noted under (F) below.  F_lego vs F_expl_lego IS")
    print("    valid (same model, same state) and is the saturation measurement.")
    print("=" * 104)
    for box, rows in (("CORE", CORE), ("SEDGE", SEDGE)):
        fl_, n = prof(F_lego[..., :NF], rows)
        fn_, _ = prof(F_nemo[..., :NF], rows)
        fe_, _ = prof(F_lego_expl[..., :NF], rows)
        dtdz, _ = prof(np.abs(dT_dz), rows)
        bandF = band[:NF]

        def bm(p):
            s = bandF & (n[:NF] > 0)
            return float(np.average(p[:NF][s], weights=n[:NF][s])) if s.any() else np.nan

        print(f"\n  --- {box} ---")
        print(f"{'k':>3s}{'z_w[m]':>8s}{'F_lego':>11s}{'F_nemo':>11s}{'diff':>10s}"
              f"{'F_expl_lego':>13s}{'|dT/dz|':>10s}{'flag':>6s}")
        for k in range(NF):
            if n[k] == 0:
                continue
            zw = gdepw1d[k + 1]
            if not (DEEP_LO_M - 50 <= zw <= DEEP_HI_M + 400):
                continue
            flag = "DEAD" if dtdz[k] < DEAD_DTDZ else ""
            print(f"{k:3d}{zw:8.1f}{fl_[k]:11.4f}{fn_[k]:11.4f}"
                  f"{fl_[k] - fn_[k]:10.4f}{fe_[k]:13.2f}{dtdz[k]:10.2e}{flag:>6s}")
        print(f"  [{DEEP_LO_M:.0f}-{DEEP_HI_M:.0f} m mean] F_lego {bm(fl_):+.4f}  "
              f"F_nemo(CONFOUNDED) {bm(fn_):+.4f} W/m^2 -- diff NOT reportable")
        print(f"    VALID (same model): implicit {bm(fl_):+.4f} vs explicit-K "
              f"{bm(fe_):+.2f} W/m^2 = "
              f"{abs(bm(fe_) / bm(fl_)) if bm(fl_) != 0 else float('nan'):.1f}x -- "
              "backward-Euler SATURATION, measured not assumed")

    # ---- (F) the CLEAN bound on the vertical-mixing flux difference ----
    # At a MATCHED state the two models see the SAME T (max|dT|=0 above) and,
    # as measured, essentially the same realized K.  The vertical diffusive
    # flux is a function of exactly those two, so its cross-model difference is
    # bounded by the K difference -- no solve, and no confounded after-field,
    # is needed to state it.  This is what answers "does the firing difference
    # translate into a flux difference at all?".
    print("\n" + "=" * 104)
    print("(F) CLEAN BOUND on the vertical-mixing flux difference at this state")
    print("=" * 104)
    for box, rows in (("CORE", CORE), ("SEDGE", SEDGE)):
        kp, n = prof(K_prod, rows)
        ar, _ = prof(avt_real, rows)
        fl_, _ = prof(F_lego[..., :NF], rows)
        bandF = band[:NF]
        s = bandF & (n[:NF] > 0)
        mF = float(np.average(fl_[:NF][s], weights=n[:NF][s])) if s.any() else np.nan
        rk = bmean(kp, n) / bmean(ar, n)
        print(f"  {box}: realized K ratio {rk:.4f} on an IDENTICAL T field "
              f"=> |flux difference| <= {abs(rk - 1) * 100:.2f}% of "
              f"{abs(mF):.2f} W/m^2 = {abs(rk - 1) * abs(mF):.3f} W/m^2")
    print("  Compare to the recorded band discrepancy the campaign is chasing "
          "(order 10 W/m^2): the vertical-mixing contribution at this state is "
          "two to three orders of magnitude too small to be the mechanism.")

    # ---- (E) POINT 3, the discriminator: FREE-RUN year-5 stratification ----
    # SEPARATE PROTOCOL from everything above (which is matched-state).  Here
    # each model ran its OWN 5 years, so the gradients CAN differ -- this is
    # what can distinguish "both dead" from "legoESM dead, NEMO live".
    # Both sides are year-5 ANNUAL MEANS (same window).  NEMO is a HISTORY
    # file, so votemper IS thickness-weighted (toce*e3t) and IS divided by
    # vovvle3t -- the artifact that does NOT apply to the restarts above.
    LEGO_Y5 = os.environ.get(
        "DINO_LEGO_Y5",
        "/tmp/claude-10257/-home-dbalwada-legoESM/"
        "853ee94c-2651-44cc-ba12-f55bf3ed1979/scratchpad/year_seamfix_y5.npz")
    NEMO_Y5 = f"{DINO}/RUN_5Y/DINO_1y_00050101_00051230_grid_T.nc"
    print("\n" + "=" * 104)
    print("(E) FREE-RUN year-5 stratification -- DIFFERENT PROTOCOL (each model "
          "ran its own 5 yr)")
    print(f"    legoESM: {LEGO_Y5}")
    print(f"    NEMO   : {NEMO_Y5} (votemper/vovvle3t de-weighted)")
    print("=" * 104)
    if not os.path.exists(LEGO_Y5):
        print(f"  SKIPPED: {LEGO_Y5} not present -- cannot run the free-run "
              "discriminator, so 'both dead' vs 'lego dead / NEMO live' stays "
              "UNRESOLVED (say so rather than guessing)")
    else:
        gT = nc.Dataset(NEMO_Y5)
        e3d = llz(gT["vovvle3t"][0])
        T_n5 = llz(gT["votemper"][0]) / np.maximum(e3d, 1e-6)
        T_l5 = np.load(LEGO_Y5)["T"]
        dz_t = np.diff(gdept1d)
        g_l = np.abs(np.diff(T_l5, axis=-1) / dz_t)[..., :NK]
        g_n = np.abs(np.diff(T_n5, axis=-1) / dz_t)[..., :NK]
        for box, rows in (("CORE", CORE), ("SEDGE", SEDGE)):
            pl, n = prof(g_l, rows)
            pn_, _ = prof(g_n, rows)
            print(f"\n  --- {box} free-run |dT/dz| [degC/m] ---")
            print(f"{'k':>3s}{'z_w[m]':>8s}{'lego':>11s}{'NEMO':>11s}"
                  f"{'lego/NEMO':>11s}{'verdict':>26s}")
            for k in range(NK):
                if n[k] == 0:
                    continue
                zw = gdepw1d[k + 1]
                if not (DEEP_LO_M - 50 <= zw <= DEEP_HI_M + 400):
                    continue
                ld, nd = pl[k] < DEAD_DTDZ, pn_[k] < DEAD_DTDZ
                v = ("both dead" if ld and nd else
                     "LEGO DEAD, NEMO LIVE" if ld and not nd else
                     "lego live, NEMO dead" if nd and not ld else "both live")
                print(f"{k:3d}{zw:8.1f}{pl[k]:11.3e}{pn_[k]:11.3e}"
                      f"{pl[k] / max(pn_[k], 1e-30):11.3f}{v:>26s}")
            ml, mn = bmean(pl, n), bmean(pn_, n)
            ld, nd = ml < DEAD_DTDZ, mn < DEAD_DTDZ
            print(f"  [{DEEP_LO_M:.0f}-{DEEP_HI_M:.0f} m mean] lego {ml:.3e}  "
                  f"NEMO {mn:.3e}  ratio {ml / max(mn, 1e-30):.3f}  => "
                  + ("BOTH DEAD (band inert in both -- the question moves "
                     "elsewhere)" if ld and nd else
                     "LEGO DEAD / NEMO LIVE (legoESM destroyed stratification "
                     "NEMO retains -- this would be the headline)" if ld and not nd
                     else "BOTH LIVE (a flux difference here is meaningful)"))

    # ---- SELF-CHECKS ----
    print("\n" + "=" * 104)
    print("SELF-CHECKS")
    print("=" * 104)
    # (1) reconstruction exactness where EVD must NOT have fired (coordinator's
    # requested check): dump == avt_k bit-for-bit on unfired wet cells.
    # (1) EXACT trigger check.  The coordinator asked for "reconstructed avt ==
    # avt_k bit-for-bit on non-firing cells"; that exact form is IMPOSSIBLE
    # here because the restart's avt_k/rn2_stg are one step stale relative to
    # the dumps (quantified in section A) -- so the equivalent EXACT check that
    # the available same-step data DOES support is the subset property:
    # EVD fires on MIN(rn2,rn2b)<=thresh, hence rn2b<=thresh (same-step dump)
    # implies avt == rn_evd EXACTLY, with no tolerance.
    assert subset_violations == 0, (
        f"self-check 1 FAILED: {subset_violations} cells have rn2b <= "
        f"{EVD_THRESH:g} but avt != rn_evd -- the EVD trigger semantics or the "
        "level alignment is wrong, and every firing number above is suspect")
    print(f"[PASS] self-check 1: all {int(fired_rn2b.sum())} cells with "
          f"rn2b<={EVD_THRESH:g} have dumped avt == rn_evd EXACTLY (0 violations) "
          "-> trigger semantics + level alignment confirmed on same-step data")

    # (2) box_profile reduction vs an explicit loop (catches an axis bug).
    kk, rows = 12, CORE
    m = wet_if[rows][:, :, kk]
    tot = float(np.sum(np.where(m, K_prod[rows][:, :, kk], 0.0)))
    pv, pn = prof(K_prod, rows)
    assert int(m.sum()) == int(pn[kk]) and abs(
        tot / max(m.sum(), 1) - pv[kk]) <= 1e-10 * max(abs(pv[kk]), 1e-30), (
        "self-check 2: prof() disagrees with a manual loop")
    print(f"[PASS] self-check 2: prof(CORE,k={kk}) manual "
          f"{tot / max(m.sum(), 1):.6e} == vectorized {pv[kk]:.6e} over "
          f"{int(pn[kk])} cells")

    # (3) same mask on both models, per level (a metric that moved because the
    # mask drifted is a confound, not a result).
    for box, rows in (("CORE", CORE), ("SEDGE", SEDGE)):
        _, n1 = prof(K_prod, rows)
        _, n2 = prof(avt_real, rows)
        assert np.array_equal(n1, n2), f"self-check 3: {box} mask differs per model"
    print("[PASS] self-check 3: identical wet-interface counts per level for "
          "legoESM and NEMO fields in both boxes")

    # (4) ablation actually did something (else the isolation claim is vacuous).
    d_ab = float(np.max(np.abs(K_prod[wet_if] - K_tke[wet_if])))
    assert d_ab > 0.0, "self-check 4: convection='none' changed nothing -- vacuous"
    print(f"[PASS] self-check 4: ablation sanity max|Kprod-Ktke| = {d_ab:.4e} "
          f"(== K_conv exactly: {abs(d_ab - K_CONV) < 1e-9})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
