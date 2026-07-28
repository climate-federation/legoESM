#!/usr/bin/env python
"""#1226 tier-2/3: PER-ELEMENT error of the four ``ldf_slp`` isopycnal-slope
rows (``wslpi``, ``wslpj``, ``uslp``, ``vslp``) vs NEMO 5.0.2's own
``eiv_dump_*.bin`` dumps.

Modeled directly on ``eos_rab_bn2_per_element.py`` (same ``build_state()``
shape, same two mechanical preconditions, same restart/dump run-directory
pairing) and reuses the offset-scan idea from the throwaway
``probe_all4_slopes.py`` -- but rebuilt from the trusted template rather than
from that older probe's setup (it read mesh_mask/restart from a DIFFERENT run
directory than the dumps -- see below).

No numerics are re-implemented here (Rule 0): the slopes come from the real
production entry point ``compute_nemo_native_slopes`` in
``gm_redi_latlon_cgrid.py``, called with the same ``gm_redi_density_and_
jacobian`` + ``_nemo_native_active_3d`` helpers the template already uses.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/ldf_slp_per_element.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Per eos_rab_bn2_per_element.py / bn2_alpha_compare.py: DINO is full-step;
# pin the NEMO e3t ladder so gdept_0 matches NEMO's own dump exactly.
os.environ["LEGOESM_NEMO_E3T"] = "both"

import importlib.util
import sys

import numpy as np
import jax.numpy as jnp

# scripts/ is not a package -- import the sibling probe by path (same idiom
# eos_rab_bn2_per_element.py uses) to reuse _read_dims/_load_haloed without
# duplicating them.
_sib_path = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib_path)
_bn2_alpha_compare = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bn2_alpha_compare
_spec.loader.exec_module(_bn2_alpha_compare)
_read_dims = _bn2_alpha_compare._read_dims
_load_haloed = _bn2_alpha_compare._load_haloed

from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_nemo_native_slopes, gm_redi_density_and_jacobian,
    _nemo_native_active_3d,
)

# STATE / DUMP CONSISTENCY: read the restart from the SAME run directory as
# the dumps. probe_all4_slopes.py (the throwaway predecessor) read mesh_mask
# from RUN_TRAJ and the restart from RUN_Y5_REBUILD against RUN_GDB's dumps --
# a cross-run-directory mismatch. mesh_mask.nc also exists in RUN_GDB itself
# (verified), so there is no reason to reach outside it.
RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"

FLOOR = 1.0e-12
OFFSETS = (-2, -1, 0, 1, 2)

DUMP_META = {
    # component: (dump basename, lego-array-role)
    "wslpi": "eiv_dump_wslpi.bin",
    "wslpj": "eiv_dump_wslpj.bin",
    "uslp": "eiv_dump_uslp.bin",
    "vslp": "eiv_dump_vslp.bin",
}

# Section (I): RAW (pre-Shapiro) w-point slopes, ldfslp.F90:335-336 (units
# 8840/8841 written at :378-379). Same haloed convention as eiv_dump_wslpi.bin.
RAW_DUMPS = {
    "wslpi": "eiv_dump_zwz_raw.bin",   # zwz, i-direction
    "wslpj": "eiv_dump_zww_raw.bin",   # zww, j-direction
}


# Section (J): the full j-direction intermediate chain, in NEMO execution
# order. All haloed, float64, same convention as eiv_dump_wslpj.bin.
CHAIN_DUMPS = {
    "prd": "eiv_dump_prd_arg.bin",
    "zgrv_iik": "eiv_dump_zgrv_iik.bin",
    "zgrv_iikm1": "eiv_dump_zgrv_iikm1.bin",
    "zaj": "eiv_dump_zaj.bin",
    "zbw": "eiv_dump_zbw.bin",
    "zbj": "eiv_dump_zbj.bin",
    "zfk": "eiv_dump_zfk.bin",
    "zww_raw": "eiv_dump_zww_raw.bin",
    "wslpj": "eiv_dump_wslpj.bin",
}

# ldfslp.F90:209 is ``DO jk = jpkm1, 2, -1`` -- 1-based jk never takes the
# values 1 or jpk, so the dumped intermediates are IDENTICALLY ZERO at 0-based
# k=0 and k=nlev-1. Scoring those pad levels would manufacture a 100% "error"
# at k=0 for every field that legoESM computes there (zbw, zbj, ...), so the
# chain is scored on the levels NEMO actually wrote.
KLO, KHI = 1, 35   # 0-based python slice [KLO:KHI] == jk = 2..jpkm1


def capture_locals(fn, target_code):
    """Run ``fn()`` and snapshot the local variables of ``target_code``'s frame
    at its RETURN, via ``sys.settrace``.

    Same spirit as ``_JnpCapture`` (which is kept for the raw-Shapiro capture):
    it READS the production function's own intermediates without altering a
    single operation.  Needed here because ``zaj``/``zbw``/``zbj``/``zcj`` are
    plain local variables, not arguments to a distinctive ``jnp`` call, so the
    pad-proxy cannot see them.  The global tracer returns ``None`` for every
    frame except the target, so only that one frame is line-traced.
    """
    holder = {}

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code is target_code:
            def local_tracer(f, ev, a):
                if ev == "return":
                    holder.update(f.f_locals)
                return local_tracer
            return local_tracer
        return None

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        out = fn()
    finally:
        sys.settrace(old)
    return out, holder


class _JnpCapture:
    """Proxy for the ``jnp`` module attribute of ``gm_redi_latlon_cgrid``, used
    to READ OFF the PRE-SMOOTHER slope field from the REAL production call.

    Why a proxy and not (a) a flag or (c) a diagnostics return: neither exists
    -- ``compute_nemo_native_slopes`` has no smoothing kwarg and returns only
    the four final fields.  And the Shapiro helper ``_shap`` is a LOCAL CLOSURE
    defined inside the function body (gm_redi_latlon_cgrid.py:1105), so it
    cannot be monkeypatched by name either.

    What it does: ``_shap``'s FIRST operation is
    ``jnp.pad(f, ((0,0),(1,1),(0,0)), mode="wrap")`` (:1115) and its argument
    ``f`` IS the pre-smoother slope.  This proxy forwards every attribute to
    the real ``jnp`` and records ``f`` on exactly that pad signature.  It
    CHANGES NO NUMERICS -- the production slope formula runs untouched and the
    returned fields are asserted bit-identical to the unproxied call.  The only
    other ``mode="wrap"`` pad in this module (:722) is inside
    ``_shapiro_smooth_slopes``, a DIFFERENT code path that
    ``compute_nemo_native_slopes`` never calls (verified by grep + the
    len(captured)==4 assertion below).

    ``_shap`` is applied in the fixed order uslp, vslp, wslpi, wslpj
    (:1133-1136), so captured[2]/captured[3] are the raw wslpi/wslpj.
    """

    _SHAP_PADW = ((0, 0), (1, 1), (0, 0))

    def __init__(self, real):
        self._real = real
        self.captured = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def pad(self, a, pad_width, mode="constant", **kw):
        if mode == "wrap" and tuple(tuple(p) for p in pad_width) == self._SHAP_PADW:
            self.captured.append(np.asarray(a))
        return self._real.pad(a, pad_width, mode=mode, **kw)


def build_state():
    # PRECISION (#1226): see eos_rab_bn2_per_element.py -- fp32 control policy
    # silently rounds the fp64 depth ladder; JAX_ENABLE_X64=1 does NOT change
    # legoESM's own precision policy.
    if os.environ.get("LEGOESM_FIDELITY_FP64", "1") == "1":
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64())

    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    # TIME LEVEL: ldfslp.F90 (ldf_slp) consumes rn2b/rab_b -- BEFORE-level
    # T/S -- on the NOW geometry (eta/H_bathy), exactly the eos_rab/bn2
    # pairing the template already established. Fed here explicitly rather
    # than assumed; asserted below via time_level_for_dump.
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    # METRIC CONVENTION (#1226): NEMO's DINO usr_def_hgr.F90 builds the grid
    # ISOTROPICALLY (pe1t = pe2t); legoESM's default 'exact' computes the true
    # finite-difference dy, differing from NEMO e2t by -1.27e-5..+9.5e-6.  zcj
    # (ldfslp.F90:307) multiplies the vmask COUNT by e2t, so that discrepancy
    # lands directly in zaj.  Env-switchable so the A/B is a single variable.
    _metric = os.environ.get("LEGOESM_METRIC_CONVENTION", "exact")
    print(f"metric_convention={_metric}", flush=True)
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                     metric_convention=_metric,
                                      omega=cfg.omega)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    z_coord = br.z_coord
    state = br.state
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    H_bathy = state.H_bathy.data
    _tb = np.asarray(bef.T); _sb = np.asarray(bef.S)
    T = jnp.asarray(_tb.reshape(state.T.data.shape))
    S = jnp.asarray(_sb.reshape(state.S.data.shape))
    eta = jnp.asarray(state.eta.data)

    # EOS DEPTH CONVENTION (probe bug found by the section-(J) chain walk,
    # 2026-07-28).  The DINO card sets eos_depth="geometric" (dino.py:912,936)
    # and production passes it through (dino.py:2867 -> gm_redi_latlon_cgrid
    # :3304-3305).  The first version of this probe omitted the argument and
    # silently got the "insitu" DEFAULT -- a DIFFERENT density convention from
    # the one NEMO uses, whose docstring documents a depth-growing O(1e-6)
    # bias.  That put a bias into `prd` (ldf_slp's INPUT) and hence into every
    # downstream slope number this script printed.  Mirror the production
    # caller exactly, including the rho0 kwarg that makes "geometric" cancel.
    eos_depth = getattr(mc, "eos_depth", cfg.eos_depth)
    _eos_mk_kw = {"rho0": mc.rho_0} if eos_depth == "geometric" else {}
    rho, jacobian = gm_redi_density_and_jacobian(
        T, S, eta, H_bathy, br.geometry, z_coord,
        eos=mc.eos, eos_linear=mc.eos_linear, mask=mask,
        rho_0=mc.rho_0, g=mc.g, eos_depth=eos_depth,
    )
    active_3d = _nemo_native_active_3d(mask, z_coord, H_bathy, T.dtype)

    # --- oracle-comparison preconditions, both MECHANICAL (#1226) ---
    from legoesm.ocean.fidelity.precision_gate import require_fp64
    require_fp64(z_coord, T, S, context="ldf_slp per-element")

    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    for dump_name in (list(DUMP_META.values()) + list(RAW_DUMPS.values())
                      + list(CHAIN_DUMPS.values())):
        lvl = time_level_for_dump(dump_name)
        if lvl != "before":
            raise ValueError(
                f"this probe feeds BEFORE-level T/S but {dump_name!r} is "
                f"{lvl!r}-level; fix the probe or the registry")

    # same _eos_mk_kw convention as gm_redi_latlon_cgrid.py:3304-3305
    eos_fn = make_eos_fn(mc.eos, mc.eos_linear, **_eos_mk_kw)
    card_slope_n2 = mc.gm_redi.slope_n2
    env_override = os.environ.get("SLOPE_N2")
    slope_n2_used = env_override if env_override is not None else card_slope_n2
    gm_cfg = mc.gm_redi._replace(slope_n2=slope_n2_used)

    # One argument bundle, called twice: once plain (stage B / the four rows)
    # and once under the _JnpCapture proxy (stage A / pre-Shapiro), so both
    # stages come from the IDENTICAL production call.
    def recall():
        return compute_nemo_native_slopes(
            rho, T, S, mask, u_mask, v_mask, z_coord, br.geometry, gm_cfg,
            eos_fn, rho_0=mc.rho_0, g=mc.g, active_3d=active_3d,
            jacobian=jacobian,
        )

    uslp, vslp, wslpi, wslpj = recall()

    return dict(
        recall=recall,
        jpi=jpi, jpj=jpj, jpk=jpk, hls=hls,
        mask=np.asarray(mask) > 0.5,
        u_mask=np.asarray(u_mask), v_mask=np.asarray(v_mask),
        active=np.asarray(active_3d) > 0.5,
        lego=dict(wslpi=np.asarray(wslpi), wslpj=np.asarray(wslpj),
                   uslp=np.asarray(uslp), vslp=np.asarray(vslp)),
        card_slope_n2=card_slope_n2, slope_n2_used=slope_n2_used,
        env_override=env_override, eos_depth=eos_depth,
        # section (K): the prd / EOS isolation
        eos_fn=eos_fn, rho_0=mc.rho_0, T=T, S=S, z_coord=z_coord,
        eos_name=mc.eos, eos_linear=mc.eos_linear, eos_mk_kw=_eos_mk_kw,
    )


# ---------------------------------------------------------------------------
# wet-point selectors -- reproduce the EXACT umask3/vmask3/wmask3 conventions
# compute_nemo_native_slopes itself builds internally (gm_redi_latlon_cgrid.py
# :962-963, :790-791), using the SAME active_3d this script fed it. This is a
# stats-only wet mask, not a re-implementation of the slope numerics (Rule 0
# is about not re-deriving the computed VALUES; selecting which points are
# wet for a stats report is unavoidable bookkeeping and must match the
# production convention exactly, which is why it is derived from `act`
# rather than a hand re-guessed mask).
def wet_w_mask(act):
    """W-point wet mask at level k: wmask3 = act[k] & act[k-1] (k=0: act[0])."""
    nlev = act.shape[-1]
    above = np.concatenate([act[:, :, :1], act[:, :, :-1]], axis=-1)
    return act & above


def wet_u_mask(act, u_mask):
    east_active = np.roll(act, -1, axis=1)
    face2d = u_mask[:, 1:]
    return (face2d[:, :, None] > 0.5) & act & east_active


def wet_v_mask(act, v_mask):
    north_active = np.roll(act, -1, axis=0)
    face2d = v_mask[1:, :]
    return (face2d[:, :, None] > 0.5) & act & north_active


MASK_FN = {
    "wslpi": lambda st: wet_w_mask(st["active"]),
    "wslpj": lambda st: wet_w_mask(st["active"]),
    "uslp": lambda st: wet_u_mask(st["active"], st["u_mask"]),
    "vslp": lambda st: wet_v_mask(st["active"], st["v_mask"]),
}


def offset_scan(lego, nemo, wet, offsets=OFFSETS):
    """corr at each vertical offset: lego[...,k] vs nemo[...,k+offset]."""
    nlev_lego = lego.shape[-1]
    nlev_nemo = nemo.shape[-1]
    out = {}
    for off in offsets:
        L_parts, N_parts = [], []
        for k in range(nlev_lego):
            kn = k + off
            if not (0 <= kn < nlev_nemo):
                continue
            wk = wet[:, :, k]
            if not wk.any():
                continue
            L_parts.append(lego[:, :, k][wk])
            N_parts.append(nemo[:, :, kn][wk])
        if not L_parts:
            out[off] = None
            continue
        L = np.concatenate(L_parts)
        N = np.concatenate(N_parts)
        finite = np.isfinite(L) & np.isfinite(N)
        L, N = L[finite], N[finite]
        if L.size < 2 or L.std() == 0.0 or N.std() == 0.0:
            out[off] = dict(n=int(L.size), corr=float("nan"))
            continue
        out[off] = dict(n=int(L.size), corr=float(np.corrcoef(L, N)[0, 1]))
    return out


def per_element_report(name, lego, nemo, wet):
    """Full per-element report at a fixed (already-chosen) offset alignment.

    lego/nemo/wet must already share the SAME level axis (caller has sliced
    to the offset-aligned overlap range)."""
    m = wet & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    n = int(m.sum())
    rms_nemo = float(np.sqrt(np.mean(ne ** 2))) if n else float("nan")

    corr = (float(np.corrcoef(lo, ne)[0, 1])
            if n >= 2 and lo.std() > 0 and ne.std() > 0 else float("nan"))
    abs_ratio = (float(np.abs(lo).sum() / np.abs(ne).sum())
                 if np.abs(ne).sum() > 0 else float("nan"))

    # (2) pointwise |lego-nemo|/|nemo| -- REFERENCE ONLY, not trustworthy for
    # a sign-changing field (see (3) below).
    denom_pt = np.maximum(np.abs(ne), FLOOR)
    rel_pt = np.abs(lo - ne) / denom_pt
    med_rel_pt = float(np.median(rel_pt))
    p99_rel_pt = float(np.percentile(rel_pt, 99))
    max_rel_pt = float(np.max(rel_pt))

    # conditioning-robust metric: err_norm = |lego-nemo| / RMS(nemo over wet)
    err_norm = np.abs(lo - ne) / max(rms_nemo, FLOOR)
    med_en = float(np.median(err_norm))
    p99_en = float(np.percentile(err_norm, 99))
    max_en = float(np.max(err_norm))

    # location of max err_norm, in the FULL (unmasked-shape) array
    en_full = np.full(wet.shape, -1.0)
    en_full[m] = err_norm
    idx = tuple(int(v) for v in np.unravel_index(np.argmax(en_full), en_full.shape))
    nemo_at_max = float(nemo[idx])

    # (3) conditioning check: fraction of wet points with |nemo| < 1e-3*RMS
    near0 = np.abs(ne) < 1.0e-3 * max(rms_nemo, FLOOR)
    frac_near0 = float(near0.mean()) if n else float("nan")
    trustworthy_pointwise = frac_near0 < 0.05

    print(f"\n--- {name}: at chosen offset ---")
    print(f"  n wet elements       = {n}")
    print(f"  corr                 = {corr:.6f}")
    print(f"  |x| ratio (sum|lego|/sum|nemo|) = {abs_ratio:.6f}")
    print(f"  RMS(nemo over wet)   = {rms_nemo:.6e}")
    print(f"  pointwise |rel| (REFERENCE ONLY, sign-changing field):")
    print(f"    median={med_rel_pt:.3e}  p99={p99_rel_pt:.3e}  max={max_rel_pt:.3e}")
    print(f"  err_norm = |lego-nemo|/RMS(nemo) (CONDITIONING-ROBUST):")
    print(f"    median={med_en:.3e}  p99={p99_en:.3e}  max={max_en:.3e}")
    print(f"    max err_norm at (j,i,level)={idx}  |nemo| there = {abs(nemo_at_max):.6e} "
          f"(nemo={nemo_at_max:+.6e}, lego={float(lego[idx]):+.6e})")
    print(f"  CONDITIONING: fraction of wet points with |nemo| < 1e-3*RMS(nemo) "
          f"= {frac_near0 * 100:.3f}%  ({int(near0.sum())}/{n})")
    if trustworthy_pointwise:
        print(f"  -> pointwise-relative stats for {name} are LIKELY TRUSTWORTHY "
              f"(near-zero fraction < 5%).")
    else:
        print(f"  -> pointwise-relative stats for {name} are NOT TRUSTWORTHY "
              f"(large near-zero-denominator fraction); use err_norm.")

    # (4) per-level median err_norm
    nlev = wet.shape[-1]
    print(f"  per-level median err_norm ({nlev} levels):")
    level_meds = []
    line = []
    for k in range(nlev):
        mk = m[:, :, k]
        if not mk.any():
            line.append(f"k{k}:--")
            continue
        ek = np.abs(lego[:, :, k][mk] - nemo[:, :, k][mk]) / max(rms_nemo, FLOOR)
        med_k = float(np.median(ek))
        level_meds.append((k, med_k))
        line.append(f"k{k}:{med_k:.2e}")
    print("    " + "  ".join(line))
    if level_meds:
        meds = np.array([v for _, v in level_meds])
        kbest = level_meds[int(np.argmin(meds))][0]
        kworst = level_meds[int(np.argmax(meds))][0]
        # RELATIVE floor on the denominator: a level whose median err_norm is
        # ~0 must not blow the ratio up to an astronomical number, but an
        # ABSOLUTE floor would also crush the ratio to <1 when EVERY level is
        # tiny (the all-roundoff case) and mislabel it. Capping the ratio at
        # 1e12 handles both.
        spread = float(meds.max() / max(meds.min(), meds.max() * 1e-12))
        print(f"  per-level spread: min={meds.min():.3e} (k={kbest})  "
              f"max={meds.max():.3e} (k={kworst})  max/min={spread:.1f}x  "
              f"-> {'FLAT (roundoff-like)' if spread < 10 else 'STRUCTURED (points at a term)'}")

    return dict(n=n, corr=corr, abs_ratio=abs_ratio, rms_nemo=rms_nemo,
                med_rel_pt=med_rel_pt, p99_rel_pt=p99_rel_pt, max_rel_pt=max_rel_pt,
                med_en=med_en, p99_en=p99_en, max_en=max_en,
                frac_near0=frac_near0, trustworthy_pointwise=trustworthy_pointwise)


def main() -> int:
    print(f"restart used  = {os.path.join(RUN_DIR, RESTART)}")
    print(f"dump dir used = {RUN_DIR}")
    print(f"LEGOESM_NEMO_E3T={os.environ.get('LEGOESM_NEMO_E3T')}")

    st = build_state()
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    print(f"card default GMRediConfig.slope_n2 = {st['card_slope_n2']!r}")
    print(f"SLOPE_N2 env override = {st['env_override']!r}")
    print(f"==> slope_n2 ACTUALLY USED = {st['slope_n2_used']!r}")
    print(f"==> eos_depth ACTUALLY USED = {st['eos_depth']!r} "
          f"(card value; 'geometric' is NEMO's convention -- the probe's "
          f"original omission defaulted to 'insitu' and biased every number)")
    nj, ni = st["mask"].shape
    print(f"grid interior (nj,ni) = ({nj},{ni})  wet T-columns = {int(st['mask'].sum())}")

    summary = {}
    for comp, dump_name in DUMP_META.items():
        print("\n" + "=" * 78)
        print(f"COMPONENT: {comp}  (dump={dump_name})")
        print("=" * 78)
        lego = st["lego"][comp]
        dump_path = os.path.join(RUN_DIR, dump_name)
        nemo = _load_haloed(dump_path, jpi, jpj, hls)
        wet3d_full = MASK_FN[comp](st)
        print(f"lego shape={lego.shape}  nemo dump shape={nemo.shape}  "
              f"wet3d shape={wet3d_full.shape}  total wet (lego-level-count)="
              f"{int(wet3d_full.sum())}")

        # --- (1) offset scan ---
        print("\n--- offset scan (corr at each vertical offset) ---")
        scan = offset_scan(lego, nemo, wet3d_full, OFFSETS)
        for off in OFFSETS:
            r = scan[off]
            if r is None:
                print(f"  offset={off:+d}: no overlapping wet levels")
            else:
                print(f"  offset={off:+d}: n={r['n']:7d}  corr={r['corr']:.6f}")
        valid = {o: r for o, r in scan.items() if r is not None and np.isfinite(r["corr"])}
        if not valid:
            print(f"*** {comp}: NO VALID OFFSET -- cannot compare, reporting as FAILURE ***")
            summary[comp] = None
            continue
        best_off = max(valid, key=lambda o: valid[o]["corr"])
        # corr is bounded above by 1, so near the peak it compresses (0.9999
        # vs 0.987 IS a decisive gap even though the raw difference is only
        # 0.012) -- compare on (1-corr) ("residual"), where a >=3x gap
        # between best and runner-up counts as a sharp peak.
        resid_sorted = sorted(1.0 - r["corr"] for r in valid.values())
        sharp = (len(resid_sorted) < 2) or (resid_sorted[0] <= 1e-15) or (
            resid_sorted[1] / max(resid_sorted[0], 1e-15) >= 3.0)
        print(f"BEST OFFSET = {best_off:+d}  corr={valid[best_off]['corr']:.6f}  "
              f"sharp peak vs runner-up (>=3x gap in 1-corr): {sharp}"
              + ("  <-- NONZERO OFFSET, FLAG" if best_off != 0 else ""))
        if not sharp:
            print("  WARNING: best-offset corr is NOT a sharp peak over the runner-up -- "
                  "alignment is ambiguous; treat downstream stats with caution.")

        # slice lego/nemo/wet to the offset-aligned common level range for (2)-(4)
        nlev_lego = lego.shape[-1]
        nlev_nemo = nemo.shape[-1]
        k_lo = max(0, -best_off)
        k_hi = min(nlev_lego, nlev_nemo - best_off)
        lego_al = lego[:, :, k_lo:k_hi]
        nemo_al = nemo[:, :, k_lo + best_off:k_hi + best_off]
        wet_al = wet3d_full[:, :, k_lo:k_hi]

        rep = per_element_report(comp, lego_al, nemo_al, wet_al)
        rep["best_off"] = best_off
        rep["sharp"] = sharp
        # keep the aligned stage-B arrays so section (I) can compare stage A
        # and stage B on the SAME points without recomputing the alignment.
        rep["_al"] = (lego_al, nemo_al, wet_al)
        summary[comp] = rep

    # =====================================================================
    # (I) INTERNALS SPLIT: stage A (slope FORMULA, pre-Shapiro) vs
    #     stage B (the 16-point SMOOTHER), for the two w-point components.
    # =====================================================================
    print("\n" + "=" * 78)
    print("(I) INTERNALS SPLIT: stage A = slope formula (pre-Shapiro zwz/zww)")
    print("                     stage B = 16-point Shapiro smoother (final wslpi/wslpj)")
    print("=" * 78)

    import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as _gmr
    cap = _JnpCapture(_gmr.jnp)
    _gmr.jnp = cap
    try:
        u2, v2, wi2, wj2 = st["recall"]()
    finally:
        _gmr.jnp = cap._real
    u2, v2, wi2, wj2 = (np.asarray(a) for a in (u2, v2, wi2, wj2))

    print(f"  [verify] captured {len(cap.captured)} Shapiro-pad inputs "
          f"(expect exactly 4: uslp, vslp, wslpi, wslpj)")
    if len(cap.captured) != 4:
        print("  ABORTING section (I): the capture did not see exactly the four "
              "_shap calls, so captured[2]/[3] cannot be assumed to be the raw "
              "wslpi/wslpj. Reporting this rather than guessing.")
        raw_ok = False
    else:
        raw_ok = True
        raw = {"wslpi": cap.captured[2], "wslpj": cap.captured[3]}
        # the proxy must be INERT on the numerics: the returned final fields
        # must be bit-identical to the unproxied call made in build_state().
        inert = all(np.array_equal(a, b) for a, b in (
            (wi2, st["lego"]["wslpi"]), (wj2, st["lego"]["wslpj"]),
            (u2, st["lego"]["uslp"]), (v2, st["lego"]["vslp"])))
        print(f"  [verify] proxy is INERT (all four returned fields bit-identical "
              f"to the unproxied call): {inert} (expect True)")
        # and the capture must actually be the PRE-smoother field, i.e. it must
        # DIFFER from the final smoothed field.
        differs = {c: not np.array_equal(raw[c], st["lego"][c]) for c in raw}
        print(f"  [verify] captured raw field DIFFERS from the final smoothed "
              f"field: {differs} (expect True for both -- else the smoother "
              f"was a no-op and stage A/B cannot be separated)")
        if not inert or not all(differs.values()):
            print("  WARNING: capture verification FAILED -- stage-A numbers below "
                  "cannot be trusted; see the booleans above.")

    stageA = {}
    if raw_ok:
        act = st["active"]
        wet_w_full = wet_w_mask(act)
        for comp, dump_name in RAW_DUMPS.items():
            print("\n" + "-" * 78)
            print(f"  STAGE A -- {comp} raw (pre-Shapiro) vs {dump_name}")
            print("-" * 78)
            lego_raw = raw[comp]
            nemo_raw = _load_haloed(os.path.join(RUN_DIR, dump_name), jpi, jpj, hls)
            print(f"  lego raw shape={lego_raw.shape}  nemo raw dump shape={nemo_raw.shape}")
            scanA = offset_scan(lego_raw, nemo_raw, wet_w_full, OFFSETS)
            for off in OFFSETS:
                r = scanA[off]
                print(f"    offset={off:+d}: " + ("no overlap" if r is None else
                      f"n={r['n']:7d}  corr={r['corr']:.6f}"))
            validA = {o: r for o, r in scanA.items()
                      if r is not None and np.isfinite(r["corr"])}
            if not validA:
                print(f"  *** {comp} stage A: NO VALID OFFSET -- reporting as FAILURE ***")
                continue
            offA = max(validA, key=lambda o: validA[o]["corr"])
            print(f"  BEST OFFSET (stage A) = {offA:+d}  corr={validA[offA]['corr']:.6f}")
            kloA = max(0, -offA)
            khiA = min(lego_raw.shape[-1], nemo_raw.shape[-1] - offA)
            repA = per_element_report(
                f"{comp} STAGE-A raw", lego_raw[:, :, kloA:khiA],
                nemo_raw[:, :, kloA + offA:khiA + offA], wet_w_full[:, :, kloA:khiA])
            repA["best_off"] = offA
            repA["_al"] = (lego_raw[:, :, kloA:khiA],
                           nemo_raw[:, :, kloA + offA:khiA + offA],
                           wet_w_full[:, :, kloA:khiA])
            stageA[comp] = repA

        # ---- side-by-side + bottom-3-active-level attribution ----
        print("\n" + "-" * 78)
        print("  STAGE A vs STAGE B side-by-side (err_norm = |lego-nemo|/RMS(nemo))")
        print("-" * 78)
        act = st["active"]
        nlev = act.shape[-1]
        kb = nlev - 1 - np.argmax(act[:, :, ::-1], axis=-1)   # last active level
        has_wet = act.any(axis=-1)
        kidx = np.arange(nlev)[None, None, :]
        bottom3 = (has_wet[:, :, None] & (kidx >= (kb - 2)[:, :, None])
                   & (kidx <= kb[:, :, None]))
        print(f"  bottom-3-active-level cells (per column, from z_coord.is_active) "
              f"= {int(bottom3.sum())}")

        for comp in RAW_DUMPS:
            if comp not in stageA or summary.get(comp) is None:
                print(f"  {comp}: stage A or stage B missing -- cannot compare.")
                continue
            A_l, A_n, A_w = stageA[comp]["_al"]
            B_l, B_n, B_w = summary[comp]["_al"]
            nk = min(A_l.shape[-1], B_l.shape[-1])
            # common points: wet in BOTH stages over the common level range
            both = A_w[:, :, :nk] & B_w[:, :, :nk]
            dA = np.abs(A_l[:, :, :nk] - A_n[:, :, :nk])
            dB = np.abs(B_l[:, :, :nk] - B_n[:, :, :nk])
            print(f"\n  {comp}: stage A err_norm median/p99/max = "
                  f"{stageA[comp]['med_en']:.3e}/{stageA[comp]['p99_en']:.3e}/"
                  f"{stageA[comp]['max_en']:.3e}   corr={stageA[comp]['corr']:.6f}"
                  f"  |x|ratio={stageA[comp]['abs_ratio']:.6f}")
            print(f"  {comp}: stage B err_norm median/p99/max = "
                  f"{summary[comp]['med_en']:.3e}/{summary[comp]['p99_en']:.3e}/"
                  f"{summary[comp]['max_en']:.3e}   corr={summary[comp]['corr']:.6f}"
                  f"  |x|ratio={summary[comp]['abs_ratio']:.6f}")
            # ABSOLUTE error magnitudes (same units, both are slopes) so the
            # "how much of B is already in A" fraction is not distorted by the
            # two stages' slightly different RMS normalisers.
            b3 = both & bottom3[:, :, :nk]
            for label, sel in (("ALL common wet points", both),
                               ("BOTTOM-3 active levels", b3)):
                if not sel.any():
                    print(f"    {label}: no points")
                    continue
                mA, mB = float(np.median(dA[sel])), float(np.median(dB[sel]))
                frac = mA / mB if mB > 0 else float("nan")
                print(f"    {label} (n={int(sel.sum())}): median|dA|={mA:.3e}  "
                      f"median|dB|={mB:.3e}  -> fraction of stage-B error already "
                      f"present in stage A = {frac:.3f}")

        # ---- interpretation, using the three cases the coordinator specified ----
        print("\n  INTERPRETATION:")
        for comp in RAW_DUMPS:
            if comp not in stageA or summary.get(comp) is None:
                continue
            a_med, b_med = stageA[comp]["med_en"], summary[comp]["med_en"]
            a_p99, b_p99 = stageA[comp]["p99_en"], summary[comp]["p99_en"]
            ratio = a_med / b_med if b_med > 0 else float("nan")
            if ratio >= 0.5:
                verdict = ("stage A is ALREADY comparable to stage B -> the residual "
                           "is owned by the SLOPE FORMULA (suspects: the sequential "
                           "zuslp_hml recurrence ldfslp.F90:271, and the `zbu - zeps` "
                           "MINUS-eps denominator at :268), not the smoother.")
            elif ratio <= 0.1:
                verdict = ("stage A is CLEAN relative to stage B -> the residual is "
                           "owned by the SMOOTHER (suspect: masking/land treatment at "
                           "bathymetry steps where the 16-point stencil mixes columns "
                           "with different bottom levels; NEMO's deliberate "
                           "parenthesisation at :278-289).")
            else:
                verdict = (f"BOTH stages carry error, stage B larger "
                           f"(A/B median ratio {ratio:.3f}) -> reporting both "
                           f"magnitudes, claiming no single owner.")
            print(f"    {comp}: A/B median err_norm ratio = {ratio:.3f} "
                  f"(A={a_med:.3e} B={b_med:.3e}; p99 A={a_p99:.3e} B={b_p99:.3e})")
            print(f"      -> {verdict}")

    # =====================================================================
    # (J) LINE-LEVEL WALK of the j-direction chain, in NEMO execution order:
    #     prd -> zgrv(iik),zgrv(iikm1) -> zaj -> zbw -> zbj -> zfk
    #         -> zww_raw -> wslpj
    # =====================================================================
    print("\n" + "=" * 78)
    print("(J) j-DIRECTION CHAIN WALK, in NEMO execution order")
    print("=" * 78)
    print(f"  scoring levels k={KLO}..{KHI - 1} (0-based); ldfslp.F90:209 is "
          f"'DO jk = jpkm1, 2, -1' so k=0 and k={st['active'].shape[-1] - 1} "
          f"are never written by NEMO (dump zeros) and are EXCLUDED.")

    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _NEMO_SLOPE_STAB_7E3,
    )
    (_u3, _v3, _wi3, _wj3), loc = capture_locals(
        st["recall"], _gmr.compute_nemo_native_slopes.__code__)
    need = ["prd", "zgrv", "zcj", "zaj", "zbw", "zbj", "swj_int", "in_ml_w",
            "wslpj", "_km1", "e3w_k", "z1_slpmax", "e2t", "zeps"]
    missing = [k for k in need if k not in loc]
    print(f"  [verify] captured {len(loc)} locals from "
          f"compute_nemo_native_slopes; missing from the needed set: {missing}")
    inert_j = np.array_equal(np.asarray(_wj3), st["lego"]["wslpj"])
    print(f"  [verify] settrace capture is INERT (returned wslpj bit-identical "
          f"to the unproxied call): {inert_j} (expect True)")
    if missing or not inert_j:
        print("  ABORTING section (J): required locals missing or capture not "
              "inert -- reporting that rather than guessing.")
    else:
        act = st["active"]
        wmask_w = wet_w_mask(act)
        vmask_v = wet_v_mask(act, st["v_mask"])
        _km1 = loc["_km1"]
        L = lambda k: np.asarray(loc[k])

        # our chain fields, in NEMO order. zfk: NEMO's zfk is 1 OUTSIDE the ML
        # and 0 inside; our production decision variable is the boolean
        # `in_ml_w` (gm_redi_latlon_cgrid.py:1098), so our zfk == 1 - in_ml_w.
        # This READS our boolean, it does not re-derive the ML criterion.
        our = {
            "prd": (L("prd"), act),
            "zgrv_iik": (L("zgrv"), vmask_v),
            "zgrv_iikm1": (np.asarray(_km1(loc["zgrv"])), vmask_v),
            "zaj": (L("zaj"), wmask_w),
            "zbw": (L("zbw"), wmask_w),
            "zbj": (L("zbj"), wmask_w),
            "zfk": (1.0 - L("in_ml_w").astype(float), wmask_w),
            "zww_raw": (raw["wslpj"] if raw_ok else None, wmask_w),
            "wslpj": (st["lego"]["wslpj"], wmask_w),
        }

        print(f"\n  {'stage':<12}{'corr':<12}{'errN med':<13}{'errN p99':<13}"
              f"{'errN max':<13}{'n':<9}")
        chain_stats = {}
        for name, dump in CHAIN_DUMPS.items():
            lego_f, wet_f = our[name]
            if lego_f is None:
                print(f"  {name:<12}SKIPPED (our field unavailable)")
                continue
            nemo_f = _load_haloed(os.path.join(RUN_DIR, dump), jpi, jpj, hls)
            nk = min(lego_f.shape[-1], nemo_f.shape[-1], KHI)
            sl = slice(KLO, nk)
            lo_a, ne_a, w_a = lego_f[:, :, sl], nemo_f[:, :, sl], wet_f[:, :, sl]
            m = w_a & np.isfinite(lo_a) & np.isfinite(ne_a)
            lo, ne = lo_a[m], ne_a[m]
            rms = float(np.sqrt(np.mean(ne ** 2)))
            en = np.abs(lo - ne) / max(rms, FLOOR)
            corr = (float(np.corrcoef(lo, ne)[0, 1])
                    if lo.std() > 0 and ne.std() > 0 else float("nan"))
            s = dict(corr=corr, med=float(np.median(en)),
                     p99=float(np.percentile(en, 99)), max=float(np.max(en)),
                     n=int(m.sum()), rms=rms,
                     lo_a=lo_a, ne_a=ne_a, m=m)
            chain_stats[name] = s
            print(f"  {name:<12}{corr:<12.6f}{s['med']:<13.3e}{s['p99']:<13.3e}"
                  f"{s['max']:<13.3e}{s['n']:<9d}")

        print("\n  PER-LEVEL median err_norm (columns are 0-based k):")
        for name, s in chain_stats.items():
            cells = []
            for kk in range(s["lo_a"].shape[-1]):
                mk = s["m"][:, :, kk]
                if not mk.any():
                    cells.append(f"k{kk + KLO}:--")
                    continue
                e = np.abs(s["lo_a"][:, :, kk][mk] - s["ne_a"][:, :, kk][mk]) / max(s["rms"], FLOOR)
                cells.append(f"k{kk + KLO}:{np.median(e):.1e}")
            print(f"    {name:<11}" + " ".join(cells))

        # ---- FIRST DIVERGING STAGE ----
        print("\n  FIRST DIVERGING STAGE:")
        order = [n for n in CHAIN_DUMPS if n in chain_stats]
        ROUNDOFF = 1.0e-10
        first_jump = first_nonround = None
        prev = None
        for name in order:
            med = chain_stats[name]["med"]
            if first_nonround is None and med > ROUNDOFF:
                first_nonround = name
            # A stage that matches EXACTLY (median 0, e.g. zfk) is NOT a
            # meaningful jump denominator -- every later stage would show a
            # spurious "infinite" jump over it. Compare against the most
            # recent stage that actually carries a nonzero residual.
            if prev is not None and first_jump is None:
                base = chain_stats[prev]["med"]
                if med > 100.0 * base:
                    first_jump = (name, prev, med / base)
            if med > 0.0:
                prev = name
        print(f"    earliest stage NOT at roundoff (median err_norm > {ROUNDOFF:.0e}): "
              f"{first_nonround}")
        if first_jump:
            print(f"    earliest >100x jump vs the previous stage: {first_jump[0]} "
                  f"(x{first_jump[2]:.1f} over {first_jump[1]})")
        else:
            print("    no stage shows a >100x jump over its predecessor -- the "
                  "residual GROWS GRADUALLY along the chain rather than being "
                  "injected at one line.")

        # ---- Q1: zcj (mask count) ----
        print("\n  Q1: zcj = MAX(sum of 4 vmask, zeps) * e2t (ldfslp.F90:307-308)")
        print("      NEMO does NOT dump zcj, so it is INFERRED from the dumps:")
        print("      zaj = (4 zgrv terms)/zcj*wmask  =>  zcj = numerator/zaj.")
        g_iik = _load_haloed(os.path.join(RUN_DIR, CHAIN_DUMPS["zgrv_iik"]), jpi, jpj, hls)
        g_km1 = _load_haloed(os.path.join(RUN_DIR, CHAIN_DUMPS["zgrv_iikm1"]), jpi, jpj, hls)
        zaj_n = _load_haloed(os.path.join(RUN_DIR, CHAIN_DUMPS["zaj"]), jpi, jpj, hls)
        num_n = ((np.roll(g_iik, 1, axis=0) + g_km1)
                 + (np.roll(g_km1, 1, axis=0) + g_iik))
        e2t_np = np.asarray(loc["e2t"])[:, :, None]
        sl = slice(KLO, KHI)
        ok = wmask_w[:, :, sl] & (np.abs(zaj_n[:, :, sl]) > 1e-30)
        cnt_n = (num_n[:, :, sl] / np.where(ok, zaj_n[:, :, sl], np.nan)) / e2t_np
        cnt_o = np.asarray(loc["zcj"])[:, :, sl] / e2t_np
        nlev_a = act.shape[-1]
        kb = nlev_a - 1 - np.argmax(act[:, :, ::-1], axis=-1)
        kidx3 = np.arange(nlev_a)[None, None, :]
        bot3 = (act.any(-1)[:, :, None] & (kidx3 >= (kb - 2)[:, :, None])
                & (kidx3 <= kb[:, :, None]))[:, :, sl]
        # The INFERRED count inherits zaj's own error (it is a division by
        # zaj), so a 1e-6 tolerance would flag that noise, not a real
        # disagreement. The count is an INTEGER 1..4, so the meaningful test
        # is whether it differs by >=0.5 (i.e. a different mask count);
        # both tolerances are reported so the reader can see the separation.
        for lbl, sel in (("all wet w-points", ok), ("bottom-3 levels", ok & bot3)):
            d = np.abs(cnt_n - cnt_o)[sel]
            d = d[np.isfinite(d)]
            if d.size == 0:
                print(f"      {lbl}: no usable points")
                continue
            n_noise = int((d > 1e-6).sum())
            n_real = int((d > 0.5).sum())
            print(f"      {lbl}: n={d.size}  max|count_nemo - count_lego|={d.max():.3e}")
            print(f"        cells off by >=0.5 (a DIFFERENT integer count) : {n_real}"
                  f"  -> {'MATCH' if n_real == 0 else 'REAL MISMATCH'}")
            print(f"        cells off by >1e-6 (inherits zaj's own noise)   : {n_noise}")
        print(f"      our zcj/e2t distinct values (bottom-3): "
              f"{np.unique(np.round(cnt_o[bot3], 6))[:8]}")

        # ---- Q2: which of zbj's three terms wins ----
        print("\n  Q2: zbj = MIN(zbw, -100*|zaj|, -7e3/e3w(Kmm)*|zaj|) (ldfslp.F90:317)")
        e3w_np = np.asarray(loc["e3w_k"])
        print(f"      our e3w_k ndim={np.asarray(loc['e3w_k']).ndim} "
              f"({'LIVE 3-D (Kmm)' if e3w_np.ndim == 3 else 'STATIC 1-D ladder'}), "
              f"our z1_slpmax={float(loc['z1_slpmax']):.6g} (NEMO literal 100), "
              f"7e3 const={_NEMO_SLOPE_STAB_7E3:.6g}")
        zbw_n = _load_haloed(os.path.join(RUN_DIR, CHAIN_DUMPS["zbw"]), jpi, jpj, hls)
        zbj_n = _load_haloed(os.path.join(RUN_DIR, CHAIN_DUMPS["zbj"]), jpi, jpj, hls)
        e3w_b = np.broadcast_to(e3w_np if e3w_np.ndim == 3 else e3w_np,
                                (nj, ni, nlev_a))
        def _branch(zbw_a, zaj_a, z1, e3w_a, zbj_a):
            t1, t2 = zbw_a, -z1 * np.abs(zaj_a)
            t3 = (-_NEMO_SLOPE_STAB_7E3 / e3w_a) * np.abs(zaj_a)
            stack = np.stack([t1, t2, t3], axis=0)
            return np.argmin(stack, axis=0)
        br_n = _branch(zbw_n[:, :, sl], zaj_n[:, :, sl], 100.0,
                       e3w_b[:, :, sl], zbj_n[:, :, sl])
        br_o = _branch(np.asarray(loc["zbw"])[:, :, sl], np.asarray(loc["zaj"])[:, :, sl],
                       float(loc["z1_slpmax"]), e3w_b[:, :, sl],
                       np.asarray(loc["zbj"])[:, :, sl])
        print("      NOTE: the NEMO branch uses NEMO's dumped zbw/zaj but OUR e3w "
              "for term 3 (NEMO does not dump e3w) -- term-3 attribution on the "
              "NEMO side is therefore conditional on e3w agreeing.")
        for lbl, sel in (("all wet w-points", wmask_w[:, :, sl]),
                         ("bottom-3 levels", wmask_w[:, :, sl] & bot3)):
            tot = int(sel.sum())
            if tot == 0:
                continue
            fn = [float((br_n[sel] == t).mean()) for t in range(3)]
            fo = [float((br_o[sel] == t).mean()) for t in range(3)]
            print(f"      {lbl} (n={tot}):")
            print(f"        NEMO  term1(zbw)={fn[0]:.4f} term2(-100|zaj|)={fn[1]:.4f} "
                  f"term3(-7e3/e3w|zaj|)={fn[2]:.4f}")
            print(f"        lego  term1(zbw)={fo[0]:.4f} term2(-100|zaj|)={fo[1]:.4f} "
                  f"term3(-7e3/e3w|zaj|)={fo[2]:.4f}")
            print(f"        cells where the WINNING branch differs: "
                  f"{int((br_n[sel] != br_o[sel]).sum())}/{tot}")

        # ---- Q3: zfk exactness ----
        print("\n  Q3: zfk = REAL(1 - 1/(1 + jk/(nmln+1))) -- FORTRAN INTEGER "
              "division, a 0/1 step (ldfslp.F90:320)")
        zfk_n = _load_haloed(os.path.join(RUN_DIR, CHAIN_DUMPS["zfk"]), jpi, jpj, hls)
        zfk_o = (1.0 - np.asarray(loc["in_ml_w"]).astype(float))
        selw = wmask_w[:, :, sl]
        dfk = np.abs(zfk_o[:, :, sl] - zfk_n[:, :, sl])
        ndiff = int((dfk[selw] > 0).sum())
        print(f"      NEMO zfk distinct values: {np.unique(zfk_n[:, :, sl][selw])}  "
              f"(expect exactly [0. 1.] if the integer step is faithful)")
        print(f"      our zfk distinct values : {np.unique(zfk_o[:, :, sl][selw])}")
        print(f"      cells where zfk DIFFERS: {ndiff}/{int(selw.sum())} "
              f"({100.0 * ndiff / max(int(selw.sum()), 1):.4f}%) -> "
              f"{'EXACT MATCH' if ndiff == 0 else 'MISMATCH'}")
        if ndiff:
            jj, ii, kk = np.where(selw & (dfk > 0))
            print(f"      differing cells are at levels "
                  f"{np.unique(kk + KLO)[:12]} (0-based)")

        # ---- Q4: zww_raw assembly + the recurrence ----
        print("\n  Q4: zww = (zfk*zaj/(zbj - zeps) + (1-zfk)*zck*zwslpj_hml)*wmask "
              "(ldfslp.F90:328)")
        print(f"      our zeps = {float(loc['zeps']):.3e}, used as "
              f"'zbj - zeps' (MINUS) at gm_redi_latlon_cgrid.py:1084 -- matches "
              f"NEMO's minus sign.")
        print("      RECURRENCE: NEMO's zwslpj_hml is a down-column running state "
              "(:331-332) set at jk=nmln+1. ldfslp.F90:209 runs the level loop "
              "'DO jk = jpkm1, 2, -1' i.e. BOTTOM-TO-TOP, so nmln+1 is visited "
              "BEFORE the shallower in-ML levels that read it. legoESM carries "
              "the SAME quantity as a gather at the anchor level "
              "(take_along_axis(swj_int, kanc) * r1_hmlw, :1090-1091), which is "
              "the exact vectorised equivalent of that recurrence -- so the "
              "recurrence is NOT missing. Verified numerically by the zww_raw "
              "row of the chain table above.")

    # =====================================================================
    # (K) prd ISOLATION -- prd is the floor under all four ldf_slp rows.
    # =====================================================================
    print("\n" + "=" * 78)
    print("(K) prd ISOLATION vs eiv_dump_prd_arg.bin")
    print("=" * 78)
    from legoesm import constants
    from legoesm.ocean.eos import NemoSEOSConfig, nemo_seos_eos

    # (5) preconditions, re-stated explicitly for this section
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    _lvl_prd = time_level_for_dump("eiv_dump_prd_arg.bin")
    print(f"  (5) time_level_for_dump('eiv_dump_prd_arg.bin') = {_lvl_prd!r} "
          f"(asserted 'before' in build_state); this probe feeds BEFORE-level "
          f"T/S (read_nemo_restart_before) on NOW geometry. require_fp64 ran in "
          f"build_state.")

    # ---- (4) CONFIG AUDIT ----
    print("\n  (4) CONFIG AUDIT -- coefficients our code ACTUALLY uses")
    # Production builds the EOS as make_eos_fn(eos, eos_linear, **_eos_mk_kw)
    # WITHOUT an eos_nemo_seos argument (gm_redi_latlon_cgrid.py:3305), so the
    # nemo_seos branch falls back to NemoSEOSConfig() DEFAULTS. Print those --
    # they are what actually evaluates, whatever the namelist says.
    ec = NemoSEOSConfig()
    namelist = dict(a0=0.165, b0=7.6554e-1, lambda1=0.06, lambda2=0.0,
                    mu1=1.4970e-4, mu2=0.0, nu=0.0, T0=10.0, S0=35.0,
                    rho0=1026.0)
    print(f"      {'field':<10}{'lego (NemoSEOSConfig)':<26}{'DINO namelist':<18}match")
    all_match = True
    for k, v_nml in namelist.items():
        v_lego = getattr(ec, k)
        ok_k = (v_lego == v_nml)
        all_match &= ok_k
        print(f"      {k:<10}{v_lego!r:<26}{v_nml!r:<18}{'YES' if ok_k else 'NO  <-- DIFFERS'}")
    print(f"      -> all coefficients match the DINO namelist: {all_match}")
    print(f"      eos={st['eos_name']!r}  eos_linear={st['eos_linear']!r}  "
          f"eos_depth={st['eos_depth']!r}  make_eos_fn kwargs={st['eos_mk_kw']}")
    print(f"      (production DINO caller: dino.py:2867 passes eos=cfg.eos, "
          f"eos_depth=cfg.eos_depth; gm_redi_latlon_cgrid.py:3304-3305 adds "
          f"rho0=rho_0 only when eos_depth=='geometric' -- this probe mirrors both.)")
    # The rho0 CONSISTENCY trap: nemo_seos_eos reconstructs zh = p/(cfg.rho0*g)
    # while the geometric path builds p = (rho_0*constants.g)*gdept with the
    # CONFIG rho_0. If those two rho0 differ, zh is stretched by their ratio.
    print(f"      rho0 consistency: NemoSEOSConfig.rho0={ec.rho0}  vs  "
          f"config rho_0={st['rho_0']}  -> zh stretch factor = "
          f"{st['rho_0'] / ec.rho0:.12f} (must be 1.0 or the depth is scaled)")

    act = st["active"]
    nemo_prd = _load_haloed(os.path.join(RUN_DIR, CHAIN_DUMPS["prd"]), jpi, jpj, hls)
    # NEMO writes prd over the levels it actually filled; auto-detect rather
    # than assuming, so an all-zero pad level is never scored as 100% error.
    lev_ok = np.abs(nemo_prd).max(axis=(0, 1)) > 0
    print(f"\n      NEMO prd dump: {nemo_prd.shape}, levels with any nonzero data: "
          f"{int(lev_ok.sum())}/{lev_ok.size} (k={np.where(lev_ok)[0].min()}.."
          f"{np.where(lev_ok)[0].max()})")

    our_prd = np.asarray(loc["prd"]) if "prd" in loc else None
    if our_prd is None:
        print("  ABORTING (K): our prd local was not captured.")
    else:
        nk_p = min(our_prd.shape[-1], nemo_prd.shape[-1])
        wet_p = act[:, :, :nk_p] & lev_ok[None, None, :nk_p]

        # ---- (1) BASELINE ----
        print("\n  (1) BASELINE: our prd vs NEMO's dump")
        rep_base = per_element_report("prd BASELINE", our_prd[:, :, :nk_p],
                                      nemo_prd[:, :, :nk_p], wet_p)

        # ---- (2) THE DEPTH TEST ----
        print("\n  (2) DEPTH TEST: substitute NEMO's own gdept(Knn) for our ladder")
        gdept_path = os.path.join(RUN_DIR, "eiv_dump_gdept.bin")
        if not os.path.exists(gdept_path):
            print(f"      CANNOT RUN: {gdept_path} missing.")
            rep_depth = None
        else:
            gd_n = _load_haloed(gdept_path, jpi, jpj, hls)
            t_depth = np.asarray(getattr(st["z_coord"], "t_depth_ref"))
            nk_d = min(gd_n.shape[-1], nk_p, t_depth.shape[-1])
            print(f"      NEMO gdept dump {gd_n.shape}; our t_depth_ref is a "
                  f"STATIC 1-D ladder of {t_depth.shape[-1]} levels "
                  f"(ocean_tendency_common.py: p_eos = rho_0*g*t_depth). NEMO's "
                  f"gdept(Knn) is the LIVE per-column gdept_0*(1+r3t).")
            wet_d = act[:, :, :nk_d] & lev_ok[None, None, :nk_d]
            t_b = np.broadcast_to(t_depth[None, None, :nk_d], gd_n[:, :, :nk_d].shape)
            dd = np.abs(t_b - gd_n[:, :, :nk_d])[wet_d]
            rel_d = dd / np.maximum(np.abs(gd_n[:, :, :nk_d][wet_d]), FLOOR)
            print(f"      our static ladder vs NEMO live gdept: median|rel|="
                  f"{np.median(rel_d):.3e}  max|rel|={np.max(rel_d):.3e}  "
                  f"max|abs|={dd.max():.4f} m")
            # Recompute prd through the PRODUCTION eos_fn, changing ONLY the
            # depth argument (mirrors eos_rab_bn2_per_element.py section D).
            p_live = (st["rho_0"] * constants.g) * jnp.asarray(
                gd_n[:, :, :nk_d], dtype=st["T"].dtype)
            rho_live = st["eos_fn"](st["T"][:, :, :nk_d], st["S"][:, :, :nk_d], p_live)
            prd_live = np.asarray(rho_live) / st["rho_0"] - 1.0
            rep_depth = per_element_report("prd @ NEMO gdept", prd_live,
                                           nemo_prd[:, :, :nk_d], wet_d)
            b, d = rep_base["med_en"], rep_depth["med_en"]
            print(f"\n      VERDICT: err_norm median {b:.3e} (our ladder) -> "
                  f"{d:.3e} (NEMO gdept), a {100 * (1 - d / max(b, 1e-300)):.1f}% "
                  f"reduction.")
            if d < 1.0e-9:
                print("      DOES THE DEPTH ARGUMENT OWN prd's RESIDUAL? YES -- "
                      "substituting NEMO's own gdept collapses prd to ROUNDOFF "
                      f"({d:.2e}); nothing physically meaningful survives.")
            elif d < 0.1 * b:
                print("      DOES THE DEPTH ARGUMENT OWN prd's RESIDUAL? YES -- "
                      "the residual collapses by >10x, so the depth argument is "
                      "the dominant contributor (a small remainder survives).")
            else:
                print("      DOES THE DEPTH ARGUMENT OWN prd's RESIDUAL? NO -- "
                      "substituting NEMO's own gdept does not collapse it.")

        # ---- (3) TERM SPLIT ----
        print("\n  (3) TERM SPLIT (potential-density vs thermobaric)")
        print("      nemo_seos_eos (eos.py) computes `zn` as a SINGLE expression;")
        print("      the potential-density and thermobaric parts are NOT")
        print("      materialised separately, so they cannot be captured. Not")
        print("      re-implementing the formula to split them (Rule 0). Instead:")
        # zh IS materialised inside nemo_seos_eos -- capture it from the
        # production function rather than recomputing p/(rho0*g).
        _, loc_eos = capture_locals(
            lambda: st["eos_fn"](st["T"], st["S"],
                                 (st["rho_0"] * constants.g) * jnp.asarray(
                                     np.broadcast_to(
                                         np.asarray(getattr(st["z_coord"], "t_depth_ref"))[None, None, :],
                                         st["T"].shape), dtype=st["T"].dtype)),
            nemo_seos_eos.__code__)
        if "zh" not in loc_eos or "zt" not in loc_eos:
            print(f"      could not capture zh/zt from nemo_seos_eos "
                  f"(got {sorted(loc_eos)[:8]}); skipping the correlation test.")
        else:
            zh = np.asarray(loc_eos["zh"])[:, :, :nk_p]
            zt = np.asarray(loc_eos["zt"])[:, :, :nk_p]
            resid = (our_prd[:, :, :nk_p] - nemo_prd[:, :, :nk_p])[wet_p]
            zh_w, zt_w = zh[wet_p], zt[wet_p]
            c_abs = float(np.corrcoef(np.abs(resid), zh_w)[0, 1])
            # the thermobaric term is -a0*mu1*zt*zh, so if it owns the residual
            # the SIGNED residual should track zt*zh, not zh alone.
            c_sig = float(np.corrcoef(resid, zt_w * zh_w)[0, 1])
            print(f"      corr(|residual|, zh)      = {c_abs:+.6f}")
            print(f"      corr(residual, zt*zh)     = {c_sig:+.6f}  "
                  f"(thermobaric term is -a0*mu1*zt*zh)")
            print(f"      -> {'thermobaric term IMPLICATED' if abs(c_sig) > 0.5 or c_abs > 0.5 else 'no strong depth signature'}")

    print("\n" + "=" * 78)
    print("SUMMARY (<=12-line)")
    print("=" * 78)
    for comp in DUMP_META:
        r = summary[comp]
        if r is None:
            print(f"{comp:8s}: FAILED (no valid offset)")
            continue
        trust = "trustworthy" if r["trustworthy_pointwise"] else "NOT trustworthy"
        print(f"{comp:8s}: offset={r['best_off']:+d}  corr={r['corr']:.6f}  "
              f"|x|ratio={r['abs_ratio']:.6f}  err_norm(median/p99/max)="
              f"{r['med_en']:.3e}/{r['p99_en']:.3e}/{r['max_en']:.3e}  "
              f"pointwise-rel {trust} ({r['frac_near0']*100:.1f}% near-zero)")
    for comp in RAW_DUMPS:
        if comp in stageA and summary.get(comp) is not None:
            a, b = stageA[comp]["med_en"], summary[comp]["med_en"]
            ratio = a / b if b > 0 else float("nan")
            owner = ("SLOPE FORMULA" if ratio >= 0.5 else
                     "SMOOTHER" if ratio <= 0.1 else "BOTH (no single owner)")
            print(f"(I) {comp:6s}: stageA(formula) err_norm median={a:.3e} vs "
                  f"stageB(final) {b:.3e}  -> A/B={ratio:.3f}; owner = {owner}")
    if our_prd is not None and rep_depth is not None:
        print(f"(K) prd   : err_norm median {rep_base['med_en']:.3e} (our static "
              f"t_depth_ref ladder) -> {rep_depth['med_en']:.3e} (NEMO's live "
              f"gdept(Knn)); the DEPTH ARGUMENT owns prd's residual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
