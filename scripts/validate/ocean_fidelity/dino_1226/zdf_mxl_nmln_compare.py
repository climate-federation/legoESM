#!/usr/bin/env python
"""#1226 zdf_mxl: legoESM's ``nmln`` (mixed-layer w-level index) vs NEMO's own
dumped ``nmln``/``hmlp``, with a SPLIT that isolates whether the residual is
in the SELECTION logic or in the N^2 (bn2) INPUT.

Function under test: :func:`legoesm.ocean.physics.lateral_mixing.
gm_redi_latlon_cgrid._nemo_mld_from_n2_integral` -- the real production path
(called from ``compute_nemo_native_slopes``/``compute_isopycnal_slopes_
latlon_cgrid`` with ``cfg.mld_criterion="n2_integral"``, the ``nemo_dino_
kamm_mlf`` recipe's setting). This probe calls that function directly; it
does NOT re-implement any of its numerics (Rule 0).

NEMO oracle side (``zdfmxl.F90:91-105``, run dir RUN_GDB, kt==nit000 dumps):
  * ``dump_nmln.bin``    : nmln, the w-level index (1-based NEMO convention,
                           stored as float64), (jpj,jpi) haloed.
  * ``dump_hmlp.bin``    : hmlp, the MLD in metres, same shape.
  * ``eiv_dump_rn2b.bin``: NEMO's OWN rn2b (linearised alpha/beta N^2 at the
                           w-interfaces), (jpk=35,jpj,jpi) haloed.
  * ``tke_dump_rn2b.bin``: the SAME rn2b array, (jpk=36,ni,nj) INTERIOR
                           (no halo). Verified bit-identical to
                           ``eiv_dump_rn2b`` over the shared 35 levels
                           (``np.array_equal`` -- see the printed check
                           below); it simply carries ONE MORE level (the
                           deepest interior interface) that ``eiv_dump_rn2b``
                           truncates. Since ``_nemo_mld_from_n2_integral``'s
                           bn2 call needs all ``nlev-1=35`` interfaces, the
                           SPLIT substitutes from ``tke_dump_rn2b`` (the
                           complete array) rather than truncating the
                           function's real input to 34 levels, which would
                           be a self-inflicted mismatch, not evidence about
                           the selection logic.

THE SPLIT: re-run ``_nemo_mld_from_n2_integral`` with NEMO's own rn2b
monkeypatched in place of ``compute_buoyancy_frequency_nemo_bn2`` (imported
INSIDE the function body, so patched at ``legoesm.ocean.eos.compute_
buoyancy_frequency_nemo_bn2``, the module attribute the function's inline
import resolves against). If the mismatch count drops to 0, the SELECTION
logic is right and 100% of the residual is our bn2; if it stays roughly the
same, the selection logic itself differs from NEMO.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/zdf_mxl_nmln_compare.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# See bn2_alpha_compare.py: DINO is full-step, e3t_1d vs e3t_0 diverge by up
# to 105 m -- pin the NEMO ladder. Reported below (this IS the knob #1226
# flags as env-var-controlled + must be surfaced).
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import importlib.util
import sys

import numpy as np
import jax.numpy as jnp

# scripts/ is not a package -- import the sibling probe by path (it is not
# importable as `scripts.validate...` without a __init__.py chain), per the
# instruction to import rather than duplicate _read_dims/_load_haloed.
_sib_path = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib_path)
_bn2_alpha_compare = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bn2_alpha_compare
_spec.loader.exec_module(_bn2_alpha_compare)
_read_dims = _bn2_alpha_compare._read_dims
_load_haloed = _bn2_alpha_compare._load_haloed
_load_interior = _bn2_alpha_compare._load_interior
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
)
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean import eos as eos_mod
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    _nemo_mld_from_n2_integral, _nemo_native_active_3d,
    gm_redi_density_and_jacobian,
)

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"


def build_state():
    # fp64 policy: the depth ladder is cast to get_policy().control, which
    # defaults to float32 and rounds NEMO's f64 gdept_1d to ~7 digits (#1226).
    if os.environ.get("LEGOESM_FIDELITY_FP64", "1") == "1":
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64())
    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    # TIME LEVEL (#1226): zdfmxl.F90:98 integrates rn2b -- the BEFORE N^2 --
    # and rn2b comes from bn2( ts(:,:,:,:,Nbb), ..., Nnn ) (stpmlf.F90:184-186):
    # T/S BEFORE, geometry NOW.  Feeding NOW T/S puts |T_now - T_before| into
    # the comparison, which is exactly what produced the phantom structure in
    # eos_rab_bn2_per_element.py's first run.
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                      omega=cfg.omega)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    z_coord = br.z_coord
    state = br.state
    mask = state.land_mask.data
    H_bathy = state.H_bathy.data
    T = jnp.asarray(np.asarray(bef.T).reshape(state.T.data.shape))
    S = jnp.asarray(np.asarray(bef.S).reshape(state.S.data.shape))
    eta = jnp.asarray(state.eta.data)

    eos_fn = make_eos_fn(mc.eos, mc.eos_linear)
    _, jacobian = gm_redi_density_and_jacobian(
        T, S, eta, H_bathy, br.geometry, z_coord,
        eos=mc.eos, eos_linear=mc.eos_linear, mask=mask,
        rho_0=mc.rho_0, g=mc.g,
    )
    active_3d = _nemo_native_active_3d(mask, z_coord, H_bathy, T.dtype)

    # --- oracle-comparison preconditions, both MECHANICAL (#1226) ---

    # Rule 1c: NEMO is fp64. Assert it on the GEOMETRY too -- an f32

    # ladder under an f64 state passes every state-level check.

    from legoesm.ocean.fidelity.precision_gate import require_fp64

    require_fp64(z_coord, T, S, context="zdf_mxl nmln compare")

    # Time level: assert the T/S we are about to compare is the level

    # this dump actually belongs to, rather than assuming 'now'.

    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    _lvl = time_level_for_dump("dump_nmln.bin")

    if _lvl != "before":

        raise ValueError(

            f"this probe feeds BEFORE-level T/S but dump_nmln.bin is "

            f"{_lvl!r}-level; fix the probe or the registry")


    return dict(
        jpi=jpi, jpj=jpj, jpk=jpk, hls=hls, ni_ni=mask.shape,
        z_coord=z_coord, mask=mask, T=T, S=S, eos_fn=eos_fn,
        gm_redi=mc.gm_redi, rho_0=mc.rho_0, g=mc.g,
        active_3d=active_3d, jacobian=jacobian,
    )


def call_mld(st):
    """The one call site under test -- unmodified production function."""
    return _nemo_mld_from_n2_integral(
        st["T"], st["S"], st["mask"], st["z_coord"], st["eos_fn"],
        st["gm_redi"].mld_rho_c, st["g"], st["rho_0"],
        active_3d=st["active_3d"], jacobian=st["jacobian"],
    )


def shift_scan(m_base, nemo_nmln, wet):
    """Empirically determine the index-convention offset -- do not assume it."""
    best = None
    for o in (0, 1, 2, 3):
        lego = np.asarray(m_base) + o
        match = int(((lego == nemo_nmln) & wet).sum())
        print(f"    offset o={o}: nmln_lego = m_base+{o} matches nemo nmln "
              f"in {match}/{int(wet.sum())} wet columns")
        if best is None or match > best[1]:
            best = (o, match)
    return best[0]


def main() -> int:
    st = build_state()
    nj, ni = st["ni_ni"]
    print(f"grid interior (nj,ni,nlev) = ({nj},{ni},{st['jpk'] - 1}) "
          f"LEGOESM_NEMO_E3T={os.environ.get('LEGOESM_NEMO_E3T')}")

    nemo_nmln = _load_haloed(os.path.join(RUN_DIR, "dump_nmln.bin"),
                             st["jpi"], st["jpj"], st["hls"])[..., 0].round().astype(int)
    nemo_hmlp = _load_haloed(os.path.join(RUN_DIR, "dump_hmlp.bin"),
                             st["jpi"], st["jpj"], st["hls"])[..., 0]
    nemo_rn2b_eiv = _load_haloed(os.path.join(RUN_DIR, "eiv_dump_rn2b.bin"),
                                 st["jpi"], st["jpj"], st["hls"])  # (nj,ni,35)
    nemo_rn2b_full = _load_interior(os.path.join(RUN_DIR, "tke_dump_rn2b.bin"),
                                    st["ni_ni"][1], st["ni_ni"][0])  # (nj,ni,36)

    # eiv_dump_rn2b (the file named in the #1226 task spec) is the SAME
    # physical array as tke_dump_rn2b, just missing its deepest interior
    # level -- verify this empirically rather than assume it (Rule 0).
    n_overlap = min(nemo_rn2b_eiv.shape[-1], nemo_rn2b_full.shape[-1])
    identical = np.array_equal(nemo_rn2b_eiv[..., :n_overlap],
                               nemo_rn2b_full[..., :n_overlap])
    print(f"[check] eiv_dump_rn2b (shape {nemo_rn2b_eiv.shape}) vs "
          f"tke_dump_rn2b (shape {nemo_rn2b_full.shape}) over the shared "
          f"{n_overlap} levels: array_equal={identical} "
          f"(expect True -- same quantity, tke has 1 more level)")
    nemo_rn2b = nemo_rn2b_full  # complete (nlev-1=35 usable) array for the split

    wet = np.asarray(st["mask"]) > 0.5
    n_wet = int(wet.sum())

    # --- (1) BASELINE ---
    print("\n=== (1) BASELINE: lego m_base vs NEMO nmln ===")
    hml, m_base = call_mld(st)
    m_base = np.asarray(m_base)
    hml = np.asarray(hml)
    best_o = shift_scan(m_base, nemo_nmln, wet)
    nmln_lego = m_base + best_o
    diff = (nmln_lego - nemo_nmln)[wet]
    n_diff = int((diff != 0).sum())
    print(f"  best offset o={best_o}")
    print(f"  total wet columns = {n_wet}, mismatched = {n_diff} "
          f"({100.0 * n_diff / n_wet:.2f}%)")
    vals, counts = np.unique(diff, return_counts=True)
    print("  histogram (lego - nemo):",
          {int(v): int(c) for v, c in zip(vals, counts)})

    m_hml = wet & np.isfinite(hml) & np.isfinite(nemo_hmlp)
    corr_hml = float(np.corrcoef(hml[m_hml], nemo_hmlp[m_hml])[0, 1])
    abs_diff_hml = np.abs(hml[m_hml] - nemo_hmlp[m_hml])
    max_abs_hml = float(abs_diff_hml.max())
    mismatched_col = (nmln_lego - nemo_nmln)[m_hml] != 0
    max_abs_hml_matched = float(abs_diff_hml[~mismatched_col].max())
    print(f"  hml vs dump_hmlp: corr={corr_hml:.6f} max|diff|={max_abs_hml:.4f} m "
          f"(max|diff| on the {n_wet - n_diff} level-MATCHED columns = "
          f"{max_abs_hml_matched:.4f} m -- the {max_abs_hml:.1f} m max is the "
          f"{n_diff} off-by-a-level columns above, a full layer thickness at "
          "depth, not a separate defect)")

    # --- (2) THE SPLIT ---
    print("\n=== (2) SPLIT: substitute NEMO's own rn2b for our computed bn2 ===")
    real_bn2 = eos_mod.compute_buoyancy_frequency_nemo_bn2
    probe_shape = {}

    def patched_bn2(T, S, gdept, gdepw_int, cfg=None, g=None):
        shp = probe_shape.setdefault("shape", real_bn2(T, S, gdept, gdepw_int,
                                                        cfg, g=g).shape)
        nk = shp[-1]
        nk_dump = nemo_rn2b.shape[-1]
        # Empirically verified alignment (this script, corr=1.000000 on both
        # `now` and `before` time levels): rn2b(1)=0 is a surface pad NEMO
        # always writes; our bn2's interior interface i == rn2b dump index
        # i+1. i.e. nemo_int = nemo_rn2b[..., 1:1+nk].
        if nk_dump < nk + 1:
            raise AssertionError(
                f"substituted rn2b has {nk_dump} levels, need >= {nk + 1} "
                f"(1 leading pad + {nk} interior interfaces matching real "
                f"bn2 shape {shp}) -- level count mismatch would silently "
                "corrupt the comparison; ABORTING split.")
        nemo_int = nemo_rn2b[..., 1:1 + nk]
        assert nemo_int.shape == shp, (
            f"substituted rn2b shape {nemo_int.shape} != real bn2 shape {shp}")
        return jnp.asarray(nemo_int)

    eos_mod.compute_buoyancy_frequency_nemo_bn2 = patched_bn2
    try:
        hml_split, m_base_split = call_mld(st)
    finally:
        eos_mod.compute_buoyancy_frequency_nemo_bn2 = real_bn2

    m_base_split = np.asarray(m_base_split)
    nmln_split = m_base_split + best_o
    diff_split = (nmln_split - nemo_nmln)[wet]
    n_diff_split = int((diff_split != 0).sum())
    print(f"  bn2 substituted with NEMO's own rn2b (shape {probe_shape['shape']}); "
          f"mismatched = {n_diff_split}/{n_wet} "
          f"({100.0 * n_diff_split / n_wet:.2f}%)  [baseline was {n_diff}]")
    if n_diff_split == 0:
        verdict = ("SELECTION LOGIC CORRECT -- residual is entirely in our "
                   "computed N^2 (bn2), not the nmln selection.")
    elif n_diff_split >= 0.8 * n_diff:
        verdict = ("SELECTION LOGIC DIFFERS from NEMO -- substituting NEMO's "
                   "own rn2b did not fix the mismatches.")
    else:
        verdict = (f"MIXED -- baseline {n_diff} -> split {n_diff_split}; "
                   "both bn2 and selection logic contribute, claiming neither "
                   "alone explains it.")
    print(f"  INTERPRETATION: {verdict}")

    # --- (3) MISMATCHED COLUMNS ---
    print("\n=== (3) MISMATCHED COLUMNS (baseline) ===")
    thresh = float(st["g"] * st["gm_redi"].mld_rho_c / st["rho_0"])
    print(f"  zN2_c threshold = {thresh:.6e}")
    jj, ii = np.where(wet & (nmln_lego - nemo_nmln != 0))
    rel_errs = []
    print(f"  {'(j,i)':<12}{'lego_nmln':<11}{'nemo_nmln':<11}{'cum@lego':<14}"
          f"{'cum@nemo':<14}{'|cum-thr|/thr (lego level)':<28}")
    for j, i in zip(jj, ii):
        ll, nn = int(nmln_lego[j, i]), int(nemo_nmln[j, i])
        # recompute cum at this column from the (unpatched) baseline call by
        # re-deriving via the same rn2b NEMO dumped, restricted to this column,
        # purely for REPORTING the threshold proximity -- not part of the
        # function under test.
        e3w_col = np.diff(np.asarray(st["z_coord"].t_depth_ref))  # (nlev-1,)
        col_rn2b = nemo_rn2b[j, i, 1:1 + len(e3w_col)]  # same alignment as the split
        contrib = np.maximum(col_rn2b, 0.0) * e3w_col
        cum = np.cumsum(contrib)
        lego_level = min(max(ll - 2, 0), len(cum) - 1)
        nemo_level = min(max(nn - 2, 0), len(cum) - 1)
        cum_lego = cum[lego_level]
        cum_nemo = cum[nemo_level]
        rel = abs(cum_nemo - thresh) / thresh
        rel_errs.append(rel)
        print(f"  ({j:3d},{i:3d})  {ll:<11d}{nn:<11d}{cum_lego:<14.6e}"
              f"{cum_nemo:<14.6e}{rel:<28.3e}")
    if rel_errs:
        rel_errs = np.asarray(rel_errs)
        print(f"\n  |cum-thresh|/thresh over mismatched columns: "
              f"min={rel_errs.min():.3e} median={np.median(rel_errs):.3e} "
              f"max={rel_errs.max():.3e}")
        n_knife = int((rel_errs < 1e-3).sum())
        print(f"  knife-edge (<1e-3): {n_knife}/{len(rel_errs)}")
    else:
        print("  (no mismatched columns)")

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("SUMMARY (paste into status doc)")
    print("=" * 60)
    print(f"e3t mode={os.environ.get('LEGOESM_NEMO_E3T')}  offset o={best_o}")
    print(f"baseline: {n_diff}/{n_wet} mismatched nmln "
          f"({100.0 * n_diff / n_wet:.2f}%)")
    print(f"hml vs hmlp: corr={corr_hml:.6f} max|diff|={max_abs_hml:.4f} m")
    print(f"split (NEMO rn2b substituted): {n_diff_split}/{n_wet} mismatched")
    print(f"verdict: {verdict}")
    if rel_errs is not None and len(rel_errs):
        print(f"mismatch proximity to threshold: min={rel_errs.min():.3e} "
              f"median={np.median(rel_errs):.3e} max={rel_errs.max():.3e} "
              f"knife-edge(<1e-3)={n_knife}/{len(rel_errs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
