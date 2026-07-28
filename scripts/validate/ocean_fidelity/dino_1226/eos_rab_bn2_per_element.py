#!/usr/bin/env python
"""#1226 tier-2: PER-ELEMENT error of ``eos_rab`` (alpha/beta) and ``bn2``
(rn2b) vs NEMO's own kt==nit000 dumps, plus the SPLIT that determines whether
bn2's residual is inherited from eos_rab or generated independently inside
bn2 itself.

This is a per-element companion to ``bn2_alpha_compare.py`` (which reports
only aggregate corr/ratio -- this script adds median/mean/max/p99 |rel| error
distributions, the location of the worst element, and the eos_rab->bn2
inheritance split. Bridge construction, NEMO dump loaders, and the calling
convention for ``gdept``/``gdepw_int`` are reused/replicated from
``bn2_alpha_compare.py`` and ``zdf_mxl_nmln_compare.py`` -- no numerics
re-derived here (Rule 0).

Calling convention for gdept/gdepw_int (replicated from the production caller
``gm_redi_latlon_cgrid._nemo_mld_from_n2_integral``, lines ~489-508): the
STATIC reference ladders (``nemo_bn2_depth_ladders``) are broadcast to the
live grid via the z*-Jacobian ``J = jacobian[..., None]`` from
``gm_redi_density_and_jacobian``: ``gdept = t_depth_ref * J``,
``gdepw_int = gdepw_int_ref * J``. Equivalently ``nemo_bn2_live_ladders``
(used by ``bn2_alpha_compare.py``) computes the same live-grid stretch from
``eta``/``H_bathy`` directly -- both are exercised in ``bn2_alpha_compare.py``
as "PRE-FIX static" vs "POST-FIX live"; this script uses the live (POST-FIX,
Jacobian) form throughout, since that is what production actually calls.

Monkeypatch target (Step 1 finding): ``compute_buoyancy_frequency_nemo_bn2``
calls ``nemo_seos_alpha_beta(T, S, gdept, cfg)`` as a bare name inside the
SAME module (``legoesm/ocean/eos.py``) -- Python resolves this via the
module's global namespace dict AT CALL TIME, so patching the module attribute
``legoesm.ocean.eos.nemo_seos_alpha_beta`` (not any importer's namespace) is
the correct patch point. Verified this is not shadowed by a local import or
alias inside ``eos.py`` (only ``jax.numpy``/``legoesm.constants``/
``legoesm.core.precision`` are imported at module scope there).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/eos_rab_bn2_per_element.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Per Step 2 / bn2_alpha_compare.py: DINO is full-step; pin the NEMO e3t
# ladder so gdept_0 matches NEMO's own dump exactly (see that script's
# header for the e3t_1d vs e3t_0 105 m divergence).
os.environ["LEGOESM_NEMO_E3T"] = "both"

import importlib.util
import sys

import numpy as np
import jax.numpy as jnp

# scripts/ is not a package -- import the sibling probe by path, exactly as
# zdf_mxl_nmln_compare.py does, to reuse _read_dims/_load_haloed/_load_interior
# without duplicating them.
_sib_path = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib_path)
_bn2_alpha_compare = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bn2_alpha_compare
_spec.loader.exec_module(_bn2_alpha_compare)
_read_dims = _bn2_alpha_compare._read_dims
_load_haloed = _bn2_alpha_compare._load_haloed
_load_interior = _bn2_alpha_compare._load_interior

from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.eos import (
    NemoSEOSConfig,
    compute_buoyancy_frequency_nemo_bn2,
    nemo_bn2_live_ladders,
)
from legoesm.ocean import eos as eos_mod
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
    gm_redi_density_and_jacobian, _nemo_native_active_3d,
)

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"

# rel = |lego - nemo| / max(|nemo|, FLOOR). alpha/beta ~ O(1e-4 - 1e-1),
# N^2 ~ O(1e-8 - 1e-4); 1e-10 is >=2 orders below the smallest real signal
# in either quantity on this state (checked empirically below against the
# actual dumped magnitudes) so it never distorts a real ratio, only guards
# division by a NEMO-dumped exact zero (e.g. masked/dry cells that slip
# through the wet-mask floor, or a genuinely-zero rn2b pad level).
FLOOR = 1.0e-10


def per_element_stats(name, lego, nemo, wet):
    """median/mean/max/p99 |rel| + aggregate ratio/corr, restricted to `wet`."""
    m = wet & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    rel = np.abs(lo - ne) / np.maximum(np.abs(ne), FLOOR)
    ratio_mean = float(lo.mean() / ne.mean())
    if ne.std() == 0.0 or lo.std() == 0.0:
        # Degenerate case (e.g. beta constant over the whole wet field on this
        # DINO state): Pearson corr is undefined (zero variance), NOT 1.0 or
        # NaN-as-failure. Report the (perfect, since lo==ne exactly here) match
        # via ratio_mean/max_rel instead of a fabricated corr.
        corr = float("nan")
    else:
        corr = float(np.corrcoef(lo, ne)[0, 1])
    stats = dict(
        n=int(m.sum()),
        median_rel=float(np.median(rel)),
        mean_rel=float(np.mean(rel)),
        max_rel=float(np.max(rel)),
        p99_rel=float(np.percentile(rel, 99)),
        ratio_mean=ratio_mean,
        corr=corr,
    )
    # location of the max |rel| in the FULL (unmasked-shape) array
    rel_full = np.full(wet.shape, -1.0)
    rel_full[m] = rel
    idx = np.unravel_index(np.argmax(rel_full), rel_full.shape)
    stats["argmax_idx"] = idx
    nlev = wet.shape[-1]
    k = idx[-1]
    depth_note = "near-surface" if k < nlev * 0.25 else (
        "deep" if k > nlev * 0.75 else "mid-column")
    stats["argmax_depth_note"] = f"level {k}/{nlev - 1} ({depth_note})"
    # p99 already characterizes the bulk tail; also report how many wet
    # elements exceed a 1% relative error, so a single max|rel| spike (e.g.
    # from a near-zero-N^2 cell where FLOOR dominates the denominator) doesn't
    # get mistaken for a widespread defect.
    n_gt_1pct = int((rel > 1.0e-2).sum())
    stats["n_gt_1pct"] = n_gt_1pct
    print(f"  {name:<28s} n={stats['n']:<8d} "
          f"median|rel|={stats['median_rel']:.3e}  mean|rel|={stats['mean_rel']:.3e}  "
          f"max|rel|={stats['max_rel']:.3e}  p99|rel|={stats['p99_rel']:.3e}  "
          f"n(rel>1%)={n_gt_1pct}")
    print(f"    {'':<28s} ratio_mean(lego/nemo)={stats['ratio_mean']:.6f}  "
          f"corr={stats['corr']:.8f}  argmax@{idx} [{stats['argmax_depth_note']}]")
    return stats


def build_state():
    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    # TIME LEVEL (#1226, corrected 2026-07-28).  NEMO stpmlf.F90:184 is
    #     CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nnn )
    # -- T/S at the BEFORE level, depth at NOW.  dump_alpha_b / tke_dump_rn2b
    # are therefore BEFORE-level quantities (bn2_alpha_compare.py says so in
    # its header).  The first version of this probe fed the NOW T/S, which put
    # |T_now - T_before| straight into the "error": that is largest in the
    # thermocline, which is exactly the levels 6-8 structure it reported.
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
    # BEFORE-level T/S (Nbb) on the NOW geometry (Nnn) -- the exact NEMO pairing.
    _tb = np.asarray(bef.T); _sb = np.asarray(bef.S)
    T = jnp.asarray(_tb.reshape(state.T.data.shape))
    S = jnp.asarray(_sb.reshape(state.S.data.shape))
    eta = jnp.asarray(state.eta.data)

    _, jacobian = gm_redi_density_and_jacobian(
        T, S, eta, H_bathy, br.geometry, z_coord,
        eos=mc.eos, eos_linear=mc.eos_linear, mask=mask,
        rho_0=mc.rho_0, g=mc.g,
    )
    # Same 3-D wet-mask source as zdf_mxl_nmln_compare.py: the exact per-column
    # bottom-level index compare (is_active), not a re-derived float compare
    # (see _nemo_native_active_3d's docstring for why the float form mismatches
    # NEMO at ~1861/10348 columns on this exact state).
    active_3d = _nemo_native_active_3d(mask, z_coord, H_bathy, T.dtype)

    return dict(
        jpi=jpi, jpj=jpj, jpk=jpk, hls=hls, ni_ni=mask.shape,
        z_coord=z_coord, mask=mask, T=T, S=S, eta=eta, H_bathy=H_bathy,
        jacobian=jacobian, active_3d=active_3d,
    )


def main() -> int:
    print(f"LEGOESM_NEMO_E3T={os.environ.get('LEGOESM_NEMO_E3T')}  "
          f"(pinned NEMO e3t ladder -- see script header / bn2_alpha_compare.py)")
    st = build_state()
    jpi, jpj, jpk, hls = st["jpi"], st["jpj"], st["jpk"], st["hls"]
    nj, ni = st["ni_ni"]
    T, S = st["T"], st["S"]
    dtype = np.asarray(T).dtype
    wet3 = np.asarray(st["mask"] > 0.5)[..., None] & np.ones(T.shape[-1], dtype=bool)
    print(f"grid interior (nj,ni,nlev) = ({nj},{ni},{T.shape[-1]})  "
          f"total wet T-columns = {int((np.asarray(st['mask']) > 0.5).sum())}  "
          f"total wet T-cells (3-D) = {int(wet3.sum())}")

    # NEMO dumps (haloed T-point files determine level counts from file size).
    def _nlev_from_size(path):
        size = os.path.getsize(path)
        return size // 8 // (jpi * jpj)

    nlev_ab = _nlev_from_size(os.path.join(RUN_DIR, "dump_alpha_b.bin"))
    nemo_alpha = _load_haloed(os.path.join(RUN_DIR, "dump_alpha_b.bin"), jpi, jpj, hls)
    nemo_beta = _load_haloed(os.path.join(RUN_DIR, "dump_beta_b.bin"), jpi, jpj, hls)
    nemo_rn2b = _load_interior(os.path.join(RUN_DIR, "tke_dump_rn2b.bin"), ni, nj)  # (nj,ni,36)
    print(f"dump_alpha_b/dump_beta_b nlev={nlev_ab}  tke_dump_rn2b nlev={nemo_rn2b.shape[-1]} "
          "(36 levels per task spec, NOT eiv_dump_rn2b)")

    # --- replicate the production calling convention for gdept/gdepw_int ---
    # (gm_redi_latlon_cgrid._nemo_mld_from_n2_integral, ~L489-508): static
    # reference ladders stretched by the z*-Jacobian. nemo_bn2_live_ladders
    # is the eta/H_bathy-driven equivalent already used by bn2_alpha_compare.py
    # -- use it directly (same live-grid quantity, avoids re-deriving the
    # Jacobian-broadcast arithmetic inline).
    gdept, gdepw_int = nemo_bn2_live_ladders(st["z_coord"], st["eta"], st["H_bathy"])
    gdept = jnp.asarray(gdept, dtype=dtype)
    gdepw_int = jnp.asarray(gdepw_int, dtype=dtype)
    print(f"gdept shape={gdept.shape}  gdepw_int shape={gdepw_int.shape}")

    cfg = NemoSEOSConfig()
    g_nemo = NEMO_CONSTANTS_CONFIG.g
    active_3d = np.asarray(st["active_3d"]) > 0.5   # (nj,ni,nlev) exact per-cell wet mask
    n_wet3d = int(active_3d.sum())
    print(f"3-D active (wet) T-cells (is_active mask, matches zdf_mxl_nmln_compare.py) "
          f"= {n_wet3d} (vs naive 2-D-column-broadcast {int((np.asarray(st['mask'])>0.5).sum() * T.shape[-1])})")

    # =========================================================================
    print("\n" + "=" * 78)
    print("(A) eos_rab (alpha, beta) per-element vs NEMO dumps, T-points")
    print("=" * 78)
    alpha_lego, beta_lego = eos_mod.nemo_seos_alpha_beta(T, S, gdept, cfg)
    alpha_lego = np.asarray(alpha_lego)
    beta_lego = np.asarray(beta_lego)
    wet_ab = active_3d[..., :nlev_ab]
    stats_alpha = per_element_stats(
        "alpha vs dump_alpha_b", alpha_lego[..., :nlev_ab], nemo_alpha, wet_ab)
    stats_beta = per_element_stats(
        "beta vs dump_beta_b", beta_lego[..., :nlev_ab], nemo_beta, wet_ab)
    print(f"  FLOOR used in rel = |lego-nemo|/max(|nemo|,FLOOR): FLOOR={FLOOR:.1e} "
          f"(alpha/beta magnitude ~O(1e-4-1e-1); floor is >=6 orders below that)")

    # =========================================================================
    print("\n" + "=" * 78)
    print("(B) bn2 (N^2) per-element vs NEMO's tke_dump_rn2b, W-points")
    print("=" * 78)
    n2_lego = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw_int, cfg, g=g_nemo))
    nk = n2_lego.shape[-1]
    # Index alignment (established in bn2_alpha_compare.py / zdf_mxl_nmln_compare.py):
    # rn2b(1)=0 surface pad; legoESM interior interface i == dump index i+1.
    nemo_n2_interior = nemo_rn2b[..., 1:1 + nk]
    # W-point (interface) wet mask: an interface is wet iff BOTH bracketing
    # T-cells are active (same convention as the iface_wet build at
    # gm_redi_latlon_cgrid.py:566-567).
    wet_w = active_3d[..., :nk] & active_3d[..., 1:1 + nk]
    stats_bn2 = per_element_stats("bn2 vs tke_dump_rn2b[1:]", n2_lego, nemo_n2_interior, wet_w)
    print(f"  FLOOR used: {FLOOR:.1e} (N^2 magnitude ~O(1e-8-1e-4); "
          f"floor is >=2 orders below that)")
    print(f"  g used (NEMO_CONSTANTS_CONFIG.g) = {g_nemo}")

    # =========================================================================
    print("\n" + "=" * 78)
    print("(C) THE SPLIT: does bn2 inherit its error from eos_rab?")
    print("=" * 78)
    real_alpha_beta = eos_mod.nemo_seos_alpha_beta

    # compute_buoyancy_frequency_nemo_bn2 needs alpha/beta at T.shape[-1]=36
    # T-levels to build its nk=35 interfaces (it indexes [...,:-1]/[...,1:]),
    # but dump_alpha_b/dump_beta_b only carry nlev_ab=35 T-levels (NEMO wrote
    # one fewer level than the grid has -- same asymmetry documented for
    # tke_dump_rn2b vs eiv_dump_rn2b in zdf_mxl_nmln_compare.py). Pad the
    # missing deepest level by repeating the last dumped level -- this ONLY
    # feeds the k=34 (deepest) interface of the split's bn2 call, which is
    # EXCLUDED from stats_split below via wet_w (the deepest interface's
    # bracketing T-cell at index 35 was never dumped, so it is not scored);
    # the pad exists purely so the function call does not shape-error.
    n_missing = T.shape[-1] - nlev_ab
    if n_missing > 0:
        pad_a = jnp.repeat(jnp.asarray(nemo_alpha[..., -1:], dtype=dtype), n_missing, axis=-1)
        pad_b = jnp.repeat(jnp.asarray(nemo_beta[..., -1:], dtype=dtype), n_missing, axis=-1)
        dumped_alpha = jnp.concatenate([jnp.asarray(nemo_alpha, dtype=dtype), pad_a], axis=-1)
        dumped_beta = jnp.concatenate([jnp.asarray(nemo_beta, dtype=dtype), pad_b], axis=-1)
    else:
        dumped_alpha = jnp.asarray(nemo_alpha, dtype=dtype)
        dumped_beta = jnp.asarray(nemo_beta, dtype=dtype)
    print(f"  padding: dump has {nlev_ab} T-levels, grid needs {T.shape[-1]} "
          f"-> repeated the deepest dumped level {n_missing}x (only feeds the "
          "excluded deepest interface, see comment above)")

    def patched_alpha_beta(T_, S_, gdept_, cfg_=None):
        return dumped_alpha, dumped_beta

    eos_mod.nemo_seos_alpha_beta = patched_alpha_beta
    try:
        n2_split = np.asarray(compute_buoyancy_frequency_nemo_bn2(
            T, S, gdept, gdepw_int, cfg, g=g_nemo))
        # verification the patch took effect: the patched fn returns the DUMPED
        # array (bit-identical over the actually-dumped levels) and DIFFERS
        # from legoESM's own computed alpha (unless coincidentally equal,
        # which would be suspicious -- check).
        patched_a, patched_b = patched_alpha_beta(T, S, gdept, cfg)
        patch_is_dump = bool(np.array_equal(np.asarray(patched_a)[..., :nlev_ab], nemo_alpha))
        patch_differs_from_lego = bool(not np.allclose(
            np.asarray(patched_a)[..., :nlev_ab][wet_ab],
            alpha_lego[..., :nlev_ab][wet_ab], rtol=0, atol=0))
    finally:
        eos_mod.nemo_seos_alpha_beta = real_alpha_beta

    print(f"  [verify] patched nemo_seos_alpha_beta returns dump bit-identical "
          f"over the {nlev_ab} dumped levels: {patch_is_dump} (expect True)")
    print(f"  [verify] patched alpha differs from legoESM's own computed alpha "
          f"(over wet T-points): {patch_differs_from_lego} (expect True)")
    if not (patch_is_dump and patch_differs_from_lego):
        print("  WARNING: patch verification FAILED -- the split result below "
              "cannot be trusted; see printed booleans above.")

    # Exclude the deepest interface (index nk-1=34) from the split's stats: it
    # is built from the PADDED (repeated, not dumped) level 35 alpha/beta, so
    # scoring it would compare against a fabricated input, not a real one.
    wet_w_split = wet_w.copy()
    if n_missing > 0:
        wet_w_split[..., -n_missing:] = False
    n_excluded = int(wet_w[..., -n_missing:].sum()) if n_missing > 0 else 0
    print(f"  excluding {n_excluded} wet interface(s) at the padded deepest "
          f"level from split stats (would score a fabricated alpha/beta input)")
    stats_split = per_element_stats(
        "bn2[split] vs tke_dump_rn2b[1:]", n2_split, nemo_n2_interior, wet_w_split)

    med_base = stats_bn2["median_rel"]
    med_split = stats_split["median_rel"]
    print(f"\n  unpatched bn2 median|rel| = {med_base:.3e}")
    print(f"  patched(dumped-alpha/beta) bn2 median|rel| = {med_split:.3e}")
    if med_split < 1.0e-12:
        verdict_c = "bn2 is EXACT and its error is 100% INHERITED from eos_rab."
    elif med_split >= 0.5 * med_base:
        verdict_c = "bn2 has its own INDEPENDENT defect (patching alpha/beta did not fix it)."
    else:
        verdict_c = (f"PARTIAL inheritance -- both eos_rab and bn2's own arithmetic "
                     f"contribute (median dropped {med_base:.3e} -> {med_split:.3e}, "
                     f"a {100 * (1 - med_split / med_base):.1f}% reduction, but did not "
                     "reach roundoff).")
    print(f"  INTERPRETATION: {verdict_c}")

    # =========================================================================
    print("\n" + "=" * 78)
    print("(D) depth-vs-T/S-vs-coefficients isolation (only if (C) shows inheritance)")
    print("=" * 78)
    if med_split >= 0.5 * med_base:
        print("  SKIPPED: (C) shows bn2's error is NOT meaningfully reduced by "
              "substituting dumped alpha/beta, so there is no meaningful eos_rab "
              "inheritance in bn2 to decompose further. (D) targets isolating "
              "eos_rab's OWN error -- run anyway below since eos_rab (A) has its "
              "own directly-measured error regardless of the bn2 split outcome.")
    do_d = True
    nemo_gdept_path = os.path.join(RUN_DIR, "eiv_dump_gdept.bin")
    if not os.path.exists(nemo_gdept_path):
        print(f"  CANNOT ISOLATE: {nemo_gdept_path} does not exist on this run dir -- "
              "no independent NEMO gdept dump is available to substitute for "
              "legoESM's gdept while holding T/S fixed. Stating this explicitly "
              "rather than fabricating a depth-vs-T/S split.")
        do_d = False
    if do_d:
        nemo_gdept_dump = _load_haloed(nemo_gdept_path, jpi, jpj, hls)
        nk_g = nemo_gdept_dump.shape[-1]
        # Compare legoESM's live gdept (the one actually fed to nemo_seos_alpha_beta
        # above) against NEMO's own dumped gdept(Kmm) -- if they are already
        # identical to roundoff, substituting one for the other is a no-op and
        # cannot demonstrate a depth-driven contribution; say so rather than
        # claim a false isolation.
        gdept_np = np.asarray(gdept)
        gdept_np_bcast = np.broadcast_to(gdept_np, (nj, ni, gdept_np.shape[-1]))
        n_common = min(nk_g, gdept_np_bcast.shape[-1])
        wet_g = active_3d[..., :n_common]
        rel_gdept = (np.abs(gdept_np_bcast[..., :n_common] - nemo_gdept_dump[..., :n_common])
                     / np.maximum(np.abs(nemo_gdept_dump[..., :n_common]), FLOOR))
        med_rel_gdept = float(np.median(rel_gdept[wet_g]))
        print(f"  median|rel| legoESM live gdept vs NEMO gdept(Kmm) dump = {med_rel_gdept:.3e} "
              f"over {int(wet_g.sum())} wet T-cells (n_common levels={n_common})")
        if med_rel_gdept < 1.0e-9:
            print("  CANNOT ISOLATE depth vs T/S vs coefficients: legoESM's gdept "
                  "is ALREADY identical to NEMO's dumped gdept(Kmm) to roundoff on "
                  "this state, so substituting NEMO's gdept for legoESM's changes "
                  "nothing -- there is no depth-argument discrepancy left to "
                  "isolate; any remaining alpha error must come from the T/S "
                  "argument or the polynomial coefficients, not from gdept.")
        else:
            alpha_nemo_gdept, beta_nemo_gdept = eos_mod.nemo_seos_alpha_beta(
                T[..., :n_common], S[..., :n_common],
                jnp.asarray(nemo_gdept_dump[..., :n_common], dtype=dtype), cfg)
            alpha_nemo_gdept = np.asarray(alpha_nemo_gdept)
            print("  Substituted NEMO's own gdept(Kmm) for legoESM's gdept, kept "
                  "legoESM's own T/S:")
            stats_d = per_element_stats(
                "alpha[NEMO-gdept] vs dump_alpha_b",
                alpha_nemo_gdept, nemo_alpha[..., :n_common], wet_g)
            print(f"  baseline (A) alpha median|rel| = {stats_alpha['median_rel']:.3e}  "
                  f"vs depth-substituted median|rel| = {stats_d['median_rel']:.3e}")
            if abs(stats_d["median_rel"] - stats_alpha["median_rel"]) < 0.1 * stats_alpha["median_rel"]:
                print("  FINDING: substituting NEMO's own gdept does NOT measurably "
                      "change alpha's error -- the depth argument is not the driver; "
                      "residual is in the T/S argument or polynomial coefficients.")
            else:
                print("  FINDING: substituting NEMO's own gdept DOES measurably "
                      "change alpha's error -- the depth argument (gdept) is a "
                      "contributor to eos_rab's residual.")

    # =========================================================================
    # (E)-(H) added after peer review: N^2 is a SIGN-CHANGING field that passes
    # through zero, so |lego-nemo|/|nemo| diverges wherever nemo N^2 ~ 0
    # REGARDLESS of whether legoESM is wrong. Sections (E)-(H) test whether
    # (B)'s tail is that artifact before any "defect" claim is allowed to stand.
    # =========================================================================
    ne_w = nemo_n2_interior[wet_w]                       # NEMO N^2 at wet interfaces
    lo_w = n2_lego[wet_w]
    rel_w = np.abs(lo_w - ne_w) / np.maximum(np.abs(ne_w), FLOOR)
    tail = rel_w > 1.0e-2
    rms_n2 = float(np.sqrt(np.mean(ne_w ** 2)))

    print("\n" + "=" * 78)
    print("(E) CONDITIONING OF THE bn2 TAIL (is rel>1% just a zero-crossing artifact?)")
    print("=" * 78)
    absn_all, absn_tail = np.abs(ne_w), np.abs(ne_w[tail])
    print(f"  |nemo_bn2| over ALL {ne_w.size} wet interfaces: "
          f"median={np.median(absn_all):.3e}  p10={np.percentile(absn_all, 10):.3e}  "
          f"p90={np.percentile(absn_all, 90):.3e}")
    near0 = absn_all < 1.0e-3 * rms_n2
    if absn_tail.size == 0:
        cond_ratio = float("nan")
        print("  tail (rel>1%) is EMPTY -- no cell exceeds 1% at all; "
              "the conditioning question is moot at this time level.")
    else:
        print(f"  |nemo_bn2| over the {int(tail.sum())} rel>1% TAIL cells:  "
              f"median={np.median(absn_tail):.3e}  p10={np.percentile(absn_tail, 10):.3e}  "
              f"p90={np.percentile(absn_tail, 90):.3e}")
        cond_ratio = float(np.median(absn_tail) / np.median(absn_all))
        print(f"  ratio median|nemo_bn2|(tail)/median|nemo_bn2|(all) = {cond_ratio:.3e}")
    print(f"  RMS(nemo_bn2 over wet) = {rms_n2:.6e}")
    print(f"  fraction of ALL wet interfaces with |nemo_bn2| < 1e-3*RMS: "
          f"{int(near0.sum())}/{ne_w.size} = {100.0 * near0.mean():.3f}%")
    frac_tail_near0 = float((near0 & tail).sum()) / max(int(tail.sum()), 1)
    print(f"  of the {int(tail.sum())} tail cells, {int((near0 & tail).sum())} "
          f"({100.0 * frac_tail_near0:.1f}%) are near-zero-N^2 (<1e-3*RMS)")
    if cond_ratio < 0.1:
        print("  PLAINLY: the tail sits on near-zero N^2 -- it is a CONDITIONING "
              "artifact of dividing by a vanishing denominator, NOT evidence of a "
              "large error in those cells.")
    else:
        print("  PLAINLY: the tail does NOT sit preferentially on near-zero N^2 -- "
              "the pointwise-relative tail is not explained by conditioning alone.")

    print("\n" + "=" * 78)
    print("(F) CONDITIONING-ROBUST METRIC: err_norm = |lego-nemo| / RMS(nemo_bn2 over wet)")
    print("=" * 78)
    err_norm = np.abs(lo_w - ne_w) / rms_n2
    print(f"  RMS used = {rms_n2:.6e} [1/s^2]")
    print(f"  err_norm: median={np.median(err_norm):.3e}  "
          f"p99={np.percentile(err_norm, 99):.3e}  max={np.max(err_norm):.3e}")
    # locate the max in 3-D
    en_full = np.full(wet_w.shape, -1.0)
    en_full[wet_w] = err_norm
    imax = np.unravel_index(np.argmax(en_full), en_full.shape)
    print(f"  max err_norm at (j,i,level)={tuple(int(v) for v in imax)}  "
          f"|nemo_bn2| there = {abs(float(nemo_n2_interior[imax])):.6e}  "
          f"(nemo_bn2={float(nemo_n2_interior[imax]):+.6e}, "
          f"lego_bn2={float(n2_lego[imax]):+.6e})")
    print("  NOTE: for a SIGN-CHANGING field like N^2 a pointwise-relative median "
          "is NOT trustworthy (denominator passes through zero); err_norm is the "
          "number that belongs in a gate.")

    print("\n" + "=" * 78)
    print("(G) DOES alpha CROSS ZERO? (if not, its pointwise-relative stats ARE valid)")
    print("=" * 78)
    a_w = nemo_alpha[wet_ab]
    print(f"  NEMO alpha over {a_w.size} wet T-cells: signed min={a_w.min():+.6e}  "
          f"signed max={a_w.max():+.6e}  min|alpha|={np.abs(a_w).min():.6e}")
    alpha_crosses = bool(a_w.min() <= 0.0 <= a_w.max()) or (
        np.abs(a_w).min() < 1.0e-3 * np.abs(a_w).mean())
    if alpha_crosses:
        print("  alpha DOES approach/cross zero -> RETRACT the 'real outlier' "
              "framing for alpha's max|rel|; its pointwise-relative tail is "
              "subject to the same conditioning artifact as bn2.")
    else:
        print("  alpha is STRICTLY ONE-SIGNED and bounded away from zero -> its "
              "pointwise-relative stats ARE trustworthy, so max|rel| is a REAL "
              "outlier worth explaining. Details of the rel>1% cells:")
        rel_a = np.abs(alpha_lego[..., :nlev_ab] - nemo_alpha) / np.maximum(
            np.abs(nemo_alpha), FLOOR)
        rel_a_masked = np.where(wet_ab, rel_a, -1.0)
        jj, ii, kk = np.where(rel_a_masked > 1.0e-2)
        gdept_np_full = np.broadcast_to(np.asarray(gdept), (nj, ni, np.asarray(gdept).shape[-1]))
        T_np, S_np = np.asarray(T), np.asarray(S)
        for j, i, k in zip(jj, ii, kk):
            below_wet = bool(active_3d[j, i, k + 1]) if k + 1 < active_3d.shape[-1] else False
            nbr = [(j + dj, i + di) for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            nbr_wet = [bool(active_3d[jn % nj, iN % ni, k]) for jn, iN in nbr]
            print(f"    (j,i,k)=({j},{i},{k})  T={T_np[j, i, k]:.4f}degC  "
                  f"S={S_np[j, i, k]:.4f}PSU  gdept={gdept_np_full[j, i, k]:.3f}m")
            print(f"        lego_alpha={alpha_lego[j, i, k]:.9e}  "
                  f"nemo_alpha={nemo_alpha[j, i, k]:.9e}  "
                  f"rel={rel_a[j, i, k]:.3e}")
            print(f"        cell-below wet={below_wet} (False => BATHYMETRY STEP / "
                  f"seafloor); 4-neighbour wet at this level={nbr_wet}; "
                  f"surface={k == 0}")
        if len(jj) > 1:
            adj = (jj.max() - jj.min() <= 2) and (ii.max() - ii.min() <= 2)
            print(f"    the {len(jj)} rel>1% cells mutually adjacent (within 2 "
                  f"cells in j and i)? {adj}")

    print("\n" + "=" * 78)
    print("(H) IS THE bn2 RESIDUAL STRUCTURED WITH DEPTH? (err_norm per interface level)")
    print("=" * 78)
    print(f"  {'level':<8}{'n_wet':<10}{'median err_norm':<20}{'p99 err_norm':<20}")
    prof = []
    for k in range(nk):
        mk = wet_w[..., k]
        if not mk.any():
            print(f"  {k:<8d}{0:<10d}{'(no wet cells)':<20}{'':<20}")
            continue
        e = np.abs(n2_lego[..., k][mk] - nemo_n2_interior[..., k][mk]) / rms_n2
        med_k, p99_k = float(np.median(e)), float(np.percentile(e, 99))
        prof.append((k, med_k))
        print(f"  {k:<8d}{int(mk.sum()):<10d}{med_k:<20.6e}{p99_k:<20.6e}")
    if prof:
        meds = np.array([p[1] for p in prof])
        kbest, kworst = prof[int(np.argmin(meds))][0], prof[int(np.argmax(meds))][0]
        spread = float(meds.max() / max(meds.min(), 1e-300))
        print(f"\n  median err_norm ranges {meds.min():.3e} (level {kbest}) .. "
              f"{meds.max():.3e} (level {kworst}); max/min spread = {spread:.1f}x")
        print("  INTERPRETATION: " + (
            "FLAT with depth (spread < 10x) -> consistent with roundoff-level "
            "noise, not a specific mis-transcribed term."
            if spread < 10 else
            f"STRUCTURED with depth ({spread:.0f}x spread, worst at level "
            f"{kworst}) -> points at a specific depth-dependent term, not roundoff."))

    # =========================================================================
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"(A) alpha: median|rel|={stats_alpha['median_rel']:.3e} "
          f"mean|rel|={stats_alpha['mean_rel']:.3e} max|rel|={stats_alpha['max_rel']:.3e} "
          f"p99|rel|={stats_alpha['p99_rel']:.3e} ratio_mean={stats_alpha['ratio_mean']:.6f} "
          f"corr={stats_alpha['corr']:.8f}")
    print(f"(A) beta:  median|rel|={stats_beta['median_rel']:.3e} "
          f"mean|rel|={stats_beta['mean_rel']:.3e} max|rel|={stats_beta['max_rel']:.3e} "
          f"p99|rel|={stats_beta['p99_rel']:.3e} ratio_mean={stats_beta['ratio_mean']:.6f} "
          f"corr={stats_beta['corr']:.8f}")
    print(f"(B) bn2:   median|rel|={stats_bn2['median_rel']:.3e} "
          f"mean|rel|={stats_bn2['mean_rel']:.3e} max|rel|={stats_bn2['max_rel']:.3e} "
          f"p99|rel|={stats_bn2['p99_rel']:.3e} ratio_mean={stats_bn2['ratio_mean']:.6f} "
          f"corr={stats_bn2['corr']:.8f}")
    print(f"(C) split: {verdict_c}")
    print(f"(E) tail conditioning: median|nemo_bn2|(tail)/median|nemo_bn2|(all) "
          f"= {cond_ratio:.3e}; {100.0 * frac_tail_near0:.1f}% of the tail is "
          f"near-zero-N^2 (<1e-3*RMS)")
    print(f"(F) ROBUST bn2 metric err_norm=|d|/RMS({rms_n2:.3e}): "
          f"median={np.median(err_norm):.3e} p99={np.percentile(err_norm, 99):.3e} "
          f"max={np.max(err_norm):.3e}")
    print(f"(G) NEMO alpha signed range [{a_w.min():+.3e}, {a_w.max():+.3e}], "
          f"min|alpha|={np.abs(a_w).min():.3e} -> "
          f"{'crosses/approaches zero (pointwise-rel NOT valid)' if alpha_crosses else 'one-signed, bounded away from zero (pointwise-rel VALID)'}")
    if prof:
        print(f"(H) per-level median err_norm spread = {spread:.1f}x "
              f"({meds.min():.3e} .. {meds.max():.3e})")
    # Does (C)'s verdict survive the conditioning check? The split compared two
    # POINTWISE-RELATIVE medians on a zero-crossing field; re-state it in the
    # robust metric so the conclusion does not rest on the fragile number.
    en_split = np.abs(n2_split[wet_w_split] - nemo_n2_interior[wet_w_split]) / rms_n2
    en_base = np.abs(n2_lego[wet_w_split] - nemo_n2_interior[wet_w_split]) / rms_n2
    print(f"(C-robust) err_norm median: unpatched={np.median(en_base):.3e} -> "
          f"dumped-alpha/beta patched={np.median(en_split):.3e} "
          f"({100.0 * (1 - np.median(en_split) / max(np.median(en_base), 1e-300)):.1f}% reduction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
