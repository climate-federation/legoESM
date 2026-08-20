"""#1226 / #1455 QUEUE ITEM 2: rebuild the missing ``dyn_cor_2d (69x/step)``
provenance probe.

The gate row's cited script (``probe_dyn_cor_2d.py``) was ``scripts/tmp/``
scratch, deleted before being committed (STALE-TUPLE SWEEP 2026-08-03: "4
rows' cited probes absent from disk"). This is a from-scratch reconstruction
of that measurement, NOT a copy of lost code -- it follows the sibling
``zu_frc_leapfrog_residual_probe.py``/``zu_frc_term_walk.py`` structure
(same RUN_DIR/DT, same ``_err_norm``/``_align_scan`` self-checks, same
production-call spy pattern) rather than re-deriving anything.

WHAT THIS MEASURES: NEMO's pre-step 2-D barotropic Coriolis removal
(``dynspg_ts.F90:296-300,359-368``, ``CALL dyn_cor_2D(puu_b(Kmm), pvv_b(Kmm),
zu_trd, zv_trd)``) vs legoESM's analogous ``_cor_u_sub``/``_cor_v_sub``
(``ocean_model_latlon_cgrid.py:3286-3290``, built via the public wrapper
:func:`barotropic_coriolis_een_pre_step`, EEN stencil,
``metric_complete=True`` -- the ``nemo_dino_kamm_mlf`` card's
``barotropic_coriolis="een_metric"``). Compared against NEMO's own LIVE
in-loop substep-1 dyn_cor_2D output (``cor2d_dump_zu_trd_substep1.bin`` /
``..zv_trd..``) -- this is the closest oracle dump to the pre-step
subtraction's own coefficients (both use the SAME EEN operator applied to a
NOW/Kmm-level transport; see ``barotropic_coriolis_een_pre_step`` docstring),
consistent with how the row's prior (lost) measurement is described in the
gate file's own metric-convention note.

Excludes the periodic-seam column (harness reindexing artifact there per the
gate row's own note, "excl. periodic-seam column, harness reindexing artifact
there, not a lego defect") from the INTERIOR comparison -- reported
separately, not silently dropped.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.probe_dyn_cor_2d

SPLIT-ARM ATTRIBUTION (2026-08, this probe, ONE variable per arm)
-----------------------------------------------------------------
The 2026-08 improvement on this row flipped THREE things at once (the AL81
triad<->mass-flux pairing fix in ``pv_flux_al81_partial_cell``; threading
``een_q_boundary`` into the barotropic path; threading ``een_e3f_scheme``),
so no row was attributable.  Re-run split, six arms, all in ONE tree epoch,
same restart / same dumps / same env (fp64, ``LEGOESM_NEMO_E3T=both``,
RUN_GDB); the pairing variable switched by restoring
``latlon_cgrid_operators.py`` from git and back (md5-verified).  FULL-interior
numbers, ``n=9758`` (u) / ``9868`` (v)::

  arm                pairing  q_boundary     e3f       u err_norm  u ratio   v err_norm  v ratio
  A0  (baseline)     pre-fix  neumann_fill   min       4.3712e-04  1.000038  3.1837e-04  0.999945
  A0b               pre-fix  nemo_live      min       3.8618e-04  1.000033  1.6556e-04  1.000024
  A1  +pairing       fixed    neumann_fill   min       2.2384e-04  1.000025  3.1924e-04  0.999947
  A2  +pairing+qb    fixed    nemo_live      min       1.6092e-04  1.000018  1.6564e-04  1.000024
  A2b +pairing+e3f   fixed    neumann_fill   nemo_avg  2.2230e-04  1.000005  3.1966e-04  0.999931
  A3  all three      fixed    nemo_live      nemo_avg  1.5983e-04  1.000000  1.6062e-04  1.000009

Single-variable attribution (each read off the arm pair that differs in that
one field only) -- the three act NEARLY ADDITIVELY on ``err_norm`` (A1 plus
the isolated qb and e3f deltas predicts u 1.594e-04 vs 1.598e-04 measured,
v 1.661e-04 vs 1.606e-04):

* ``err_norm`` and ``ratio`` attribute to DIFFERENT variables -- do not quote
  one and imply the other.
* u ``err_norm``: the PAIRING owns it (A0->A1, -48.8%); qb adds -28.1%
  (A1->A2); e3f is inert (-0.7%).
* v ``err_norm``: ``een_q_boundary`` owns it (A1->A2, -48.1%); the pairing is
  INERT on v (A0->A1, +0.3%, i.e. marginally worse); e3f -3.0% (A2->A3).
* u ``|ratio-1|``: ``een_e3f_scheme`` owns it (A1->A2b, 2.5e-5 -> 5e-6, -80%;
  A2->A3, 1.8e-5 -> 1e-6), the variable that is inert on u ``err_norm``.
* v ``|ratio-1|``: qb then e3f (5.3e-5 -> 2.4e-5 -> 9e-6); e3f WITHOUT qb
  makes it worse (5.3e-5 -> 6.9e-5), so those two are not independent on v.

NOT AT BAR: every arm, including A3, prints DEBT.  The best v ratio (1.000009)
misses the |ratio-1| <= 1e-6 bar by 9x.  This block attributes an improvement;
it does not clear the row.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_before_state_topo, bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import barotropic_coriolis_een_pre_step
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

# Reuse wholesale (module-reuse rule) -- same RUN_DIR/DT/loaders/self-checks
# every sibling #1226 probe in this package already shares.
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_full,
)
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_u_structure_probe import (
    _err_norm, _align_scan,
)

JPI, JPJ, HLS = 56, 203, 2
SEAM_COL = 49  # already-documented periodic-seam/DINO-sill column (sill_lon_m_deg=1.0)


def _capture_cor_sub(model, st, sf):
    """Spy ``state_mid.u/v`` (NOW/Kmm, post-slow-tendency momentum) via the
    same ``barotropic_substeps_latlon_cgrid`` spy point the sibling probes
    already use -- the caller of that function is given ``F_slow_u/v``
    AFTER the ``_cor_u_sub``/``_cor_v_sub`` subtraction (a bare local inside
    ``_step_impl``, no public accessor), so this recomputes the subtraction
    INDEPENDENTLY via the public :func:`barotropic_coriolis_een_pre_step`
    wrapper on the SAME captured ``state_mid.u/v``, with ``h_k_pre`` built
    the same way ``_step_impl`` builds it
    (``compute_layer_thickness(state.eta, state.H_bathy, z_coord,
    min_water_column_m)`` -- NOW level, momentum update never touches eta,
    matching ``zu_frc_leapfrog_residual_probe.py``'s own ``h_k_nn``)."""
    captured = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        result = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        if kw.get("eta_init") is not None and "state_mid_u" not in captured:
            captured["state_mid_u"] = state_mid.u.data
            captured["state_mid_v"] = state_mid.v.data
        return result

    ocmod.barotropic_substeps_latlon_cgrid = _spy_baro
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
    assert "state_mid_u" in captured, "barotropic solver never called with a seed"
    return captured


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="probe_dyn_cor_2d")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"barotropic_coriolis_split={dcfg.barotropic_coriolis_split!r}  "
          f"barotropic_coriolis={dcfg.barotropic_coriolis!r}  "
          f"barotropic_een_seed={dcfg.barotropic_een_seed!r}")
    assert dcfg.barotropic_coriolis_split == "live"

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="probe_dyn_cor_2d twin state")
    # Read the flag from the SAME resolved field production reads
    # (self.config.barotropic.barotropic_coriolis), not the raw DINOConfig
    # (adversarial-review note: removes a latent divergence risk if the
    # from_flat threading ever remapped it).
    metric_complete = mc.barotropic.barotropic_coriolis in ("een_metric",)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5

    from legoesm.ocean.vertical import compute_layer_thickness

    captured = _capture_cor_sub(model, st, sf)
    U_Nnn = captured["state_mid_u"]
    V_Nnn = captured["state_mid_v"]
    h_k_pre = compute_layer_thickness(
        st.eta.data, st.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m)
    _min_wc = jnp.asarray(model.config.min_water_column_m, dtype=U_Nnn.dtype)

    # Mirror the PRODUCTION call site exactly (ocean_model_latlon_cgrid.py,
    # the barotropic_coriolis_een_pre_step call under barotropic_coriolis_
    # split="live"): it forwards the card's een_q_boundary/een_e3f_scheme --
    # NEMO's dyn_cor_2D_init reads the SAME raw ff_f/e3f_vor as vor_een
    # (dynspg_ts.F90:1517-1531), so the barotropic EEN must use the same two
    # rules the 3-D EEN does.  Taking them from the RESOLVED model config
    # (mc), not the raw DINOConfig, for the same reason metric_complete does.
    print(f"een_q_boundary={mc.een_q_boundary!r}  "
          f"een_e3f_scheme={mc.een_e3f_scheme!r}")
    cor_u_sub, cor_v_sub = barotropic_coriolis_een_pre_step(
        U_Nnn, V_Nnn, h_k_pre, br.geometry,
        st.land_mask.data, st.u_mask.data, st.v_mask.data, _min_wc,
        U_Nnn.dtype, metric_complete=metric_complete,
        een_q_boundary=mc.een_q_boundary,
        een_e3f_scheme=mc.een_e3f_scheme,
        # ...including dz_ref, which the production call site also forwards
        # and which is LIVE on this card's een_e3f_scheme="nemo_avg" branch
        # (fully-dry-vertex e3f_0 fallback).  Measured inert on the
        # tendency, but "mirror exactly" has to mean exactly.
        dz_ref=model.z_coord.dz_ref)
    cor_u_sub = np.asarray(cor_u_sub)
    cor_v_sub = np.asarray(cor_v_sub)

    nemo_zu_trd = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zu_trd_substep1.bin"), JPI, JPJ, HLS)
    nemo_zv_trd = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zv_trd_substep1.bin"), JPI, JPJ, HLS)

    def _to_nemo_u(a):
        return np.asarray(a)[:, 1:]

    def _to_nemo_v(a):
        return np.asarray(a)[1:, :]

    lego_u = _to_nemo_u(cor_u_sub)
    lego_v = _to_nemo_v(cor_v_sub)
    n_lat_u, n_lon_u = min(lego_u.shape[0], nemo_zu_trd.shape[0]), min(lego_u.shape[1], nemo_zu_trd.shape[1])
    n_lat_v, n_lon_v = min(lego_v.shape[0], nemo_zv_trd.shape[0]), min(lego_v.shape[1], nemo_zv_trd.shape[1])
    lego_u = lego_u[:n_lat_u, :n_lon_u]
    nemo_u = nemo_zu_trd[:n_lat_u, :n_lon_u]
    lego_v = lego_v[:n_lat_v, :n_lon_v]
    nemo_v = nemo_zv_trd[:n_lat_v, :n_lon_v]
    um = umask2[:n_lat_u, :n_lon_u]
    vm = vmask2[:n_lat_v, :n_lon_v]

    print("\n" + "=" * 78)
    print("dyn_cor_2d: _cor_u_sub/_cor_v_sub (EEN pre-step) vs NEMO substep-1 zu_trd/zv_trd")
    print("=" * 78)
    _align_scan("cor_u_sub", lego_u, nemo_u, um)
    _align_scan("cor_v_sub", lego_v, nemo_v, vm)

    def _stat(name, lego, nemo, mask, exclude_col=None):
        m = mask.copy()
        if exclude_col is not None and exclude_col < m.shape[1]:
            m[:, exclude_col] = False
        m = m & np.isfinite(lego) & np.isfinite(nemo)
        a, b = lego[m], nemo[m]
        corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
        rms_b = float(np.sqrt(np.mean(b ** 2))) if b.size else float("nan")
        ratio = float(np.sqrt(np.mean(a ** 2)) / rms_b) if rms_b > 0 else float("nan")
        err_norm, rms, maxdiff, _ = _err_norm(lego, nemo, mask if exclude_col is None else m)
        print(f"  {name}: n={a.size}  corr={corr:.8f}  ratio={ratio:.6f}  "
              f"err_norm={err_norm:.4e}  rms(nemo)={rms:.4e}  max|diff|={maxdiff:.4e}")
        return corr, ratio

    print("\n-- FULL interior (includes periodic-seam column) --")
    _stat("u", lego_u, nemo_u, um)
    _stat("v", lego_v, nemo_v, vm)

    print(f"\n-- excluding periodic-seam column {SEAM_COL} (harness reindexing artifact) --")
    corr_u, ratio_u = _stat("u", lego_u, nemo_u, um, exclude_col=SEAM_COL)
    corr_v, ratio_v = _stat("v", lego_v, nemo_v, vm, exclude_col=SEAM_COL)

    bar_ok_u = corr_u >= 1.0 - 1e-9 and abs(ratio_u - 1.0) <= 1e-6
    bar_ok_v = corr_v >= 1.0 - 1e-9 and abs(ratio_v - 1.0) <= 1e-6
    print("\n" + "=" * 78)
    print(f"dyn_cor_2d (69x/step): u corr={corr_u:.6f} ratio={ratio_u:.6f} "
          f"({'AT BAR' if bar_ok_u else 'DEBT'})")
    print(f"dyn_cor_2d (69x/step): v corr={corr_v:.6f} ratio={ratio_v:.6f} "
          f"({'AT BAR' if bar_ok_v else 'DEBT'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
