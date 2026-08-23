#!/usr/bin/env python
"""#1226 W3 workstream (nemo_faithful_ocean_implementation_plan.md): close the
"traldf_iso_lap tendency" measurement GAP.

WHAT THIS MEASURES: legoESM's ``nemo_iso_lap_tracer_tendency_latlon_cgrid``
(gm_redi_latlon_cgrid.py:1891, selected via ``gm_redi_slope_scheme=
"nemo_iso_lap"`` -- CONFIRMED set by both ``nemo_dino_kamm``/
``nemo_dino_kamm_mlf`` cards, dino.py:942,1142) vs NEMO's own ``tra_ldf``
(dispatches to ``traldf_iso_lap``, ``np_lap_i``, DINO's ``nldf_tra``) via the
NOW-EXISTING bracket:

  stage 22  stp_dump_ts_krhs('before_traldf')  ts(:,:,:,:,Nrhs)  PRE  tra_ldf
            (stpmlf.F90:436, right before CALL tra_ldf at :437)
  stage 23  stp_dump_ts_krhs('after_traldf')   ts(:,:,:,:,Nrhs)  POST tra_ldf
            (stpmlf.F90:438, right after the CALL returns)

Both are the SAME running Nrhs accumulator (a RATE, K/s and PSU/s -- same
convention ``coverage_rows_measure.measure_tra_qsr`` already uses for its
own bracketed-difference comparison, NOT a state), and stpmlf.F90's own
comment (:429-435) confirms "NO routine writes ts(:,:,:,:,Nrhs) between the
two stp_dump_ts_krhs calls" -- so stage23-stage22 isolates EXACTLY
traldf_iso_lap's own Krhs increment, a genuine (not tautological) bracket.

CONFIRMED (read directly, this task): ``traldf_iso.F90:96-97,102``
(``traldf_iso_lap``) reads the tracer at ``Kbb`` ("before" fields, "computed
using before fields (forward in time)" -- the docstring's own words) and
writes into ``Krhs``, called at ``stpmlf.F90:437`` as
``CALL tra_ldf(kstp, Nbb, Nnn, ts, Nrhs)``. legoESM's ``_leapfrog_step``
evaluates its OWN GM/Redi call (inside ``_step_impl``, line ~3917's
``gm_redi_tracer_tendency_latlon`` -> dispatch to
``nemo_iso_lap_tracer_tendency_latlon_cgrid`` for slope_scheme=
"nemo_iso_lap") on the Nbb DISSIPATIVE pass (``_leapfrog_step``'s own
docstring: "evaluate dyn_ldf(Kbb)/tra_ldf(Kbb) ... on the BEFORE state") --
the SAME time level NEMO uses. ``slope_positions="nemo_native"`` is set for
both kamm cards (dino.py:2806), routing through
``compute_nemo_native_slopes`` + the ``native_slopes=`` branch (mode-b
negation does NOT apply here).

METHOD: spy the MODULE-LEVEL function
``nemo_iso_lap_tracer_tendency_latlon_cgrid`` (monkeypatch on the module
object, same spy pattern as ``probe_dyn_cor_2d.py``/``dyn_zdf_probe.py``)
during a real ``model.step()`` call and capture its T-tracer and S-tracer
outputs from the Nbb dissipative pass (the FIRST two calls per step, per
the ``_dT``/``dS_dt`` call order at gm_redi_latlon_cgrid.py:3706-3724 --
Nnn advective pass runs first inside ``_leapfrog_step``, so it captures
BOTH passes' calls; the Nbb pass's calls are identified as the ones whose
input tracer equals ``state.T_before``/``state.S_before``, matching NEMO's
Kbb input exactly).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.traldf_iso_lap_probe
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import dataclasses

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.precision_gate import (
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gmmod
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
)

# Reuse wholesale (module-reuse rule) -- DT (same value every lane) + err_norm
# self-check every sibling #1226 probe already shares. RUN_DIR/RESTART route
# through dump_lane (#1455 shared selector) instead of zu_frc_term_walk's own
# (still-hardcoded-to-gdb_y5) RUN_DIR -- this probe must be lane-switchable
# independently of that (untouchable) sibling.
from scripts.validate.ocean_fidelity.dino_1226 import dump_lane
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import DT
from scripts.validate.ocean_fidelity.dino_1226.cancelling_rows_per_element import (
    per_element_stats,
)
from scripts.validate.ocean_fidelity.dino_1226.bn2_alpha_compare import (
    _read_dims, _load_haloed, _shift_scan,
)

RUN_DIR = dump_lane.RUN_DIR
RESTART = dump_lane.RESTART

set_policy(PrecisionPolicy.fp64())

# --- Precondition: register this script's dumps' NEMO time level (fail-closed
# registry, Rule 1d) -- an unregistered dump raises via time_level_for_dump.
register_dump(
    "stp_dump_22_before_traldf_tem.bin", "now",
    "stpmlf.F90:436 CALL stp_dump_ts_krhs(kstp,22,'before_traldf',"
    "ts(:,:,:,:,Nrhs)), right before CALL tra_ldf(kstp,Nbb,Nnn,ts,Nrhs) at "
    ":437 -- the RUNNING Nrhs accumulator (K/s rate) immediately before "
    "tra_ldf adds its own increment. traldf_iso_lap itself reads the "
    "tracer INPUT at Kbb (traldf_iso.F90:102 'in: at kbb'), so 'now' here "
    "labels this dump's OWN provenance (call-site order), not the tracer "
    "argument's time level -- matching ldf_dump_du.bin's convention "
    "(zu_frc_term_walk.py) for the analogous momentum-side dyn_ldf row.")
register_dump("stp_dump_22_before_traldf_sal.bin", "now", "same call site as _tem.")
register_dump(
    "stp_dump_23_after_traldf_tem.bin", "now",
    "stpmlf.F90:438 CALL stp_dump_ts_krhs(kstp,23,'after_traldf',"
    "ts(:,:,:,:,Nrhs)), right after the CALL tra_ldf(...) at :437 returns. "
    "stpmlf.F90:429-435's own comment confirms NO routine writes "
    "ts(:,:,:,:,Nrhs) between the two dumps, so stage23-stage22 isolates "
    "EXACTLY traldf_iso_lap's own Krhs increment.")
register_dump("stp_dump_23_after_traldf_sal.bin", "now", "same call site as _tem.")
for _name in (
    "stp_dump_22_before_traldf_tem.bin", "stp_dump_22_before_traldf_sal.bin",
    "stp_dump_23_after_traldf_tem.bin", "stp_dump_23_after_traldf_sal.bin",
):
    time_level_for_dump(_name)


def _capture_iso_lap(model, st, sf, T_before, S_before, wet_mask3):
    """Spy the MODULE-LEVEL ``nemo_iso_lap_tracer_tendency_latlon_cgrid`` on
    a real ``model.step()`` call. legoESM calls it 4x per step(): the Nnn
    advective pass (T then S) followed by the Nbb dissipative pass (T then
    S) -- CONFIRMED this task by instrumented spy run (isolated debug
    script, not committed): call 0 == state.T.data (Nnn) exactly on the wet
    mask, call 1 == state.S.data, call 2 == state.T_before.data (Nbb)
    exactly on the wet mask (max|diff|=0.0), call 3 == state.S_before.data.
    Land cells differ (fill-value convention, irrelevant off-mask), so
    identification uses a WET-MASKED exact match, not strict
    ``np.array_equal`` over the full padded array."""
    captured = {"all_calls": []}
    _real = gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid

    def _spy(q, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid,
             kappa_Redi, active_3d=None, **kwargs):
        result = _real(q, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian,
                        grid, kappa_Redi, active_3d, **kwargs)
        out = result[0] if (isinstance(result, tuple) and
                             kwargs.get("return_bolus")) else result
        captured["all_calls"].append((np.asarray(q), np.asarray(out)))
        return result

    gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = _spy
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = _real

    n_calls = len(captured["all_calls"])
    assert n_calls == 4, (
        f"nemo_iso_lap_tracer_tendency_latlon_cgrid called {n_calls} times "
        "(want exactly 4 -- T,S x {Nnn advective, Nbb dissipative} passes) "
        "-- check config.gm_redi / gm_redi_slope_scheme dispatch or "
        "_leapfrog_step's call structure for a change")

    def _wet_equal(a, b):
        n0, n1, n2 = (min(a.shape[i], b.shape[i], wet_mask3.shape[i])
                      for i in range(3))
        m = wet_mask3[:n0, :n1, :n2]
        aa, bb = a[:n0, :n1, :n2], b[:n0, :n1, :n2]
        return bool(np.array_equal(aa[m], bb[m]))

    Tb, Sb = np.asarray(T_before), np.asarray(S_before)
    dT_bb = dS_bb = None
    match_idx = []
    for i, (q_in, out) in enumerate(captured["all_calls"]):
        if _wet_equal(q_in, Tb):
            dT_bb = out
            match_idx.append((i, "T_before"))
        elif _wet_equal(q_in, Sb):
            dS_bb = out
            match_idx.append((i, "S_before"))
    print(f"  Nbb-pass call identification (wet-masked exact match): {match_idx}")
    assert dT_bb is not None and dS_bb is not None, (
        "could not identify the Nbb-pass T/S calls by wet-masked exact "
        "input match -- T_before/S_before argument identity assumption "
        "violated (check _step_impl's T_mid/S_mid construction for a "
        "change from the confirmed bit-identical-on-Nbb-pass behaviour)")
    assert {i for i, _ in match_idx} == {2, 3}, (
        f"Nbb-pass calls landed at indices {[i for i, _ in match_idx]}, "
        "expected {2, 3} (the confirmed call order) -- the call-order "
        "assumption in this probe's docstring is now wrong, re-verify "
        "before trusting the result")
    return dT_bb, dS_bb


def main() -> int:
    print(dump_lane.banner())
    e3t_mode = require_explicit_e3t_mode(context="traldf_iso_lap_probe")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"outer_integrator={dcfg.outer_integrator!r}  "
          f"gm_redi_slope_scheme={dcfg.gm_redi_slope_scheme!r}")
    assert dcfg.outer_integrator == "leapfrog"
    assert dcfg.gm_redi_slope_scheme == "nemo_iso_lap"

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                      omega=dcfg.omega)
    before = read_nemo_restart_before(
        os.path.join(RUN_DIR, RESTART), nn_hls=0)
    st_with_before = bridge_before_state_topo(
        br._replace(state=br.state), g, before, periodic_i=True)
    br = br._replace(state=st_with_before)
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                               sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    print(f"  mc.gm_redi.slope_scheme={mc.gm_redi.slope_scheme!r}  "
          f"mc.gm_redi.slope_positions={mc.gm_redi.slope_positions!r}")
    assert mc.gm_redi is not None
    assert mc.gm_redi.slope_scheme == "nemo_iso_lap"
    assert mc.gm_redi.slope_positions == "nemo_native"
    require_fp64(br.z_coord, br.state.T.data, br.state.S.data,
                 br.geometry.dx_u, context="traldf_iso_lap_probe")
    print(f"  br.state.T.data.dtype={br.state.T.data.dtype}  "
          f"br.geometry.dx_u.dtype={br.geometry.dx_u.dtype}  (want float64 both)")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    tmask3 = np.asarray(g.tmask) > 0.5
    dT_bb, dS_bb = _capture_iso_lap(
        model, br.state, sf, br.state.T_before.data, br.state.S_before.data,
        tmask3)
    print(f"  dT_bb.dtype={dT_bb.dtype}  shape={dT_bb.shape}")

    # ---- Load NEMO's stage-22 (pre) / stage-23 (post) ts(Nrhs), bracket. ---
    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    print(f"{dump_lane.LANE} dims: jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls}")
    tem_pre = _load_haloed(os.path.join(RUN_DIR, "stp_dump_22_before_traldf_tem.bin"),
                            jpi, jpj, hls)
    tem_post = _load_haloed(os.path.join(RUN_DIR, "stp_dump_23_after_traldf_tem.bin"),
                             jpi, jpj, hls)
    sal_pre = _load_haloed(os.path.join(RUN_DIR, "stp_dump_22_before_traldf_sal.bin"),
                            jpi, jpj, hls)
    sal_post = _load_haloed(os.path.join(RUN_DIR, "stp_dump_23_after_traldf_sal.bin"),
                             jpi, jpj, hls)
    print(f"  tem_pre.dtype={tem_pre.dtype}  shape={tem_pre.shape}")

    nemo_dT = tem_post - tem_pre   # bracketed increment, [K/s]
    nemo_dS = sal_post - sal_pre   # bracketed increment, [PSU/s]
    print(f"  [self-check] traldf_iso_lap increment: nonzero levels "
          f"tem={int(np.sum(np.any(np.abs(nemo_dT) > 0, axis=(0, 1))))}/"
          f"{nemo_dT.shape[-1]}  sal="
          f"{int(np.sum(np.any(np.abs(nemo_dS) > 0, axis=(0, 1))))}/{nemo_dS.shape[-1]}")

    n0, n1, n2 = (min(dT_bb.shape[i], nemo_dT.shape[i], tmask3.shape[i])
                  for i in range(3))
    lego_dT = dT_bb[:n0, :n1, :n2]
    lego_dS = dS_bb[:n0, :n1, :n2]
    nemo_dT_c = nemo_dT[:n0, :n1, :n2]
    nemo_dS_c = nemo_dS[:n0, :n1, :n2]
    mask3 = tmask3[:n0, :n1, :n2]

    print("\n" + "=" * 78)
    print("traldf_iso_lap tendency: legoESM (Nbb pass) vs NEMO bracketed "
          "Krhs increment (stage23-stage22)")
    print("=" * 78)
    _shift_scan("traldf_iso_lap tem", lego_dT, nemo_dT_c, mask3)
    _shift_scan("traldf_iso_lap sal", lego_dS, nemo_dS_c, mask3)

    r_tem = per_element_stats("traldf_iso_lap tem [K/s]", lego_dT, nemo_dT_c,
                               mask3, sign_changing=True)
    r_sal = per_element_stats("traldf_iso_lap sal [PSU/s]", lego_dS, nemo_dS_c,
                               mask3, sign_changing=True)

    print("\n" + "=" * 78)
    print(f"traldf_iso_lap: tem {'AT BAR' if r_tem['at_bar'] else 'DEBT'}  "
          f"(bar_metric={r_tem['bar_metric']:.3e}, n={r_tem['n']})")
    print(f"traldf_iso_lap: sal {'AT BAR' if r_sal['at_bar'] else 'DEBT'}  "
          f"(bar_metric={r_sal['bar_metric']:.3e}, n={r_sal['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
