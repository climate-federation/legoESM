#!/usr/bin/env python
"""#1226 W3 workstream (nemo_faithful_ocean_implementation_plan.md): close the
"dyn_zdf (momentum implicit vertical solve)" measurement GAP.

ORIGINAL PLAN (this task's brief): bracket legoESM's momentum implicit-
vertical-mixing solve (``LatLonCGridOceanModel._apply_implicit_vertical_
mixing``, ``do_momentum=True``, ocean_model_latlon_cgrid.py:5223, called from
``_leapfrog_step`` at :7170 with ``naa_expl`` as input) against NEMO's own
stage-7 (pre, ``stp_dump_state_and_bt('dynspg')``, stpmlf.F90:293) / stage-8
(post, ``stp_dump_state_and_bt('dynzdf')``, stpmlf.F90:312) dump pair, both
dumped as ``uu/vv(Naa)``.

CONFIRMED DEAD END (re-derived independently this task, THEN found already
documented as a dead end by the sibling probe ``spg_substep_chain.py``
lines ~21-38 -- same conclusion, not new): stage-7's ``uu(:,:,:,Naa)`` is
NOT a velocity state. Trace: stpmlf.F90:478-481 ``Nrhs=Nbb; Nbb=Nnn;
Nnn=Naa; Naa=Nrhs`` runs at the END of the step to rotate the 3-level index
scheme for the NEXT step -- so DURING the step body (incl. lines 288/293),
``Naa`` and ``Nrhs`` are the SAME index (this is a fixed 3-slot array of
time-level ALIASES, not 4 distinct buffers). In the MLF (non-RK3, confirmed
DINO's cpp_DINO.fcm has no key_RK3) branch of ``dyn_spg_ts``
(dynspg_ts.F90:1168-1170, the "Correct velocities" block), the barotropic
correction is written into ``puu(:,:,jk,Kmm)`` -- ``puu(:,:,jk,Kaa)`` (same
memory as ``Krhs``) is NEVER written by ``dyn_spg``/``dyn_spg_ts`` as a
velocity state; what stage-7 captures is the momentum RHS/tendency
accumulator (units ~m/s^2). Confirmed quantitatively THIS task: stage-7
(``stp_dump_07_dynspg_u.bin``) has |mean|=1.75e-7, range (-4.7e-6,+3.0e-6);
stage-6 (``stp_dump_06_dynhpg_du.bin``, the CONFIRMED Krhs accumulator right
before ``dyn_spg`` runs) has the SAME order of magnitude, |mean|=3.9e-6,
range (-6.4e-4,+7.3e-4) -- both ~m/s^2 tendency scale, five-plus orders
below the ~0.01-0.5 m/s velocity scale stage-8's genuine post-``dyn_zdf``
state carries (``dyn_zdf`` IS the first routine to populate the 3-D
``puu(:,:,:,Kaa)`` velocity, at dynzdf.F90:139-140:
``puu(Kaa) = (puu(Kbb) + rDt*puu(Krhs)) * umask``, run BEFORE its own
tridiagonal solve). This task's own probe run (below) reproduces the same
signature: feeding legoESM's real pre-solve u/v against stage-7 gives
corr=-0.29 (u) / 0.08 (v), |x|ratio~1e5, rms(nemo)~5e-7 -- an
apples-to-tendency comparison, not evidence of a defect (same verdict
``spg_substep_chain.py`` already reached; kept here as the SELF-CHECK gate,
printed but not treated as a bar result).

NO NEMO DUMP EXISTS that captures the true pre-dyn_zdf Krhs (stage 6 is
BEFORE dyn_spg's own pressure-gradient + barotropic corrections are added
into Krhs, at dynspg.F90:171 and dynspg_ts.F90:1156-1159; no dump brackets
Krhs AFTER those additions). Reconstructing that Krhs from legoESM's own
production tendencies would require porting NEMO's dyn_spg Krhs-correction
formula independently -- the same "real port, not bracket-and-diff" scope
exclusion the row's own prior note already applied to the drag fold itself.
**Gap is at the instrumentation level, not fixable by a probe alone.**

WHAT IS MEASURABLE, and reported below AS SUCH (not as "the solver alone"):
legoESM's own full momentum state AFTER its implicit vertical-mixing call
(do_momentum=True, fed by legoESM's OWN upstream naa_expl, not NEMO's Krhs)
vs NEMO's stage-8 (post, GENUINE velocity state -- unlike stage 7). This is
an END-TO-END momentum comparison AT this point in the step (inherits any
upstream advection/vorticity/ldf/hpg/barotropic mismatch, not isolated to
the vertical solve) -- reported honestly as such, not as a solver-only bar.

METHOD: spy ``LatLonCGridOceanModel._apply_implicit_vertical_mixing`` (same
monkeypatch-spy pattern as ``probe_dyn_cor_2d.py``'s ``_capture_cor_sub``)
on the ``do_momentum=True`` call inside ``_leapfrog_step``, capturing its
``naa_expl`` input (u,v) and its returned state's u,v.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.dyn_zdf_probe
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
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
)

# Reuse wholesale (module-reuse rule) -- same DT + err_norm/align-scan
# self-checks every sibling #1226 probe already shares.
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    DT, _load_full_3d,
)
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_u_structure_probe import (
    _err_norm,
)
from scripts.validate.ocean_fidelity.dino_1226.bn2_alpha_compare import _read_dims
# Lane selector (#1455): RUN_DIR/RESTART are lane-dependent, NOT
# zu_frc_term_walk's own hardcoded RUN_GDB/year-5 constants.
from scripts.validate.ocean_fidelity.dino_1226 import dump_lane

RUN_DIR = dump_lane.RUN_DIR
RESTART = dump_lane.RESTART

set_policy(PrecisionPolicy.fp64())

# --- Precondition: register this script's dumps' NEMO time level (fail-closed
# registry, Rule 1d) -- an unregistered dump raises via time_level_for_dump.
register_dump(
    "stp_dump_07_dynspg_u.bin", "after",
    "stpmlf.F90:293 CALL stp_dump_state_and_bt(kstp,7,'dynspg',uu(:,:,:,Naa),"
    "vv(:,:,:,Naa),...) -- CONFIRMED DEAD END (this task, independently "
    "re-derived + matches spg_substep_chain.py's prior finding): Naa/Nrhs "
    "are the SAME array index DURING the step body (stpmlf.F90:478-481 only "
    "rotates them at the END of the step), and dyn_spg_ts's MLF branch "
    "(dynspg_ts.F90:1168-1170) writes its velocity correction into Kmm, "
    "NEVER Kaa -- so this dump is actually the Krhs momentum-tendency "
    "accumulator (units ~m/s^2, confirmed by magnitude match against the "
    "known-Krhs stage-6 dynhpg dump), NOT a velocity state at any time "
    "level. Registered 'after' only to satisfy the registry API; DO NOT "
    "treat this as a genuine (t,level) state -- see module docstring.")
register_dump("stp_dump_07_dynspg_v.bin", "after", "same call site as _u.")
register_dump(
    "stp_dump_08_dynzdf_u.bin", "after",
    "stpmlf.F90:312 CALL stp_dump_state_and_bt(kstp,8,'dynzdf',uu(:,:,:,Naa),"
    "vv(:,:,:,Naa)) -- dumped Naa level immediately after the CALL dyn_zdf "
    "at :305 returns. This is the POST-dyn_zdf state (the row's measurement "
    "target).")
register_dump("stp_dump_08_dynzdf_v.bin", "after", "same call site as _u.")
for _name in (
    "stp_dump_07_dynspg_u.bin", "stp_dump_07_dynspg_v.bin",
    "stp_dump_08_dynzdf_u.bin", "stp_dump_08_dynzdf_v.bin",
):
    time_level_for_dump(_name)


def _capture_zdf_momentum(model, st, sf):
    """Spy ``_apply_implicit_vertical_mixing``'s do_momentum=True call inside
    ``_leapfrog_step`` -- same monkeypatch-spy pattern as
    ``probe_dyn_cor_2d.py._capture_cor_sub``. Captures the PRE (naa_expl,
    the method's first positional arg) and POST (this call's returned state)
    u,v. NOTE (see module docstring): PRE is legoESM's OWN upstream state,
    not NEMO's Krhs (no NEMO dump captures that), so this does NOT isolate
    the solver in the comparison below -- it is an end-to-end bracket at
    this point in the step."""
    captured = {}
    model_cls = type(model)
    _real = model_cls._apply_implicit_vertical_mixing

    def _spy(self, state, dt, surface_forcing, *args, **kwargs):
        if kwargs.get("do_momentum", True) and "pre_u" not in captured:
            captured["pre_u"] = np.asarray(state.u.data)
            captured["pre_v"] = np.asarray(state.v.data)
        result = _real(self, state, dt, surface_forcing, *args, **kwargs)
        if kwargs.get("do_momentum", True) and "post_u" not in captured:
            out_state = result[0] if isinstance(result, tuple) else result
            captured["post_u"] = np.asarray(out_state.u.data)
            captured["post_v"] = np.asarray(out_state.v.data)
        return result

    model_cls._apply_implicit_vertical_mixing = _spy
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        model_cls._apply_implicit_vertical_mixing = _real
    assert {"pre_u", "pre_v", "post_u", "post_v"} <= captured.keys(), (
        "_apply_implicit_vertical_mixing(do_momentum=True) never called -- "
        "check config.implicit_vertical_mixing / outer_integrator")
    return captured


def _stat(name, lego, nemo, mask):
    m = mask & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    n = int(m.sum())
    corr = float(np.corrcoef(lo, ne)[0, 1]) if n >= 2 and lo.std() > 0 and ne.std() > 0 else float("nan")
    rms_n = float(np.sqrt(np.mean(ne ** 2))) if n else float("nan")
    ratio = float(np.sqrt(np.mean(lo ** 2)) / rms_n) if rms_n > 0 else float("nan")
    rel = np.abs(lo - ne) / np.maximum(np.abs(ne), 1e-12)
    med_rel, p99_rel, max_rel = (float(np.median(rel)), float(np.percentile(rel, 99)),
                                 float(np.max(rel))) if n else (float("nan"),) * 3
    err_norm, rms, maxdiff, _ = _err_norm(lego, nemo, mask)
    print(f"  {name:<10s} n={n:<8d} corr={corr:.8f}  |x|ratio={ratio:.8f}  "
          f"err_norm={err_norm:.4e}  rms(nemo)={rms:.4e}")
    print(f"  {'':<10s} pointwise|rel| median={med_rel:.3e} p99={p99_rel:.3e} "
          f"max={max_rel:.3e}  max|diff|={maxdiff:.4e}")
    return corr, ratio, med_rel, p99_rel, max_rel


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="dyn_zdf_probe")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")
    print(dump_lane.banner())

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"outer_integrator={dcfg.outer_integrator!r}  "
          f"zdf_drag_in_matrix={dcfg.zdf_drag_in_matrix!r}  "
          f"barotropic_drag_substep={getattr(dcfg, 'barotropic_drag_substep', None)!r}")
    assert dcfg.outer_integrator == "leapfrog"
    assert dcfg.zdf_drag_in_matrix is True, (
        "row's mechanism (bottom-drag-in-matrix fold) requires "
        "zdf_drag_in_matrix=True on this recipe -- if this ever flips False "
        "the probe is no longer testing the row's stated code path")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                      omega=dcfg.omega)
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                               sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    print(f"  mc.implicit_vertical_mixing={mc.implicit_vertical_mixing!r}  "
          f"mc.zdf_drag_in_matrix={mc.zdf_drag_in_matrix!r}  "
          f"mc.bottom_drag.bottom_drag_scheme={mc.bottom_drag.bottom_drag_scheme!r}")
    assert mc.implicit_vertical_mixing is True
    assert mc.zdf_drag_in_matrix is True
    assert mc.bottom_drag.bottom_drag_scheme in ("nemo_quadratic", "nemo_loglayer")
    require_fp64(br.z_coord, br.state.T.data, br.state.S.data,
                 br.geometry.dx_u, context="dyn_zdf_probe")
    print(f"  br.state.u.data.dtype={br.state.u.data.dtype}  "
          f"br.geometry.dx_u.dtype={br.geometry.dx_u.dtype}  (want float64 both)")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    captured = _capture_zdf_momentum(model, br.state, sf)
    for k, v in captured.items():
        print(f"  captured[{k}].dtype={v.dtype}  shape={v.shape}")

    # ---- Load NEMO's stage-7 (pre) / stage-8 (post) full 3-D u/v(Naa) -----
    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    print(f"{dump_lane.LANE} dims: jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls}")
    nemo_pre_u = _load_full_3d(os.path.join(RUN_DIR, "stp_dump_07_dynspg_u.bin"),
                                jpi, jpj, jpk - 1, hls)
    nemo_pre_v = _load_full_3d(os.path.join(RUN_DIR, "stp_dump_07_dynspg_v.bin"),
                                jpi, jpj, jpk - 1, hls)
    nemo_post_u = _load_full_3d(os.path.join(RUN_DIR, "stp_dump_08_dynzdf_u.bin"),
                                 jpi, jpj, jpk - 1, hls)
    nemo_post_v = _load_full_3d(os.path.join(RUN_DIR, "stp_dump_08_dynzdf_v.bin"),
                                 jpi, jpj, jpk - 1, hls)
    print(f"  nemo_pre_u.dtype={nemo_pre_u.dtype}  shape={nemo_pre_u.shape}")

    umask3 = np.asarray(g.umask) > 0.5
    vmask3 = np.asarray(g.vmask) > 0.5

    lego_pre_u, lego_pre_v = captured["pre_u"], captured["pre_v"]
    lego_post_u, lego_post_v = captured["post_u"], captured["post_v"]

    n0u, n1u, n2u = (min(lego_pre_u.shape[i], nemo_pre_u.shape[i], umask3.shape[i])
                     for i in range(3))
    n0v, n1v, n2v = (min(lego_pre_v.shape[i], nemo_pre_v.shape[i], vmask3.shape[i])
                     for i in range(3))
    lego_pre_u = lego_pre_u[:n0u, :n1u, :n2u]
    lego_post_u = lego_post_u[:n0u, :n1u, :n2u]
    nemo_pre_u_c = nemo_pre_u[:n0u, :n1u, :n2u]
    nemo_post_u_c = nemo_post_u[:n0u, :n1u, :n2u]
    um = umask3[:n0u, :n1u, :n2u]

    lego_pre_v = lego_pre_v[:n0v, :n1v, :n2v]
    lego_post_v = lego_post_v[:n0v, :n1v, :n2v]
    nemo_pre_v_c = nemo_pre_v[:n0v, :n1v, :n2v]
    nemo_post_v_c = nemo_post_v[:n0v, :n1v, :n2v]
    vm = vmask3[:n0v, :n1v, :n2v]

    print("\n" + "=" * 78)
    print("dyn_zdf (momentum implicit vertical solve)")
    print("=" * 78)
    print("\n-- DEAD-END REPRODUCTION: legoESM's real pre-solve u,v vs NEMO "
          "stage-7 'dynspg' dump --")
    print("   EXPECTED to fail (confirms stage-7 is Krhs, not a velocity "
          "state -- see module docstring + spg_substep_chain.py's prior, "
          "independent finding). NOT a bar result; printed for the record.")
    _stat("pre_u", lego_pre_u, nemo_pre_u_c, um)
    _stat("pre_v", lego_pre_v, nemo_pre_v_c, vm)
    print(f"   rms(nemo stage-7 u)~{float(np.sqrt(np.mean(nemo_pre_u_c**2))):.3e} "
          f"vs rms(legoESM real velocity)~"
          f"{float(np.sqrt(np.mean(lego_pre_u**2))):.3e} -- five-plus orders "
          "apart (m/s^2 tendency vs m/s velocity), confirming the units "
          "mismatch, not a solver defect.")

    print("\n-- MEASURABLE COMPARISON (NOT solver-isolated -- end-to-end at "
          "this point in the step, fed by legoESM's OWN upstream state, not "
          "NEMO's Krhs): legoESM POST (implicit-solve output) vs NEMO "
          "stage-8 'dynzdf' dump (a genuine velocity state) --")
    corr_u, ratio_u, med_u, p99_u, max_u = _stat("post_u", lego_post_u, nemo_post_u_c, um)
    corr_v, ratio_v, med_v, p99_v, max_v = _stat("post_v", lego_post_v, nemo_post_v_c, vm)

    bar_ok_u = corr_u >= 1.0 - 1e-9 and abs(ratio_u - 1.0) <= 1e-6
    bar_ok_v = corr_v >= 1.0 - 1e-9 and abs(ratio_v - 1.0) <= 1e-6
    print("\n" + "=" * 78)
    print(f"dyn_zdf: u corr={corr_u:.6f} ratio={ratio_u:.6f} "
          f"median|rel|={med_u:.3e} p99={p99_u:.3e} max={max_u:.3e} "
          f"({'AT BAR' if bar_ok_u else 'DEBT'} on the end-to-end metric; "
          "NOT a solver-isolated verdict -- see docstring)")
    print(f"dyn_zdf: v corr={corr_v:.6f} ratio={ratio_v:.6f} "
          f"median|rel|={med_v:.3e} p99={p99_v:.3e} max={max_v:.3e} "
          f"({'AT BAR' if bar_ok_v else 'DEBT'} on the end-to-end metric; "
          "NOT a solver-isolated verdict -- see docstring)")
    print("\nVERDICT: the row remains a GENUINE INSTRUMENTATION GAP -- no "
          "existing NEMO dump brackets dyn_zdf's true Krhs input (stage 6 "
          "predates dyn_spg's own Krhs additions; stage 7 is mislabelled, "
          "not a state). Closing it for real needs either (a) a NEW NEMO "
          "dump of Krhs taken AFTER dyn_spg returns but BEFORE dyn_zdf runs "
          "(a rebuild), or (b) porting dyn_spg's Krhs-correction formula "
          "into legoESM to reconstruct NEMO's true pre-solve input -- both "
          "out of this probe-writing task's scope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
