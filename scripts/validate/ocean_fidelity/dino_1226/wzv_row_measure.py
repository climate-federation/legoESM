"""#1455 queue item: measure the "wzv (vertical velocity)" row -- promoted
from UNMEASURED (``fidelity_bar_gate.py``'s ``PER_ELEMENT["wzv (vertical
velocity)"]`` entry is currently ``(None, None, ...)``).

STALE-NOTE CORRECTION (Rule: prove the path executes / verify a claim before
writing it down): that row's note, ``coverage_rows_measure.py``, and
``ww_inheritance_walk.py`` all say NEMO's ``ww`` (``wzv_MLF``, ``sshwzv.
F90:168``) is "NEVER dumped anywhere in cfgs/DINO/MY_SRC/*.F90". That claim
predates the FACE10/GDB-era instrumentation and is now FALSE for the run
this whole #1226/#1455 campaign uses (``RUN_DIR = .../cfgs/DINO/RUN_GDB``,
``zu_frc_term_walk.py``): direct dumps exist --

    wzv_dump_ww_call1.bin, wzv_dump_ww_call2.bin   (RUN_GDB, single restart kt)

-- written at ``sshwzv.F90:289-295`` (verified directly, this file) and
ALREADY REGISTERED in ``time_levels.py`` (``_DUMP_TIME_LEVEL["wzv_dump_ww_
call1.bin"]``/``["...call2.bin"]``, both "now"). No instrumentation rebuild
needed; this script is a pure measurement using dumps that already exist.

CALL-SITE STRUCTURE (stpmlf.F90, read directly, this file):
    :244  CALL wzv( kstp, Nbb, Nnn, Naa, ww )   -- BEFORE dyn_adv (:265)
                                                    => call1 = the ww
                                                       dyn_adv/dyn_zad
                                                       CONSUMES this step.
    :302  CALL div_hor (2nd call, post dyn_spg_ts)
    :315  CALL wzv( kstp, Nbb, Nnn, Naa, ww )   -- AFTER dyn_zdf, RE-uses
                                                    the barotropic-corrected
                                                    divergence
                                                    => call2 = the ww
                                                       tra_adv (:417)
                                                       CONSUMES this step.
So: lego's vertical-momentum-advection ``w`` (the one fed to
``nemo_advective_vertical_momentum_advection`` / ZAD, per the #1455 ZAD
substitution finding already in the gate's "dyn_adv ZAD" note) is
STRUCTURALLY call1, not call2. Both are measured below; the row records
call1 as the primary number, call2 reported for completeness (per task
instruction 3) with an explicit statement of which lego's w corresponds to.

lego's w: the PRODUCTION diagnosed w -- ``_bc_vertical_and_depthmean_
velocity`` (``ocean_pe_latlon_cgrid.py:1328``) -> ``diagnose_w_from_flux_div``
(``:1356``), called from ``ocean_pe_latlon_cgrid.py:3953`` inside the same
``LatLonCGridOceanModel.tendencies_with_diagnostics`` entry point used by
every other #1226/#1455 row -- captured via the SAME spy-on-call pattern
``ww_inheritance_walk.py`` already used (not a re-derivation; the public API
does not expose ``w`` as a diagnostics field, so a call-site capture is the
only way to get the EXACT array fed downstream).

Uses ``cancelling_rows_per_element.py::per_element_stats`` (this campaign's
canonical corr/ratio/err_norm convention) wholesale -- not re-derived.

Population: active-only 3-D t-mask (w lives on the T-column, w-grid;
``tmask`` from the mesh_mask, matching the "e3t=both" / active-3D-mask
convention every other #1226/#1455 row uses).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.wzv_row_measure
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config

# Reuse wholesale: the established RUN_DIR/DT (zu_frc_term_walk.py) and the
# canonical per-element stats convention (cancelling_rows_per_element.py).
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import RUN_DIR, DT
from scripts.validate.ocean_fidelity.dino_1226.cancelling_rows_per_element import per_element_stats

JPI, JPJ, JPK, HLS = 56, 203, 36, 2  # w-grid: full jpk=36 levels (jk=1..jpk)


def _load_wzv(path: str) -> np.ndarray:
    """wzv dumps are DO jk = 1, jpk (36 levels, full w-grid, incl. bottom
    BC level jpk where pww=0) -- NOT jpkm1 like the tendency-increment dumps
    (_load_full_3d in zu_frc_term_walk.py is jpkm1-only, so not reused here)."""
    a = np.fromfile(path, dtype="<f8").reshape(JPK, JPJ, JPI)
    a = a[:, HLS:-HLS, HLS:-HLS]
    return np.moveaxis(a, 0, -1)  # (n_lat, n_lon, 36)


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="wzv_row_measure")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")
    assert e3t_mode == "both", "run with LEGOESM_NEMO_E3T=both (task rule)"

    for name in ("wzv_dump_ww_call1.bin", "wzv_dump_ww_call2.bin"):
        print(f"time_level({name}) = {time_level_for_dump(name)!r}")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.vertical_momentum_scheme={dcfg.vertical_momentum_scheme!r}")
    assert dcfg.vertical_momentum_scheme == "nemo_advective"

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, br.state, context="wzv_row_measure")
    print("dtype check: u", br.state.u.data.dtype, "z_coord.h_partial", br.z_coord.h_partial.dtype)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)

    # Capture lego's PRODUCTION diagnosed w via the exact call site the
    # model itself uses (public API exposes no `w` diagnostics field).
    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pemod

    captured = {}
    _real_bc_vert = pemod._bc_vertical_and_depthmean_velocity

    def _spy_bc_vert(*a, **kw):
        result = _real_bc_vert(*a, **kw)
        if "w" not in captured:
            captured["w"] = result[3]  # (h_u, h_v, flux_div_k, w, u_prime, v_prime)
        return result

    pemod._bc_vertical_and_depthmean_velocity = _spy_bc_vert
    try:
        with jax.disable_jit():
            _tend, _diag = model.tendencies_with_diagnostics(br.state, surface_forcing=None, dt=DT)
    finally:
        pemod._bc_vertical_and_depthmean_velocity = _real_bc_vert

    assert "w" in captured, "PROVE-THE-PATH-EXECUTES check failed: spy never fired"
    w_lego = np.asarray(captured["w"])
    print(f"\nPATH-EXECUTES CONFIRMED: _bc_vertical_and_depthmean_velocity fired, "
          f"w_lego shape={w_lego.shape} dtype={w_lego.dtype}")

    tmask3 = np.asarray(g.tmask) > 0.5  # (n_lat, n_lon, nlev) active-3D-mask population

    ww_call1 = _load_wzv(os.path.join(RUN_DIR, "wzv_dump_ww_call1.bin"))
    ww_call2 = _load_wzv(os.path.join(RUN_DIR, "wzv_dump_ww_call2.bin"))

    n_lat = min(w_lego.shape[0], ww_call1.shape[0], tmask3.shape[0])
    n_lon = min(w_lego.shape[1], ww_call1.shape[1], tmask3.shape[1])
    # SURFACE-ANCHORED alignment (reviewer nit, physics-validator SHIP note):
    # lego's w has nlev+1=37 interfaces (index 0 = surface, index 36 = padded
    # bottom zero); NEMO's dump has jpk=36 (jk=1 = surface). min() keeps
    # indices 0..35 aligned surface-to-surface and drops ONLY lego's padded
    # bottom zero -- correct precisely BECAUSE both fields are surface-first
    # and both ends are zero by construction. If diagnose_w_from_flux_div's
    # interface ordering ever changes, this truncation silently misaligns:
    # re-verify then (the all-dry NEMO jk=jpk dummy level hides it from the
    # wet-point population).
    n_lev = min(w_lego.shape[2], ww_call1.shape[2], tmask3.shape[2])
    wet3 = tmask3[:n_lat, :n_lon, :n_lev]
    lo = w_lego[:n_lat, :n_lon, :n_lev]
    ne1 = ww_call1[:n_lat, :n_lon, :n_lev]
    ne2 = ww_call2[:n_lat, :n_lon, :n_lev]

    print("\n" + "=" * 78)
    print("wzv (vertical velocity) ROW -- lego production w vs NEMO ww,")
    print("active-only 3-D tmask population, per_element_stats convention")
    print("=" * 78)
    print("\n-- CALL1 (stpmlf.F90:244, pre-dyn_adv -- the w dyn_zad CONSUMES; "
          "PRIMARY row number) --")
    stats1 = per_element_stats("wzv vs ww_call1", lo, ne1, wet3)

    print("\n-- CALL2 (stpmlf.F90:315, post-dyn_zdf/2nd div_hor -- the w "
          "tra_adv CONSUMES; reported per task instruction 3, NOT the "
          "structural match for lego's diagnosed w used in dyn_zad) --")
    stats2 = per_element_stats("wzv vs ww_call2", lo, ne2, wet3)

    # Per-level profile (localisation, task instruction 2 if DEBT).
    print("\n" + "=" * 78)
    print("PER-LEVEL err_norm=|lego-ww_call1|/RMS(ww_call1), active tmask")
    print("=" * 78)
    err1 = np.abs(lo - ne1)
    per_level = []
    for k in range(n_lev):
        m_k = wet3[..., k]
        if not m_k.any():
            per_level.append(float("nan"))
            continue
        rms_k = float(np.sqrt(np.mean(ne1[..., k][m_k] ** 2)))
        e_k = float(np.sqrt(np.mean(err1[..., k][m_k] ** 2)))
        per_level.append(e_k / max(rms_k, 1e-12))
    print(np.array2string(np.array(per_level), precision=3, max_line_width=200))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  call1 (structural match): n={stats1['n']} corr={stats1['corr']:.9f} "
          f"|x|ratio={stats1['ratio_abs']:.9f} bar_metric(median {'err_norm' if stats1['sign_changing'] else '|rel|'})"
          f"={stats1['bar_metric']:.3e} p99={stats1['p99_en'] if stats1['sign_changing'] else stats1['p99_rel']:.3e} "
          f"max={stats1['max_en'] if stats1['sign_changing'] else stats1['max_rel']:.3e} at_bar={stats1['at_bar']}")
    print(f"  call2 (reported, non-structural): n={stats2['n']} corr={stats2['corr']:.9f} "
          f"|x|ratio={stats2['ratio_abs']:.9f} bar_metric={stats2['bar_metric']:.3e} at_bar={stats2['at_bar']}")

    return 0 if stats1["n"] > 0 and stats2["n"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
