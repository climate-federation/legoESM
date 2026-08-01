"""#1226 ITEM 1 (scope-capped task) -- ONE-OFF: corr/ratio for the ZAD gate row
under the CORRECTED population (3-D umask), vs the union population.

READ-ONLY measurement. Reuses zad_level29_onset_walk.py's exact load path
(RUN_GDB restart kt=57601, LEGOESM_NEMO_E3T=both, nemo_advective scheme) --
not re-derived. Does NOT touch any UNTOUCHABLE walk script. Purpose: the gate
row's provenance script (run_dyn_zad_probe.py) uses a SYNTHETIC-input
harness; this script gets corr/ratio on the REAL restart-dump comparison
(the population the union/active-only err_norm numbers in
zad_level29_onset_walk.py / dyn_zad_ldf_walk.py already established), with
the ACTIVE-ONLY (3-D umask) population as the row's headline metric per the
task's ITEM 1 instruction ("mask the ZAD dump with the 3-D umask before
comparing").

Run::
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.zad_gate_corr_ratio_1226
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config

from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import RUN_DIR, DT, _load_full_3d

register_dump("zad_dump_du.bin", "now", "dynadv.F90:97 dyn_zad Krhs increment (stock dynzad.F90:86-119).")
time_level_for_dump("zad_dump_du.bin")


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _corr_ratio(a, b):
    a, b = np.asarray(a), np.asarray(b)
    corr = float(np.corrcoef(a, b)[0, 1])
    ratio = float(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(b ** 2))) if np.any(b) else float("nan")
    return corr, ratio


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="_zad_gate_corr_ratio_1226")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")
    assert e3t_mode == "both"

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    assert dcfg.vertical_momentum_scheme == "nemo_advective"

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    restart_path = os.path.join(RUN_DIR, "DINO_00057600_restart.nc")
    print(f"restart path: {restart_path}  step kt=57601 (established #1226 probe point)")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(restart_path, nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, br.state, context="_zad_gate_corr_ratio_1226")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    with jax.disable_jit():
        _tend, diag = model.tendencies_with_diagnostics(br.state, surface_forcing=None, dt=DT)

    vertadv_u_3d = np.asarray(diag.vertadv_u.data)
    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    umask3 = np.asarray(g.umask) > 0.5  # the 3-D mask NEMO's dynzdf.F90:121 actually applies

    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)
    vertadv_u_f = _u_to_nemo(vertadv_u_3d)

    n_lat = min(vertadv_u_f.shape[0], nemo_zad_du.shape[0], umask2.shape[0], umask3.shape[0])
    n_lon = min(vertadv_u_f.shape[1], nemo_zad_du.shape[1], umask2.shape[1], umask3.shape[1])
    n_lev = min(vertadv_u_f.shape[2], nemo_zad_du.shape[2], umask3.shape[2])
    lo = vertadv_u_f[:n_lat, :n_lon, :n_lev]
    ne = nemo_zad_du[:n_lat, :n_lon, :n_lev]
    m2 = umask2[:n_lat, :n_lon]
    m3 = umask3[:n_lat, :n_lon, :n_lev]

    # UNION population: the OLD (harness-buggy) metric -- surface u-mask
    # broadcast across every level, so below-seafloor cells NEMO discards
    # (dynzdf.F90:121) are still compared.
    m2_bcast = np.broadcast_to(m2[:, :, None], lo.shape)
    a_union, b_union = lo[m2_bcast], ne[m2_bcast]
    corr_union, ratio_union = _corr_ratio(a_union, b_union)
    err_norm_union = float(np.sqrt(np.mean((a_union - b_union) ** 2)) / np.sqrt(np.mean(b_union ** 2)))

    # ACTIVE-ONLY population: the CORRECTED metric -- the full 3-D umask,
    # i.e. exactly the population dynzdf.F90:121 keeps.
    a_act, b_act = lo[m3], ne[m3]
    corr_act, ratio_act = _corr_ratio(a_act, b_act)
    err_norm_act = float(np.sqrt(np.mean((a_act - b_act) ** 2)) / np.sqrt(np.mean(b_act ** 2)))

    print(f"n cells: union(surface-mask bcast)={a_union.size}  active-only(3-D umask)={a_act.size}")
    print(f"UNION       (surface u-mask bcast, OLD/buggy pop): corr={corr_union:.6f} ratio={ratio_union:.6f} "
          f"err_norm={err_norm_union:.4e}")
    print(f"ACTIVE-ONLY (3-D umask, CORRECTED pop, = NEMO's kept cells): corr={corr_act:.6f} ratio={ratio_act:.6f} "
          f"err_norm={err_norm_act:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
