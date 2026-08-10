#!/usr/bin/env python
"""Per-level avm error, measured WITHOUT the fed-state confound.

``s17_coeff_arms.py`` proved that substituting NEMO's ``avm`` collapses the
dyn_zdf levels 0-8 residual ~200x, so ``avm`` OWNS it.  Its K3 alignment scan
also reported err_norm 1.63e-01 for avm -- but that number was taken in a run
where the solve's input state had been replaced by NEMO's, so legoESM's TKE
closure saw a DIFFERENT velocity (hence a different shear) than NEMO's zdf_phy
did.  That is a confound for the avm number itself (not for the arms).

Here the model runs on its OWN state (no substitution), which is the same
state NEMO's zdf_phy ran on (both start from the restart), so the avm
comparison is controlled.  Reference: ``dump_avm.bin`` = ldftra.F90:956,
written at stpmlf.F90:203, i.e. AFTER zdf_phy (:190) and BEFORE dyn_zdf (:305).

Mask: NEMO's ``wmask`` at the W-level the interface actually sits on (legoESM
interface j <-> NEMO 0-based level j+1), not tmask at level j.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.s17_avm_perlevel
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import dataclasses
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
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
)
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, _load_full_3d,
)
from scripts.validate.ocean_fidelity.dino_1226.bn2_alpha_compare import _read_dims
from scripts.validate.ocean_fidelity.dino_1226.s17_dynzdf_bracket import (
    _SUB, _run_with_hook, _en, RESTART,
)

set_policy(PrecisionPolicy.fp64())

_CAP: dict[str, object] = {"A": None, "n": 0}


def main() -> int:
    require_explicit_e3t_mode(context="s17_avm_perlevel")
    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                     omega=dcfg.omega)
    br = br._replace(state=bridge_before_state_topo(br, g, bef,
                                                    periodic_i=True))
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                              sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.z_coord, br.state.T.data, br.state.S.data,
                 br.geometry.dx_u, context="s17_avm_perlevel")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    sf = dino_step_surface_forcing(
        dino_lat_lon_surface_forcing_arrays(br.geometry, cfg))

    import legoesm.ocean.physics.vertical_mixing as VM
    real = VM.compute_vertical_K_profiles

    def spy(*a, **kw):
        out = real(*a, **kw)
        _CAP["n"] = int(_CAP["n"]) + 1
        if _CAP["A"] is None:
            _CAP["A"] = np.asarray(out[1])
        return out

    VM.compute_vertical_K_profiles = spy
    try:
        _SUB["u"] = _SUB["v"] = None
        _run_with_hook(model, br.state, sf)          # NO substitution
    finally:
        VM.compute_vertical_K_profiles = real
    A_lego = _CAP["A"]
    print(f"[control] K-profile spy fired {_CAP['n']} times "
          f"{'PASS' if _CAP['n'] >= 1 else '*** HOOK NEVER FIRED ***'}")
    if _CAP["n"] < 1:
        raise SystemExit("spy never fired")

    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    avm = _load_full_3d(os.path.join(RUN_DIR, "dump_avm.bin"), jpi, jpj,
                        jpk - 1, hls)                       # (199,52,35)
    nif = A_lego.shape[-1]
    A_nemo = np.zeros_like(A_lego)
    n = min(nif, avm.shape[-1] - 1)
    A_nemo[..., :n] = avm[..., 1:1 + n]      # interface j <-> NEMO 0-based j+1
    wmask = np.asarray(g.wmask) if hasattr(g, "wmask") else np.asarray(g.tmask)
    which = "wmask" if hasattr(g, "wmask") else "tmask(FALLBACK)"
    M = np.zeros(A_lego.shape, dtype=bool)
    M[..., :n] = wmask[..., 1:1 + n] > 0.5
    print(f"[mask] {which}  wet interfaces = {int(M.sum())}  "
          f"dtypes lego={A_lego.dtype} nemo={A_nemo.dtype}")

    e, rn, rl, mx, cc = _en(A_lego, A_nemo, M)
    print(f"\nGLOBAL avm: err_norm={e:.4e} corr={cc:.6f} "
          f"RMS(nemo)={rn:.4e} RMS(lego)={rl:.4e} ratio={rl/rn:.6f} "
          f"max|d|={mx:.4e}")
    # Is the error a CONTINUOUS closure difference, or discrete zdf_evd
    # (rn_evd=100 m2/s, ocean.output:738) on/off FLIPS?  A flip contributes
    # |d| ~= rn_evd exactly; a closure difference does not pile up there.
    dd = np.abs(A_lego - A_nemo)[M]
    tot = float((dd ** 2).sum())
    for lo, hi, lab in ((0, 1, "|d|<1 (closure-scale)"),
                        (1, 50, "1..50"),
                        (50, 95, "50..95"),
                        (95, 105, "95..105  <-- rn_evd=100 FLIP band"),
                        (105, 1e9, ">105")):
        sel = (dd >= lo) & (dd < hi)
        print(f"  |d| in {lab:34s} n={int(sel.sum()):8d} "
              f"({100*sel.mean():6.3f}% of wet)  "
              f"share of sum(d^2) = {100*float((dd[sel]**2).sum())/tot:6.2f}%")
    ev_l = float((A_lego[M] > 95).mean())
    ev_n = float((A_nemo[M] > 95).mean())
    print(f"  fraction of wet interfaces at EVD level (>95 m2/s): "
          f"lego={100*ev_l:.3f}%  nemo={100*ev_n:.3f}%")
    print("\n  j  NEMO 0-based lv   err_norm      corr     RMS(nemo)   RMS(lego)"
          "    ratio")
    for j in list(range(0, 12)) + [15, 20, 25, 30, 33]:
        if j >= nif or not M[..., j].any():
            continue
        e, rn, rl, mx, cc = _en(A_lego[..., j:j+1], A_nemo[..., j:j+1],
                                M[..., j:j+1])
        print(f" {j:3d}   {j+1:8d}      {e:.4e}  {cc:8.5f}  {rn:.4e}  "
              f"{rl:.4e}  {rl/rn if rn else float('nan'):7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
