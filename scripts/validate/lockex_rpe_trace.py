"""Per-step RPE / variance trace for the lock-exchange sorted-RPE sign puzzle.

Question this answers: WHERE does the negative sorted-RPE drift accumulate?
Every step logs, on the same state:

  RPE_fix   sorted RPE on FIXED Boussinesq volumes (area * dz_ref)
  RPE_mov   sorted RPE on MOVING z-star volumes   (area * h(eta))
  M2        wet-volume-weighted tracer variance   sum(A h (T - Tbar)^2)
  T_min/max ocean-only tracer bounds
  heat_mov  wet heat content on the SAME moving volumes, sum(A h T)  [K m3]
  vol_mov   total wet moving volume, sum(A h)  [m3]
  Tbar      heat_mov / vol_mov; heat residual net of volume drift is
            heat_mov - heat_mov0 - Tbar0*(vol_mov - vol_mov0)  (offline)
  viol_mov  volume-weighted bounds violation, sum(A h ((T-30)_+ + (5-T)_+))
  eta_max   |eta| max
  lat/lon_Tmax, lat/lon_Tmin  degrees; WHERE the ocean-masked extremes live
            (localises a violation: front vs southern wall vs north fold)

Discriminators:
  * M2 down + RPE_fix down TOGETHER, smoothly     -> metric-vs-physics mismatch
    accumulates continuously (sampling/EOS class).
  * RPE_fix drops in bursts uncorrelated with M2  -> event-driven (e.g. fronts
    crossing cell boundaries, remap).
  * RPE_fix and RPE_mov diverge from step 1       -> the eta volume-weight term.
  * T bounds escape [5, 30] when RPE drops        -> over/undershoot after all.
  * heat_mov/vol_mov drift with RPE_mov           -> tracer/volume
    NON-CONSERVATION, not rearrangement: the sorted metric only means mixing
    when the (rho, vol) multiset is conserved.
  * viol_mov tracks the RPE_mov drop              -> the drop IS the
    over/undershoot (anti-diffusive variance production at the front).

--tracer-adv overrides the matrix's matched advection scheme (latlon/tripole/
mpas arms only; FESOM's is internal). Output goes to
rpe_trace_<grid>_<scheme>.csv so A/B arms never clobber the control.

Writes one CSV per grid to <out>/rpe_trace_<grid>.csv. GPU-ready
(JAX_PLATFORMS honoured; diagnostics pull to host each sample).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# MPAS scan carries fp32 leaves unless the global policy is fp64 (the matrix
# runner sets this in main(); a standalone driver must too).
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "matrix"))

_RES = {"latlon": "36x72", "mpas": "ico3", "fesom": "pi", "tripole": "eorca1"}

# Schemes each arm's model constructor accepts (--tracer-adv validation;
# MPASOceanModel raises only at construction time, deep in setup).
_VALID_ADV = {
    "latlon": {"upwind", "centered", "tvd", "superbee", "ppm_fct", "fct2",
               "ppm", "dst3", "dst3_multidim", "som", "weno5", "weno7"},
    "tripole": {"upwind", "centered", "tvd", "superbee", "ppm_fct", "fct2",
                "ppm", "dst3", "dst3_multidim", "som", "weno5", "weno7"},
    "mpas": {"upwind", "tvd", "superbee"},
}


def _load_matrix():
    spec = importlib.util.spec_from_file_location(
        "rm", _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rm"] = mod
    spec.loader.exec_module(mod)
    return mod


def trace(grid_type: str, days: float, dt: float, sample_every: int,
          out_dir: Path, m, tracer_adv: str | None = None,
          uniform_t: bool = False) -> Path:
    if tracer_adv is not None:
        if grid_type == "fesom":
            raise SystemExit(
                "--tracer-adv has no effect on the FESOM arm (its advection "
                "is internal to fesom_jax); refusing to run a lie.")
        if tracer_adv not in _VALID_ADV[grid_type]:
            raise SystemExit(
                f"--tracer-adv {tracer_adv!r} is not accepted by the "
                f"{grid_type} model; valid: {sorted(_VALID_ADV[grid_type])}")
        # All three legoESM arms read the module global at setup-call time
        # (latlon via setdefault, mpas/tripole via explicit kw), so patching
        # it swaps the scheme without touching the matched A_v/K_v.
        # Restored in the finally below so an in-process control run after an
        # override arm is not silently poisoned (codex).
    # BOTH globals: the lock-exchange C-grid arms read
    # LOCKEX_CGRID_TRACER_ADV (fct2 default), everything else
    # MATCHED_TRACER_ADV -- an override must reach whichever the arm uses.
    old_adv = m.MATCHED_TRACER_ADV
    old_cgrid_adv = m.LOCKEX_CGRID_TRACER_ADV
    if tracer_adv is not None:
        m.MATCHED_TRACER_ADV = tracer_adv
        m.LOCKEX_CGRID_TRACER_ADV = tracer_adv
    try:
        return _trace_body(grid_type, days, dt, sample_every, out_dir, m,
                           tracer_adv, uniform_t)
    finally:
        m.MATCHED_TRACER_ADV = old_adv
        m.LOCKEX_CGRID_TRACER_ADV = old_cgrid_adv


def _trace_body(grid_type, days, dt, sample_every, out_dir, m,
                tracer_adv, uniform_t) -> Path:
    if grid_type == "fesom" and abs(dt - m.DEFAULT_DT) > 1e-9:
        raise SystemExit(
            f"--dt {dt} unsupported on the FESOM arm: FesomOceanModel's SSH "
            f"operator is built for dt={m.DEFAULT_DT} in _create_ocean_setup "
            f"and step() refuses a mismatch.")
    tc = SimpleNamespace(grid_type=grid_type, resolution=_RES[grid_type],
                         case="lock_exchange", run_kwargs={})
    grid, z, cfg, model, ck, lon, lat = m._create_ocean_setup(
        tc, nlev=20, H_max=20.0)
    if grid_type == "fesom":
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            create_lock_exchange_state)
        from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig
        st = create_lock_exchange_state(grid.mesh, LockExchangeConfig())
    else:
        st = m._create_rest_state(tc, grid, z, H_max=20.0)
        st = m._init_lock_exchange(st, grid_type, grid, z)
    if uniform_t:
        # Consistency-with-continuity probe: with T=15 uniform the tracer
        # equation must reduce discretely to continuity, so ANY departure of
        # T from 15 is advecting-flux/thickness-update inconsistency --
        # limiter-independent, and the candidate source of the over/undershoot
        # that survived the fct2/upwind scheme swaps. A uniform-density basin
        # at rest would stay at rest, so an eta step (+/-5 cm across the
        # lock-exchange front) forces barotropic sloshing over the uniform
        # tracer.
        import jax.numpy as _jnp
        if grid_type == "fesom":
            raise SystemExit("--uniform-t not wired for the FESOM arm.")
        # eta step sign taken from the lock-exchange front ALREADY in T
        # (warm side +5 cm, cold side -5 cm) -- guaranteed the right shape
        # and the same front geometry as the real case.
        sign = _jnp.where(st.T.data[..., 0] > 15.0, 1.0, -1.0)
        eta_pert = st.eta.replace(
            data=(0.05 * sign).astype(st.eta.data.dtype))
        T15 = st.T.replace(data=_jnp.full_like(st.T.data, 15.0))
        st = st._replace(T=T15, eta=eta_pert)

    from legoesm.ocean.eos import linear_eos
    from legoesm import constants
    import jax.numpy as jnp

    def diags(s):
        T, S, area_bc, mask_bc, z_c, h = m._rpe_extract(s, grid_type, grid, z)
        rho = np.asarray(linear_eos(
            jnp.array(T), jnp.array(S), jnp.zeros_like(jnp.array(T)),
            rho_ref=constants.rho_ocean, alpha_T=2.0e-4, beta_S=0.0,
            T_ref=15.0), dtype=np.float64)
        dz_ref = np.asarray(z.dz_ref, dtype=np.float64)
        oc3 = np.broadcast_to((mask_bc > 0.5)[..., None], T.shape)
        total_area = float(np.sum(area_bc[mask_bc > 0.5]))
        vol_fix = np.broadcast_to(
            area_bc[..., None] * dz_ref, T.shape)[oc3]
        vol_mov = np.broadcast_to(area_bc[..., None] * h, T.shape)[oc3]
        rho_o = rho[oc3]
        if not (np.all(np.isfinite(rho_o))
                and np.all(np.isfinite(vol_mov))):
            raise ValueError(f"non-finite at sample, grid={grid_type}")
        if not np.all(vol_mov > 0.0):
            raise ValueError(
                f"non-positive moving volume at sample, grid={grid_type}: "
                f"min={vol_mov.min():.3e} (finite but would corrupt the sort)")
        Tbar = float(np.sum(T[oc3] * vol_mov) / np.sum(vol_mov))
        m2 = float(np.sum(vol_mov * (T[oc3] - Tbar) ** 2))
        heat_mov = float(np.sum(T[oc3] * vol_mov))
        vol_tot = float(np.sum(vol_mov))
        viol_mov = float(np.sum(vol_mov * (
            np.maximum(T[oc3] - 30.0, 0.0) + np.maximum(5.0 - T[oc3], 0.0))))
        # WHERE the extremes live (argmax metadata rule): lat/lon of the
        # ocean-masked T max/min, to localise a violation (front vs walls vs
        # north fold) without a separate probe.
        T_masked = np.where(oc3, T, np.nan)
        spatial = T.shape[:-1]
        lat_a = np.asarray(lat, dtype=np.float64)
        lon_a = np.asarray(lon, dtype=np.float64)
        if lat_a.size != int(np.prod(spatial)):
            # latlon arm: 1-D axes -> 2-D tracer-point fields
            lat_a, lon_a = np.meshgrid(lat_a, lon_a, indexing="ij")
        lat_a = lat_a.reshape(spatial)
        lon_a = lon_a.reshape(spatial)
        i_max = np.unravel_index(np.nanargmax(T_masked), T.shape)[:-1]
        i_min = np.unravel_index(np.nanargmin(T_masked), T.shape)[:-1]
        loc = (float(lat_a[i_max]), float(lon_a[i_max]),
               float(lat_a[i_min]), float(lon_a[i_min]))
        eta = np.asarray(s.eta.data, dtype=np.float64)
        # Committed Phase-C metric (legoesm.ocean.rpe.compute_rpe). Its
        # packing was FIXED 2026-08-10 to densest-at-bottom (it used to
        # invert the sign against its own docstring); it now delegates to
        # the same pack_sorted_rpe kernel as RPE_fix/RPE_mov and differs
        # only in its EOS configuration and volume construction.
        from legoesm.ocean.rpe import compute_rpe as _committed_rpe
        rpe_committed = float(_committed_rpe(
            s, z, grid_type=grid_type, grid=grid, eos="linear"))
        # Shared packing kernel from the matrix runner -- the SAME code the
        # RPE_rel gate uses, applied to the two volume choices (codex: no
        # duplicated sorted-RPE math that can go stale).
        return (m._pack_sorted_rpe(rho_o, vol_fix, total_area),
                m._pack_sorted_rpe(rho_o, vol_mov, total_area),
                rpe_committed,
                m2, float(T[oc3].min()), float(T[oc3].max()),
                heat_mov, vol_tot, Tbar, viol_mov,
                float(np.abs(eta[mask_bc > 0.5]).max()), *loc)

    n_steps = int(days * 86400 / dt)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tracer_adv}" if tracer_adv is not None else ""
    if uniform_t:
        suffix += "_uniformT"
    out = out_dir / f"rpe_trace_{grid_type}{suffix}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "t_days", "RPE_fix", "RPE_mov", "RPE_committed",
                    "M2", "T_min", "T_max",
                    "heat_mov", "vol_mov", "Tbar", "viol_mov", "eta_max",
                    "lat_Tmax", "lon_Tmax", "lat_Tmin", "lon_Tmin"])
        w.writerow([0, 0.0, *diags(st)])
        for n in range(1, n_steps + 1):
            st = model.step(st, dt)
            if n % sample_every == 0 or n == n_steps:
                w.writerow([n, n * dt / 86400.0, *diags(st)])
                fh.flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", required=True, choices=sorted(_RES))
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=300.0)
    ap.add_argument("--sample-every", type=int, default=6,
                    help="steps between samples (6 = every 30 min at dt=300)")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "results" / "lockex_rpe_trace")
    ap.add_argument("--tracer-adv", default=None,
                    help="override the matrix's matched tracer_advection for "
                         "this arm (e.g. fct2, ppm_fct, upwind); default = "
                         "matrix MATCHED_TRACER_ADV. Refused for fesom.")
    ap.add_argument("--uniform-t", action="store_true",
                    help="consistency-with-continuity probe: uniform T=15 "
                         "under an eta step; T moving off 15 = advecting-"
                         "flux/thickness inconsistency (limiter-independent)")
    a = ap.parse_args()
    m = _load_matrix()
    out = trace(a.grid, a.days, a.dt, a.sample_every, a.out, m,
                tracer_adv=a.tracer_adv, uniform_t=a.uniform_t)
    print(f"COMPLETED {a.grid}: {out}")


if __name__ == "__main__":
    main()
