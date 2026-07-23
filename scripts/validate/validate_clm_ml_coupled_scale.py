"""S5 validation: the coupled CLM-ML land tile compiles O(1) in the column count.

S3 replaced the CLM-ML canopy's per-column Python loop with a ``jax.lax.scan`` over
columns.  This validator confirms the PAYOFF end-to-end through the coupler: build the
coupled ``ModelDriver`` with ``--land-surface-scheme clm_ml`` at two atmosphere
resolutions (→ two land column counts), warm-start the canopy (eager, O(ncol) — kept
OUT of the compile window), then lower + compile ONE jitted coupled land-tile step at
each ncol and compare.

The O(1) signal is the **HLO program size** of the lowered step (deterministic,
CI-robust): with the S3 scan the canopy contributes a single ``while`` regardless of
ncol, so the coupled step's HLO is ~constant in ncol; the pre-S3 O(ncol) Python-loop
unroll made it grow ~linearly.  Wall-clock compile time is reported too, but only as
informational context (it is not the pass/fail signal — see ``assess_o1``).

Synthetic land data (uniform loam, patched loaders) is used deliberately: this is a
COMPILE-SCALING benchmark, not a physics-accuracy check (that is
``validate_clm_ml_canopy.py`` at the EC site).  Requires the vendored CLM-ML backend.

Run:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
      PYTHONPATH="<pkgs>" python scripts/validate/validate_clm_ml_coupled_scale.py
Exit 0 if the coupled compile is sub-linear in ncol (O(1)-ish), 1 otherwise.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# O(1) verdict — PURE, deterministic (unit-tested; no driver / JAX needed)
# ---------------------------------------------------------------------------
# The coupled compile is "sub-linear in ncol" (the S3-scan payoff) when the HLO
# program grows markedly slower than the column count.  A pre-S3 O(ncol) unroll
# grows ~linearly (ratio ≈ ncol_ratio); the scan grows ~constant (ratio ≈ 1).
# Pass when the size ratio is at most this fraction of the ncol ratio.
SUBLINEAR_FRAC: float = 0.75


def assess_o1(rows):
    """Given ascending-ncol measurements, decide if compile is sub-linear in ncol.

    ``rows``: list of dicts with keys ``ncol`` and ``hlo_chars`` (>=2 rows, ncol
    strictly increasing).  Returns ``(is_o1, detail)``.  Uses the FIRST and LAST
    row so it works for 2+ resolutions.  Deterministic — the unit test drives it
    with synthetic numbers (linear must FAIL, sub-linear must PASS).
    """
    if len(rows) < 2:
        raise ValueError("assess_o1 needs >=2 measurements")
    ncols = [r["ncol"] for r in rows]
    if any(b <= a for a, b in zip(ncols, ncols[1:])):
        raise ValueError(f"ncol must be strictly increasing, got {ncols}")
    lo, hi = rows[0], rows[-1]
    ncol_ratio = hi["ncol"] / lo["ncol"]
    size_ratio = hi["hlo_chars"] / max(lo["hlo_chars"], 1)
    is_o1 = size_ratio <= SUBLINEAR_FRAC * ncol_ratio
    detail = dict(ncol_ratio=ncol_ratio, size_ratio=size_ratio,
                  threshold=SUBLINEAR_FRAC * ncol_ratio, is_o1=is_o1)
    return is_o1, detail


# ---------------------------------------------------------------------------
# Synthetic coupled-driver setup (benchmark only) + measurement
# ---------------------------------------------------------------------------
def _fake_surface_map(path, lat_deg, lon_deg):
    """Uniform-loam, all-bare-soil CLM map sized to the requested columns."""
    n = int(np.asarray(lat_deg).size)
    pft = np.zeros((n, 17))
    pft[:, 0] = 1.0
    o = np.ones(n)
    return dict(
        pft_fractions=jnp.asarray(pft),
        theta_wp=jnp.asarray(0.12 * o), theta_fc=jnp.asarray(0.30 * o),
        glacier_frac=jnp.asarray(np.zeros(n)),
        pct_sand=jnp.asarray(40.0 * o), pct_clay=jnp.asarray(20.0 * o),
        theta_r=jnp.asarray(0.05 * o), theta_sat=jnp.asarray(0.45 * o),
        alpha_vg=jnp.asarray(2.0 * o), n_vg=jnp.asarray(1.4 * o),
        K_sat=jnp.asarray(1.0e-5 * o),
    )


def _apply_synthetic_land():
    """Point the CLM-surfdata + land-fraction loaders at synthetic data (benchmark)."""
    import legoesm.land.clm_surface_map as clm
    import legoesm.grids.topography as topo
    clm.download_clm_surfdata = lambda *a, **k: "synthetic"
    clm.load_clm_surface = _fake_surface_map
    topo.load_land_fraction = lambda grid, path, *a, **k: jnp.full(grid.lat.shape, 0.5)


def _build_cfg(resolution):
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig)
    return ExperimentConfig(
        grid=GridConfig(resolution=resolution, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        land_mask_path="synthetic.nc",
        use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
        land_surface_scheme="clm_ml",
    )


def measure(resolution):
    """Build the coupled clm_ml driver at ``resolution``; return ncol + HLO size +
    compile wall time of one jitted coupled land-tile step (warm-start excluded)."""
    from legoesm.driver.model_driver import ModelDriver
    _apply_synthetic_land()
    with tempfile.TemporaryDirectory() as td:
        driver = ModelDriver(_build_cfg(resolution), output_dir=td)
        t0 = time.time()
        driver.setup()                      # eager warm-start (O(ncol)) — excluded below
        t_setup = time.time() - t0
        ncol = int(driver._land_ml_state.T_soil.shape[0])
        tile = driver.physics._step_multilayer_land_tile

        def step(land_ml, T, p_s, q_v, u, v):
            return tile(land_ml, jnp.full(ncol, 400.0), jnp.full(ncol, 350.0),
                        T, p_s, q_v, u, v, None, 600.0,
                        cos_zenith_col=jnp.full(ncol, 0.7))

        args = (driver._land_ml_state, driver.state.T.data, driver.state.p_s.data,
                driver.q_v, driver.state.u.data, driver.state.v.data)
        t0 = time.time()
        lowered = jax.jit(step).lower(*args)
        hlo = lowered.as_text()
        t_lower = time.time() - t0
        t0 = time.time()
        compiled = lowered.compile()
        t_compile = time.time() - t0
        _, T_sfc, _ = compiled(*args)
        T_sfc = np.asarray(T_sfc)
    return dict(res=resolution, ncol=ncol, hlo_chars=len(hlo),
                t_setup=t_setup, t_compile_wall=t_lower + t_compile,
                T_sfc_finite=bool(np.all(np.isfinite(T_sfc))),
                T_sfc_min=float(T_sfc.min()), T_sfc_max=float(T_sfc.max()))


def main(resolutions=(2, 3)):
    import inspect
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLCanopyFluxesMod as _m
    if "cos_zenith_device" not in inspect.signature(_m.MLCanopyFluxes).parameters:
        print("SKIP: vendored CLM-ML backend lacks cos_zenith_device=")
        return 0
    rows = [measure(r) for r in resolutions]
    print("\n=== S5: coupled CLM-ML compile scaling in ncol ===")
    print("res | ncol | warm-start(s) | compile-wall(s) | HLO chars | Tsfc[min,max] finite")
    for r in rows:
        print("%3d | %4d | %12.1f | %14.2f | %9d | [%.1f,%.1f] %s" % (
            r["res"], r["ncol"], r["t_setup"], r["t_compile_wall"], r["hlo_chars"],
            r["T_sfc_min"], r["T_sfc_max"], r["T_sfc_finite"]))
    is_o1, d = assess_o1(rows)
    print("\nncol_ratio=%.2fx  HLO_size_ratio=%.2fx  (threshold <= %.2fx)  => O(1): %s" % (
        d["ncol_ratio"], d["size_ratio"], d["threshold"], is_o1))
    finite = all(r["T_sfc_finite"] for r in rows)
    print("skin T finite at all scales: %s" % finite)
    return 0 if (is_o1 and finite) else 1


if __name__ == "__main__":
    sys.exit(main())
