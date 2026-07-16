"""FAST end-to-end check of the COMPLETE correction loop against REAL ERA5.

The smoke (``smoke_compare_reanalysis.py``) validates CONSTRUCTION on synthetic data; the
C_K-sensitivity pre-flight measures the C_K LEVERAGE on real data; the OSSE recovers a known
parameter in a perfect-model twin.  This fills the remaining gap: does the WHOLE loop —
real-ERA5 ingest → compare → rank → LES spin-off → diagnose → re-run → monotonic gate — RUN
end-to-end on YOUR real ERA5 archive?  It uses a single-state model (no slow climatology
time-mean) + a SMALL LES regime so it finishes in MINUTES, not the multi-day production
campaign (whose LES spin-off dominates the runtime — iter 499/501/502).

This validates that the loop COMPOSES + the gate behaves on real data; it is NOT a science
result (the small LES under-resolves turbulence, and gray radiation gives a C_K-INSENSITIVE
bias the gate correctly REJECTS — iter 412/503: bias 11.24 K, gate rejects).  The production
SCIENCE run (rrtmgp / high-res / equilibrated, where the bias is C_K-controllable) is the
HPC-scale campaign.  Exits 0 when the complete loop ran end-to-end (finite biases + ≥1 worst
column processed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the sibling ``scripts.*`` entry points importable when run as a standalone CLI.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_arg_parser() -> argparse.ArgumentParser:
    """The CLI parser as a factory so the flag contract is unit-testable."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-era5-dir", required=True,
                   help="a local NCAR-RDA ll025 ERA5 archive (e5.oper.an.{pl,sfc}.*.ll025*.nc)")
    p.add_argument("--local-era5-date", required=True, help="YYYYMMDD")
    p.add_argument("--resolution", type=int, default=4)
    p.add_argument("--nlev", type=int, default=10)
    p.add_argument("--era5-n-times", type=int, default=2)
    p.add_argument("--n-worst", type=int, default=2)
    return p


def _run_coupled(c_k, *, resolution, nlev):
    """A tiny single-state coupled (CMIP) clubb_lite run at the given C_K (no time-mean).

    ``c_k`` may be a SCALAR (the baseline / a uniform C_K) OR a per-column ``(n_columns,)``
    field — the correction loop injects a per-column C_K for the re-run, and clubb_lite
    broadcasts it via ``broadcast_column_param`` (iter 507: ``float(c_k)`` crashed on the
    1-D field, so the per-column re-run was never actually exercised by this tool)."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig, TurbulenceConfig
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    c_k_arg = c_k if getattr(c_k, "ndim", 0) > 0 else float(c_k)
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=int(resolution), nlev=int(nlev)),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=1), radiation="gray", days=1,
        turbulence="clubb_lite",
        turbulence_override=TurbulenceConfig(
            scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=c_k_arg)))
    driver = CoupledESMDriver(atm, PRESETS["aquaplanet"](), ocean_grid=None)
    driver.setup()
    driver.run()
    return driver


def main(argv: list[str] | None = None) -> int:
    """Run the complete real-ERA5 loop once and report the bias trajectory + gate verdict."""
    args = _build_arg_parser().parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    from functools import partial

    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig, run_forced_les
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.compare_reanalysis import (
        column_state_from_carry,
        column_state_from_hydrostatic,
    )
    from legoesm.training.correction_loop import make_compare_fn, run_correction_iteration
    from legoesm.training.era5_to_state import (
        WB2_PRESSURE_LEVELS,
        TrainingERA5Config,
        load_era5_time_mean,
    )

    from scripts.data.load_local_era5 import open_local_era5_dataset
    from scripts.run.run_correction_campaign import (
        fast_validation_les_regime,
        make_les_diagnose_fn,
    )
    from scripts.validate.compare_amip_era5 import canonical_grid_type, select_era5_regrid

    print("[real-loop] baseline coupled model (res "
          f"{args.resolution}, gray) ...", flush=True)
    d0 = _run_coupled(0.4, resolution=args.resolution, nlev=args.nlev)
    a = d0._atm
    model0 = column_state_from_hydrostatic(a.state, a.q_v, sst_K=d0._ocean_state.T_sfc.data)
    grid, sigma = a.grid, a.sigma
    n_lat, n_lon, _ = model0.T.shape

    print(f"[real-loop] ingesting REAL ERA5 ({args.local_era5_date}) → reference on the "
          f"{n_lat}x{n_lon} grid ...", flush=True)
    ds = open_local_era5_dataset(args.local_era5_dir, args.local_era5_date)
    era5 = load_era5_time_mean(
        TrainingERA5Config(levels=WB2_PRESSURE_LEVELS,
                           surface_variables=("surface_pressure", "skin_temperature")),
        range(int(args.era5_n_times)), ds=ds)
    canon = canonical_grid_type("latlon")
    reference = column_state_from_carry(select_era5_regrid(canon)(era5, grid, sigma))
    print(f"[real-loop] real ERA5 reference: T ∈ [{float(jnp.min(reference.T)):.1f}, "
          f"{float(jnp.max(reference.T)):.1f}] K, p_s ∈ [{float(jnp.min(reference.p_s)):.0f}, "
          f"{float(jnp.max(reference.p_s)):.0f}] Pa", flush=True)

    rad2deg = 180.0 / float(jnp.pi)
    compare_fn = make_compare_fn(
        reference=reference, sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=jnp.asarray(grid.grid_lat) * rad2deg,
        lon_deg=jnp.asarray(grid.grid_lon) * rad2deg,
        area_weights=jnp.ones((n_lat, n_lon)), n_worst=int(args.n_worst),
        run_amip_fn=lambda clubb: column_state_from_hydrostatic(
            (dd := _run_coupled(clubb.C_K, resolution=args.resolution, nlev=args.nlev))._atm.state,
            dd._atm.q_v, sst_K=dd._ocean_state.T_sfc.data),
        coordinate=sigma)

    # Use the DIMENSIONALLY-CORRECT clubb_coefficient diagnosis (C_K = K_m/(ℓ·√wp2)),
    # threading the GCM's l_mix_max so the diagnosed ℓ matches the model's — exactly as the
    # production campaign/OSSE do (auto-populate, iter 379). The legacy "eddy_diffusivity"
    # diagnoses a DIMENSIONAL K [m²/s] that CANNOT be injected as the dimensionless
    # clubb_lite_C_K (it just saturates the bounds-clamp) — iter-503 wrongly used it, so this
    # tool now validates the production-correct path (iter 507).
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma,
        les_config=ColumnLESConfig(regime=fast_validation_les_regime(),
                                   diagnosis_method="clubb_coefficient",
                                   clubb_l_mix_max=CLUBBLiteConfig().l_mix_max,
                                   gate_les_realism=False),
        run_les_fn=partial(run_forced_les, dt_s=0.5, n_steps=2))

    print("[real-loop] running the FULL correction iteration vs REAL ERA5 "
          "(compare → rank → LES → diagnose → re-run → gate) ...", flush=True)
    res = run_correction_iteration(
        CLUBBLiteConfig(C_K=0.4), compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=(n_lat, n_lon), background=0.4,
        diagnosis_method="clubb_coefficient", les_budget=int(args.n_worst))

    base, upd = float(res.bias.baseline_bias), float(res.bias.updated_bias)
    ran = np.isfinite(base) and np.isfinite(upd) and res.n_corrected >= 1
    print("\n[real-loop] RESULT — the COMPLETE real-ERA5 loop:")
    print(f"  bias vs real ERA5: {base:.5g} → {upd:.5g}  (improved={bool(res.bias.improved)})")
    print(f"  worst columns processed: {res.n_corrected}; valid LES diagnoses: "
          f"{res.n_diagnoses_valid}")
    print(f"  complete loop ran end-to-end: {ran}  "
          "(NOT a science result — small LES + gray → C_K-insensitive bias; the gate "
          "correctly REJECTS. The SCIENCE run is the HPC campaign with rrtmgp/high-res.)")
    return 0 if ran else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
