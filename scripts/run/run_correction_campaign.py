"""Offline LES-informed correction CAMPAIGN driver — the HPC entry point.

Composes the full loop (``docs/COMPARE_REANALYSIS.md``) into a runnable campaign:

    run AMIP/CMIP → time-mean → compare to ERA5 → rank worst columns
      → (cluster) → LES-diagnose → correct ``clubb_lite.C_K`` → RE-RUN → repeat.

Everything below the comparison is the already-tested training package; this
module supplies the two composition pieces that turn it into a runnable campaign:

* :func:`make_clubb_build_driver` — wraps a mode-specific ``build_base_driver``
  (AMIP ``ModelDriver`` / CMIP ``CoupledESMDriver``) so each run injects the
  corrected ``clubb_lite`` config via ``ExperimentConfig.turbulence_override``
  (iter 35) — the parameter-update mechanism.
* :func:`make_les_diagnose_fn` — turns the model column state (the loop's
  ``model_ctx``) into the LES closure coefficient via
  ``run_column_les.process_column`` (extract the GCM column forcing → spin off a
  real plane LES → diagnose ``K`` / ``w_e``).

:func:`build_correction_campaign` assembles ``make_run_fn`` ∘ ``make_compare_fn``
∘ the LES ``diagnose_fn`` and runs ``run_correction_campaign``; the ``main`` CLI
loads a base config + an ERA5 reference (reusing
``scripts/validate/compare_amip_era5.py``) and drives it on real data at scale.
The bias *sign* is an empirical result of the real run — this module only wires
the campaign.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from legoesm.atmosphere.physics.turbulence.config import (
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.training.correction_loop import make_compare_fn, run_correction_campaign
from legoesm.training.run_to_column_mean import make_run_fn


def make_clubb_build_driver(
    base_atm_config: Any,
    build_base_driver: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """``build_driver(clubb_cfg)`` that injects ``clubb_cfg`` via ``turbulence_override``.

    ``base_atm_config`` is the base :class:`ExperimentConfig` (its ``turbulence``
    MUST already be ``"clubb_lite"`` — the campaign corrects ``clubb_lite.C_K``,
    it does NOT switch schemes; a non-clubb base config raises rather than being
    silently rewritten).  Each call sets only ``turbulence_override`` to the
    corrected :class:`CLUBBLiteConfig` and hands the result to
    ``build_base_driver`` (which constructs + ``setup()``-s the AMIP/CMIP driver —
    the run mode stays injected, so this is grid/mode-agnostic).
    """
    if getattr(base_atm_config, "turbulence", None) != "clubb_lite":
        raise ValueError(
            "make_clubb_build_driver: base_atm_config.turbulence must be "
            f"'clubb_lite' (the campaign corrects clubb_lite.C_K, it does not "
            f"switch schemes); got {getattr(base_atm_config, 'turbulence', None)!r}."
        )

    def build_driver(clubb_cfg: Any) -> Any:
        cfg = base_atm_config._replace(
            turbulence_override=TurbulenceConfig(
                scheme="clubb_lite", clubb_lite=clubb_cfg),
        )
        return build_base_driver(cfg)

    return build_driver


def make_les_diagnose_fn(
    grid: Any,
    sigma: Any,
    *,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
) -> Callable[[Any, Any], Any]:
    """``diagnose_fn(record, model_ctx)`` that spins off + diagnoses a column LES.

    ``model_ctx`` is the model :class:`ColumnState` the loop passes through (from
    :func:`make_compare_fn`); ``run_column_les.process_column`` extracts that
    column's GCM large-scale forcing, runs the plane LES (``run_les_fn``), and
    diagnoses the closure coefficient.  ``grid`` / ``sigma`` are the model grid +
    vertical coordinate the forcing extractor needs.
    """
    from legoesm.atmosphere.dynamics.column_les import process_column

    def diagnose_fn(record: Any, model_ctx: Any) -> Any:
        if getattr(model_ctx, "u", None) is None or getattr(model_ctx, "v", None) is None:
            raise ValueError(
                "make_les_diagnose_fn: model_ctx needs cell-centred u/v for the "
                "column-LES forcing extraction (an MPAS edge-velocity state is "
                "not supported)."
            )
        return process_column(
            record,
            T=model_ctx.T, q_v=model_ctx.q_v, u=model_ctx.u,
            v=model_ctx.v, p_s=model_ctx.p_s,
            grid=grid, sigma=sigma, config=les_config, run_les_fn=run_les_fn,
        )

    return diagnose_fn


def build_correction_campaign(
    *,
    base_atm_config: Any,
    build_base_driver: Callable[[Any], Any],
    extract_column_state: Callable[..., Any],
    reference: Any,
    sigma: Any,
    grid: Any,
    area_weights: Any,
    n_iterations: int,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
    n_worst: int,
    les_budget: int | None = None,
    env_scales: Any | None = None,
    initial_clubb: Any | None = None,
    lat_deg: Any | None = None,
    lon_deg: Any | None = None,
):
    """Assemble + run the LES-informed ``clubb_lite.C_K`` correction campaign.

    Composes :func:`make_clubb_build_driver` → :func:`make_run_fn` →
    :func:`make_compare_fn` (vs the regridded ERA5 ``reference``) → the LES
    :func:`make_les_diagnose_fn`, and drives
    :func:`~legoesm.training.correction_loop.run_correction_campaign` for
    ``n_iterations`` rounds (``les_budget`` caps the LES count by environment
    clustering).  ``grid_shape`` is taken from ``reference.T`` (the model grid).
    Returns the :class:`CampaignResult`.
    """
    import jax.numpy as jnp
    import numpy as np

    grid_shape = tuple(int(d) for d in reference.T.shape[:-1])
    rad2deg = 180.0 / np.pi
    if lat_deg is None:
        lat_deg = jnp.asarray(np.asarray(grid.grid_lat) * rad2deg)
    if lon_deg is None:
        lon_deg = jnp.asarray(np.asarray(grid.grid_lon) * rad2deg)

    build_driver = make_clubb_build_driver(base_atm_config, build_base_driver)
    run_fn = make_run_fn(build_driver, extract_column_state)
    compare_fn = make_compare_fn(
        reference=reference,
        sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=lat_deg, lon_deg=lon_deg, area_weights=area_weights,
        n_worst=n_worst, run_amip_fn=run_fn,
    )
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn)

    background = float(CLUBBLiteConfig().C_K)
    return run_correction_campaign(
        initial_clubb if initial_clubb is not None else CLUBBLiteConfig(),
        int(n_iterations),
        compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=grid_shape,
        background=background, les_budget=les_budget, env_scales=env_scales,
    )


def make_base_driver_builder(
    mode: str, *, coupled_preset: Any = None, ocean_grid: Any = None
):
    """Return ``(build_base_driver, extract_column_state)`` for the run ``mode``.

    * ``"amip"`` → a :class:`ModelDriver` (prescribed SST) + ``amip_column_state``.
    * ``"cmip"`` → a :class:`CoupledESMDriver` (interactive ocean; ``coupled_preset``
      required) + ``cmip_column_state``.  ``ocean_grid=None`` (the default) makes
      the coupled driver use its OWN atmosphere grid for the ocean — same-grid
      coupling with an identity remap (and the ocean SST lands on the atm column
      shape).  Pass an explicit ``ocean_grid`` only for a genuinely different
      ocean grid.

    Each ``build_base_driver(cfg)`` constructs + ``setup()``-s the driver.  Raises
    on an unknown mode (dispatch hardening) or a ``"cmip"`` call missing
    ``coupled_preset`` — so a typo selects nothing silently.
    """
    from legoesm.training.run_to_column_mean import (
        amip_column_state,
        cmip_column_state,
    )

    if mode == "amip":
        from legoesm.driver.model_driver import ModelDriver

        def build_amip(cfg: Any) -> Any:
            driver = ModelDriver(cfg)
            driver.setup()
            return driver

        return build_amip, amip_column_state

    if mode == "cmip":
        if coupled_preset is None:
            raise ValueError(
                "make_base_driver_builder: mode='cmip' requires coupled_preset."
            )
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver

        def build_cmip(cfg: Any) -> Any:
            # ocean_grid=None ⇒ the coupled driver uses its own atm grid (same
            # object ⇒ identity remap; ocean SST on the atm column shape).
            driver = CoupledESMDriver(cfg, coupled_preset, ocean_grid=ocean_grid)
            driver.setup()
            return driver

        return build_cmip, cmip_column_state

    raise ValueError(
        f"make_base_driver_builder: unknown mode {mode!r}; choose 'amip' or 'cmip'."
    )


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="base ExperimentConfig JSON")
    p.add_argument("--mode", choices=("amip", "cmip"), default="amip",
                   help="AMIP (prescribed SST) or CMIP (coupled ocean)")
    p.add_argument("--coupled-preset", default="aquaplanet",
                   help="coupled_config preset name (CMIP mode)")
    p.add_argument("--era5-zarr", required=True, help="ERA5 zarr (reference)")
    p.add_argument("--era5-cache", default=None, help="ERA5 local cache dir")
    p.add_argument("--era5-time-idx", type=int, default=0)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--n-worst", type=int, default=20)
    p.add_argument("--les-budget", type=int, default=None,
                   help="cap LES to K env-cluster representatives (default: all)")
    p.add_argument("--les-dt", type=float, default=1.0, help="LES timestep [s]")
    p.add_argument("--les-hours", type=float, default=2.0, help="LES duration [h]")
    p.add_argument("--out", default="corrected_clubb_config.json")
    return p


def _area_weights(grid):  # pragma: no cover - HPC path
    """Per-column quadrature weights for the bias aggregation.

    Prefers the grid's true cell areas (``grid_area`` — incl. Gaussian quadrature
    weights — or ``area``); falls back to cos-latitude (a lat-lon proxy) with a
    warning if the grid exposes neither.
    """
    import warnings

    import jax.numpy as jnp
    import numpy as np

    for attr in ("grid_area", "area"):
        a = getattr(grid, attr, None)
        if a is not None:
            return jnp.asarray(a)
    warnings.warn(
        "run_correction_campaign: grid exposes no cell-area weights; using "
        "cos-latitude (a lat-lon proxy) for the bias aggregation.", stacklevel=2)
    return jnp.cos(jnp.deg2rad(jnp.asarray(np.asarray(grid.grid_lat))))


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - heavy I/O
    """CLI: load a base config + an ERA5 reference, run the campaign, write the
    corrected ``clubb_lite`` config + a per-round bias report.

    ``--mode amip`` (prescribed SST) or ``--mode cmip`` (coupled ocean on the same
    grid, ``--coupled-preset``) — the mode selects the driver + SST source via
    :func:`make_base_driver_builder`.  Heavy I/O (ERA5, the driver, the LES) is
    imported lazily here so the composition helpers above stay importable +
    unit-testable without it.  Reuses the ERA5→model-grid regrid of
    ``scripts/validate/compare_amip_era5.py``.

    Advanced :func:`build_correction_campaign` knobs (``env_scales`` for the
    clustering metric, ``initial_clubb`` to warm-start from a saved corrected
    config) are exposed only via that API, not this CLI; ``--les-budget`` is the
    one clustering knob surfaced here.
    """
    import json

    import numpy as np
    from legoesm.atmosphere.dynamics.column_les import ColumnLESConfig, run_forced_les
    from legoesm.driver.config import experiment_config_from_dict
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.training.compare_reanalysis import column_state_from_carry
    from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice

    from scripts.validate.compare_amip_era5 import (
        canonical_grid_type,
        select_era5_regrid,
    )

    args = _build_arg_parser().parse_args(argv)
    with open(args.config) as f:
        base_cfg = experiment_config_from_dict(json.load(f))

    # Grid + vertical coordinate the EXACT way the run builds them (so the ERA5
    # regrid + LES forcing extraction match the run's grid and vertical coord —
    # hybrid vs sigma — with no rebuilt-grid mismatch), but WITHOUT the heavy
    # setup() side effects (output dirs, dycore/physics build): the campaign is
    # single-rank offline orchestration, so the driver's own grid constructor
    # suffices.  ``_bootstrap_runtime`` is run first so the precision policy is
    # applied (else a non-default-precision sigma would get the wrong dtype).
    _probe = ModelDriver(base_cfg)
    _probe._bootstrap_runtime()
    _probe._create_grid()
    grid, sigma = _probe.grid, _probe.sigma

    # Run mode: AMIP (prescribed SST) or CMIP (coupled ocean on the SAME grid as
    # the atmosphere — ocean_grid=None lets the coupled driver use its own atm
    # grid, an identity remap, so the ocean SST lands on the atm column shape).
    coupled_preset = None
    if args.mode == "cmip":
        from legoesm.driver.coupled_config import PRESETS
        if args.coupled_preset not in PRESETS:
            raise SystemExit(
                f"unknown --coupled-preset {args.coupled_preset!r}; "
                f"choose from {sorted(PRESETS)}")
        coupled_preset = PRESETS[args.coupled_preset]()
    build_base_driver, extract_fn = make_base_driver_builder(
        args.mode, coupled_preset=coupled_preset, ocean_grid=None)

    # ERA5 reference regridded to the model grid + sigma (same regrid as the
    # one-shot compare driver).
    canon = canonical_grid_type(base_cfg.grid.grid_type)
    era5_slice = load_era5_slice(
        TrainingERA5Config(zarr_store=args.era5_zarr, local_cache_dir=args.era5_cache),
        args.era5_time_idx)
    reference = column_state_from_carry(select_era5_regrid(canon)(era5_slice, grid, sigma))

    n_steps = int(args.les_hours * 3600.0 / args.les_dt)
    result = build_correction_campaign(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, reference=reference, sigma=sigma,
        grid=grid, area_weights=_area_weights(grid), n_iterations=args.iterations,
        les_config=ColumnLESConfig(), n_worst=args.n_worst,
        les_budget=args.les_budget,
        run_les_fn=partial(run_forced_les, dt_s=args.les_dt, n_steps=n_steps),
    )
    biases = [(float(it.bias.baseline_bias), float(it.bias.updated_bias),
               bool(it.bias.improved)) for it in result.iterations]
    for i, (b0, b1, imp) in enumerate(biases):
        print(f"[campaign] round {i}: bias {b0:.5g} -> {b1:.5g} "
              f"({'IMPROVED' if imp else 'no improvement'})")
    with open(args.out, "w") as f:
        json.dump({"C_K": np.asarray(result.final_config.C_K).tolist(),
                   "biases": biases}, f, indent=2)
    print(f"[campaign] wrote corrected clubb config to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
