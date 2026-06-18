"""Perfect-model (identical-twin) OSSE driver for the LES correction loop.

A cheap, fully-controlled GO/NO-GO precursor to the real-ERA5 campaign
(``scripts/run/run_correction_campaign.py``): the "truth" is the model itself run
with a KNOWN turbulence coefficient, so we can check not just that the loop lowers
the bias but that it RECOVERS the known parameter (see
:mod:`legoesm.training.perfect_model_osse`).  If the loop cannot recover a known
parameter here, it will not help against real ERA5 — run this FIRST.

``build_perfect_model_osse`` wires the real AMIP/CMIP driver + LES exactly like the
campaign (reusing the shared helpers in ``run_correction_campaign``); ``main`` is
the CLI.  The heavy run/LES are injected so the wiring is unit-testable with mocks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from legoesm.training.run_to_column_mean import make_run_fn

from scripts.run.run_correction_campaign import (
    COEFFICIENT_SPEC_MAP,
    METHOD_PROMOTION,
    METHODS_NEED_LMIX,
    compose_compare_fn,
    grid_latlon_deg,
    make_clubb_build_driver,
    make_les_diagnose_fn,
    maybe_env_grid_fn,
)


def _production_loop_defaults(osse_kwargs: dict, sigma: Any) -> None:
    """Match the production campaign's loop defaults so the twin is not falsely
    optimistic: clip the diagnosed coefficient to its registered bounds (as
    ``build_correction_campaign`` does), and wire the env-generalization grid fn
    when ``feedback_strategy='environment'`` (else the loop raises on a None)."""
    osse_kwargs.setdefault("clip_to_bounds", True)
    strategy = osse_kwargs.get("feedback_strategy", "static")
    osse_kwargs.setdefault("env_grid_fn", maybe_env_grid_fn(strategy, sigma))


def build_perfect_model_osse(
    *,
    base_atm_config: Any,
    build_base_driver: Callable[[Any], Any],
    extract_column_state: Callable[..., Any],
    sigma: Any,
    grid: Any,
    area_weights: Any,
    true_clubb: Any,
    biased_clubb: Any,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
    n_worst: int,
    n_iterations: int,
    lat_deg: Any | None = None,
    lon_deg: Any | None = None,
    **osse_kwargs: Any,
):
    """Wire the real driver/LES and run a perfect-model OSSE.

    ``true_clubb`` carries the KNOWN true coefficient (the pseudo-truth is the
    model run with it); ``biased_clubb`` is the biased start the loop corrects.
    The diagnosed coefficient (``les_config.diagnosis_method``) selects the
    promotable field exactly as the campaign does.  Returns an
    :class:`~legoesm.training.perfect_model_osse.OSSEResult`.
    """
    from legoesm.training.perfect_model_osse import run_perfect_model_osse

    if getattr(les_config, "diagnosis_methods", None) is not None:
        # process_column would return a {method: diagnosis} dict the single-
        # coefficient OSSE cannot reduce (mirrors build_correction_campaign).
        raise ValueError(
            "build_perfect_model_osse is single-coefficient: set "
            "les_config.diagnosis_method, not diagnosis_methods.")
    lat_deg, lon_deg = grid_latlon_deg(grid, lat_deg, lon_deg)
    method = les_config.diagnosis_method
    if method not in METHOD_PROMOTION:
        raise ValueError(
            f"unknown diagnosis_method {method!r}; choose from "
            f"{tuple(METHOD_PROMOTION)}.")
    promotion_key, field_name = METHOD_PROMOTION[method]
    # C_K / c_eps evaluate the GCM mixing length → match it to the clubb config.
    if method in METHODS_NEED_LMIX and \
            getattr(les_config, "clubb_l_mix_max", None) is None:
        les_config = les_config._replace(clubb_l_mix_max=float(biased_clubb.l_mix_max))

    build_driver = make_clubb_build_driver(base_atm_config, build_base_driver)
    run_fn = make_run_fn(build_driver, extract_column_state)

    def build_compare_fn(reference):
        return compose_compare_fn(
            base_atm_config=base_atm_config, build_base_driver=build_base_driver,
            extract_column_state=extract_column_state, reference=reference,
            sigma=sigma, area_weights=area_weights, n_worst=n_worst,
            lat_deg=lat_deg, lon_deg=lon_deg)

    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn)
    grid_shape = tuple(int(s) for s in grid.grid_shape_2d)
    _production_loop_defaults(osse_kwargs, sigma)

    return run_perfect_model_osse(
        true_config=true_clubb, biased_config=biased_clubb,
        coefficient_field=field_name, run_fn=run_fn,
        build_compare_fn=build_compare_fn, diagnose_fn=diagnose_fn,
        promotion_key=promotion_key, grid_shape=grid_shape,
        diagnosis_method=method, n_iterations=n_iterations, **osse_kwargs)


def build_multi_perfect_model_osse(
    *,
    base_atm_config: Any,
    build_base_driver: Callable[[Any], Any],
    extract_column_state: Callable[..., Any],
    sigma: Any,
    grid: Any,
    area_weights: Any,
    true_clubb: Any,
    biased_clubb: Any,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
    n_worst: int,
    n_iterations: int,
    coefficients: tuple[str, ...] = ("C_K", "Pr_t", "C_eps"),
    sequential: bool = False,
    lat_deg: Any | None = None,
    lon_deg: Any | None = None,
    **osse_kwargs: Any,
):
    """Wire the real driver/LES and run a SIMULTANEOUS multi-coefficient OSSE.

    Mirrors ``build_multi_correction_campaign`` (one LES per column diagnosed for
    every coefficient's method) but in the perfect-model twin: recovers EVERY known
    coefficient in ``coefficients`` from a single pseudo-truth run.  Returns a
    :class:`~legoesm.training.perfect_model_osse.MultiOSSEResult`.
    """
    from legoesm.training.correction_loop import CorrectionSpec
    from legoesm.training.perfect_model_osse import run_multi_perfect_model_osse

    if not coefficients:
        raise ValueError("coefficients must be a non-empty tuple.")
    lat_deg, lon_deg = grid_latlon_deg(grid, lat_deg, lon_deg)

    specs = []
    for name in coefficients:
        if name not in COEFFICIENT_SPEC_MAP:
            raise ValueError(
                f"unknown coefficient {name!r}; choose from "
                f"{tuple(COEFFICIENT_SPEC_MAP)}.")
        key, method = COEFFICIENT_SPEC_MAP[name]
        # The biased start is the round-0 base for each coefficient (the twin's bias).
        specs.append(CorrectionSpec(key, method, float(getattr(biased_clubb, name))))

    # Diagnose every coefficient's method from ONE LES run (dedup, keep order).
    methods = tuple(dict.fromkeys(s.diagnosis_method for s in specs))
    les_config = les_config._replace(diagnosis_methods=methods)
    if {"clubb_coefficient", "c_eps"}.intersection(methods) and \
            getattr(les_config, "clubb_l_mix_max", None) is None:
        les_config = les_config._replace(clubb_l_mix_max=float(biased_clubb.l_mix_max))

    build_driver = make_clubb_build_driver(base_atm_config, build_base_driver)
    run_fn = make_run_fn(build_driver, extract_column_state)

    def build_compare_fn(reference):
        return compose_compare_fn(
            base_atm_config=base_atm_config, build_base_driver=build_base_driver,
            extract_column_state=extract_column_state, reference=reference,
            sigma=sigma, area_weights=area_weights, n_worst=n_worst,
            lat_deg=lat_deg, lon_deg=lon_deg)

    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn)
    grid_shape = tuple(int(s) for s in grid.grid_shape_2d)
    _production_loop_defaults(osse_kwargs, sigma)

    return run_multi_perfect_model_osse(
        true_config=true_clubb, biased_config=biased_clubb, specs=specs,
        run_fn=run_fn, build_compare_fn=build_compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=grid_shape, n_iterations=n_iterations, sequential=sequential,
        **osse_kwargs)


def _build_argparser():  # pragma: no cover - thin CLI plumbing
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="base experiment config (YAML/JSON)")
    p.add_argument("--mode", default="amip", choices=("amip", "cmip"))
    p.add_argument("--coupled-preset", default=None)
    p.add_argument("--true-ck", type=float, required=True,
                   help="the KNOWN true coefficient (the pseudo-truth uses it)")
    p.add_argument("--biased-ck", type=float, required=True,
                   help="the biased start the loop corrects")
    p.add_argument("--diagnosis-method", default="clubb_coefficient",
                   choices=tuple(METHOD_PROMOTION))
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--n-worst", type=int, default=8)
    p.add_argument("--les-hours", type=float, default=6.0)
    p.add_argument("--les-dt", type=float, default=0.5,
                   help="LES timestep [s]. Default 0.5 keeps the acoustic Courant < 1 "
                        "at the shallow-regime dx=50 m; a larger dt is rejected by "
                        "run_forced_les' acoustic-CFL pre-flight.")
    return p


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - heavy I/O
    """CLI: build the real driver, run the OSSE, print the recovery verdict."""
    from functools import partial

    from legoesm.atmosphere.dynamics.column_les import (
        ColumnLESConfig,
        run_forced_les,  # the only module that defines it (rce_diagnostics never did)
    )
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.perfect_model_osse import osse_verdict

    from scripts.run.run_correction_campaign import (
        _area_weights,
        load_base_config_and_grid,
        make_base_driver_builder,
    )

    args = _build_argparser().parse_args(argv)
    base_cfg, grid, sigma = load_base_config_and_grid(args.config)
    build_base_driver, extract_fn = make_base_driver_builder(
        args.mode, coupled_preset=args.coupled_preset)
    n_steps = int(args.les_hours * 3600.0 / args.les_dt)
    run_les = partial(run_forced_les, dt_s=args.les_dt, n_steps=n_steps)

    field = METHOD_PROMOTION[args.diagnosis_method][1]
    result = build_perfect_model_osse(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, sigma=sigma, grid=grid,
        area_weights=_area_weights(grid),
        true_clubb=CLUBBLiteConfig(**{field: args.true_ck}),
        biased_clubb=CLUBBLiteConfig(**{field: args.biased_ck}),
        les_config=ColumnLESConfig(diagnosis_method=args.diagnosis_method),
        run_les_fn=run_les, n_worst=args.n_worst, n_iterations=args.iterations)

    verdict = osse_verdict(result)
    print(f"[osse] true={result.true_value:.4g} biased={result.initial_value:.4g} "
          f"recovered={result.recovered_value:.4g}")
    print(f"[osse] bias {result.initial_bias:.5g} -> {result.final_bias:.5g} "
          f"(kept {result.n_accepted}/{result.n_rounds}); param error "
          f"{result.initial_param_error:.4g} -> {result.final_param_error:.4g}")
    print(f"[osse] {verdict.status.upper()}: {verdict.message}")
    return 0 if verdict.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
