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

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Run-as-a-script bootstrap (iter 321): a direct ``python scripts/validate/...`` invocation puts
# the script's OWN directory on ``sys.path``, not the repo root, so the module-level
# ``from scripts.run... import`` below would raise ``ModuleNotFoundError``. Add the repo root (it
# already worked via ``-m`` / pytest, which put the CWD on path). MUST precede the ``scripts.*``
# import, so the following imports trip E402 (module-import-not-at-top) — exempt in pyproject.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    print_realism_summary,
)


def _production_loop_defaults(osse_kwargs: dict, sigma: Any) -> None:
    """Match the production campaign's loop defaults so the twin is not falsely
    optimistic: clip the diagnosed coefficient to its registered bounds (as
    ``build_correction_campaign`` does), and wire the env-generalization grid fn
    when ``feedback_strategy='environment'`` (else the loop raises on a None)."""
    osse_kwargs.setdefault("clip_to_bounds", True)
    strategy = osse_kwargs.get("feedback_strategy", "static")
    osse_kwargs.setdefault("env_grid_fn", maybe_env_grid_fn(strategy, sigma))


def _build_osse_harness(
    *, base_atm_config: Any, build_base_driver: Callable[[Any], Any],
    extract_column_state: Callable[..., Any], sigma: Any, grid: Any, area_weights: Any,
    les_config: Any, run_les_fn: Callable[[Any], Any], n_worst: int,
    lat_deg: Any, lon_deg: Any, phis: Any,
):
    """The driver/LES harness SHARED by the single- + multi-coefficient OSSE builders
    (and reusable for a per-grid cross-resolution build): the corrected-config build
    driver, the run→time-mean ``run_fn``, the reference→``compare_fn`` closure, the LES
    ``diagnose_fn``, and the ``grid_shape``.  The single-vs-multi difference is ONLY the
    ``les_config`` (``diagnosis_method`` vs ``diagnosis_methods``) which the caller
    resolves BEFORE this — so the harness wiring lives in ONE place, not copy-pasted
    (CLAUDE.md: no duplicate harness wiring).  Returns
    ``(run_fn, build_compare_fn, diagnose_fn, grid_shape)``.
    """
    build_driver = make_clubb_build_driver(base_atm_config, build_base_driver)
    run_fn = make_run_fn(build_driver, extract_column_state)

    def build_compare_fn(reference):
        return compose_compare_fn(
            base_atm_config=base_atm_config, build_base_driver=build_base_driver,
            extract_column_state=extract_column_state, reference=reference,
            sigma=sigma, area_weights=area_weights, n_worst=n_worst,
            lat_deg=lat_deg, lon_deg=lon_deg)

    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn, phis=phis)
    grid_shape = tuple(int(s) for s in grid.grid_shape_2d)
    return run_fn, build_compare_fn, diagnose_fn, grid_shape


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
    phis: Any | None = None,
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

    run_fn, build_compare_fn, diagnose_fn, grid_shape = _build_osse_harness(
        base_atm_config=base_atm_config, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, sigma=sigma, grid=grid,
        area_weights=area_weights, les_config=les_config, run_les_fn=run_les_fn,
        n_worst=n_worst, lat_deg=lat_deg, lon_deg=lon_deg, phis=phis)
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
    phis: Any | None = None,
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

    run_fn, build_compare_fn, diagnose_fn, grid_shape = _build_osse_harness(
        base_atm_config=base_atm_config, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, sigma=sigma, grid=grid,
        area_weights=area_weights, les_config=les_config, run_les_fn=run_les_fn,
        n_worst=n_worst, lat_deg=lat_deg, lon_deg=lon_deg, phis=phis)
    _production_loop_defaults(osse_kwargs, sigma)

    return run_multi_perfect_model_osse(
        true_config=true_clubb, biased_config=biased_clubb, specs=specs,
        run_fn=run_fn, build_compare_fn=build_compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=grid_shape, n_iterations=n_iterations, sequential=sequential,
        **osse_kwargs)


def build_cross_resolution_osse(
    *,
    build_base_driver: Callable[[Any], Any],
    extract_column_state: Callable[..., Any],
    base_atm_config_coarse: Any,
    coarse_grid: Any,
    coarse_sigma: Any,
    area_weights_coarse: Any,
    base_atm_config_fine: Any,
    fine_grid: Any,
    fine_sigma: Any,
    area_weights_fine: Any,
    true_clubb: Any,
    biased_clubb: Any,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
    n_worst: int,
    n_iterations: int,
    phis_coarse: Any | None = None,
    phis_fine: Any | None = None,
    **osse_kwargs: Any,
):
    """Wire the real driver/LES on TWO grids and run a cross-resolution OSSE.

    Learns a grid-agnostic env→coefficient kernel on the cheap ``coarse`` grid
    (``feedback_strategy='environment'``) and deploys it on the expensive ``fine``
    grid, measuring the paired fine-grid bias change WITH vs WITHOUT the correction —
    the iter-69/70 cross-resolution env-kernel transfer, validated in a twin BEFORE a
    real high-res run pays for it.

    ``build_base_driver`` is grid-AGNOSTIC (the resolution lives in the config it is
    handed), so the SAME builder serves both grids; only the ``base_atm_config`` differs
    (coarse vs fine ``ExperimentConfig``).  Single-coefficient like
    :func:`build_perfect_model_osse` (the deployed kernel maps env → ONE field).
    Returns a :class:`~legoesm.training.perfect_model_osse.CrossResOSSEResult`.
    """
    from legoesm.training.perfect_model_osse import run_cross_resolution_osse

    if getattr(les_config, "diagnosis_methods", None) is not None:
        raise ValueError(
            "build_cross_resolution_osse is single-coefficient: set "
            "les_config.diagnosis_method, not diagnosis_methods.")
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

    lat_c, lon_c = grid_latlon_deg(coarse_grid, None, None)
    coarse_run, coarse_compare, coarse_diagnose, coarse_shape = _build_osse_harness(
        base_atm_config=base_atm_config_coarse, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, sigma=coarse_sigma, grid=coarse_grid,
        area_weights=area_weights_coarse, les_config=les_config, run_les_fn=run_les_fn,
        n_worst=n_worst, lat_deg=lat_c, lon_deg=lon_c, phis=phis_coarse)

    lat_f, lon_f = grid_latlon_deg(fine_grid, None, None)
    fine_run, fine_compare, _fine_diagnose, fine_shape = _build_osse_harness(
        base_atm_config=base_atm_config_fine, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, sigma=fine_sigma, grid=fine_grid,
        area_weights=area_weights_fine, les_config=les_config, run_les_fn=run_les_fn,
        n_worst=n_worst, lat_deg=lat_f, lon_deg=lon_f, phis=phis_fine)

    # The coarse campaign clips the diagnosed coefficient to its bounds (production
    # parity); the env_grid_fns are passed EXPLICITLY below, so do NOT route through
    # _production_loop_defaults (which would inject a duplicate env_grid_fn kwarg).
    osse_kwargs.setdefault("clip_to_bounds", True)

    return run_cross_resolution_osse(
        true_config=true_clubb, biased_config=biased_clubb,
        coefficient_field=field_name, promotion_key=promotion_key,
        diagnosis_method=method,
        coarse_grid_shape=coarse_shape, coarse_run_fn=coarse_run,
        coarse_build_compare_fn=coarse_compare, coarse_diagnose_fn=coarse_diagnose,
        coarse_env_grid_fn=maybe_env_grid_fn("environment", coarse_sigma),
        fine_grid_shape=fine_shape, fine_run_fn=fine_run,
        fine_build_compare_fn=fine_compare,
        fine_env_grid_fn=maybe_env_grid_fn("environment", fine_sigma),
        n_iterations=n_iterations, **osse_kwargs)


def _build_argparser():  # pragma: no cover - thin CLI plumbing
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="base experiment config (YAML/JSON)")
    p.add_argument("--mode", default="amip", choices=("amip", "cmip"))
    p.add_argument("--coupled-preset", default=None)
    p.add_argument("--true-ck", type=float, default=None,
                   help="the KNOWN true coefficient (the pseudo-truth uses it); REQUIRED for "
                        "the single-coefficient mode (omit when using --coefficients)")
    p.add_argument("--biased-ck", type=float, default=None,
                   help="the biased start the loop corrects (single-coefficient mode)")
    p.add_argument("--diagnosis-method", default="clubb_coefficient",
                   choices=tuple(METHOD_PROMOTION))
    p.add_argument("--coefficients", default=None,
                   help="comma-list (e.g. C_K,Pr_t,C_eps) for a SIMULTANEOUS multi-coefficient "
                        "OSSE — mirrors the campaign's --coefficients so a multi-coefficient run "
                        "is pre-flightable (iter 472). The TRUE values are the model defaults; "
                        "the biased start is each default x --multi-bias-factor.")
    p.add_argument("--multi-bias-factor", type=float, default=1.5,
                   help="(--coefficients) the biased start = each coefficient's true default x "
                        "this factor (default 1.5); the loop must recover the defaults.")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--fine-resolution", type=int, default=None,
                   help="if set, run a CROSS-RESOLUTION OSSE instead of same-grid: learn "
                        "the env→coefficient kernel on --config's (coarse) grid, then deploy "
                        "+ measure the paired bias change on a FINE grid built from the same "
                        "base config at this resolution. Validates the iter-69/70 "
                        "env-kernel transfer in a twin before an expensive high-res run.")
    p.add_argument("--n-worst", type=int, default=8)
    p.add_argument("--les-hours", type=float, default=6.0)
    p.add_argument("--les-dt", type=float, default=0.5,
                   help="LES timestep [s]. Default 0.5 keeps the acoustic Courant < 1 "
                        "at the shallow-regime dx=50 m; a larger dt is rejected by "
                        "run_forced_les' acoustic-CFL pre-flight.")
    p.add_argument("--orographic-forcing", choices=("auto", "on", "off"),
                   default="auto",
                   help="orographic geostrophic LES-forcing term over terrain — MUST "
                        "match the real campaign's setting so the OSSE go/no-go "
                        "faithfully predicts it. 'auto' (default): use the model's OWN "
                        "static topography if any (flat models stay flat). 'on': "
                        "REQUIRE terrain (error if flat). 'off': force flat.")
    p.add_argument("--surface-flux", action="store_true",
                   help="give each spin-off LES the SST-driven surface buoyancy flux (iter "
                        "364/465) — MUST match the real campaign's --surface-flux so the OSSE "
                        "go/no-go faithfully predicts the REALISTIC diagnosis path (it changes "
                        "HOW the closure is diagnosed, not just which columns). The OSSE twin's "
                        "runs carry SST, so the shared make_les_diagnose_fn threads it.")
    p.add_argument("--quick", action="store_true",
                   help="WIRING SMOKE ONLY (single-coefficient mode): replace the production "
                        "LES with the tiny shared fast_validation_les_regime so the WHOLE OSSE "
                        "harness (truth run -> biased run -> spin-off LES -> diagnose -> gate) "
                        "runs in MINUTES on a small config. The under-resolved LES gives a "
                        "GARBAGE coefficient, so the recovery/bias verdict is NOT meaningful — "
                        "the exit code reflects only whether the harness COMPOSED end-to-end. "
                        "Drop --quick (production LES) for the real go/no-go.")
    return p


def _reject_quick_with_multi_or_cross(args: Any) -> None:
    """``--quick`` (the fast single-coefficient WIRING smoke) is incompatible with the
    multi-coefficient (``--coefficients``) and cross-resolution (``--fine-resolution``)
    modes — fail LOUD rather than silently ignoring ``--quick`` (which would run the slow
    production LES the operator was trying to avoid).  Pure + called BEFORE the heavy
    ``_build_run_setup`` so the misuse is caught instantly and is unit-testable."""
    if getattr(args, "quick", False) and (
            args.coefficients is not None or args.fine_resolution is not None):
        raise SystemExit(
            "--quick (the fast single-coefficient WIRING smoke) is incompatible with "
            "--coefficients / --fine-resolution. Run the single-coefficient OSSE for the "
            "wiring smoke, or drop --quick for the production multi/cross-resolution go/no-go.")


def _resolve_fine_resolution(fine_resolution: int, coarse_resolution: int) -> int:
    """Validate ``--fine-resolution`` for a cross-resolution OSSE: a POSITIVE grid
    resolution DIFFERENT from ``--config``'s.

    Both failures are fail-loud (``SystemExit``), not a silent obscure crash / no-op:
    a non-positive resolution would crash deep in grid construction, and ``fine ==
    coarse`` is the SAME grid — the coarse-learned kernel would trivially 'transfer' to
    itself and report a falsely-reassuring ``transferred``.  For a same-grid go/no-go,
    run the plain OSSE WITHOUT ``--fine-resolution``.  Returns the validated int.
    """
    if fine_resolution <= 0:
        raise SystemExit(
            f"--fine-resolution must be a positive grid resolution (got {fine_resolution}).")
    if fine_resolution == coarse_resolution:
        raise SystemExit(
            f"--fine-resolution {fine_resolution} equals --config's resolution: that is the "
            f"SAME grid, so the coarse-learned kernel would trivially 'transfer' to itself "
            f"(a falsely-reassuring no-op). Run the same-grid OSSE WITHOUT --fine-resolution, "
            f"or choose a different resolution.")
    return fine_resolution


def _run_cross_resolution_main(
    args, *, base_cfg, coarse_grid, coarse_sigma, build_base_driver, extract_fn,
    run_les, phis_coarse,
):  # pragma: no cover - heavy I/O
    """--fine-resolution: build the FINE grid from the SAME base config (only the
    resolution differs) the identical way the run does, then run the cross-resolution
    OSSE — learn the kernel coarse, deploy + measure the paired bias change fine — and
    print the transfer verdict (exit 0 iff the kernel TRANSFERRED)."""
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.training.perfect_model_osse import cross_res_osse_verdict

    from scripts.run.run_correction_campaign import (
        _area_weights,
        _build_grid_for_config,
        resolve_orographic_phis,
    )

    field = METHOD_PROMOTION[args.diagnosis_method][1]
    fine_res = _resolve_fine_resolution(args.fine_resolution, base_cfg.grid.resolution)
    fine_cfg = base_cfg._replace(grid=base_cfg.grid._replace(resolution=fine_res))
    fine_grid, fine_sigma = _build_grid_for_config(fine_cfg)
    # The fine grid's OWN static topography (the fine run's forcing), resolved the
    # identical 'auto'/'on'/'off' way as the coarse phis.
    phis_fine = resolve_orographic_phis(
        args.orographic_forcing,
        lambda: ModelDriver(fine_cfg).static_topography_phis())

    result = build_cross_resolution_osse(
        build_base_driver=build_base_driver, extract_column_state=extract_fn,
        base_atm_config_coarse=base_cfg, coarse_grid=coarse_grid,
        coarse_sigma=coarse_sigma, area_weights_coarse=_area_weights(coarse_grid),
        base_atm_config_fine=fine_cfg, fine_grid=fine_grid, fine_sigma=fine_sigma,
        area_weights_fine=_area_weights(fine_grid),
        true_clubb=CLUBBLiteConfig(**{field: args.true_ck}),
        biased_clubb=CLUBBLiteConfig(**{field: args.biased_ck}),
        les_config=ColumnLESConfig(diagnosis_method=args.diagnosis_method,
                                   surface_flux=args.surface_flux),
        run_les_fn=run_les, n_worst=args.n_worst, n_iterations=args.iterations,
        phis_coarse=phis_coarse, phis_fine=phis_fine)

    verdict = cross_res_osse_verdict(result)
    print(f"[xres] coarse bias {result.coarse_initial_bias:.5g} -> "
          f"{result.coarse_final_bias:.5g} (reduced={result.coarse_bias_reduced})")
    print(f"[xres] fine bias {result.fine_bias_uncorrected:.5g} -> "
          f"{result.fine_bias_corrected:.5g} (reduction "
          f"{result.fine_bias_reduction:.5g})")
    print(f"[xres] coverage in_hull={result.fraction_in_hull:.3f} "
          f"covered={result.fraction_covered:.3f} "
          f"thresh={result.coverage_threshold:.3f} (n_fine={result.n_fine_columns})")
    print(f"[xres] {verdict.status.upper()}: {verdict.message}")
    print_realism_summary(run_les, prefix="[xres]")
    return 0 if verdict.ok else 1


def _run_multi_osse_main(args, *, base_cfg, grid, sigma, build_base_driver, extract_fn,
                         run_les, phis):
    """SIMULTANEOUS multi-coefficient OSSE (``--coefficients``, iter 472): recover the model's
    DEFAULT coefficients from a ``--multi-bias-factor``-perturbed start, mirroring the campaign's
    ``--coefficients`` so a multi-coefficient run is pre-flightable. Exit-code-gated on the
    verdict (0 = all recovered + bias fell), like the single mode."""
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.perfect_model_osse import multi_osse_verdict

    from scripts.run.run_correction_campaign import _area_weights

    coefficients = tuple(c.strip() for c in args.coefficients.split(",") if c.strip())
    if not coefficients:
        raise SystemExit("--coefficients is empty; pass e.g. --coefficients C_K,Pr_t,C_eps.")
    f = float(args.multi_bias_factor)
    if f <= 0.0:
        raise SystemExit(f"--multi-bias-factor must be > 0, got {f}.")
    true_clubb = CLUBBLiteConfig()                    # the model defaults are the pseudo-truth
    biased_clubb = true_clubb._replace(
        **{c: float(getattr(true_clubb, c)) * f for c in coefficients})

    result = build_multi_perfect_model_osse(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, sigma=sigma, grid=grid,
        area_weights=_area_weights(grid),
        true_clubb=true_clubb, biased_clubb=biased_clubb, coefficients=coefficients,
        les_config=ColumnLESConfig(surface_flux=args.surface_flux),
        run_les_fn=run_les, n_worst=args.n_worst, n_iterations=args.iterations, phis=phis)

    verdict = multi_osse_verdict(result)
    n = len(result.per_coefficient)
    n_rec = sum(1 for c in result.per_coefficient.values()
                if c.final_param_error < c.initial_param_error)
    print(f"[osse-multi] bias {result.initial_bias:.5g} -> {result.final_bias:.5g} "
          f"(kept {result.n_accepted}/{result.n_rounds}); {n_rec}/{n} coefficients recovered")
    for c in result.per_coefficient.values():
        moved = "yes" if c.final_param_error < c.initial_param_error else "NO"
        print(f"[osse-multi]   {c.field}: true={c.true_value:.4g} biased={c.initial_value:.4g} "
              f"recovered={c.recovered_value:.4g} (toward truth: {moved})")
    print(f"[osse-multi] {verdict.status.upper()}: {verdict.message}")
    print_realism_summary(run_les, prefix="[osse-multi]")
    return 0 if verdict.ok else 1


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - heavy I/O
    """CLI: build the real driver, run the OSSE, print the recovery verdict."""
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.perfect_model_osse import osse_verdict

    from scripts.run.run_correction_campaign import _area_weights, _build_run_setup

    args = _build_argparser().parse_args(argv)
    _reject_quick_with_multi_or_cross(args)   # fail fast before the heavy setup (iter 504)
    # Shared run-setup preamble (iter 295): config/grid/sigma + driver builder (with the
    # RESOLVED coupled preset — the OSSE previously passed the RAW --coupled-preset name to
    # make_base_driver_builder, the iter-240 'str' has no attribute ocean_mode crash in
    # CMIP mode) + the CFL-checked LES runner + the orographic phis (identical to the real
    # campaign's, so this go/no-go predicts the real run — iter 126/127).
    base_cfg, grid, sigma, build_base_driver, extract_fn, run_les, phis = _build_run_setup(
        args)

    if args.coefficients is not None:
        if args.fine_resolution is not None:
            raise SystemExit(
                "--coefficients (simultaneous multi-coefficient) + --fine-resolution "
                "(cross-resolution) is unsupported: the cross-resolution OSSE is "
                "single-coefficient. Run them separately.")
        return _run_multi_osse_main(
            args, base_cfg=base_cfg, grid=grid, sigma=sigma,
            build_base_driver=build_base_driver, extract_fn=extract_fn,
            run_les=run_les, phis=phis)

    if args.fine_resolution is not None:
        return _run_cross_resolution_main(
            args, base_cfg=base_cfg, coarse_grid=grid, coarse_sigma=sigma,
            build_base_driver=build_base_driver, extract_fn=extract_fn,
            run_les=run_les, phis_coarse=phis)

    if args.true_ck is None or args.biased_ck is None:
        raise SystemExit(
            "the single-coefficient OSSE requires --true-ck and --biased-ck (or pass "
            "--coefficients for the simultaneous multi-coefficient mode).")
    field = METHOD_PROMOTION[args.diagnosis_method][1]
    # --quick swaps in the tiny shared validation regime (wiring smoke); production otherwise.
    les_kwargs = dict(diagnosis_method=args.diagnosis_method, surface_flux=args.surface_flux)
    if args.quick:
        from scripts.run.run_correction_campaign import fast_validation_les_regime
        les_kwargs["regime"] = fast_validation_les_regime()
    result = build_perfect_model_osse(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, sigma=sigma, grid=grid,
        area_weights=_area_weights(grid),
        true_clubb=CLUBBLiteConfig(**{field: args.true_ck}),
        biased_clubb=CLUBBLiteConfig(**{field: args.biased_ck}),
        les_config=ColumnLESConfig(**les_kwargs),
        run_les_fn=run_les, n_worst=args.n_worst, n_iterations=args.iterations,
        phis=phis)

    verdict = osse_verdict(result)
    print(f"[osse] true={result.true_value:.4g} biased={result.initial_value:.4g} "
          f"recovered={result.recovered_value:.4g}")
    print(f"[osse] bias {result.initial_bias:.5g} -> {result.final_bias:.5g} "
          f"(kept {result.n_accepted}/{result.n_rounds}); param error "
          f"{result.initial_param_error:.4g} -> {result.final_param_error:.4g}")
    print(f"[osse] {verdict.status.upper()}: {verdict.message}")
    print_realism_summary(run_les, prefix="[osse]")   # WHY any spin-off LES was rejected
    if args.quick:
        import math
        ran = math.isfinite(result.initial_bias) and math.isfinite(result.final_bias)
        print("[osse] *** --quick WIRING SMOKE ***: the tiny validation LES is DELIBERATELY "
              "under-resolved, so the recovery/bias verdict above is NOT meaningful. This "
              "exit code reflects only whether the OSSE harness COMPOSED end-to-end; run "
              "WITHOUT --quick (production LES) for the real go/no-go.")
        return 0 if ran else 1
    return 0 if verdict.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
