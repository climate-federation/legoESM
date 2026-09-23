"""Mode-specific model / ERA5 / loss builders for WeatherBench scale training.

Thin adapter that reuses the INSTALLED lat-lon training stack (the reference
``run_aimip_latlon.py`` is WIP against a newer API, so this binds to the real
package signatures) and exposes exactly what
``scripts/run/train_weatherbench_scale.py`` needs for the data-parallel loop:

    build_mode_components(cfg, yml) -> (model, grid, sigma, params, make_run_seg, loss_config, dt)
    load_era5_samples(cfg, yml, grid, sigma) -> list[(ic, target, forcing)]
    evaluate_wb2(cfg, yml, model, params, make_run_seg, grid, sigma)

The three modes differ ONLY in the ``make_run_seg`` factory (per
``training_driver._build_train_step``): physics feeds
``TrainablePhysicsParams.to_segment_kwargs()``; neural_gcm builds a
``NeuralPhysics`` step; sfno builds an SFNO lat-lon step. On the SPECTRAL
core, ``physics`` is the six-family classical model shared with AIMIP
(``make_aimip_classical_spectral_physics``), not the old 6-knob
``TrainablePhysicsParams`` stack — that one survives only on the lat-lon path
below and in ``train_physics_params_spectral``.

End-to-end validation is the single-GPU smoke (``--mode physics --smoke``) +
the cluster launch; the data-parallel gradient average is unit-tested separately.
"""
from __future__ import annotations

import calendar
import importlib.util
import math
from pathlib import Path

import numpy as np

import logging

logger = logging.getLogger(__name__)
# One-shot process flag: the hold-fixed warning fires once, not per shard/epoch.
_WARNED_ERA5_FLUX_HOLD_FIXED = False

_REPO = Path(__file__).resolve().parents[4]   # packages/ml/legoesm/training/ -> repo root


def _resolve_n_days(cfg, yml):
    """Training-window length [days] per train year (#1047 ask a).

    Precedence: explicit ``cfg.n_days`` (CLI ``--n-days``) > YAML
    ``n_training_days`` > 3 (the historical hardcoded default). ``--smoke``
    always forces 1 (single-window wiring check). This replaced a hardcoded
    ``n_days = 3`` in both ERA5 loaders that capped every WeatherBench arm at
    ~60 samples/year and left the column-MLP data-starved (#1047 finding 2).
    """
    if getattr(cfg, "smoke", False):
        return 1
    n = getattr(cfg, "n_days", None)
    if n is None:
        n = yml.get("n_training_days", 3)
    n = int(n)
    if n < 1:
        raise ValueError(f"n_days (training window) must be >= 1, got {n}")
    return n


def _resolve_windows(cfg, yml):
    """Scattered training windows as ``[(year, day_offset, n_days), ...]`` or None.

    Precedence: ``cfg.windows`` > YAML ``train_windows`` > None. ``None`` selects
    the consecutive back-compat mode (the first ``n_days`` of each ``train_years``
    entry). A window list instead scatters MANY SHORT windows across the calendar
    -- the AIMIP scale config's ``(year, day_offset, n_days)`` house pattern --
    so a WB arm sees independent synoptic scenes in every season rather than one
    autocorrelated January block (#1047 finding 2 / design note 1). ``--smoke``
    is honoured downstream (one window), so it is not special-cased here.
    """
    raw = getattr(cfg, "windows", None)
    if raw is None:
        raw = yml.get("train_windows")
    if not raw:
        return None
    out = []
    for w in raw:
        if len(w) != 3:
            raise ValueError(
                f"train window must be (year, day_offset, n_days), got {w!r}")
        year, off, nd = int(w[0]), int(w[1]), int(w[2])
        if off < 0 or nd < 1:
            raise ValueError(
                f"train window needs day_offset>=0 and n_days>=1, got {w!r}")
        # The IC days must stay within `year`: otherwise the window silently pulls
        # samples from the NEXT year while still labelling them `year`, and the
        # forcing calendar (DOY relative to year-01-01) is then wrong / leap-
        # phase-shifted. Targets may cross New Year (i_ic + stride is only a few
        # hours ahead); it is the IC span that is validated.
        days_in_year = 366 if calendar.isleap(year) else 365
        if off + nd > days_in_year:
            raise ValueError(
                f"train window ICs must stay within {year}: "
                f"day_offset+n_days={off + nd} > {days_in_year} days, got {w!r}")
        out.append((year, off, nd))
    return out


def _training_sample_indices(cfg, yml, times, snaps_per_day, stride):
    """Yield ``(year, i_ic, i_tg)`` ERA5 index pairs for the training samples.

    Shared by both the lat-lon and spectral loaders so the window logic lives in
    ONE place. Scattered mode (``_resolve_windows`` -> a list) walks each
    ``(year, day_offset, n_days)`` window; consecutive mode (default) walks the
    first ``n_days`` days of each ``train_years`` entry. A pair whose target
    index runs past the ERA5 record is skipped; ``cfg.smoke`` stops after the
    first window (single-window wiring check, matching the prior behaviour).
    """
    windows = _resolve_windows(cfg, yml)
    if windows is None:
        n_days = _resolve_n_days(cfg, yml)
        windows = [(int(y), 0, n_days) for y in yml["train_years"]]
    for (year, day_offset, n_days) in windows:
        base = int(np.searchsorted(times, np.datetime64(f"{year}-01-01")))
        start = base + int(day_offset) * snaps_per_day
        for d in range(int(n_days) * snaps_per_day):
            i_ic, i_tg = start + d, start + d + stride
            if i_tg >= len(times):
                break
            yield year, i_ic, i_tg
        if getattr(cfg, "smoke", False):
            break


def _sharded_indices(cfg, yml, times, snaps_per_day, stride, rank, nproc):
    """This rank's contiguous shard of the ``(year, i_ic, i_tg)`` index list.

    #1286 fix B: the ERA5 index tuples are cheap (no GPU arrays), so we
    materialize the full ORDERED list here and hand each rank ONLY its slice —
    the loaders then build carries for that slice alone, never the global set.
    The partition is BIT-IDENTICAL to the previous
    ``shard_samples(build_all(), rank, nproc)`` (data_parallel.shard_samples,
    drop_remainder): contiguous ``per = n // nproc``, rank ``p`` gets
    ``idx[p*per:(p+1)*per]``.  Single process -> the full list.
    """
    idx = list(_training_sample_indices(cfg, yml, times, snaps_per_day, stride))
    if nproc <= 1:
        return idx
    per = len(idx) // nproc          # drop_remainder: balanced shards
    start = rank * per
    return idx[start:start + per]


def _sample_to_host(sample):
    """Pull a built ``(ic, target, forcing)`` off-device to host numpy leaves.

    #1286 fix A: the per-run ``samples`` list is kept host-resident so the whole
    training set is NOT parked in device memory; the training loop
    ``device_put``s one sample at a time.  ``jax.device_get`` converts every
    device-array leaf to numpy and passes non-array leaves through unchanged;
    the build's transient device allocation is freed once this returns.
    """
    import jax
    return jax.device_get(sample)


def _load_run_amip():
    """Exec scripts/run/run_amip.py as a module to reuse its arg parser + config builder."""
    path = _REPO / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_for_scale", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _validated_wb_radiation(cfg, yml) -> str:
    """The WB campaign's radiation backend, pinned to rrtmgp.

    Shared by both training cores so neither can drift; ``--smoke`` keeps the
    cheap gray path (see campaign_driver.validate_campaign_radiation).
    """
    from legoesm.training.campaign_driver import validate_campaign_radiation

    return validate_campaign_radiation(
        str(yml.get("radiation", "rrtmgp")), campaign="wb",
        smoke=bool(getattr(cfg, "smoke", False)),
    )


def build_latlon_config(cfg, yml):
    """ExperimentConfig for a lat-lon C-grid PE run, via run_amip's parser."""
    ra = _load_run_amip()
    # Pure-dycore WB modes (neural_gcm / sfno) replace the physics pipeline with
    # a neural net -- their rollout runs NO boundary-layer scheme, so the
    # turbulence field feeds ONLY _create_friction (fric_decay), nothing else
    # (dt/grid/sigma/dycore are turbulence-independent; physics_pipeline is
    # unused for these modes).  Declare turbulence="none" for them so the #931
    # double-count gate KEEPS the Held-Suarez Rayleigh drag ON -- that
    # fric_decay is their SOLE #797 adjoint dissipation of the otherwise
    # undamped dycore (the epoch-0 zero-init rollout is the bare dycore).  A
    # "louis" label would gate it to a no-op and reintroduce the NaN-gradient
    # blow-up (#797 bug 11).  Verified byte-identical to the pre-#931 always-on
    # drag: turbulence="none" reproduces the exact HS Rayleigh fric_decay while
    # leaving dt/grid/sigma unchanged.  Physics mode genuinely runs louis and
    # keeps it (louis owns BOTH the surface stress and the adjoint damping).
    _mode = getattr(cfg, "mode", None)
    _turbulence = ("none" if _mode in ("neural_gcm", "sfno")
                   else yml.get("turbulence", "louis"))
    argv = [
        "--grid-type", "latlon",
        "--discretization", "latlon_cgrid",
        "--resolution", str(int(yml["n_lat"])),
        "--nlev", str(int(yml["nlev"])),
        "--vertical-coord", "sigma",
        "--dt", str(float(yml["dt"])),
        # WB is a campaign: rrtmgp, unless this is a --smoke wiring check. The
        # lat-lon path forwarded the raw key, so a production WB config could
        # still run gray while the spectral path was pinned (codex).
        "--radiation", _validated_wb_radiation(cfg, yml),
        "--convection", str(yml.get("convection", "sbm")),
        "--turbulence", str(_turbulence),
        "--microphysics", str(yml.get("microphysics", "kessler")),
        "--gravity-wave-drag", str(yml.get("gravity_wave_drag", "hines")),
        # run_amip's --rad-update-steps defaults to None (its main() auto-sets
        # None -> floor(3600/dt) inside _postprocess_args, which we deliberately
        # skip here — it enforces AMIP forcing-path args the WB/ERA5 path lacks).
        # Pass an explicit value so cfg.rad_update_steps is a valid int (the
        # driver's _prepare_run_context reads it bare): radiation every step (1)
        # unless the YAML overrides.
        "--rad-update-steps", str(int(yml.get("rad_update_steps", 1))),
    ]
    parsed = ra.build_arg_parser().parse_args(argv)
    return ra.build_config_from_args(parsed)


# --- campaign-YAML key validation (2026-08-24) ------------------------------
# Every top-level key this training path actually reads.  A key outside this
# set is a HARD ERROR: the lr/optimizer keys sat inert in the decks for a
# whole campaign (v2 trained at 3e-4 while its deck said 1.5e-3), and a typo
# like ``learning_rate`` would silently reintroduce exactly that.
_WB_CONSUMED_TOP_KEYS = frozenset({
    "classical", "convection", "dt", "era5_cadence_hours",
    "era5_cloud_condensate", "era5_cloud_zarr", "era5_flux_zarr",
    "era5_surface_fluxes", "era5_zarr", "eval_years",
    "grad_accum", "grad_clip_norm", "gravity_wave_drag", "loss", "lr",
    "microphysics", "n_epochs", "n_lat", "n_lon", "n_training_days",
    "neural_gcm", "nlev", "optimizer", "rad_update_steps", "radiation",
    "sfno", "smoke_radiation", "spectral", "train_windows", "train_years",
    "turbulence", "warmup_steps", "weight_decay",
})
# Deck metadata this trainer never reads but the decks legitimately carry
# (documentation / other tooling).  Kept OUT of the consumed set so the
# distinction stays visible; anything here is tolerated, not honoured.
_WB_METADATA_TOP_KEYS = frozenset({"grid", "cache_dir"})
# Exactly the keys _spectral_pe_config + the dt lookup read from the
# ``spectral`` block.  ``si_substep`` (typo) silently reverting a deck to the
# unstable full step is the same defect class as the inert lr.
_WB_SPECTRAL_KEYS = frozenset({
    "n_max", "dt", "hyperdiff_coeff", "hyperdiff_order", "time_integrator",
    "semi_implicit", "si_T_ref", "si_alpha", "si_substeps", "sponge_sigma",
    "sponge_tau", "spectral_filter_order", "spectral_filter_strength",
    "fix_mass", "anchor_mass_to_initial", "vertical_advection_scheme",
    "frictional_heating",
})
# Exactly the keys the classical / neural_gcm / sfno builders read from their
# nested blocks (#1663 review: these were unvalidated, so a nested typo like
# ``nn_hiden`` silently fell back to the default -- the inert-key defect class).
_WB_CLASSICAL_KEYS = frozenset({
    "cloud", "clubb_top_press_hpa", "convection", "gwd", "microphysics",
    "convective_rain_to_surface", "orbital_insolation",
    "param_fixed", "param_init",
    "rad_update_interval_steps", "rrtmgp_column_chunk_size",
    "rrtmgp_gpoint_batch_size", "spatial_init_std",
    "spatial_seed", "spatial_surface", "surface_bulk", "trainable_schemes",
    "turbulence",
})
# neural_gcm and sfno share one arch/drag key surface (both neural cores); kept
# permissive across the two so a valid arch key is never falsely rejected.
_WB_NEURAL_KEYS = frozenset({
    "surface_drag", "surface_drag_scheme", "surface_drag_confounded",
    "nn_hidden", "nn_layers", "gauss_n_max", "sfno_embed_dim", "sfno_n_blocks",
    "sfno_mlp_expansion", "spatial_embedding",
})


def _pinned_trainable_names(param_fixed: dict) -> set:
    """Deck-pinned fields that are ALSO registered trainable parameters.

    Most pins are structural (an overlap choice, a sub-column count, a switch)
    and have no leaf to exclude. The few that overlap must be excluded from the
    trainable bundle: the pin is spliced after the trained value, so such a
    leaf would be optimized, never reach the model, and still show up as
    "trained" in the run's parameter report.
    """
    from legoesm.training.param_collector import build_registry
    known = {m.qualified_name for m in build_registry()}
    return {f"{key}.{field}" for key, fields in param_fixed.items()
            for field in fields} & known


def wb_needs_land_frac(mode: str, yml: dict) -> bool:
    """Whether the WB sample loader must carry an ERA5 land fraction.

    True for the classical ``physics`` arm (convection's land branch),
    when the top-level ``era5_surface_fluxes`` flag is on (the flux planes
    ship with the mask) or when the selected learned arm's block enables
    ``spatial_embedding`` (the land fraction is one of its static inputs).
    """
    if mode == "physics" or bool(yml.get("era5_surface_fluxes", False)):
        return True
    if mode in ("neural_gcm", "sfno"):
        ov = yml.get(mode, {}) or {}
        return bool(ov.get("spatial_embedding", False))
    return False


def validate_wb_campaign_yaml(yml: dict) -> None:
    """Refuse a campaign YAML carrying keys this training path does not read.

    Unknown keys raise with a nearest-match hint.  Covers the top level and
    the ``spectral``, ``loss``, ``classical``, ``neural_gcm`` and ``sfno``
    blocks (#1663: the last three were unchecked, so a nested typo silently
    defaulted).
    """
    import difflib

    # Common synonyms difflib's ratio cannot reach (``learning_rate`` vs
    # ``lr`` share two characters).
    aliases = {"learning_rate": "lr", "epochs": "n_epochs",
               "weight-decay": "weight_decay", "grad_clip": "grad_clip_norm"}

    def _reject(unknown, known, where):
        # Suggestions draw ONLY from consumed keys — pointing a typo at a
        # tolerated-metadata key would "fix" it into a knob that does nothing
        # (GLM review). Advisory only: nothing is auto-translated.
        hints = []
        lower_map = {kk.lower(): kk for kk in known}
        for k in sorted(unknown):
            if k in aliases and aliases[k] in known:
                close = [aliases[k]]
            else:
                # match case-insensitively so ``w_t`` suggests ``w_T``
                close = [lower_map[m] for m in difflib.get_close_matches(
                    k.lower(), list(lower_map), n=1)]
            hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)"
                                     if close else ""))
        raise SystemExit(
            f"unknown {where} key(s) in campaign YAML: {', '.join(hints)}. "
            "Keys this trainer does not read are a hard error: inert deck "
            "keys are how the wb_classical_v2 campaign trained at the wrong "
            "learning rate.")

    top_known = _WB_CONSUMED_TOP_KEYS | _WB_METADATA_TOP_KEYS
    unknown = set(yml) - top_known
    if unknown:
        _reject(unknown, sorted(_WB_CONSUMED_TOP_KEYS), "top-level")
    spec = yml.get("spectral") or {}
    unknown = set(spec) - _WB_SPECTRAL_KEYS
    if unknown:
        _reject(unknown, sorted(_WB_SPECTRAL_KEYS), "spectral-block")
    # Loss block: make_loss_config silently DROPS keys outside
    # LossConfig._fields ("if k in valid"), so a ``w_t`` typo would fall back
    # to the default weight — the same inert-key failure mode (codex + GLM).
    loss = yml.get("loss") or {}
    if loss:
        from legoesm.training.losses import LossConfig

        unknown = set(loss) - set(LossConfig._fields)
        if unknown:
            _reject(unknown, sorted(LossConfig._fields), "loss-block")
    # Nested per-scheme blocks (#1663): typo'd keys here silently defaulted.
    for _blk, _known in (("classical", _WB_CLASSICAL_KEYS),
                         ("neural_gcm", _WB_NEURAL_KEYS),
                         ("sfno", _WB_NEURAL_KEYS)):
        _b = yml.get(_blk) or {}
        _u = set(_b) - _known
        if _u:
            _reject(_u, sorted(_known), f"{_blk}-block")


def make_loss_config(cfg, yml):
    from legoesm.training.losses import LossConfig
    lb = dict(yml.get("loss", {}))
    valid = set(LossConfig()._fields)
    kwargs = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in lb.items() if k in valid}
    return LossConfig(**kwargs)


def _spectral_pe_config(yml):
    """SpectralPEConfig for the WB spectral training core, from ``yml["spectral"]``.

    Defaults mirror the AIMIP spectral trainer's proven values
    (``NeuralGCMSpectralConfig``: hyperdiff 2.5e15 order-2, exponential filter
    0.01/8) with ONE deliberate difference: ``semi_implicit`` defaults **True**
    — the whole point of the spectral core is the Hoskins–Simmons SI step whose
    implicit gravity-wave treatment keeps the training ADJOINT bounded (#817
    blocker 1; the explicit lat-lon core's adjoint grows ~x1.3/step).  Set
    ``spectral: {semi_implicit: false}`` to opt back into the explicit
    integrator (then use an explicit-CFL-safe ``spectral.dt``).

    Conservation knobs are forwarded so the WB lane can run the SAME
    conserved-quantity constraints as the AMIP/AIMIP lane
    (``run_aimip._build_spectral_config`` forwards the identical four keys):
    the dry-mass anchor (``fix_mass`` + ``anchor_mass_to_initial``, both
    honoured by ``spectral_rollout``, which recomputes the target mass from
    each rollout's own initial state) and the energy numerics
    (``vertical_advection_scheme``, ``frictional_heating``).  Defaults are
    ``SpectralPEConfig``'s own, so every existing WB run is byte-identical:
    the anchor stays OFF unless a YAML asks for it, and the energy-conserving
    Simmons-Burridge transport + frictional heating stay ON (the code default
    since the -0.48 K/day upwind T leak was measured).
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

    _PE_DEFAULTS = SpectralPEConfig()
    spec = dict(yml.get("spectral", {}))
    return SpectralPEConfig(
        hyperdiff_coeff=float(spec.get("hyperdiff_coeff", 2.5e15)),
        hyperdiff_order=int(spec.get("hyperdiff_order", 2)),
        time_integrator=str(spec.get("time_integrator", "ssp_rk3")),
        semi_implicit=bool(spec.get("semi_implicit", True)),
        si_T_ref=float(spec.get("si_T_ref", 300.0)),
        si_alpha=float(spec.get("si_alpha", 0.5)),
        si_substeps=int(spec.get("si_substeps", 1)),
        sponge_sigma=float(spec.get("sponge_sigma", 0.1)),
        sponge_tau=float(spec.get("sponge_tau", 0.0)),
        spectral_filter_order=int(spec.get("spectral_filter_order", 8)),
        spectral_filter_strength=float(spec.get("spectral_filter_strength", 0.01)),
        fix_mass=bool(spec.get("fix_mass", _PE_DEFAULTS.fix_mass)),
        anchor_mass_to_initial=bool(spec.get(
            "anchor_mass_to_initial", _PE_DEFAULTS.anchor_mass_to_initial)),
        vertical_advection_scheme=str(spec.get(
            "vertical_advection_scheme",
            _PE_DEFAULTS.vertical_advection_scheme)),
        frictional_heating=bool(spec.get(
            "frictional_heating", _PE_DEFAULTS.frictional_heating)),
    )


def _build_mode_components_spectral(cfg, yml):
    """Spectral (Gaussian + semi-implicit) training-core components (#817).

    Same 7-tuple contract as the lat-lon path, with the rollout routed through
    the differentiable spectral PE core (``spectral_rollout`` +
    ``_make_spectral_integrator``, #829): the SI step treats the fast
    gravity-wave terms implicitly, so the training adjoint stays bounded where
    the explicit lat-lon core's grows ~x1.3/step (#817 blocker 1) — and the
    Gaussian grid has no pole-cell CFL clamp, so ``dt`` stays at the configured
    value (1800 s default under SI) instead of collapsing to seconds, which is
    what turned a 6 h lead into thousands of BPTT steps (blocker 2's
    amplifier).

    The returned ``make_run_seg(trainable).raw(ic, n_steps, forcing)`` maps a
    Gaussian-grid SegmentCarry through carry->spectral -> SI rollout ->
    spectral->carry, so the trainer's ``combined_loss(pred, target)`` and the
    data-parallel loop are unchanged.  ``model`` is None (the functional
    spectral rollout has no model object; only the WB2 pointer hook received
    it).
    """
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.model_registry import build_variant
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        compute_sponge_factor, compute_spectral_filter,
    )
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state,
        spectral_state_to_carry,
        spectral_rollout,
        make_sfno_spectral_physics,
        make_column_mlp_spectral_physics,
        make_physics_params_spectral_physics,
    )

    spec = dict(yml.get("spectral", {}))
    n_max = int(spec.get("n_max", max(21, int(yml["n_lat"]) // 3)))
    nlev = int(yml["nlev"])
    grid = create_gaussian_grid(n_max)
    sigma = create_sigma_coordinate(nlev)
    pe_config = _spectral_pe_config(yml)
    # No pole-cell clamp on the Gaussian grid.  The SI step is NOT stable at
    # arbitrary dt: only the linear gravity-wave subsystem is implicit, and
    # the explicit RK3 advection has a spectral CFL bound
    # (u * n_max * dt_sub / a < sqrt(3), dt_sub = dt / si_substeps).  Measured
    # 2026-08-23 at T63: dt=1800 with si_substeps=1 blows up exponentially
    # within 12 steps (any level count); dt<=900, or dt=1800 with
    # si_substeps>=3, is stable.  Campaign YAMLs must satisfy the bound —
    # gated by tests/unit/test_wb_campaign_dt_stability.py.
    dt = float(spec.get("dt", 1800.0 if pe_config.semi_implicit else 600.0))

    sponge_factor = None
    if pe_config.sponge_tau > 0:
        sponge_factor = compute_sponge_factor(
            sigma.sigma_full, pe_config.sponge_sigma, pe_config.sponge_tau, dt)
    spectral_filter = None
    if pe_config.spectral_filter_strength > 0:
        spectral_filter = compute_spectral_filter(
            grid.ls, grid.n_max,
            order=pe_config.spectral_filter_order,
            cutoff_fraction=pe_config.spectral_filter_strength)

    loss_config = make_loss_config(cfg, yml)

    # Mode -> (params pytree, physics_fn factory, does the fn take forcing?).
    # The learned fns (column MLP / SFNO) accept the traced ``forcing`` dict
    # (prescribed-SST pathway); the classical stack does NOT — its signature is
    # (state, grid, sigma), same as in the AIMIP trainer, so the WB classical
    # rollout is unforced exactly as the AIMIP classical one is. The
    # prescribed-SST AMIP path is a different entry point in both campaigns.
    if cfg.mode == "physics":
        # WB classical IS AIMIP classical since 2026-08-12: the same
        # six-family scheme-parameter model, built by the same factory. It used
        # to be a 6-knob, 2-family, gray-radiation model — a different
        # experiment wearing the same name, and not a baseline for this one.
        from legoesm.training.aimip_params import (
            make_aimip_classical_spectral_physics,
        )
        _radiation = _validated_wb_radiation(cfg, yml)
        _cl = dict(yml.get("classical", {}))
        params = build_variant(
            "classical", nlev=nlev,
            overrides={
                "spatial_surface": bool(_cl.get("spatial_surface", False)),
                "spatial_init_std": float(_cl.get("spatial_init_std", 0.0)),
                "spatial_seed": int(_cl.get("spatial_seed", 0)),
            })
        # Every family named explicitly in the YAML: a classical model carries
        # one of each, and defaulting them silently is how a campaign ends up
        # comparing models that differ in more than the variable under test.
        _schemes = dict(
            convection_scheme=str(_cl.get("convection", "tiedtke")),
            turbulence_scheme=str(_cl.get("turbulence", "louis")),
            gwd_scheme=str(_cl.get("gwd", "mcfarlane")),
            microphysics_scheme=str(_cl.get("microphysics", "sundqvist")),
            cloud_scheme=str(_cl.get("cloud", "xu_randall")),
            surface_bulk_scheme=str(_cl.get("surface_bulk", "constant")),
        )
        _rad_interval = int(_cl.get("rad_update_interval_steps", 6))

        # Any scheme in any family may be swapped, and the TRAINABLE
        # PARAMETERS follow the swap: the hand-written knob set only covers
        # one scheme per family, so an arm that selects e.g. bechtold or clubb
        # would otherwise run it at fixed defaults with no gradient. This is
        # the same spec-driven bundle run_aimip exposes as
        # ``aimip_trainable_schemes``; WB reads it from ``classical.trainable_
        # schemes`` and defaults to the same tier so the two campaigns train
        # the same thing for the same selection.
        _param_init = dict(_cl.get("param_init", {}) or {})
        # Static scheme-config values the deck pins: {scheme_key: {field: v}}.
        # For settings that are NOT trainable parameters -- a cloud-overlap
        # choice, a sub-column count, a switch -- so a training arm can run the
        # same configuration a production run does.
        _param_fixed = {k: dict(v) for k, v in
                        (_cl.get("param_fixed", {}) or {}).items()}
        _tier = _cl.get("trainable_schemes", "extended")
        if _param_init and not _tier:
            raise ValueError(
                "classical.param_init needs classical.trainable_schemes: with "
                "no spec-driven bundle there is no leaf to seed, so every "
                "starting value would be silently ignored.")
        if _tier:
            from legoesm.training.aimip_params import (
                AIMIPTrainableBundle,
                aimip_legacy_owned_fields,
                aimip_scheme_keys_for,
            )
            from legoesm.training.param_collector import build_trainable_params

            _active = aimip_scheme_keys_for(
                convection=_schemes["convection_scheme"],
                turbulence=_schemes["turbulence_scheme"],
                gwd=_schemes["gwd_scheme"],
                microphysics=_schemes["microphysics_scheme"],
                radiation=_radiation,
                cloud=_schemes["cloud_scheme"],
            )
            _scheme_params = build_trainable_params(
                active_scheme_keys=_active,
                tier=(_tier if isinstance(_tier, str) else "extended"),
                # Deck-supplied starting values ("scheme_key.field": value),
                # e.g. an arm that begins from the tuned AMIP configuration
                # rather than from library defaults. Seeded rather than
                # overridden, so the parameter still trains; a name that
                # reaches no selected parameter raises.
                init_values=_param_init,
                # Only what the hand-written route cannot reach: the splice
                # runs after it, so a doubly-covered field would silently zero
                # the legacy leaf's gradient. A field the deck PINS is excluded
                # for the same reason one step later: the pin is applied after
                # the trained value, so training it would optimize a leaf that
                # never reaches the model, and the trained-parameter report
                # would show it moving.
                exclude=tuple(sorted(
                    set(aimip_legacy_owned_fields(
                        cloud_scheme=_schemes["cloud_scheme"]))
                    | _pinned_trainable_names(_param_fixed))),
            )
            params = AIMIPTrainableBundle(
                classical=params, schemes=_scheme_params)

        # SPLIT radiation, as the AIMIP arm runs it: RRTMGP once every
        # `rad_update_interval_steps` scan steps instead of on every RK stage.
        # Without this the interval key is INERT and the SI SSP-RK3 step calls
        # RRTMGP three times per step — ~18x the intended rate at interval 6,
        # and the combined wrapper also freezes radiation's solar time (codex).
        # CAM trop_cloud_top_press for CLUBB [hPa in the YAML, Pa inside]:
        # None/absent -> CLUBBConfig's default 0.0 = feature off. The
        # 32-level arm sets 50 hPa — the diagnostic scheme's measured
        # stratospheric excursion lives 16-50 hPa there. A non-finite or
        # non-positive value is a config error, not a silent off-switch.
        _clubb_top_hpa = _cl.get("clubb_top_press_hpa")
        if _clubb_top_hpa is None:
            _clubb_top = None
        else:
            _clubb_top = float(_clubb_top_hpa) * 100.0
            if not math.isfinite(_clubb_top) or _clubb_top <= 0.0:
                raise ValueError(
                    "classical.clubb_top_press_hpa must be a finite "
                    f"positive pressure in hPa, got {_clubb_top_hpa!r}; "
                    "omit the key to leave the CAM trop-cloud-top taper "
                    "off.")

        # Radiation g-point block size: the documented compile/memory
        # tradeoff (see make_aimip_classical_spectral_physics).  The
        # value-and-grad step needs ~50.5 GiB of scratch at T63/L32 with the
        # default 16, which is at the limit of an 80 GB card; halving the
        # block halves the radiation activations the backward pass holds, at
        # the cost of more blocks to walk.  Exposed here so a run can be made
        # to fit without editing code.
        # Radiation COLUMN block size: the solve is split into blocks of this
        # many columns and each block is checkpointed, so the backward pass
        # holds one block's activations instead of every column's.  Needed
        # when max-random overlap expands the solver's column axis by the
        # sub-column count (8 sub-columns at T63/L32 asks for 490 GiB
        # otherwise).  0 = off, and the value must divide the solver's column
        # count (n_sub * ncol when sub-columns are on).
        _col_chunk = int(_cl.get("rrtmgp_column_chunk_size", 0))
        if _col_chunk < 0:
            raise ValueError(
                "classical.rrtmgp_column_chunk_size must be >= 0, got "
                f"{_col_chunk}; 0 disables column chunking.")
        _gpt_batch = int(_cl.get("rrtmgp_gpoint_batch_size", 16))
        if _gpt_batch < 1:
            raise ValueError(
                "classical.rrtmgp_gpoint_batch_size must be >= 1, got "
                f"{_gpt_batch}; use the scan path by setting it to 1 rather "
                "than 0 or a negative value.")

        def make_physics_fn(p):
            return make_aimip_classical_spectral_physics(
                p, grid, dt, radiation=_radiation, split_rad=True,
                rad_update_interval_steps=_rad_interval,
                rrtmgp_gpoint_batch_size=_gpt_batch,
                rrtmgp_column_chunk_size=_col_chunk,
                param_overrides=(_param_fixed or None),
                orbital_insolation=bool(
                    _cl.get("orbital_insolation", False)),
                convective_rain_to_surface=bool(
                    _cl.get("convective_rain_to_surface", False)),
                clubb_top_press=_clubb_top, **_schemes)
        # The sample's forcing carries ERA5 skin temperature and the scene's
        # real calendar.  It used to be dropped here: the surface then sat at
        # the lowest model level's own air temperature (zero sensible heat
        # flux by construction) and radiation ran every scene on a
        # spring-equinox noon sun.  The rollout now anchors the bulk-flux
        # surface temperature to the prescribed field and advances the real
        # calendar through the window.
        uses_forcing = True
        split_rad_interval = _rad_interval

    elif cfg.mode == "neural_gcm":
        ov = yml.get("neural_gcm", {})
        # ``nn_hidden``/``nn_layers`` are this lane's historical key names; the
        # registry's vocabulary is AIMIP's. The WB-suite adapter retires the
        # aliases; until then map them here rather than teach the registry two
        # spellings.
        _cn = {}
        if "nn_hidden" in ov:
            _cn["nn_hidden_dim"] = int(ov["nn_hidden"])
        if "nn_layers" in ov:
            _cn["n_layers"] = int(ov["nn_layers"])
        # NeuralGCM-style spatial embedding (mode block) + the six ERA5
        # surface-flux input planes (top-level campaign flag).
        _cn["spatial_embedding"] = bool(ov.get("spatial_embedding", False))
        _cn["era5_surface_fluxes"] = bool(yml.get("era5_surface_fluxes", False))
        params = build_variant("column_nn", nlev=nlev, grid=grid, overrides=_cn)

        # #1464: the learned arm has no momentum head, so without this it runs
        # with NO surface turbulent drag while the `physics` arm it is scored
        # against inherits TurbulenceConfig.scheme="smagorinsky". Opt-in so no
        # existing campaign changes silently; `neural_gcm.surface_drag: true`
        # in the campaign YAML makes the two arms differ in thermodynamics
        # only. Built ONCE here, not per make_physics_fn call, so the closure
        # is a compile-time constant.
        _drag_fn = None
        if bool(ov.get("surface_drag", False)):
            from legoesm.training.neural_gcm_spectral import (
                make_turbulence_only_spectral_physics,
            )
            _drag_fn = make_turbulence_only_spectral_physics(
                dt, turbulence_scheme=str(ov.get("surface_drag_scheme",
                                                 "smagorinsky")))

        def make_physics_fn(p):
            return make_column_mlp_spectral_physics(
                p, grid, momentum_physics_fn=_drag_fn)
        uses_forcing = True
        split_rad_interval = None

    elif cfg.mode == "sfno":
        # Channel count, forcing planes, residual_prediction=False and the
        # epoch-0 decoder zero-init (so the first rollout is the pure SI
        # dycore) now live in the registry, shared with AIMIP.
        ov = yml.get("sfno", {})
        # NOTE this lane's default SFNO size moved 256/8 -> the registry's
        # 128/4 for a YAML that pins neither; every shipped config under
        # config/wb/ pins both, so no existing run moves. `sfno_mlp_expansion`
        # is newly honoured here (it was silently ignored before).
        _sf = {k: int(ov[k]) for k in
               ("sfno_embed_dim", "sfno_n_blocks", "sfno_mlp_expansion")
               if k in ov}
        # ACE2-style land-fraction plane (mode block) + the six ERA5
        # surface-flux input planes (top-level campaign flag).
        _sf_spatial = bool(ov.get("spatial_embedding", False))
        _sf_fluxes = bool(yml.get("era5_surface_fluxes", False))
        _sf["spatial_embedding"] = _sf_spatial
        _sf["era5_surface_fluxes"] = _sf_fluxes
        params = build_variant("sfno_physics", nlev=nlev, grid=grid,
                               overrides=_sf)

        def make_physics_fn(p):
            return make_sfno_spectral_physics(
                p, grid, spatial_embedding=_sf_spatial,
                era5_surface_fluxes=_sf_fluxes)
        uses_forcing = True
        split_rad_interval = None

    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    class _SpectralRunSeg:
        """`.raw(ic, n_steps, forcing)` contract of build_segment_fn, on the
        spectral core.  ``raw`` = non-donating, non-jit — safe inside
        eqx.filter_value_and_grad (buffer-donation doctrine)."""

        def __init__(self, physics_fn):
            # The classical factory returns (non_rad_fn, rad_fn) under
            # split_rad; every other mode returns one callable.
            if split_rad_interval is not None:
                self._physics_fn, self._rad_fn = physics_fn
            else:
                self._physics_fn, self._rad_fn = physics_fn, None

        def raw(self, ic_carry, n_steps, forcing):
            state0 = carry_to_spectral_state(ic_carry, grid)
            forcing_base = forcing if uses_forcing else None
            # Classical convection reads the grid's mask, not PhysicsState.
            # Keep the per-sample plane traced and the shared grid immutable.
            physics_grid = grid
            if cfg.mode == "physics":
                if forcing is None or forcing.get("land_frac") is None:
                    raise ValueError(
                        "physics mode requires land_frac from "
                        "_load_era5_samples_spectral (via build_spectral_forcing)")
                physics_grid = grid._replace(land_frac=forcing["land_frac"])
            # Spectral radiation does not read land_frac; surface BCs come from forcing.
            gated = ({} if self._rad_fn is None else
                     {"rad_physics_fn": self._rad_fn,
                      "rad_update_interval": split_rad_interval})
            final = spectral_rollout(
                state0, self._physics_fn, physics_grid, sigma, pe_config,
                dt, int(n_steps),
                sponge_factor, spectral_filter,
                forcing_base=forcing_base,
                **gated,
            )
            return spectral_state_to_carry(final, grid, sigma)

    def make_run_seg(trainable):
        return _SpectralRunSeg(make_physics_fn(trainable))

    return None, grid, sigma, params, make_run_seg, loss_config, dt


def check_surface_drag_confound(yml, mode, training_core):
    """Refuse — or name — a run whose learned arm has no surface stress.

    Returns ``None`` when the two arms are equalised, or a one-line note when
    the campaign has DECLARED that they cannot be, so a caller can record the
    confound beside the numbers it produced.  Raises ``SystemExit`` for the one
    declaration that is simply false on the selected core.

    Two declarations exist, and they are not the same kind of thing:

    ``core_does_not_read_the_key``
        The learned column cannot be given the classical arm's surface stress
        because the core it runs never reads the key.  True of the lat-lon
        core, FALSE of the spectral one.  Selecting the spectral core with such
        a config scores a learned arm carrying no surface stress against a
        classical arm that has one -- the #1464 confound, back with a label on
        it.  Refused.

    ``builder_refuses_classical_scheme``
        The classical arm runs a PROGNOSTIC turbulence scheme that the drag
        builder cannot reproduce, so no choice of ``surface_drag_scheme``
        equalises the arms.  The declaration is checked against the builder's
        real refusal set by ``tests/unit/test_neural_momentum_sink.py``.  This
        one is legitimate, but the run still produces a table whose two arms
        differ by a momentum sink, so it is NAMED rather than refused.

    Lives here, not in a driver, because this is the module every entry point
    that builds a learned spectral arm already goes through -- guarding one
    driver left the evaluation driver free to write a confounded scorecard.
    """
    neural = yml.get("neural_gcm") or {}
    if not isinstance(neural, dict):
        return None
    if mode != "neural_gcm" or neural.get("surface_drag") is True:
        return None
    declared = neural.get("surface_drag_confounded")
    if declared == "core_does_not_read_the_key" and training_core == "spectral":
        raise SystemExit(
            "this config declares surface_drag_confounded: "
            "'core_does_not_read_the_key', which is only true on the lat-lon "
            "core -- the spectral core DOES read neural_gcm.surface_drag, so "
            "this run would give the learned arm no surface stress while the "
            "classical arm it is compared against has one (#1464). Either run "
            "--training-core latlon, or set neural_gcm.surface_drag: true with "
            "surface_drag_scheme equal to classical.turbulence.")
    if declared == "builder_refuses_classical_scheme":
        classical = yml.get("classical")
        scheme = classical.get("turbulence") if isinstance(classical, dict) else None
        return (
            f"learned arm has NO surface stress: the classical arm runs "
            f"{scheme!r}, a prognostic scheme the drag builder cannot "
            f"reproduce (declared surface_drag_confounded="
            f"'builder_refuses_classical_scheme'). The two arms differ by a "
            f"momentum sink as well as by the model -- this is not an "
            f"equalised comparison (#1464).")
    return None

def build_mode_components(cfg, yml):
    """Return (model, grid, sigma, params, make_run_seg, loss_config, dt) for cfg.mode."""
    import jax

    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.training.training_driver import build_training_segment

    core = getattr(cfg, "training_core", "latlon")
    # Every entry point that builds a learned arm comes through here -- train
    # and eval alike -- so the #1464 surface-stress guard belongs here and not
    # in one driver, where the other one simply walked past it.
    note = check_surface_drag_confound(yml, getattr(cfg, "mode", None), core)
    if note:
        import logging
        logging.getLogger("scale_build").warning("CONFOUNDED RUN: %s", note)
    if core == "spectral":
        return _build_mode_components_spectral(cfg, yml)
    if core != "latlon":
        raise ValueError(
            f"unknown training_core {core!r}; choose 'latlon' (explicit "
            f"C-grid production core) or 'spectral' (Gaussian semi-implicit "
            f"training core, #817)")
    config = build_latlon_config(cfg, yml)
    driver = ModelDriver(config)
    driver.setup()
    grid, sigma, model = driver.grid, driver.sigma, driver.model
    # The driver's setup applies BOTH stability clamps (the factory's
    # pole-cell advective clamp AND the gravity-wave CFL reduction) and
    # stores the final safe value in config.dycore.dt — use it verbatim.
    # Re-deriving it here is how the old _cfl_dt handed the training
    # segment dt=81.8 s (pole-cell CFL 1.47) at the C32 smoke size: it
    # used the meridional spacing only, and the pure-dycore modes
    # (zero-init neural_gcm / sfno) blew up to loss=nan (#797).
    dt = float(driver.config.dycore.dt)
    # The driver's BL Rayleigh-friction profile: the training rollout needs
    # the same dissipation as production — the adjoint through an undamped
    # dycore returns NaN gradients (#797 bug 11; bites the pure-dycore
    # epoch-0 neural_gcm/sfno modes hardest).
    fric_decay = driver.fric_decay
    physics_pipeline = build_physics_pipeline(grid, sigma, config)
    loss_config = make_loss_config(cfg, yml)

    if cfg.mode == "physics":
        from legoesm.training.trainable_params import TrainablePhysicsParams
        params = TrainablePhysicsParams.from_defaults()
        step_unified = physics_pipeline.build_step_unified()

        def make_run_seg(trainable):
            return build_training_segment(
                model, step_unified, grid, sigma, dt, fric_decay=fric_decay,
                **trainable.to_segment_kwargs())

    elif cfg.mode == "neural_gcm":
        from legoesm.atmosphere.physics.neural_physics import (
            NeuralPhysics, make_neural_step_unified,
        )
        from legoesm.core.grid_adapters import make_adapter
        ov = yml.get("neural_gcm", {})
        adapter = make_adapter(grid)
        # Prescribed-input options: spatial_embedding is a per-model-block
        # key, era5_surface_fluxes is deck-global (the same key the loader
        # reads); the pos-embed table needs the column count.
        spatial_embedding = bool(ov.get("spatial_embedding", False))
        params = NeuralPhysics(
            nlev=int(yml["nlev"]), hidden_dim=int(ov.get("nn_hidden", 256)),
            n_layers=int(ov.get("nn_layers", 4)), key=jax.random.PRNGKey(0),
            spatial_embedding=spatial_embedding,
            era5_surface_fluxes=bool(yml.get("era5_surface_fluxes", False)),
            n_columns=adapter.ncol if spatial_embedding else None)

        def make_run_seg(nn_phys):
            return build_training_segment(
                model, make_neural_step_unified(nn_phys, adapter), grid, sigma,
                dt, fric_decay=fric_decay)

    elif cfg.mode == "sfno":
        import equinox as eqx
        import jax.numpy as jnp

        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.grids.regridding import compute_latlon_to_voronoi_weights
        from legoesm.training.model_registry import sfno_extra_input_channels
        from legoesm.training.sfno_dycore_coupling import (
            SFNOPhysics, make_sfno_step_unified_latlon,
        )
        from legoesm.ml.sfno import SFNO, SFNOConfig

        ov = yml.get("sfno", {})
        nlev = int(yml["nlev"])
        spatial_embedding = bool(ov.get("spatial_embedding", False))
        era5_surface_fluxes = bool(yml.get("era5_surface_fluxes", False))
        n_ch = 4 * nlev + 2   # SFNOPhysics packs [u, v, T, q_v, lnps, phis]
        # Prescribed input channels appended by SFNOPhysics (land fraction,
        # then the six ERA5 surface-flux planes); the decoder still predicts
        # only the n_ch tendency channels.
        n_max = int(ov.get("gauss_n_max", max(21, int(yml["n_lat"]) // 3)))
        gauss = create_gaussian_grid(n_max)
        sfno = SFNO(
            SFNOConfig(
                in_channels=n_ch + sfno_extra_input_channels(
                    spatial_embedding=spatial_embedding,
                    era5_surface_fluxes=era5_surface_fluxes),
                out_channels=n_ch,
                embed_dim=int(ov.get("sfno_embed_dim", 256)),
                n_blocks=int(ov.get("sfno_n_blocks", 8)),
                # tendencies, NOT state residuals: with the default
                # residual_prediction=True the "tendency" would contain the
                # full state and destroy the rollout in one step.
                residual_prediction=False,
            ),
            gauss, key=jax.random.PRNGKey(0))
        # Epoch-0 stability contract (same as the NeuralPhysics zero-init):
        # the untrained SFNO must emit exactly-zero tendencies so the first
        # rollout is the pure dycore.
        sfno = eqx.tree_at(
            lambda m: (m.decoder.weight, m.decoder.bias), sfno,
            (jnp.zeros_like(sfno.decoder.weight),
             jnp.zeros_like(sfno.decoder.bias)))
        params = SFNOPhysics(
            sfno=sfno, grid=gauss, nlev=nlev,
            spatial_embedding=spatial_embedding,
            era5_surface_fluxes=era5_surface_fluxes)

        def _flat_points(lat_1d, lon_1d):
            lon2d, lat2d = np.meshgrid(np.asarray(lon_1d), np.asarray(lat_1d))
            return lat2d.ravel(), lon2d.ravel()

        g_lat_f, g_lon_f = _flat_points(gauss.lat, gauss.lon)
        ll_lat_f, ll_lon_f = _flat_points(grid.lat, grid.lon)
        w_ll2g = compute_latlon_to_voronoi_weights(
            np.asarray(grid.lat), np.asarray(grid.lon), g_lat_f, g_lon_f,
        )._replace(target_shape=(int(gauss.n_lat), int(gauss.n_lon)))
        w_g2ll = compute_latlon_to_voronoi_weights(
            np.asarray(gauss.lat), np.asarray(gauss.lon), ll_lat_f, ll_lon_f,
        )._replace(target_shape=(int(grid.n_lat), int(grid.n_lon)))

        def make_run_seg(sfno_ph):
            step = make_sfno_step_unified_latlon(sfno_ph, w_ll2g, w_g2ll)
            return build_training_segment(model, step, grid, sigma, dt,
                                           fric_decay=fric_decay)

    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    return model, grid, sigma, params, make_run_seg, loss_config, dt


def rollout_hours(cfg, yml):
    hrs = cfg.multi_step_hours or tuple(yml.get("loss", {}).get("multi_step_hours", ()) or ())
    return float(hrs[0]) if hrs else 6.0     # first lead = the base rollout horizon


def era5_time_to_forcing_calendar(time_ns, year):
    """(day_of_year 1-based INTEGER, seconds_of_day) for spectral_rollout.

    Calendar convention of ``spectral_rollout._forcing_at`` (codex #817
    adversarial finding): the rollout advances
    ``doy_eff = day_of_year + (t + seconds_of_day)/86400``, so the base
    ``day_of_year`` must be the integer 1-based day and the intra-day fraction
    must ride ``seconds_of_day`` alone — a 0-based/fractional doy is wrapped a
    full year off (Jan 1 00Z -> day 365), and a fractional doy would
    double-count the hours (which also rules out ``day_to_calendar``'s
    fractional doy here).
    """
    elapsed_s = float((time_ns - np.datetime64(f"{year}-01-01"))
                      / np.timedelta64(1, "s"))
    doy_1based = float(np.floor(elapsed_s / 86400.0)) + 1.0
    sod = float(elapsed_s % 86400.0)
    return doy_1based, sod


def validate_carry_holds_scheme(carry, microphysics: str, *, context: str) -> int:
    """Refuse a scheme whose species the BUILT carry cannot hold.

    Mirrors the production driver, which counts the leading non-None slots of
    its live tracer state and validates that count.  Counting instead the
    registry the scheme itself selected would be tautological — the two can
    only ever agree, so such a check could never catch the failure it claims to
    (codex).  Counting the carry catches a seeding path that silently produced
    fewer slots than the scheme writes, which is exactly how the WeatherBench
    classical arm trained to a NaN: nine-species microphysics on a
    three-species state, six tendencies discarded per evaluation.

    Deliberately NOT placed in the shared microphysics bridge: several dycore
    tests build partial tracer states on purpose and rely on its tolerance, so
    turning that into a hard error is a separate policy decision.
    """
    from legoesm.core.tracers import make_full_moisture_registry
    from legoesm.driver.physics_pipeline import validate_microphysics_tracer_slots

    have = 0
    for name in make_full_moisture_registry().names:
        if getattr(carry, name, None) is None:
            break
        have += 1
    return validate_microphysics_tracer_slots(microphysics, have, context=context)


def _era5_config(cfg, yml):
    """The ERA5 store config for a training run.

    ``era5_cloud_condensate: true`` in the campaign YAML pulls cloud liquid and
    cloud ice into the initial condition from ``era5_cloud_zarr`` (ARCO-ERA5 by
    default).  It is OFF unless asked for: the WeatherBench2 store carries no
    condensate at all, so every sample would otherwise start cloud-free and the
    microphysics parameters could not influence a six-hour forecast.
    """
    from legoesm.training.era5_to_state import TrainingERA5Config

    c = TrainingERA5Config(dt_hours=int(yml.get("era5_cadence_hours", 6)))
    c = c._replace(zarr_store=yml["era5_zarr"],
                   load_cloud_condensate=bool(yml.get("era5_cloud_condensate",
                                                      False)))
    if yml.get("era5_cloud_zarr"):
        c = c._replace(cloud_zarr=str(yml["era5_cloud_zarr"]))
    c = c._replace(
        load_surface_fluxes=bool(yml.get("era5_surface_fluxes", False)),
        load_land_frac=wb_needs_land_frac(getattr(cfg, "mode", None), yml))
    if "era5_flux_zarr" in yml:
        # An explicit "" selects the STATE store as the flux store (the
        # loader's documented option), so the key's presence decides.
        c = c._replace(flux_zarr=str(yml["era5_flux_zarr"] or ""))
    return c


def prescribed_surface_planes(sl, grid):
    """Regridded prescribed surface planes the slice carries -> dict.

    ONE plane builder shared by the spectral forcing dict and the lat-lon
    ``pack_forcing`` loader, so a model is fed exactly the same planes
    whenever the slice carries them (i.e. whenever its
    ``TrainingERA5Config`` asked for them).  Keys are the SPECTRAL forcing
    names; signs are already the model's (``era5_to_state`` did the ERA5
    flips).  Planes stay grid-shaped (n_lat, n_lon) — the spectral caller
    flattens them to columns.
    """
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import regrid_2d_to_gaussian

    planes = {}
    if sl.sfc_shf is not None:
        for _key in ("sfc_shf", "sfc_lhf", "sfc_tau_x", "sfc_tau_y",
                     "sfc_sw_up", "sfc_sw_down", "sfc_lw_up"):
            planes[_key] = jnp.asarray(regrid_2d_to_gaussian(
                getattr(sl, _key), sl.lat, sl.lon, grid))
    if sl.land_frac is not None:
        # Static 0..1 mask; interpolation can overshoot, clip back.
        planes["land_frac"] = jnp.asarray(np.clip(regrid_2d_to_gaussian(
            sl.land_frac, sl.lat, sl.lon, grid), 0.0, 1.0))
    return planes


def build_spectral_forcing(sl, grid, time_ns, year):
    """The traced per-sample forcing dict of the spectral core, from a slice.

    ONE builder for the training loader and the WB2 forecast eval, so a model
    is scored with exactly the forcing it was trained with: the prescribed
    surface-flux planes and the land fraction ride along whenever the slice
    carries them (i.e. whenever its ``TrainingERA5Config`` asked for them),
    regridded and flattened exactly like ``T_sfc``.  Signs are already the
    model's: ``era5_to_state`` did the ERA5 flips.
    """
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import regrid_2d_to_gaussian

    def _plane(field):
        return jnp.asarray(regrid_2d_to_gaussian(
            field, sl.lat, sl.lon, grid)).reshape(-1)

    sst = _plane(sl.sst)
    doy_1based, sod = era5_time_to_forcing_calendar(time_ns, year)
    forcing = {
        "T_sfc": sst,
        "sic": jnp.zeros_like(sst),
        "day_of_year": jnp.asarray(doy_1based),
        "seconds_of_day": jnp.asarray(sod),
    }
    # Shared plane builder (grid-shaped there), flattened to columns here so
    # the output dict is identical to the pre-helper version.
    for _key, _plane_2d in prescribed_surface_planes(sl, grid).items():
        forcing[_key] = _plane_2d.reshape(-1)
    return forcing


def _load_era5_samples_spectral(cfg, yml, grid, sigma, *,
                                rank=0, nproc=1, host_resident=False):
    """(ic, target, forcing) samples on the Gaussian grid for the spectral core.

    ``ic``/``target`` are SegmentCarry on the Gaussian grid
    (``era5_to_spectral_carry``); ``forcing`` is the traced dict the spectral
    rollout's prescribed-SST pathway consumes (``forcing_base``: flat
    ``T_sfc``/``sic`` (ncol,) + calendar scalars — SegmentForcing doctrine, all
    traced so per-sample values never retrace).  Consumed only inside the
    spectral ``make_run_seg(...).raw`` (opaque to the trainer loop).

    #1286: with ``nproc > 1`` this builds ONLY ``rank``'s contiguous shard (fix
    B — no global materialization); with ``host_resident`` each built sample is
    moved off-device to host numpy (fix A — the loop ``device_put``s per batch).
    """
    from legoesm.training.era5_to_state import (
        era5_to_spectral_carry,
        load_era5_slice,
        open_era5_zarr,
    )

    # ic/target slices need the state only; the sample-start slice (2-D
    # fields only: asking it for condensate would re-read the cloud store per
    # sample for data it throws away) is the one that carries the prescribed
    # surface planes.
    sst_cfg = _era5_config(cfg, yml)._replace(load_cloud_condensate=False)
    era5_cfg = _era5_config(cfg, yml)._replace(
        load_surface_fluxes=False, load_land_frac=False)
    surface_fluxes = sst_cfg.load_surface_fluxes
    flux_ds = None
    global _WARNED_ERA5_FLUX_HOLD_FIXED
    ds = open_era5_zarr(era5_cfg.zarr_store)
    if surface_fluxes:
        # Opened once for the whole sharded loop; the loader honours flux_ds.
        # A land fraction alone is NOT a reason to open it: the state store
        # (WB2) carries the mask and the loader falls back lazily otherwise.
        # flux_zarr == "" means "read the fluxes from the state store".
        flux_ds = open_era5_zarr(sst_cfg.flux_zarr) if sst_cfg.flux_zarr else ds
    if surface_fluxes and not _WARNED_ERA5_FLUX_HOLD_FIXED:
        _WARNED_ERA5_FLUX_HOLD_FIXED = True
        logger.warning(
            "era5_surface_fluxes: hourly interpolation of the ERA5 surface "
            "fluxes is not implemented — every flux plane is HELD FIXED at "
            "its sample-start value for the whole training window "
            "(owner decision 2026-09-15).")
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    snaps_per_day = 24 // era5_cfg.dt_hours
    roll_h = rollout_hours(cfg, yml)
    stride = int(roll_h) // era5_cfg.dt_hours
    # The condensate store is opened ONCE and threaded through every slice --
    # re-opening a remote zarr per sample would dominate the load.
    cloud_ds = (open_era5_zarr(era5_cfg.cloud_zarr)
                if era5_cfg.load_cloud_condensate else None)
    # Which water species the state must carry follows the scheme the arm
    # selects, and the carry builder already knows how to seed them — it was
    # simply never told which scheme was running here, so every state was built
    # for the three warm-rain slots.  A nine-species microphysics then had its
    # ice / snow / graupel / number tendencies silently dropped downstream.
    # The learned arms run no microphysics or turbulence scheme at all, so they
    # keep the three-slot carry and are byte-unchanged.
    #
    # Microphysics only. A stateful turbulence scheme (CLUBB, MYNN, EDMF) would
    # also seed a prognostic energy carry, but the spectral rollout threads no
    # turbulence state between steps and the spectral->carry conversion has
    # nowhere to put one, so seeding it would make the forecast carry
    # structurally different from the initial condition it is scored against.
    # Carrying turbulence energy across steps on this path is separate work.
    _cl = dict(yml.get("classical", {})) if cfg.mode == "physics" else {}
    _micro = str(_cl.get("microphysics", "none"))


    samples = []
    for year, i_ic, i_tg in _sharded_indices(
            cfg, yml, times, snaps_per_day, stride, rank, nproc):
        ic = era5_to_spectral_carry(
            load_era5_slice(era5_cfg, i_ic, ds=ds, cloud_ds=cloud_ds),
            grid, sigma, microphysics=_micro)
        target = era5_to_spectral_carry(
            load_era5_slice(era5_cfg, i_tg, ds=ds, cloud_ds=cloud_ds),
            grid, sigma, microphysics=_micro)
        if not samples:
            validate_carry_holds_scheme(
                ic, _micro, context=f"WB {cfg.mode} arm initial condition")
        sst_src = load_era5_slice(sst_cfg, i_ic, ds=ds, flux_ds=flux_ds)
        forcing = build_spectral_forcing(sst_src, grid, times[i_ic], year)
        sample = (ic, target, forcing)
        samples.append(_sample_to_host(sample) if host_resident else sample)
    return samples


def load_era5_samples(cfg, yml, grid, sigma, *,
                      rank=0, nproc=1, host_resident=False):
    """(ic, target, forcing) samples over the train windows, on the lat-lon grid.

    #1286: ``nproc > 1`` builds ONLY ``rank``'s contiguous shard (fix B);
    ``host_resident`` keeps the samples on host numpy (fix A).  Defaults
    (``rank=0, nproc=1, host_resident=False``) reproduce the previous
    global-eager behaviour byte-for-byte for existing callers.
    """
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import (
        era5_to_latlon_carry,
        load_era5_slice,
        open_era5_zarr,
        regrid_2d_to_gaussian,
    )
    from legoesm.driver.compiled_segments import pack_forcing

    if getattr(cfg, "training_core", "latlon") == "spectral":
        return _load_era5_samples_spectral(
            cfg, yml, grid, sigma,
            rank=rank, nproc=nproc, host_resident=host_resident)

    global _WARNED_ERA5_FLUX_HOLD_FIXED

    era5_cfg = _era5_config(cfg, yml)
    ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    snaps_per_day = 24 // era5_cfg.dt_hours
    roll_h = rollout_hours(cfg, yml)
    stride = int(roll_h) // era5_cfg.dt_hours

    # Opened once, like the state store — a per-sample remote zarr open would
    # dominate the load (see the spectral loader).
    cloud_ds = (open_era5_zarr(era5_cfg.cloud_zarr)
                if era5_cfg.load_cloud_condensate else None)
    # 2-D only, but keeps load_surface_fluxes / load_land_frac so the
    # sample-start slice carries the prescribed planes.
    sst_cfg = era5_cfg._replace(load_cloud_condensate=False)
    # Flux store opened once, only when fluxes are prescribed ("" = the
    # state store); a land fraction alone comes from the state store.
    flux_ds = None
    if sst_cfg.load_surface_fluxes:
        flux_ds = open_era5_zarr(sst_cfg.flux_zarr) if sst_cfg.flux_zarr else ds
    _cl = dict(yml.get("classical", {})) if cfg.mode == "physics" else {}
    _micro = str(_cl.get("microphysics", "none"))

    config = build_latlon_config(cfg, yml)
    driver = _driver_for_ctx(config)
    ctx = driver._prepare_run_context(0, config.start_day, restore_carry=False)

    # Spectral forcing-dict name -> driver-path (pack_forcing) name for the
    # planes the two lanes name differently; the radiative planes and the
    # land fraction already share their names.
    _spec_to_driver = {
        "sfc_shf": "sfc_shflx_override",
        "sfc_lhf": "sfc_lhflx_override",
        "sfc_tau_x": "sfc_taux_override",
        "sfc_tau_y": "sfc_tauy_override",
    }

    samples = []
    for year, i_ic, i_tg in _sharded_indices(
            cfg, yml, times, snaps_per_day, stride, rank, nproc):
        ic = era5_to_latlon_carry(
            load_era5_slice(era5_cfg, i_ic, ds=ds, cloud_ds=cloud_ds),
            grid, sigma, microphysics=_micro)
        target = era5_to_latlon_carry(
            load_era5_slice(era5_cfg, i_tg, ds=ds, cloud_ds=cloud_ds),
            grid, sigma, microphysics=_micro)
        sst_src = load_era5_slice(sst_cfg, i_ic, ds=ds, flux_ds=flux_ds)
        sst = regrid_2d_to_gaussian(sst_src.sst, sst_src.lat, sst_src.lon, grid)
        # Prescribed surface planes the slice carries (spectral names),
        # regridded exactly like sst, then renamed to the driver-path
        # override names for pack_forcing.
        planes = prescribed_surface_planes(sst_src, grid)
        if "sfc_shf" in planes and not _WARNED_ERA5_FLUX_HOLD_FIXED:
            _WARNED_ERA5_FLUX_HOLD_FIXED = True
            logger.warning(
                "era5_surface_fluxes: hourly interpolation of the ERA5 surface "
                "fluxes is not implemented — every flux plane is HELD FIXED at "
                "its sample-start value for the whole training window "
                "(owner decision 2026-09-15).")
        plane_kwargs = {
            _spec_to_driver.get(_key, _key): _plane
            for _key, _plane in planes.items()}
        doy = float((times[i_ic] - np.datetime64(f"{year}-01-01"))
                    / np.timedelta64(1, "D"))
        forcing = pack_forcing(
            sst=sst, sic=jnp.zeros_like(sst),
            day_of_year=jnp.asarray(doy), seconds_of_day=jnp.asarray(0.0),
            solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
            **plane_kwargs)
        sample = (ic, target, forcing)
        samples.append(_sample_to_host(sample) if host_resident else sample)
    return samples


def _driver_for_ctx(config):
    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    return driver


def evaluate_wb2(cfg, yml, model, params, make_run_seg, grid, sigma):
    """Pointer to the standalone WB2 checkpoint eval driver (issue #919).

    The scorecard is produced by ``scripts/validate/run_weatherbench_eval.py``,
    which reuses :func:`build_mode_components` + ``make_run_seg`` + the
    ``evaluations.wb_orchestrator`` scorer against the ``epoch_NNNN.eqx``
    checkpoint this run wrote. The driver is NOT imported here on purpose: the
    dependency direction is ``evaluations -> legoesm`` only (importing
    ``evaluations`` from this package would invert it), and the PBS job's
    ``PYTHONPATH`` (``scripts/cluster/wb_forecast/env.sh``) does not put the repo
    root on the path, so an import here would fail inside the job. Rank-0 logs
    the ready-to-run command; run it as a separate login-node/eval step.
    """
    import logging
    log = logging.getLogger("wb_scale")
    core = getattr(cfg, "training_core", "latlon")
    # v1 of the WB2 driver is spectral-only; flag a non-spectral core so a
    # default-core (latlon) run doesn't copy-paste a command the CLI rejects.
    note = ("" if core == "spectral" else
            f"\n  NOTE: --training-core {core!r} checkpoints are NOT WB2-evaluable "
            "in v1 (the driver is spectral-only); retrain/score with a spectral core.")
    log.info(
        "WB2 eval is a separate driver (issue #919); this run's checkpoints are "
        "under %s. Score them with:\n"
        "  python scripts/validate/run_weatherbench_eval.py --config %s --mode %s "
        "--training-core %s --checkpoint %s/epoch_NNNN.eqx --eval-year %s "
        "--out %s/wb2_scorecard.json%s",
        cfg.out_dir, cfg.config_path, cfg.mode, core, cfg.out_dir,
        yml["eval_years"][0], cfg.out_dir, note)
    return None
