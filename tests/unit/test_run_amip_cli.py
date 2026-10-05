"""CLI coverage for the real AMIP entrypoint."""

from __future__ import annotations

import io
import sys

import pytest

from legoesm import constants
from scripts.run.run_amip import (
    _apply_aimip_classical_overrides,
    _apply_spectral_scheme_fallback,
    _postprocess_args,
    _print_forcing_activity,
    _require_full_physics_for_amip,
    build_arg_parser,
    build_config_from_args,
)


def test_spectral_scheme_fallback():
    p = build_arg_parser()
    base = ["--grid-type", "gaussian", "--discretization", "spectral",
            "--truncation", "21"]

    # Default (prognostic tiedtke/mcfarlane) on spectral -> auto diagnostic.
    a = _apply_spectral_scheme_fallback(p.parse_args(base), base)
    assert a.convection == "sbm", "spectral did not fall back to diagnostic convection"
    assert "mcfarlane" not in str(a.gravity_wave_drag), \
        "spectral did not fall back off prognostic GWD"
    assert a.gravity_wave_drag == "rayleigh"

    # Explicit --convection is honoured verbatim (user's call, even if prognostic).
    argv = base + ["--convection", "tiedtke"]
    a2 = _apply_spectral_scheme_fallback(p.parse_args(argv), argv)
    assert a2.convection == "tiedtke", "explicit --convection was overridden"

    # Non-spectral grid: no-op (keeps the prognostic defaults).
    cs = ["--grid-type", "cubed_sphere", "--discretization", "cdgrid"]
    a3 = _apply_spectral_scheme_fallback(p.parse_args(cs), cs)
    assert a3.convection == "tiedtke", "non-spectral grid wrongly swapped schemes"


def test_multilayer_land_flags_flow_to_config():
    parser = build_arg_parser()
    # default: slab land (multilayer off)
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.use_multilayer_land is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--land-mask-file", "lsm.nc",
        "--use-multilayer-land",
        "--multilayer-n-layers", "8",
        "--multilayer-soil-depth", "4.5",
    ]), parser))
    assert cfg_on.use_multilayer_land is True
    assert cfg_on.multilayer_n_layers == 8
    assert cfg_on.multilayer_soil_depth == 4.5


def test_land_update_seconds_flows_to_config_and_validates():
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.land_update_seconds == 0.0

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--land-mask-file", "lsm.nc",
        "--use-multilayer-land",
        "--land-update-seconds", "300",
    ]), parser))
    assert cfg_on.land_update_seconds == 300.0

    # A cadence without the multilayer land is a silently-inert knob and
    # must be refused by validate_strict.
    import pytest
    cfg_bad = cfg_off._replace(land_update_seconds=300.0)
    with pytest.raises(ValueError, match="land_update_seconds"):
        cfg_bad.validate_strict()
    with pytest.raises(ValueError, match="land_update_seconds"):
        cfg_on._replace(land_update_seconds=-1.0).validate_strict()


def test_hard_saturation_adjustment_flag_flows_to_config():
    """--hard-saturation-adjustment round-trips to ExperimentConfig (opt-in
    warm-rain hard saturation-adjustment guard; default OFF)."""
    parser = build_arg_parser()
    # default OFF: byte-identical smooth microphysics path
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.hard_saturation_adjustment is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--hard-saturation-adjustment",
    ]), parser))
    assert cfg_on.hard_saturation_adjustment is True

    # BooleanOptionalAction exposes the explicit --no- off switch.
    cfg_no = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--no-hard-saturation-adjustment",
    ]), parser))
    assert cfg_no.hard_saturation_adjustment is False


def test_use_clubb_cloud_fraction_flag_flows_to_config():
    """--use-clubb-cloud-fraction round-trips to ExperimentConfig (marine-Sc
    albedo lever; radiation then reads diagnostic CLUBB's PDF cloud fraction)."""
    parser = build_arg_parser()
    # default OFF: RH grid-scale cloud fraction (byte-identical path)
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.use_clubb_cloud_fraction is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "clubb",
        "--use-clubb-cloud-fraction",
    ]), parser))
    assert cfg_on.use_clubb_cloud_fraction is True
    # explicit negation restores the default (BooleanOptionalAction)
    cfg_neg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--no-use-clubb-cloud-fraction",
    ]), parser))
    assert cfg_neg.use_clubb_cloud_fraction is False


def test_clubb_prognostic_flag_reaches_the_turbulence_kernel():
    """--clubb-prognostic must round-trip AND reach the nested scheme config.

    The round-trip alone is not enough: ``turbulence_config_for`` is the single
    source every dycore's kernel is built from, so a flag that reaches
    ``ExperimentConfig`` but not that function is inert on every backend while
    reporting success. Both halves are asserted here.
    """
    from legoesm.driver.physics_pipeline import turbulence_config_for

    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.clubb_prognostic is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "clubb",
        "--clubb-prognostic",
    ]), parser))
    assert cfg_on.clubb_prognostic is True
    # the half that actually decides what runs
    assert turbulence_config_for(cfg_on).clubb.prognostic is True
    # and the diagnostic default really is the other value, so the assertion
    # above cannot pass by the field simply defaulting True.  Without the flag
    # the nested sub-config is left as None and dispatch substitutes a fresh
    # CLUBBConfig(), so ask what dispatch will actually run rather than reading
    # a slot that is legitimately empty.
    from legoesm.atmosphere.physics.turbulence.integration import (
        materialize_sub_config,
    )
    cfg_diag = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "clubb",
    ]), parser))
    diag_tc = materialize_sub_config(turbulence_config_for(cfg_diag))
    assert diag_tc.clubb.prognostic is False


def test_clubb_prognostic_under_another_closure_is_refused():
    """Prognostic moments are a CLUBB feature; asking for them under Louis is a
    configuration error, not something to ignore quietly."""
    from legoesm.driver.physics_pipeline import turbulence_config_for

    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "louis",
        "--clubb-prognostic",
    ]), parser))
    with pytest.raises(ValueError, match="requires turbulence='clubb'"):
        turbulence_config_for(cfg)


def test_clubb_prognostic_not_swallowed_by_a_turbulence_override():
    """An explicit turbulence override is authoritative, but it must not make
    the prognostic request vanish.

    Found by review: the threading sat inside the no-override branch, so a run
    setting BOTH an override and clubb_prognostic resolved to the DIAGNOSTIC
    closure while reporting nothing -- the inert-knob failure this repo keeps
    hitting. Refusing is correct; silently honouring the flag would instead
    rewrite an object the caller declared authoritative.
    """
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import turbulence_config_for

    base = ExperimentConfig(turbulence="clubb", clubb_prognostic=True)
    diagnostic_override = base._replace(
        turbulence_override=TurbulenceConfig(scheme="clubb"))
    with pytest.raises(ValueError, match="turbulence_override"):
        turbulence_config_for(diagnostic_override)

    # an override that DOES select it resolves normally
    ok = base._replace(turbulence_override=TurbulenceConfig(
        scheme="clubb", clubb=CLUBBConfig(prognostic=True)))
    assert turbulence_config_for(ok).clubb.prognostic is True
    # and an override with the flag OFF is untouched (no false positive)
    off = ExperimentConfig(turbulence="clubb", clubb_prognostic=False)._replace(
        turbulence_override=TurbulenceConfig(scheme="clubb"))
    assert turbulence_config_for(off).scheme == "clubb"


def test_convective_precip_efficiency_cli_wiring_929():
    """#929: the shared --convective-precip-efficiency knob reaches the config
    for BOTH Tiedtke and Bechtold; UNSET is the ``None`` sentinel (each scheme
    keeps its own default) — never a silent 0.0 that would disable Bechtold's
    ON-by-default rain split."""
    parser = build_arg_parser()

    # Unset -> None sentinel (NOT 0.0): Bechtold keeps its own 0.7 default.
    cfg_unset = build_config_from_args(_postprocess_args(
        parser.parse_args(
            ["--dataset", "analytical", "--convection", "bechtold"]),
        parser))
    assert cfg_unset.convective_precip_efficiency is None

    # Bechtold + explicit PE now ALLOWED (the guard was Tiedtke-only) and
    # reaches the config.
    cfg_bech = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--convective-precip-efficiency", "0.7",
    ]), parser))
    assert cfg_bech.convective_precip_efficiency == 0.7

    # Explicit 0.0 for Bechtold (legacy no-split) is accepted and threaded.
    cfg_bech0 = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--convective-precip-efficiency", "0.0",
    ]), parser))
    assert cfg_bech0.convective_precip_efficiency == 0.0

    # Tiedtke still reaches the config.
    cfg_tied = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "tiedtke",
        "--convective-precip-efficiency", "0.5",
    ]), parser))
    assert cfg_tied.convective_precip_efficiency == 0.5


def test_convective_precip_efficiency_rejected_for_non_massflux_929():
    """#929: --convective-precip-efficiency>0 requires a mass-flux scheme
    (tiedtke or bechtold); other schemes ignore the knob, so the run-guard
    rejects it rather than silently no-op.  (0.0 / unset are fine everywhere.)"""
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--convection", "sbm",
            "--convective-precip-efficiency", "0.7",
        ]), parser)


def test_no_use_multilayer_land_overrides_yaml_default():
    """--no-use-multilayer-land flips a set_defaults(True) (i.e. a --config YAML
    that enables the multilayer land) back off — needed to run a production
    YAML on the MPAS/spectral standalone backends (#869 MPAS probe)."""
    parser = build_arg_parser()
    parser.set_defaults(use_multilayer_land=True)  # what a YAML would do
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--no-use-multilayer-land",
    ]), parser))
    assert cfg.use_multilayer_land is False


def test_no_sponge_overrides_yaml_default():
    """--no-sponge flips a set_defaults(True) (a --config YAML enabling the
    #836 top sponge) back off — needed for the #847 drift-lever walk's
    sponge-off leg against amip_production.yaml."""
    parser = build_arg_parser()
    parser.set_defaults(sponge_enabled=True)  # what a YAML would do
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--no-sponge",
    ]), parser))
    assert cfg.sponge_enabled is False


def test_no_surface_tiled_overrides_yaml_default():
    """--no-surface-tiled flips a set_defaults(True) (a --config YAML enabling
    the tiled mosaic surface) back off — same MPAS/spectral escape hatch as
    --no-use-multilayer-land (the standalone backends don't run the tiled
    coupled pipeline; validate_strict otherwise demands an active land tile)."""
    parser = build_arg_parser()
    parser.set_defaults(surface_tiled=True)  # what a YAML would do
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--no-surface-tiled",
    ]), parser))
    assert cfg.surface_tiled is False


def test_multilayer_land_accepted_on_mpas():
    """use_multilayer_land + MPAS grid parses (MPAS port,
    tasks/mpas_land_port.md): the tile is stepped in the MPAS driver loop with
    skin-T feedback via forcing['T_sfc'].  The old early argparse rejection
    (VoronoiMesh had no lat/lat2d) is retired — setup now reads latCell."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-mask-file", "land_mask.nc",
        "--grid-type", "mpas", "--discretization", "mpas",
        "--use-multilayer-land",
    ]), parser)
    cfg = build_config_from_args(args)
    assert cfg.use_multilayer_land is True
    cfg.validate_strict()


def test_multilayer_land_still_rejected_on_spectral():
    """The SPECTRAL standalone path keeps the early rejection (passive land
    tile, no soil-column stepping wired there)."""
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical",
            "--truncation", "21", "--discretization", "spectral",
            "--use-multilayer-land",
        ]), parser)


def test_clm_ml_rejected_on_mpas_but_two_leaf_is_not():
    """Only ``clm_ml`` is refused on the MPAS lane.

    It needs the coupled pipeline's per-column canopy grid threading, which
    this lane does not have, so selecting it must fail early rather than
    silently run something else. ``two_leaf`` is not refused: the MPAS land
    step dispatches to it, and refusing it would leave a resistance-based land
    surface unreachable on the lane the AMIP campaign runs on.

    SCOPE: this pins the ARGUMENT GUARD, nothing downstream. Whether the
    canopy's solved humidity and fluxes actually reach the atmosphere is a
    separate question — that hand-over sits behind its own switch, which
    defaults off — and is pinned by the land-boundary tests, not here.
    """
    parser = build_arg_parser()
    mpas_args = ["--dataset", "analytical",
                 "--grid-type", "mpas", "--discretization", "mpas",
                 "--use-multilayer-land", "--land-surface-scheme"]
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args(mpas_args + ["clm_ml"]), parser)
    args = _postprocess_args(parser.parse_args(mpas_args + ["two_leaf"]),
                             parser)
    assert args.land_surface_scheme == "two_leaf"


def test_clm_surfdata_path_flows_to_config():
    """--clm-surfdata-path round-trips into ExperimentConfig (empty default =>
    UCAR download; a set path lets a compute node with no internet use a staged
    surfdata NetCDF for the multilayer land)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.clm_surfdata_path == ""

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--clm-surfdata-path", "/data/clm_surfdata.nc",
    ]), parser))
    assert cfg.clm_surfdata_path == "/data/clm_surfdata.nc"


def test_transient_land_cover_flags_flow_to_config():
    """--transient-land-cover / --land-cover-surfdata round-trip into
    ExperimentConfig (off + empty by default => static single-year cover)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.transient_land_cover is False
    assert cfg_default.land_cover_surfdata == ""

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--land-mask-file", "lsm.nc", "--use-multilayer-land",
        "--transient-land-cover",
        "--land-cover-surfdata", "/data/luh2_transient_surfdata.nc",
    ]), parser))
    assert cfg.transient_land_cover is True
    assert cfg.land_cover_surfdata == "/data/luh2_transient_surfdata.nc"


def test_transient_land_cover_validate_strict_requires_multilayer_and_surfdata():
    """validate_strict() rejects transient cover without a multilayer tile or
    without a surfdata path — both would silently no-op the requested LULC."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--land-mask-file", "lsm.nc", "--use-multilayer-land",
        "--transient-land-cover",
        "--land-cover-surfdata", "/data/luh2_transient_surfdata.nc",
    ]), parser))
    cfg.validate_strict()  # complete config: no error

    # transient cover but no surfdata path -> reject
    with pytest.raises(ValueError, match="land_cover_surfdata"):
        cfg._replace(land_cover_surfdata="").validate_strict()
    # transient cover but slab land (no multilayer) -> reject
    with pytest.raises(ValueError, match="use_multilayer_land"):
        cfg._replace(use_multilayer_land=False).validate_strict()


def test_land_ic_path_flows_to_config():
    """--land-ic round-trips into ExperimentConfig.land_ic_path (#746): a
    spun-up MultiLayerLandState restart from run_land_spinup replaces the
    cold-start soil column in a coupled multilayer AMIP run."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.land_ic_path == ""

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--land-ic", "/scratch/land_spinup/land_ic.npz",
    ]), parser))
    assert cfg.land_ic_path == "/scratch/land_spinup/land_ic.npz"


def test_snow_albedo_feedback_flag_flows_to_config():
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.snow_albedo_feedback is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--snow-albedo-feedback",
    ]), parser))
    assert cfg_on.snow_albedo_feedback is True


def test_moisture_advection_flag_flows_to_config():
    """Issue #771: resolved-wind moisture advection is OPT-IN (default OFF,
    bit-identical legacy path); --moisture-advection turns it on."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.moisture_advection is False   # default off

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--moisture-advection",
    ]), parser))
    assert cfg_on.moisture_advection is True

    cfg_off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--no-moisture-advection",
    ]), parser))
    assert cfg_off.moisture_advection is False


def test_radiation_column_chunk_flag_flows_to_config():
    """--radiation-column-chunk round-trips into ExperimentConfig
    (rrtmgp_column_chunk_size). 0 (default) = off / byte-identical; a >0 value
    caps the rrtmgp XLA compile time by mapping the solve over fixed-size
    column blocks (numerically exact — radiation columns are independent)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.rrtmgp_column_chunk_size == 0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--radiation-column-chunk", "256",
    ]), parser))
    assert cfg.rrtmgp_column_chunk_size == 256


def test_land_gs_max_flag_flows_to_config():
    """--land-gs-max round-trips into ExperimentConfig (the global stomatal
    canopy-conductance calibration knob for land ET, issue #730). Default 0.3
    matches StomataConfig.gs_max; a lower value raises canopy resistance."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.land_gs_max == 0.3

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-gs-max", "0.15",
    ]), parser))
    assert cfg.land_gs_max == 0.15


def test_land_gs_max_validate_strict_rejects_nonpositive_or_nonfinite():
    """validate_strict() rejects a non-positive / non-finite gs_max — such a
    value would zero or NaN the entire land latent-heat flux."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="land_gs_max"):
            cfg._replace(land_gs_max=bad).validate_strict()


def test_land_interface_flux_flag_flows_to_config():
    """--land-interface-flux round-trips into ExperimentConfig; default is the
    byte-identical 'legacy_dual'; 'unified' selects the single-flux-law slab
    SEB (one flux law at the land-air interface; the semi-implicit
    discretization term remains — see _step_slab_land)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.land_interface_flux == "legacy_dual"

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-interface-flux", "unified",
        "--turbulence", "holtslag_boville", "--slab-land-active",
        "--topography", "etopo",
    ]), parser))
    assert cfg.land_interface_flux == "unified"
    cfg.validate_strict()

    # slab_land_active over FLAT topography derives f_land == 0 (no land) —
    # 'unified' would silently never step; validate_strict must reject it
    # (codex R2 finding: the tile-presence gate must see through the flag).
    flat = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-interface-flux", "unified",
        "--turbulence", "holtslag_boville", "--slab-land-active",
    ]), parser))
    with pytest.raises(ValueError, match="land_interface_flux"):
        flat.validate_strict()

    # Unknown value is rejected by argparse choices (before validate_strict).
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "analytical",
                           "--land-interface-flux", "both"])


def test_land_interface_flux_unified_requires_turbulence():
    """validate_strict rejects unified without a turbulence scheme (there is
    no atmosphere-side surface-layer law to unify with)."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-interface-flux", "unified",
        "--turbulence", "none", "--slab-land-active",
        "--topography", "etopo",
    ]), parser))
    with pytest.raises(ValueError, match="land_interface_flux"):
        cfg.validate_strict()


def test_c_land_flag_flows_to_config():
    """--c-land round-trips into ExperimentConfig.C_land (previously the
    config field was silently inert — never threaded to the pipeline)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.C_land == 2.0e5

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--c-land", "5e5",
    ]), parser))
    assert cfg.C_land == 5.0e5
    cfg.validate_strict()

    bad = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--c-land", "0.0",
    ]), parser))
    with pytest.raises(ValueError, match="C_land"):
        bad.validate_strict()


def test_land_soil_moisture_init_frac_flag_flows_to_config():
    """--land-soil-moisture-init-frac round-trips (issue #730 drier-cold-start
    knob); default 0.5 is byte-identical to the init default."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.land_soil_moisture_init_frac == 0.5

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-soil-moisture-init-frac", "0.25",
    ]), parser))
    assert cfg.land_soil_moisture_init_frac == 0.25
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="land_soil_moisture_init_frac"):
            cfg._replace(land_soil_moisture_init_frac=bad).validate_strict()


def test_land_surface_scheme_flag_flows_to_config():
    """--land-surface-scheme round-trips (issue #730 two-leaf canopy selector).

    The DEFAULT is now the two-leaf canopy. It used to be SimpleSEB, which
    measurement retired: on a well-watered column SimpleSEB evaporates at
    potential with stomata off, and with them on its humidity gradient
    self-extinguishes.

    The config carries "two_leaf" either way. A canopy needs a multilayer land
    tile to run inside, and when there is none the DRIVER resolves back to
    SimpleSEB and warns at setup — the config value is not rewritten, so that
    fallback is not visible here.
    """
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.land_surface_scheme == "two_leaf"
    cfg_ml_default = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--use-multilayer-land",
        "--land-mask-file", "lsm.nc",
    ]), parser))
    assert cfg_ml_default.land_surface_scheme == "two_leaf"

    # Canopy schemes require --use-multilayer-land (they run inside the multilayer
    # land tile); the flag round-trips with it set.
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-surface-scheme", "two_leaf",
        "--use-multilayer-land",
    ]), parser))
    assert cfg.land_surface_scheme == "two_leaf"

    # clm_ml is a first-class selector (S4-AMIP plumbing); validate_strict must
    # ACCEPT it (the coupled ncol>1 capability gate lives in the driver setup,
    # not here — single-point CLM-ML runs today via run_lmip).
    cfg_clm = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-surface-scheme", "clm_ml",
        "--use-multilayer-land", "--land-mask-file", "lsm.nc",
    ]), parser))
    assert cfg_clm.land_surface_scheme == "clm_ml"
    cfg_clm.validate_strict()  # must not raise


def test_clm_ml_use_surfdata_pft_flag_flows_to_config():
    """--clm-ml-use-surfdata-pft round-trips (per-column dominant PFT for CLM-ML ->
    mixed-PFT columns handled by the group-by-structure scan).  Default off."""
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.clm_ml_use_surfdata_pft is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-surface-scheme", "clm_ml",
        "--use-multilayer-land", "--clm-ml-use-surfdata-pft",
    ]), parser))
    assert cfg_on.clm_ml_use_surfdata_pft is True


def test_canopy_scheme_requires_multilayer_land():
    """A canopy surface scheme without --use-multilayer-land is a hard CLI error,
    not a silent drop to the slab land (dispatch-hardening)."""
    parser = build_arg_parser()
    for scheme in ("two_leaf", "clm_ml"):
        argv = ["--dataset", "analytical", "--land-surface-scheme", scheme]
        with pytest.raises(SystemExit):
            _postprocess_args(parser.parse_args(argv), parser, argv)
    # simple_seb is a no-op on the slab and is NOT gated.  (It is no longer the
    # default — the two-leaf canopy is — but an EXPLICIT canopy request without
    # the tile is still refused, which is what the loop above checks.)
    _postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-surface-scheme", "simple_seb",
    ]), parser)


def test_sponge_flags_flow_to_config():
    """--sponge / --sponge-coeff-per-day / --sponge-sigma-top round-trip (#836
    top-of-atmosphere sponge); default OFF with the config default coeff/base."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.sponge_enabled is False
    assert cfg_default.sponge_coeff_per_day == 2.0
    assert cfg_default.sponge_sigma_top == 0.15

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--sponge",
        "--sponge-coeff-per-day", "4.0", "--sponge-sigma-top", "0.2",
    ]), parser))
    assert cfg.sponge_enabled is True
    assert cfg.sponge_coeff_per_day == 4.0
    assert cfg.sponge_sigma_top == 0.2


def test_hard_sat_override_flags_flow_and_validate():
    """--hard-sat-adjust-threshold / --hard-sat-max-heating-k round-trip;
    gate + spec-bounds enforcement (day-137 drain-capacity lever)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.hard_sat_adjust_threshold is None
    assert cfg_default.hard_sat_max_heating_K is None

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--hard-saturation-adjustment",
        "--hard-sat-max-heating-k", "10.0",
        "--hard-sat-adjust-threshold", "1.05",
    ]), parser))
    assert cfg.hard_sat_max_heating_K == 10.0
    assert cfg.hard_sat_adjust_threshold == 1.05
    cfg.validate_strict()

    # Override without the boolean gate -> refused (silently-inert config).
    cfg_nogate = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--hard-sat-max-heating-k", "10.0",
    ]), parser))
    with pytest.raises(ValueError, match="hard_saturation_adjustment=True"):
        cfg_nogate.validate_strict()

    # Spec bounds enforced.
    cfg_oob = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--hard-saturation-adjustment",
        "--hard-sat-max-heating-k", "80.0",
    ]), parser))
    with pytest.raises(ValueError, match="hard_sat_max_heating_K"):
        cfg_oob.validate_strict()


def test_mpas_land_boundary_flags_flow_to_config():
    """--mpas-land-lapse-k-per-km / --mpas-land-beta round-trip (MPAS land
    surface boundary, 2026-07-23 speckle fix); defaults byte-identical OFF."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.mpas_land_lapse_K_per_km == 0.0
    assert cfg_default.mpas_land_beta == 1.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--grid-type", "voronoi", "--discretization", "mpas",
        "--mpas-land-lapse-k-per-km", "6.5", "--mpas-land-beta", "0.6",
    ]), parser))
    assert cfg.mpas_land_lapse_K_per_km == 6.5
    assert cfg.mpas_land_beta == 0.6
    # validate_strict is exercised in test_mpas_land_boundary (the bare CLI
    # invocation here has topography='flat', which the inert-corner guard
    # correctly refuses).


def test_mpas_qv_smoothing_flag_flows_to_config():
    """--mpas-qv-smooth-del2-m2s round-trip (MPAS horizontal moisture
    smoothing, 2026-07-23 speckle fix); default byte-identical OFF."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.mpas_qv_smooth_del2_m2s == 0.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--grid-type", "voronoi", "--discretization", "mpas",
        "--mpas-qv-smooth-del2-m2s", "2e5",
    ]), parser))
    assert cfg.mpas_qv_smooth_del2_m2s == 2.0e5
    # validate_strict bounds/lane guards live in test_mpas_qv_smoothing.


def test_snow_albedo_ageing_flags_flow_to_config():
    """--snow-age-activation-K and --land-snow-tau-days round-trip.

    Both exist because every snow-covered polar cell measured the fully-aged
    albedo 0.521 against an observed 0.70-0.82, and an arm moving the activation
    temperature ALONE left it unchanged: the e-folding time is the binding
    parameter and used to be a hardcoded calibrated constant no run could
    select.  Neither had a round-trip test before.
    """
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.snow_age_activation_K is None
    assert cfg_default.land_snow_tau_days == 150.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--snow-age-activation-K", "5000",
        "--land-snow-tau-days", "60",
    ]), parser))
    assert cfg.snow_age_activation_K == 5000.0
    assert cfg.land_snow_tau_days == 60.0
    cfg.validate_strict()


def test_land_soil_freeze_thaw_round_trip_and_decks():
    """The flag reaches ExperimentConfig both ways, and each deck states its
    value explicitly: production ON (user 2026-09-26), the preserved old deck
    OFF so it still reproduces the arms it exists for."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--use-multilayer-land"]
    cfg0 = build_config_from_args(_postprocess_args(parser.parse_args(base), parser))
    assert cfg0.land_soil_freeze_thaw is False
    cfg1 = build_config_from_args(_postprocess_args(parser.parse_args(
        base + ["--land-soil-freeze-thaw"]), parser))
    assert cfg1.land_soil_freeze_thaw is True
    cfg2 = build_config_from_args(_postprocess_args(parser.parse_args(
        base + ["--no-land-soil-freeze-thaw"]), parser))
    assert cfg2.land_soil_freeze_thaw is False
    for deck, want in (("amip_production.yaml", True),
                       ("amip_sundqvist_l36.yaml", False)):
        p = build_arg_parser()
        rows = load_yaml_config(str(_repo_root() / "config" / "amip" / deck), p)
        assert rows.get("land_soil_freeze_thaw") is want, deck
        p.set_defaults(**rows)
        cfg = build_config_from_args(_postprocess_args(
            p.parse_args(_AMIP_DUMMY_PATHS), p))
        assert cfg.land_soil_freeze_thaw is want, deck


def test_land_soil_ice_impedance_exponent_round_trip_deck_and_range():
    """CLM5 e_ice reaches ExperimentConfig, production states it explicitly
    (user 2026-09-28), and a value outside [0, 10] is refused."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--use-multilayer-land"]
    cfg0 = build_config_from_args(_postprocess_args(parser.parse_args(base), parser))
    assert cfg0.land_soil_ice_impedance_exponent == 6.0
    cfg1 = build_config_from_args(_postprocess_args(parser.parse_args(
        base + ["--land-soil-ice-impedance-exponent", "3.5"]), parser))
    assert cfg1.land_soil_ice_impedance_exponent == 3.5
    p = build_arg_parser()
    rows = load_yaml_config(str(_repo_root() / "config" / "amip" / "amip_production.yaml"), p)
    assert rows.get("land_soil_ice_impedance_exponent") == 6.0
    for bad in ("-1", "11", "nan"):
        cfg = build_config_from_args(_postprocess_args(parser.parse_args(
            base + ["--land-soil-ice-impedance-exponent", bad]), parser))
        with pytest.raises(ValueError, match="land_soil_ice_impedance_exponent"):
            cfg.validate_strict()
    slab = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--land-soil-ice-impedance-exponent", "3"]), parser))
    with pytest.raises(ValueError, match="requires use_multilayer_land"):
        slab.validate_strict()


def test_land_soil_freeze_thaw_without_multilayer_land_is_refused():
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--land-soil-freeze-thaw"]), parser))
    with pytest.raises(ValueError, match="land_soil_freeze_thaw"):
        cfg.validate_strict()


def test_land_canopy_flags_round_trip_and_production_pin():
    """--land-canopy-stress-b0 / --land-canopy-interception reach
    ExperimentConfig both ways, and the production deck pins the current
    values (b0 stressed, interception off)."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--use-multilayer-land",
            "--land-surface-scheme", "two_leaf"]
    cfg0 = build_config_from_args(_postprocess_args(parser.parse_args(base), parser))
    assert cfg0.land_canopy_stress_b0 is True
    assert cfg0.land_canopy_interception is False
    cfg1 = build_config_from_args(_postprocess_args(parser.parse_args(
        base + ["--no-land-canopy-stress-b0", "--land-canopy-interception"]),
        parser))
    assert cfg1.land_canopy_stress_b0 is False
    assert cfg1.land_canopy_interception is True
    try:   # flat analytical topography fails validation for its own reason
        cfg1.validate_strict()
    except ValueError as e:
        assert "land_canopy" not in str(e)
    p = build_arg_parser()
    rows = load_yaml_config(
        str(_repo_root() / "config" / "amip" / "amip_production.yaml"), p)
    assert rows.get("land_canopy_stress_b0") is True
    assert rows.get("land_canopy_interception") is False


@pytest.mark.parametrize("flag", ["--land-canopy-interception",
                                  "--no-land-canopy-stress-b0"])
def test_land_canopy_flags_without_two_leaf_are_refused(flag):
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--use-multilayer-land",
         "--land-surface-scheme", "simple_seb", flag]), parser))
    with pytest.raises(ValueError, match="two_leaf"):
        cfg.validate_strict()


@pytest.mark.parametrize("bad", ["0.1", "500"])
def test_land_snow_tau_days_out_of_range_is_refused(bad):
    """0.5 d is melting spring snow and 400 d spans the cold plateau; outside
    that the value is not a snow-ageing timescale."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-snow-tau-days", bad]), parser))
    with pytest.raises(ValueError, match="land_snow_tau_days"):
        cfg.validate_strict()


def test_snow_albedo_responds_to_the_ageing_timescale():
    """The parameter must be load-bearing, not merely present: at a fixed snow
    age, lengthening the e-folding time has to RAISE the albedo, and at the
    calibrated 3.674 days a two-month-old pack has to sit on its floor -- which
    is the defect that motivated the knob."""
    import jax.numpy as jnp
    from legoesm.surface_albedo import LandAlbedoConfig, snow_albedo
    age_s = jnp.asarray([60.0 * 86400.0])
    short = LandAlbedoConfig(tau_snow_decay=3.674 * 86400.0)
    long_ = LandAlbedoConfig(tau_snow_decay=60.0 * 86400.0)
    a_short = float(snow_albedo(age_s, short)[0])
    a_long = float(snow_albedo(age_s, long_)[0])
    assert a_short == pytest.approx(short.alpha_snow_min, abs=1e-3)
    assert a_long > a_short + 0.05


def test_mpas_qv_biharmonic_flag_flows_to_config():
    """--mpas-qv-smooth-del4-m4s round-trip (scale-selective companion to the
    Laplacian, 2026-09-11); default OFF, and the two coefficients are
    independent so an arm can move one without the other."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.mpas_qv_smooth_del4_m4s == 0.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--grid-type", "voronoi", "--discretization", "mpas",
        "--mpas-qv-smooth-del4-m4s", "3.6e14",
    ]), parser))
    assert cfg.mpas_qv_smooth_del4_m4s == 3.6e14
    assert cfg.mpas_qv_smooth_del2_m2s == 0.0


def test_mpas_land_beta_soil_flag_flows_to_config():
    """--mpas-land-beta-soil round-trip (#1312 phase 2b traced beta_soil);
    default byte-identical OFF, --no- form revertible from a YAML True."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.mpas_land_beta_soil is False

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--grid-type", "voronoi", "--discretization", "mpas",
        "--use-multilayer-land", "--mpas-land-beta-soil",
    ]), parser))
    assert cfg.mpas_land_beta_soil is True

    parser2 = build_arg_parser()
    parser2.set_defaults(mpas_land_beta_soil=True)   # simulates a YAML pin
    cfg_off = build_config_from_args(_postprocess_args(parser2.parse_args([
        "--dataset", "analytical", "--no-mpas-land-beta-soil",
    ]), parser2))
    assert cfg_off.mpas_land_beta_soil is False
    # validate_strict inert-corner guards live in
    # test_mpas_multilayer_land_port (refusal without multilayer land).
def test_land_surface_scheme_validate_strict_rejects_unknown():
    """validate_strict() rejects an unknown surface scheme (dispatch hardening —
    a typo must fail early, not silently fall through in model_driver)."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    with pytest.raises(ValueError, match="land_surface_scheme"):
        cfg._replace(land_surface_scheme="two_leff").validate_strict()


def test_orbital_insolation_flag_flows_to_config():
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.orbital_insolation is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--orbital-insolation",
    ]), parser))
    assert cfg_on.orbital_insolation is True


def test_convective_cloud_boolean_optional_action_can_disable_config_default():
    """--no-convective-cloud must override a config-file-enabled default.

    convective_cloud is BooleanOptionalAction (not store_true): a YAML
    ``--config`` (e.g. amip_production.yaml) sets convective_cloud=True via
    parser.set_defaults, and a store_true flag could never turn that back OFF
    from the CLI. This guards the paired --convective-cloud / --no-convective-cloud
    behaviour AND the set_defaults(True) + --no-... override that the AMIP-optimal
    (prescribed-SST) run relies on.
    """
    parser = build_arg_parser()
    # default (no flag): OFF, and flows to the config
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.convective_cloud is False

    # explicit --convective-cloud: ON (backward-compatible with the old store_true)
    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convective-cloud",
    ]), parser))
    assert cfg_on.convective_cloud is True

    # the NEW capability: a config default of True (as amip_production.yaml sets)
    # can be turned OFF with --no-convective-cloud (impossible under store_true).
    parser2 = build_arg_parser()
    parser2.set_defaults(convective_cloud=True)          # simulates the YAML default
    assert parser2.parse_args(["--dataset", "analytical"]).convective_cloud is True
    cfg_off = build_config_from_args(_postprocess_args(parser2.parse_args([
        "--dataset", "analytical", "--no-convective-cloud",
    ]), parser2))
    assert cfg_off.convective_cloud is False


def test_build_config_includes_joint_physics_parameterization_flags():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--radiation", "rrtmgp",
        "--convection", "mass_flux",
        "--turbulence", "louis",
        "--gravity-wave-drag", "rayleigh",
        "--time-var", "month",
        "--lat-var", "ylat",
        "--lon-var", "xlon",
        "--held-suarez-forcing",
        "--physics-parameterization", "ml",
        "--physics-parameterization-checkpoint", "physics.eqx",
        "--physics-parameterization-stats", "physics_stats.npz",
        "--physics-parameterization-hidden-dim", "192",
        "--physics-parameterization-layers", "4",
        "--physics-parameterization-seed", "7",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.radiation == "rrtmg"
    assert cfg.convection == "mass_flux"
    assert cfg.turbulence == "louis"
    assert cfg.gravity_wave_drag == "rayleigh"
    assert cfg.time_var == "month"
    assert cfg.lat_var == "ylat"
    assert cfg.lon_var == "xlon"
    assert cfg.held_suarez_forcing is True
    assert cfg.physics_parameterization == "ml"
    assert cfg.physics_parameterization_checkpoint == "physics.eqx"
    assert cfg.physics_parameterization_stats == "physics_stats.npz"
    assert cfg.physics_parameterization_hidden_dim == 192
    assert cfg.physics_parameterization_layers == 4
    assert cfg.physics_parameterization_seed == 7


def test_enable_latlon_spmd_flag_flows_to_config():
    """--enable-latlon-spmd round-trips into ExperimentConfig (default off)."""
    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--time-var", "month",
            "--lat-var", "ylat", "--lon-var", "xlon"]
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(base), parser))
    assert cfg_off.enable_latlon_spmd is False

    cfg_on = build_config_from_args(_postprocess_args(
        parser.parse_args(base + ["--enable-latlon-spmd"]), parser))
    assert cfg_on.enable_latlon_spmd is True


def test_latlon_spmd_compiled_segments_flag_flows_to_config():
    """--latlon-spmd-compiled-segments (M2b) round-trips into
    ExperimentConfig (default off), and validate_strict rejects it without
    --enable-latlon-spmd (silent-no-op hardening) while accepting the pair."""
    import pytest

    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--time-var", "month",
            "--lat-var", "ylat", "--lon-var", "xlon"]
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(base), parser))
    assert cfg_off.latlon_spmd_compiled_segments is False

    cfg_on = build_config_from_args(_postprocess_args(
        parser.parse_args(base + ["--enable-latlon-spmd", "--grid-type",
                                  "latlon", "--latlon-spmd-compiled-segments"]),
        parser))
    assert cfg_on.latlon_spmd_compiled_segments is True
    assert cfg_on.enable_latlon_spmd is True
    cfg_on.validate_strict()                       # valid pair passes

    cfg_orphan = build_config_from_args(_postprocess_args(
        parser.parse_args(base + ["--latlon-spmd-compiled-segments"]), parser))
    assert cfg_orphan.latlon_spmd_compiled_segments is True
    with pytest.raises(ValueError, match="requires\\s+enable_latlon_spmd"):
        cfg_orphan.validate_strict()


def test_convection_cli_choices_match_config_single_source():
    """--convection CLI choices MUST equal the driver config's authoritative
    VALID_CONVECTION_SCHEMES.  Regression guard: a stale hardcoded CLI choices
    list ({none,sbm,dca,kuo,mass_flux,edmf}) once rejected --convection tiedtke
    while ExperimentConfig.validate_strict accepted it, so every tiedtke AMIP
    job died at argparse."""
    from legoesm.driver.config import VALID_CONVECTION_SCHEMES

    parser = build_arg_parser()
    conv = next(a for a in parser._actions if a.dest == "convection")
    assert set(conv.choices) == set(VALID_CONVECTION_SCHEMES)
    # the profile-prognostic schemes wired in #477 must be selectable
    for scheme in ("tiedtke", "bechtold", "zhang_mcfarlane",
                   "kain_fritsch", "emanuel"):
        assert scheme in conv.choices


def test_convection_tiedtke_parses_and_flows_to_config():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--convection", "tiedtke"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.convection == "tiedtke"


def test_build_config_includes_surfdata_path():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--land-mask-file", "sftlf.nc",
        "--surfdata", "legoesm_surfdata.nc",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.land_mask_path == "sftlf.nc"
    assert cfg.surfdata_path == "legoesm_surfdata.nc"


def test_build_config_surfdata_defaults_empty():
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.surfdata_path == ""


def test_surfdata_without_land_mask_warns(capsys):
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--surfdata", "sd.nc"])
    _postprocess_args(args, parser)
    assert "ignored without --land-mask-file" in capsys.readouterr().out


def test_joint_parameterization_requires_mass_flux_and_louis():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--physics-parameterization", "ml",
        "--physics-parameterization-checkpoint", "physics.eqx",
        "--physics-parameterization-stats", "physics_stats.npz",
    ])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)


def test_precision_flag_flows_to_config_and_validates():
    parser = build_arg_parser()
    for mode in ("fp32", "fp64", "mixed"):
        args = parser.parse_args(["--dataset", "analytical",
                                  "--precision", mode])
        args = _postprocess_args(args, parser)
        cfg = build_config_from_args(args)
        assert cfg.precision == mode
        cfg.validate_strict()  # membership-validated scheme Literal


def test_precision_default_is_fp32():
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]),
                             parser)
    assert build_config_from_args(args).precision == "fp32"


def test_precision_unknown_mode_rejected():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "analytical", "--precision", "bf16"])


def test_spectral_postprocess_promotes_gaussian_grid():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--discretization", "spectral",
        "--truncation", "42",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.grid.grid_type == "gaussian"
    assert cfg.grid.resolution == 42
    assert cfg.dycore.discretization == "spectral"


def test_ic_era5_threads_through_to_config():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--ic", "era5",
        "--ic-path", "/tmp/era5_test.zarr",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.ic == "era5"
    assert cfg.ic_path == "/tmp/era5_test.zarr"


def test_ic_default_is_backward_compatible():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.ic == "default"
    assert cfg.ic_path == ""


def test_ic_era5_without_ic_path_fails():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--ic", "era5"])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)


# ---------------------------------------------------------------------------
# --aerosol-ccn grid gating: MPAS is now wired (Phase C); spectral standalone
# is still blocked.
# ---------------------------------------------------------------------------

_AEROSOL_CCN_BASE = [
    "--dataset", "analytical",
    "--aerosol-ccn",
    "--aerosol-forcing", "external",
    "--aerosol-file", "/tmp/aer.nc",   # external forcing requires a file
    "--microphysics", "morrison",
]


def test_aerosol_ccn_allowed_on_mpas():
    parser = build_arg_parser()
    args = parser.parse_args(_AEROSOL_CCN_BASE + [
        "--grid-type", "voronoi",
        "--discretization", "mpas",
    ])
    # Must NOT raise now that the MPAS combined-physics path fills the
    # specified-Nc field.
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.nc_from_aerosol is True


def test_aerosol_ccn_still_blocked_on_spectral():
    parser = build_arg_parser()
    args = parser.parse_args(_AEROSOL_CCN_BASE + [
        "--discretization", "spectral",
    ])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)


def test_aerosol_ccn_requires_external_forcing():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--aerosol-ccn",
        "--microphysics", "morrison", "--grid-type", "voronoi",
        "--discretization", "mpas",
    ])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)
def test_surface_tiled_flags_flow_to_config():
    """--surface-tiled / --surface-z0-land round-trip into ExperimentConfig
    and validate together with --slab-land-active + --turbulence louis."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--land-mask-file", "land_mask.nc",
        "--turbulence", "louis",
        "--surface-bulk-scheme", "coare3",
        "--slab-land-active",
        "--surface-tiled",
        "--surface-z0-land", "0.15",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.surface_tiled is True
    assert cfg.surface_z0_land == 0.15
    assert cfg.slab_land_active is True
    assert cfg.turbulence == "louis"
    # The full combination must be self-consistent under strict validation.
    assert cfg.validate_strict() is None


def test_surface_tiled_defaults_off():
    """Without the flag, tiling is off (byte-identical legacy behaviour)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.surface_tiled is False


def test_surface_tiled_unsupported_turbulence_rejected():
    """--surface-tiled with a scheme that cannot consume the injected tiled
    surface_flux tuple (e.g. holtslag_boville) fails strict validation."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "holtslag_boville",
        "--slab-land-active",
        "--surface-tiled",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="surface_tiled.*louis"):
        cfg.validate_strict()


@pytest.mark.parametrize("scheme", ["louis", "clubb_lite", "clubb"])
def test_surface_tiled_accepts_flux_consuming_schemes(scheme):
    """--surface-tiled validates with EVERY kernel that accepts the injected
    tiled surface_flux=(tau_x, tau_y, shflx, lhflx, ustar) BC: louis and the
    CLUBB family (clubb routes it through clubb_step's kinematic interface)."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--land-mask-file", "land_mask.nc",
        "--turbulence", scheme,
        "--surface-bulk-scheme", "coare3",
        "--slab-land-active",
        "--surface-tiled",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.turbulence == scheme
    assert cfg.validate_strict() is None


def test_soil_bucket_flags_flow_to_config():
    """--land-soil-bucket and its parameters round-trip into ExperimentConfig
    and validate together with an active land tile."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--land-mask-file", "land_mask.nc",
        "--turbulence", "louis",
        "--slab-land-active",
        "--land-soil-bucket",
        "--land-bucket-w-max", "120.0",
        "--land-beta-min", "0.2",
        "--land-bucket-w-init-frac", "0.4",
        "--land-k-infiltration", "3.3e-6",
        "--land-infil-suction-boost", "1.5",
        "--no-land-infiltration-excess",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.land_soil_bucket is True
    assert cfg.land_bucket_w_max == 120.0
    assert cfg.land_beta_min == 0.2
    assert cfg.land_bucket_w_init_frac == 0.4
    assert cfg.land_K_infiltration == 3.3e-6
    assert cfg.land_infil_suction_boost == 1.5
    assert cfg.land_infiltration_excess is False
    assert cfg.validate_strict() is None


def test_infiltration_params_reject_nan_and_negative():
    """NaN/negative infiltration params fail strict validation (NaN-safe guards:
    a bare ``x < 0`` would let NaN slip through and poison the infiltration cap)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--land-mask-file", "land_mask.nc", "--slab-land-active", "--land-soil-bucket"]),
        parser)
    base = build_config_from_args(args)
    assert base.validate_strict() is None          # baseline is valid
    for bad in (float("nan"), -1.0, 0.0):
        with pytest.raises(ValueError, match="land_K_infiltration"):
            base._replace(land_K_infiltration=bad).validate_strict()
    for bad in (float("nan"), -0.5):
        with pytest.raises(ValueError, match="land_infil_suction_boost"):
            base._replace(land_infil_suction_boost=bad).validate_strict()


def test_soil_bucket_defaults_off():
    """Without the flag, the bucket is off (byte-identical legacy behaviour)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.land_soil_bucket is False


def test_soil_bucket_requires_active_land_rejected():
    """--land-soil-bucket without an active land tile fails strict validation."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--land-soil-bucket",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="land_soil_bucket.*land tile"):
        cfg.validate_strict()


def test_land_stomatal_beta_flag_flows_to_config():
    """--land-stomatal-beta round-trips and validates with the bucket on."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--land-mask-file", "land_mask.nc",
        "--turbulence", "louis",
        "--slab-land-active",
        "--land-soil-bucket",
        "--land-stomatal-beta",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.land_stomatal_beta is True
    assert cfg.validate_strict() is None


def test_land_stomatal_beta_defaults_off():
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    assert build_config_from_args(args).land_stomatal_beta is False


def test_land_stomatal_beta_requires_bucket_rejected():
    """--land-stomatal-beta without the bucket (no beta_soil) is rejected."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--slab-land-active",
        "--land-stomatal-beta",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="land_stomatal_beta.*land_soil_bucket"):
        cfg.validate_strict()


def test_soil_bucket_rejects_bad_beta_min():
    """beta_min outside (0, 1] is rejected."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--slab-land-active",
        "--land-soil-bucket",
        "--land-beta-min", "1.5",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="land_beta_min"):
        cfg.validate_strict()


def test_bechtold_cape_threshold_flows_to_config():
    """--bechtold-cape-threshold round-trips into ExperimentConfig and the
    Bechtold convection config (the coarse-resolution precip-deficit lever)."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--convection", "bechtold",
        "--bechtold-cape-threshold", "10.0",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.bechtold_cape_threshold == 10.0
    assert cfg.validate_strict() is None


def test_bechtold_rain_vapor_sink_flows_to_config_and_kernel():
    """The rain vapour-sink selector threads CLI -> ExperimentConfig ->
    BechtoldConfig; the default is the formation-local debit and the legacy
    vapour-mass spread stays selectable for the A/B; validate_strict refuses
    anything else."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--convection", "bechtold"]), parser))
    assert cfg.bechtold_rain_vapor_sink == "formation"
    cfg_legacy = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--convection", "bechtold",
         "--bechtold-rain-vapor-sink", "vapour_mass"]), parser))
    assert cfg_legacy.bechtold_rain_vapor_sink == "vapour_mass"
    with pytest.raises(ValueError, match="bechtold_rain_vapor_sink"):
        cfg._replace(bechtold_rain_vapor_sink="bogus").validate_strict()
    from legoesm.driver.physics_pipeline import _resolve_convection
    assert _resolve_convection(cfg)[1].rain_vapor_sink == "formation"
    assert _resolve_convection(cfg_legacy)[1].rain_vapor_sink == "vapour_mass"


def test_bechtold_subsidence_solve_flows_to_config():
    """--bechtold-subsidence-solve round-trips into ExperimentConfig (the
    day-65 blowup-bisect stability escape hatch); unset matches the
    BechtoldConfig default (byte-identical); unknown value rejected at
    the parser (choices) and by validate_strict membership."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--convection", "bechtold",
        "--bechtold-subsidence-solve", "advective",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.bechtold_subsidence_solve == "advective"
    assert cfg.validate_strict() is None

    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.bechtold_subsidence_solve == "implicit_flux"
    with pytest.raises(SystemExit):
        parser.parse_args(["--bechtold-subsidence-solve", "bogus"])
    with pytest.raises(ValueError, match="bechtold_subsidence_solve"):
        cfg._replace(bechtold_subsidence_solve="bogus").validate_strict()


def test_bechtold_cape_threshold_defaults_to_scheme_default():
    """Unset → matches BechtoldConfig.cape_threshold (byte-identical)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.bechtold_cape_threshold == 70.0


def test_bechtold_cape_threshold_rejects_negative():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--bechtold-cape-threshold", "-5.0",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="bechtold_cape_threshold"):
        cfg.validate_strict()


def test_issue484_new_amip_flags_flow_to_config():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--hyperdiff-scale", "1.25",
        "--div-damp-scale", "0.75",
        "--no-conservation-fixer",
        "--no-fix-mass",
        "--max-wallclock-seconds", "7200",
        "--restart-buffer-seconds", "900",
        "--checkpoint-format", "zarr",
        "--seed", "123",
        "--forcing-update-days", "2.5",
        "--solar-s0", "1362.5",
        "--tau-equator", "8.1",
        "--tau-pole", "2.2",
        "--unfused-radiation",
        "--rrtmgp-gpoint-batch-size", "16",
        "--volcanic-aerosol-lw",
        "--t-ice-k", str(constants.T_freeze_ocean + 0.25),
        "--albedo-ice", "0.7",
        "--albedo-ocean", "0.08",
        "--sfc-emissivity", "0.96",
        "--emissivity-ice", "0.94",
        "--k-bl-max-per-day", "1.5",
        "--k-free-per-day", "0.2",
        "--aerosol-forcing", "external",
        "--aerosol-file", "/dummy/aero.nc",   # external forcing requires a file
        "--microphysics", "morrison",
        "--nc-from-aerosol",
        "--a-h-scale", "0.25",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    # a_h_scale is the SECOND-ORDER Laplacian viscosity multiplier. It is not
    # scale-selective (damps as k^2) so it reaches the baroclinic eddies that
    # drive the midlatitude jet; component_factory records "crushing the
    # midlatitude eddy-driven jets" at 16x the default. It was a DycoreConfig
    # field with no CLI flag until 2026-07-31.
    assert cfg.dycore.a_h_scale == 0.25
    assert cfg.dycore.hyperdiff_scale == 1.25
    assert cfg.dycore.div_damp_scale == 0.75
    assert cfg.dycore.conservation_fixer is False
    assert cfg.dycore.fix_mass is False
    assert cfg.output.max_wallclock_seconds == 7200
    assert cfg.output.restart_buffer_seconds == 900
    assert cfg.output.checkpoint_format == "zarr"
    assert cfg.seed == 123
    assert cfg.forcing_update_days == 2.5
    assert cfg.S_0 == 1362.5
    assert cfg.tau_equator == 8.1
    assert cfg.tau_pole == 2.2
    assert cfg.unfused_radiation is True
    assert cfg.rrtmgp_gpoint_batch_size == 16
    assert cfg.volcanic_aerosol_lw is True
    assert cfg.T_ice == constants.T_freeze_ocean + 0.25
    assert cfg.albedo_ice == 0.7
    assert cfg.albedo_ocean == 0.08
    assert cfg.sfc_emissivity == 0.96
    assert cfg.emissivity_ice == 0.94
    assert cfg.k_BL_max_per_day == 1.5
    assert cfg.k_free_per_day == 0.2
    assert cfg.nc_from_aerosol is True
    cfg.validate_strict()


def test_tuned_slab_knobs_flow_to_config():
    """The tuned air-sea + cloud knobs (mirroring run_coupled) round-trip into
    ExperimentConfig so AMIP can run with the tuned slab parameters; the defaults
    keep the prior AMIP behaviour (constant / 0 / None / off)."""
    parser = build_arg_parser()
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.surface_bulk_scheme == "constant"
    # None = scheme-native gustiness (coare3: 600 m AeroBulk default, others
    # off).  With the default "constant" scheme this is still off.
    assert d.surface_gustiness_zi is None
    assert d.cloud_q_c_diagnostic is None
    assert d.cloud_rh_crit is None
    assert d.convective_cloud is False

    c = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--surface-bulk-scheme", "coare3",
        "--gustiness-zi", "300",
        "--q-c-diagnostic", "3e-4",
        "--rh-crit", "0.8",
        "--convective-cloud",
    ]), parser))
    assert c.surface_bulk_scheme == "coare3"
    assert c.surface_gustiness_zi == 300.0
    assert c.cloud_q_c_diagnostic == pytest.approx(3e-4)
    assert c.cloud_rh_crit == 0.8
    assert c.convective_cloud is True


def test_aimip_classical_checkpoint_flag():
    """--aimip-classical-checkpoint round-trips (default None); the trained-param
    injection itself is validated end-to-end by the AMIP run."""
    parser = build_arg_parser()
    assert parser.parse_args(["--dataset", "analytical"]).aimip_classical_checkpoint is None
    a = parser.parse_args(["--dataset", "analytical",
                           "--aimip-classical-checkpoint", "x/epoch_0019.eqx"])
    assert a.aimip_classical_checkpoint == "x/epoch_0019.eqx"


def _serialise_aimip_defaults(tmp_path):
    """Write a default AIMIPClassicalParams checkpoint for the override tests."""
    import equinox as eqx
    from legoesm.training.aimip_params import AIMIPClassicalParams

    ckpt = tmp_path / "epoch_defaults.eqx"
    eqx.tree_serialise_leaves(str(ckpt), AIMIPClassicalParams.from_defaults())
    return str(ckpt)


def test_aimip_classical_overrides_force_sundqvist_microphysics(tmp_path):
    """The AIMIP-classical override forces the trained scheme set AND turns on
    sundqvist microphysics when none was requested — without a precip sink,
    tiedtke detrains condensate into q_c with no removal (CWV water trap)."""
    ckpt = _serialise_aimip_defaults(tmp_path)
    parser = build_arg_parser()
    # explicit --microphysics none is the water-trap case the override repairs
    # (the CLI default is now sundqvist under the full-physics policy)
    args = parser.parse_args(["--dataset", "analytical", "--microphysics", "none",
                              "--aimip-classical-checkpoint", ckpt])
    assert args.microphysics == "none"
    out = _apply_aimip_classical_overrides(args)
    # the full classical scheme set is forced on...
    assert out.convection == "tiedtke"
    assert out.turbulence == "louis"
    assert out.gravity_wave_drag == "mcfarlane"
    assert out.clouds == "xu_randall"
    # ...and microphysics is promoted none -> sundqvist (closes the budget)
    assert out.microphysics == "sundqvist"
    assert out._aimip_params is not None
    # the trained sundqvist leaves build a real config
    sq = out._aimip_params.to_sundqvist_config()
    assert float(sq.auto_rate) > 0.0
    # the forcing reaches the BUILT ExperimentConfig — this is the field every
    # physics builder (incl. the MPAS/spectral run()-time rebuild) reads, so the
    # water sink closes on every backend even if the trained leaves only inject
    # onto the finite-volume PhysicsPipeline.
    parser2 = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(out, parser2))
    assert cfg.microphysics == "sundqvist"


# --- full-physics policy: AMIP must never run a parameterization slot 'none' ---

def test_amip_default_physics_all_active():
    """The DEFAULT AMIP config has every parameterization active (no 'none') —
    convection/microphysics/turbulence/gravity_wave_drag/clouds + radiation."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    for slot in ("convection", "microphysics", "turbulence",
                 "gravity_wave_drag"):
        assert getattr(cfg, slot) != "none", f"{slot} defaulted to none"
    assert cfg.cloud_scheme != "none"   # config field for the --clouds arg
    assert cfg.radiation in ("gray", "rrtmg", "rrtmgp")  # never none


def test_amip_default_passes_full_physics_guard():
    """The default args satisfy the guard (no SystemExit)."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    _require_full_physics_for_amip(args, parser)   # no raise


def test_amip_rejects_disabled_physics_slot():
    """A 'none' slot without an escape flag fails LOUDLY (SystemExit), and the
    message names the offending slot."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--microphysics", "none"])
    with pytest.raises(SystemExit):
        _require_full_physics_for_amip(args, parser)
    # also catches turbulence / gwd / clouds / convection
    for slot, flag in [("turbulence", "--turbulence"),
                       ("gravity_wave_drag", "--gravity-wave-drag"),
                       ("clouds", "--clouds"), ("convection", "--convection")]:
        a = parser.parse_args(["--dataset", "analytical", flag, "none"])
        with pytest.raises(SystemExit):
            _require_full_physics_for_amip(a, parser)


def test_amip_allow_disabled_physics_escape():
    """--allow-disabled-physics permits a 'none' slot (idealized/dry run)."""
    parser = build_arg_parser()
    assert parser.parse_args(
        ["--dataset", "analytical"]).allow_disabled_physics is False
    args = parser.parse_args(["--dataset", "analytical", "--microphysics", "none",
                              "--allow-disabled-physics"])
    _require_full_physics_for_amip(args, parser)   # no raise


def _all_none_args(parser, *extra):
    return parser.parse_args(
        ["--dataset", "analytical", "--convection", "none", "--microphysics",
         "none", "--turbulence", "none", "--gravity-wave-drag", "none",
         "--clouds", "none", *extra])


def test_amip_held_suarez_fully_dry_exempt():
    """A fully-dry Held-Suarez run bypasses the guard without the escape flag."""
    parser = build_arg_parser()
    _require_full_physics_for_amip(_all_none_args(parser, "--held-suarez-forcing"),
                                   parser)         # no raise


def test_amip_spmd_alone_not_exempt():
    """--enable-latlon-spmd is NOT exempt on its own: an all-param-'none' SPMD run
    still leaves radiation active (the SPMD driver rejects it), so it must go
    through Held-Suarez or --allow-disabled-physics.  Bare SPMD -> SystemExit;
    Held-Suarez SPMD and --allow-disabled-physics SPMD both pass."""
    parser = build_arg_parser()
    bare = _all_none_args(parser, "--grid-type", "latlon", "--enable-latlon-spmd")
    with pytest.raises(SystemExit):
        _require_full_physics_for_amip(bare, parser)
    hs_spmd = _all_none_args(parser, "--grid-type", "latlon",
                             "--enable-latlon-spmd", "--held-suarez-forcing")
    _require_full_physics_for_amip(hs_spmd, parser)        # no raise
    allowed = _all_none_args(parser, "--grid-type", "latlon",
                             "--enable-latlon-spmd", "--allow-disabled-physics")
    _require_full_physics_for_amip(allowed, parser)        # no raise


def test_amip_mixed_held_suarez_partial_none_still_rejected():
    """Held-Suarez exempts only a FULLY-dry stack — HS with microphysics='none'
    but default tiedtke convection / xu_randall clouds is the dangerous mixed
    case (tiedtke condensate with no precip sink) and must still fail."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--microphysics", "none",
                              "--held-suarez-forcing"])
    # convection/turbulence/gwd/clouds stay at their active defaults -> mixed
    with pytest.raises(SystemExit):
        _require_full_physics_for_amip(args, parser)


def test_aimip_classical_overrides_explicit_sundqvist_keeps_trained(tmp_path):
    """Passing --microphysics sundqvist explicitly keeps sundqvist and still
    carries the trained params (the injection gate fires on resolved scheme)."""
    ckpt = _serialise_aimip_defaults(tmp_path)
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical",
                              "--microphysics", "sundqvist",
                              "--aimip-classical-checkpoint", ckpt])
    out = _apply_aimip_classical_overrides(args)
    assert out.microphysics == "sundqvist"
    assert out._aimip_params is not None


def test_aimip_classical_overrides_warn_mpas_backend(tmp_path, capsys):
    """On the SUPPORTED MPAS backend (voronoi grid + --discretization mpas) the
    schemes are still forced on (water sink closes via config.microphysics) but
    a loud warning fires that the trained leaves only apply on the FV path."""
    ckpt = _serialise_aimip_defaults(tmp_path)
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--grid-type", "voronoi",
                              "--discretization", "mpas",
                              "--aimip-classical-checkpoint", ckpt])
    out = _apply_aimip_classical_overrides(args)
    assert out.microphysics == "sundqvist"   # water sink still closes
    assert out.convection == "tiedtke"
    assert "WARNING" in capsys.readouterr().out


def test_aimip_classical_overrides_spectral_refused(tmp_path):
    """Spectral refuses tiedtke (profile-prognostic carry not threaded, #405) so
    it would crash deep in setup before the microphysics sink runs — the override
    fails early with a clear SystemExit instead of an honest-looking warning."""
    ckpt = _serialise_aimip_defaults(tmp_path)
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical",
                              "--discretization", "spectral",
                              "--aimip-classical-checkpoint", ckpt])
    with pytest.raises(SystemExit, match="spectral"):
        _apply_aimip_classical_overrides(args)


def test_aimip_classical_overrides_respect_explicit_microphysics(tmp_path):
    """An explicit prognostic microphysics (morrison) is NOT overridden to
    sundqvist — the user's choice wins and its config is left for the pipeline
    (the trained sundqvist leaves only apply to sundqvist)."""
    ckpt = _serialise_aimip_defaults(tmp_path)
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical",
                              "--microphysics", "morrison",
                              "--aimip-classical-checkpoint", ckpt])
    out = _apply_aimip_classical_overrides(args)
    assert out.microphysics == "morrison"
    assert out.convection == "tiedtke"


def test_aimip_classical_overrides_noop_without_flag():
    """No checkpoint -> no scheme forcing; scheme fields are left untouched."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    # snapshot the VALUES before the call (out is args, so comparing references
    # post-call would be vacuous): immutable-string snapshots prove no mutation.
    before = (args.convection, args.turbulence, args.gravity_wave_drag,
              args.clouds, args.microphysics)
    out = _apply_aimip_classical_overrides(args)
    assert out._aimip_params is None
    assert (out.convection, out.turbulence, out.gravity_wave_drag,
            out.clouds, out.microphysics) == before


def test_max_wallclock_seconds_threads_to_config():
    """--max-wallclock-seconds must wire into OutputConfig.max_wallclock_seconds."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--max-wallclock-seconds", "41400",
        "--checkpoint-days", "30",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.output.max_wallclock_seconds == 41400.0
    assert cfg.output.checkpoint_days == 30


def test_slurm_ntasks_1_does_not_trigger_distributed(monkeypatch):
    """SLURM_NTASKS=1 (set in every sbatch job) must NOT set distributed=True.
    Regression guard: prior bug made all single-task GPU sbatch jobs enter the
    MPI path and crash on SingleRankLayout.ownership."""
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.delenv("PMI_SIZE", raising=False)
    monkeypatch.delenv("MPI_LOCALNRANKS", raising=False)
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    assert not args.distributed, "SLURM_NTASKS=1 must not trigger distributed mode"


def test_slurm_ntasks_4_triggers_distributed(monkeypatch):
    """SLURM_NTASKS>1 (real multi-task MPI job) MUST set distributed=True."""
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.delenv("PMI_SIZE", raising=False)
    monkeypatch.delenv("MPI_LOCALNRANKS", raising=False)
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    assert args.distributed, "SLURM_NTASKS=4 must trigger distributed mode"


# ---------------------------------------------------------------------------
# G4a: _print_forcing_activity summary
# ---------------------------------------------------------------------------

def _capture_forcing_activity(extra_args: list[str]) -> str:
    """Parse args, run _print_forcing_activity, return captured stdout."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"] + extra_args)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        _print_forcing_activity(args)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_forcing_activity_all_channels_active_rrtmg():
    """All external forcing channels should print ACTIVE with rrtmg."""
    out = _capture_forcing_activity([
        "--radiation", "rrtmg",
        "--ghg-forcing", "external", "--ghg-file", "/tmp/ghg.nc",
        "--ozone-forcing", "external", "--ozone-file", "/tmp/o3.nc",
        "--aerosol-forcing", "external", "--aerosol-file", "/tmp/aer.nc",
        "--volcanic-aerosol-file", "/tmp/vol.nc",
        "--solar-source", "file", "--solar-file", "/tmp/solar.nc",
    ])
    assert "SST/SIC" in out and "ACTIVE" in out
    # Each radiation-gated channel should be ACTIVE
    assert out.count("ACTIVE") >= 4
    assert "inert" not in out


def test_forcing_activity_gray_marks_channels_inert():
    """With gray radiation, GHG/ozone/aerosol channels must be inert."""
    out = _capture_forcing_activity([
        "--radiation", "gray",
        "--ghg-forcing", "external", "--ghg-file", "/tmp/ghg.nc",
        "--ozone-forcing", "external", "--ozone-file", "/tmp/o3.nc",
    ])
    # GHG and ozone are radiation-gated → inert under gray
    assert "inert" in out or "gray radiation" in out


def test_forcing_activity_solar_constant_label():
    """Default solar (constant) should be labeled as constant, not ACTIVE file."""
    out = _capture_forcing_activity(["--radiation", "rrtmg"])
    assert "constant S_0" in out or "constant" in out


def test_slab_land_active_flag_threads_to_config():
    """--slab-land-active must reach ExperimentConfig.slab_land_active."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--slab-land-active"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.slab_land_active is True


def test_slab_land_active_default_is_false():
    """slab_land_active must default to False (passive land is the safe default)."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.slab_land_active is False


def test_gustiness_zi_threads_to_config():
    """--gustiness-zi and --surface-bulk-scheme must reach ExperimentConfig."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--surface-bulk-scheme", "coare3",
        "--gustiness-zi", "300",
        "--turbulence", "holtslag_boville",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.surface_bulk_scheme == "coare3"
    assert cfg.surface_gustiness_zi == 300.0


@pytest.mark.parametrize("value", ["3e-4", "1e-5"])
def test_q_c_diagnostic_threads_to_config(value: str):
    """--q-c-diagnostic must reach ExperimentConfig.cloud_q_c_diagnostic.

    ``1e-5`` is the sub-production condensate-floor rung the AMIP campaign needs
    to test coupled: it exercises the whole CLI route (parse -> postprocess ->
    build -> validate_strict), not just the bound tuple.
    """
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--q-c-diagnostic", value,
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.cloud_q_c_diagnostic == pytest.approx(float(value))
    cfg.validate_strict()


def test_gustiness_defaults_scheme_native():
    """--gustiness-zi unset = None = scheme-native (AeroBulk parity): off for
    the default "constant" scheme, 600 m built-in for coare3; explicit 0
    forces off.  cloud_q_c_diagnostic stays None = CloudConfig default."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.surface_gustiness_zi is None
    assert cfg.cloud_q_c_diagnostic is None
    args0 = parser.parse_args(["--dataset", "analytical", "--gustiness-zi", "0"])
    cfg0 = build_config_from_args(_postprocess_args(args0, parser))
    assert cfg0.surface_gustiness_zi == 0.0


def test_bulk_thermo_convention_flag_flows_to_config():
    """--bulk-thermo-convention must reach
    ExperimentConfig.surface_thermo_convention (#762): default "legoesm"
    (constant L_v / dry c_pd, byte-identical); "aerobulk" = NEMO/AeroBulk
    parity, and passes the validate_strict membership check."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.surface_thermo_convention == "legoesm"

    args = parser.parse_args([
        "--dataset", "analytical",
        "--surface-bulk-scheme", "coare3",
        "--turbulence", "holtslag_boville",
        "--bulk-thermo-convention", "aerobulk",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.surface_thermo_convention == "aerobulk"
    cfg.validate_strict()


def test_cloud_tuning_flags_thread_to_config():
    """--rh-crit / --cloud-conv-cloud-max reach ExperimentConfig and pass
    strict validation within their bounds."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--rh-crit", "0.83",
        "--cloud-conv-cloud-max", "0.25",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.cloud_rh_crit == pytest.approx(0.83)
    assert cfg.cloud_conv_cloud_max == pytest.approx(0.25)
    assert cfg.validate_strict() is None


def test_cloud_tuning_flags_default_none():
    """Unset cloud-tuning knobs stay None (scheme defaults, byte-identical)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.cloud_rh_crit is None
    assert cfg.cloud_conv_cloud_max is None


def test_rh_crit_out_of_bounds_rejected():
    """An rh_crit outside (0.5, 0.99) must fail strict validation."""
    parser = build_arg_parser()
    args = _postprocess_args(
        parser.parse_args(["--dataset", "analytical", "--rh-crit", "1.5"]), parser)
    cfg = build_config_from_args(args)
    with pytest.raises((ValueError, AssertionError)):
        cfg.validate_strict()


def test_conv_cloud_condensate_flag_resolves_to_cloudconfig():
    """--conv-cloud-condensate round-trips into ExperimentConfig and resolves
    onto the hot-loop anvil CloudConfig.conv_cloud_condensate; unset leaves the
    scheme default (1.5e-4) => byte-identical anvil optics."""
    from legoesm.atmosphere.physics.clouds.config import (
        CloudConfig,
        build_cloud_config,
    )
    parser = build_arg_parser()
    # Explicit flag => ExperimentConfig scalar => resolved (hot-loop) CloudConfig.
    args = _postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--conv-cloud-condensate", "3e-5",
    ]), parser)
    cfg = build_config_from_args(args)
    assert cfg.cloud_conv_cloud_condensate == pytest.approx(3e-5)
    assert cfg.validate_strict() is None
    resolved = build_cloud_config(
        cfg.cloud_scheme,
        conv_cloud_condensate=cfg.cloud_conv_cloud_condensate)
    assert resolved.conv_cloud_condensate == pytest.approx(3e-5)
    # Unset => None scalar => CloudConfig keeps its default anvil condensate.
    cfg_def = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_def.cloud_conv_cloud_condensate is None
    resolved_def = build_cloud_config(
        cfg_def.cloud_scheme,
        conv_cloud_condensate=cfg_def.cloud_conv_cloud_condensate)
    default_condensate = CloudConfig._field_defaults["conv_cloud_condensate"]
    assert default_condensate == pytest.approx(1.5e-4)  # documented anvil default
    assert resolved_def.conv_cloud_condensate == pytest.approx(default_condensate)


def test_conv_cloud_condensate_out_of_bounds_rejected():
    """A conv_cloud_condensate outside (1e-5, 1e-3) must fail strict validation."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--conv-cloud-condensate", "1e-2",
    ]), parser)
    cfg = build_config_from_args(args)
    with pytest.raises((ValueError, AssertionError)):
        cfg.validate_strict()


def test_subgrid_autoconv_flag_threads_to_config():
    """--subgrid-autoconv round-trips into ExperimentConfig (#613)."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--microphysics", "morrison",
        "--subgrid-autoconv",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.subgrid_autoconversion is True
    assert cfg.microphysics == "morrison"


def test_subgrid_autoconv_defaults_off():
    """Sub-grid warm-rain closure is opt-in (byte-identical legacy default)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.subgrid_autoconversion is False


def test_convective_cloud_flag_threads_to_config():
    """--convective-cloud round-trips into ExperimentConfig (parity with the
    CMIP slab config config/cmip/cmip_ocean_slab.yaml)."""
    parser = build_arg_parser()
    args = _postprocess_args(
        parser.parse_args(["--dataset", "analytical", "--convective-cloud"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.convective_cloud is True


def test_convective_cloud_defaults_off():
    """Convective cloud cover is opt-in (byte-identical legacy default)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.convective_cloud is False


# --- --config YAML loader (authoritative AMIP production config) --------------

def _repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


# Machine-path flags the YAML deliberately omits (supplied at runtime); dummy
# values are fine — config construction does not stat the paths.
_AMIP_DUMMY_PATHS = [
    "--ic-path", "/dummy/era5.zarr", "--forcing-path", "/dummy/sst.nc",
    "--sic-path", "/dummy/sic.nc", "--solar-file", "/dummy/solar.nc",
    "--ozone-file", "/dummy/o3.nc", "--ghg-file", "/dummy/ghg.nc",
    "--aerosol-file", "/dummy/aero.nc", "--volcanic-aerosol-file", "/dummy/volc.nc",
    "--topography", "/dummy/etopo.nc", "--days", "10", "--output", "/dummy/out",
]


def _amip_config_yamls():
    """Every shipped config/amip/*.yaml — so a typo'd key in ANY of them (not just
    amip_production) is caught, incl. amip_sota.yaml and future configs."""
    return sorted((_repo_root() / "config" / "amip").glob("*.yaml"))


@pytest.mark.parametrize("cfg_file", _amip_config_yamls(),
                         ids=lambda p: p.name)
def test_config_yaml_loads_all_keys_are_valid_dests(cfg_file):
    """Every key in each shipped AMIP config is a real run_amip dest — a typo'd /
    dropped override is a hard error (dispatch-hardening)."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    defaults = load_yaml_config(str(cfg_file), parser)  # raises on unknown key
    assert defaults  # non-empty
    valid_dests = {a.dest for a in parser._actions}
    assert set(defaults).issubset(valid_dests)


def test_amip_sota_config_builds_valid_experiment_config():
    """config/amip/amip_sota.yaml (SOTA: multilayer land + conv-cloud-off)
    builds a valid ExperimentConfig — the SOTA knobs are consistent (e.g.
    multilayer land waives the slab-bucket requirement for stomata; morrison +
    external aerosol stay on as the aerosol_ccn prereqs)."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    cfg_file = _repo_root() / "config" / "amip" / "amip_sota.yaml"
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(str(cfg_file), parser))
    args = _postprocess_args(parser.parse_args(
        _AMIP_DUMMY_PATHS + ["--clm-surfdata-path", "/dummy/surfdata.nc",
                             "--land-mask-file", "/dummy/lsm.nc"]), parser)
    cfg = build_config_from_args(args)
    cfg.validate_strict()  # raises if the SOTA combo is inconsistent
    assert cfg.use_multilayer_land is True
    # aerosol_ccn is DISABLED in the shipped SOTA config (#745): as wired the
    # indirect effect is ~15x too strong (-24 W/m^2 vs IPCC -1 to -1.7); it
    # returns after the Twomey/lifetime split + autoconv calibration (#730).
    # The prereqs (morrison + external aerosol forcing) stay on.
    assert cfg.nc_from_aerosol is False
    assert cfg.microphysics == "morrison"
    assert cfg.aerosol_forcing == "external"
    assert cfg.convective_cloud is False
    assert cfg.convection == "sbm"


def test_config_yaml_round_trips_authoritative_values():
    """`run_amip.py --config config/amip/amip_production.yaml` reproduces the
    production AMIP parametrization.

    Re-baselined 2026-09-23 (user decision): the production atmospheric physics
    suite is CAM6 -- the CAM 32-level hybrid table, CLUBB cloud fraction,
    the Zhang-McFarlane convection port, orographic-only gravity waves, the
    CESM Large-Yeager surface fluxes and the 1800 s physics step with three
    macro/micro sub-steps.  The outgoing Sundqvist/L36 deck is asserted
    separately below so both remain pinned.

    The grid keys are recipe-sensitive together, so a change here is a
    stability A/B, not an edit."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    cfg_file = _repo_root() / "config" / "amip" / "amip_production.yaml"
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(str(cfg_file), parser))
    args = _postprocess_args(parser.parse_args(_AMIP_DUMMY_PATHS), parser)
    # resolution/nlev/discretization are CLI dests baked into cfg.grid, so
    # assert them at the args level the YAML controls.
    assert args.resolution == 6
    assert args.nlev == 32                 # CAM6 L32 hybrid table
    assert args.vertical_coord == "cam_l32"
    assert args.discretization == "mpas"
    # The deck spells the mesh "voronoi"; the parser normalises the family's
    # spellings to one name, so assert the resolved value the run uses.
    assert args.grid_type == "mpas"
    assert args.dt == 112.5
    cfg = build_config_from_args(args)
    assert cfg.convection == "zhang_mcfarlane"   # CAM6 zm_conv port
    # CAM6 f09 L32 (non-WACCM) runs OROGRAPHIC drag only: build-namelist sets
    # use_gw_front and use_gw_convect_dp false below 60 levels.
    assert cfg.gravity_wave_drag == "mcfarlane"
    assert cfg.microphysics == "morrison"        # MG2 port is run 2
    assert cfg.cloud_scheme == "cam6_clubb"
    assert cfg.use_clubb_cloud_fraction is True
    assert cfg.radiation == "rrtmg"              # rrtmgp builder alias
    assert cfg.turbulence == "clubb"
    assert cfg.clubb_prognostic is True
    assert cfg.surface_bulk_scheme == "large_yeager_cesm"
    # CAM6 cadence: physics every 1800 s, radiation hourly, CLUBB+micro
    # sub-cycled three times at 600 s inside each physics step.
    assert args.physics_update_steps == 16
    assert args.rad_update_steps == 32
    assert args.cld_macmic_num_steps == 3
    # No tiled surface on this lane -- the tiled port is open work.
    assert cfg.surface_tiled is False
    assert cfg.start_year == 1979
    # convective_cloud is a real scientific lever, not boilerplate: it was
    # flipped OFF on 2026-08-22 because the tropical-rain runs that reproduced
    # observed ocean rain all ran with it off.  cam6_clubb refuses it anyway
    # (deepcu is CAM6's own deep-convective cloud term), so it stays pinned.
    assert cfg.convective_cloud is False
    assert cfg.surface_gustiness_zi == 300.0
    assert cfg.convective_precip_efficiency == pytest.approx(0.8)
    # User table 2026-09-24, "production now, matches CAM6", three rows set
    # EXPLICITLY so nothing rests on a code default (the defaults are the
    # opposite of all three): the closure's liquid handed to cloud water; the
    # post-step supersaturation drain off (CAM clubb_do_liqsupersat=.false.);
    # and the drain's ice-curve blend off.  Plus the two floors the partition's
    # own guard forces to zero.  Pinned THROUGH the driver on every config the
    # lever reaches, because a row that lands on ExperimentConfig but not on
    # the nested scheme config would report success while changing nothing.
    assert cfg.clubb_liquid_partition is True
    assert cfg.hard_saturation_adjustment is False
    assert cfg.hard_sat_ice_curve is False
    assert cfg.cloud_q_c_diagnostic == 0.0
    assert cfg.cloud_conv_cloud_condensate == 0.0
    from legoesm.driver.physics_pipeline import (
        _resolve_microphysics, turbulence_config_for,
    )
    from legoesm.atmosphere.physics.turbulence.integration import (
        materialize_sub_config,
    )
    assert materialize_sub_config(
        turbulence_config_for(cfg)).liquid_partition is True
    assert _resolve_microphysics(cfg)[1].liquid_from_closure is True
    # User decision 2026-09-25 (run 4): NO in-cloud inhomogeneity thinning,
    # matching CAM6 whose RRTMG applies no such factor.  "constant" with
    # factor 1.0 is the no-thinning path.  Pinned on the RESOLVED cloud
    # config and on its EFFECT: the liquid path handed to radiation equals
    # the raw grid-mean path, which two_region (the old row) would thin.
    assert cfg.cloud_optics_inhomogeneity == "constant"
    assert cfg.cloud_inhomogeneity_factor == 1.0
    _assert_optics_thinning(cfg, thinned=False)


def _assert_optics_thinning(cfg, *, thinned: bool):
    """Build the lane's cloud config from ``cfg`` and measure the thinning.

    A saturated, cloudy, warm column: with no inhomogeneity factor the liquid
    water path handed to radiation is EXACTLY ``q_c * dp / g``; two_region at
    fsd 1.0 thins it.  Asserting the effect, not the field, is what makes the
    production pin non-vacuous against a renamed or ignored selector.
    """
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties,
    )
    # The MPAS lane's own builder (model_driver._run_mpas), so the pin reads
    # the config the run would radiate with, not the YAML.
    from legoesm.driver.model_driver import _standalone_cloud_config
    cloud_cfg = _standalone_cloud_config(
        cfg, cfg.cloud_scheme, allow_convective_cloud=True)
    assert cloud_cfg.cloud_partial_coverage_optics == "none"
    T = jnp.full((1, 4), 285.0)
    p_full = jnp.array([[70000.0, 80000.0, 90000.0, 95000.0]])
    dp = jnp.full((1, 4), 5000.0)
    q_v = jnp.full((1, 4), 8.0e-3)
    q_c = jnp.full((1, 4), 4.0e-4)
    p_half = jnp.array([[65000.0, 75000.0, 85000.0, 92500.0, 97500.0]])
    props = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp, config=cloud_cfg, q_cloud=q_c,
        q_ice=jnp.zeros_like(q_c),
        cloud_fraction_override=jnp.full((1, 4), 0.5),
        # cam6_clubb's tropopause switch needs these; harmless to the others.
        lat=jnp.zeros((1,)), p_half=p_half)
    raw = q_c * dp / constants.g
    ratio = float(jnp.sum(props.lwp) / jnp.sum(raw))
    if thinned:
        assert ratio < 0.95, f"expected two_region thinning, got ratio {ratio}"
    else:
        assert ratio == 1.0, f"expected NO thinning, got ratio {ratio}"


def test_sundqvist_l36_deck_still_round_trips():
    """The outgoing production deck stays RUNNABLE and unchanged under its new
    name, so every result measured against it reproduces -- including
    cam6_base60, the paired baseline of the CAM6 run-1 comparison.  It must
    carry NEITHER numerical lever, which is what makes it that baseline."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    cfg_file = _repo_root() / "config" / "amip" / "amip_sundqvist_l36.yaml"
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(str(cfg_file), parser))
    args = _postprocess_args(parser.parse_args(_AMIP_DUMMY_PATHS), parser)
    assert args.resolution == 6
    assert args.nlev == 36
    assert args.vertical_coord == "sigma"
    assert args.discretization == "mpas"
    assert args.dt == 112.5
    assert args.mpas_vert_advection_scheme == "van_leer"
    cfg = build_config_from_args(args)
    # The deck is silent on the damper, so the CLI dest stays None and the
    # CODE default is what the run gets.  Assert the RESOLVED value, which is
    # what makes this deck the neither-lever baseline.
    assert cfg.dycore.mpas_div_damp4_scale == 0.0
    assert cfg.convection == "bechtold"
    assert cfg.cloud_scheme == "sundqvist"
    assert cfg.gravity_wave_drag == "mcfarlane+e3sm_cam"
    assert cfg.surface_bulk_scheme == "coare3"
    assert cfg.cloud_rh_crit == pytest.approx(0.85)
    assert cfg.cloud_q_c_diagnostic == pytest.approx(5e-6)
    # This deck RUNS PROGNOSTIC CLUBB, so the 2026-09-24 three-row decision
    # could reach it -- it must not.  The campaign pair's pre-registration
    # rests on this deck resolving to the OPPOSITE of all three rows, through
    # the driver, on every config the lever reaches.
    assert cfg.turbulence == "clubb" and cfg.clubb_prognostic is True
    assert cfg.clubb_liquid_partition is False
    assert cfg.hard_saturation_adjustment is True
    assert cfg.hard_sat_ice_curve is True
    from legoesm.driver.physics_pipeline import (
        _resolve_microphysics, turbulence_config_for,
    )
    from legoesm.atmosphere.physics.turbulence.integration import (
        materialize_sub_config,
    )
    assert materialize_sub_config(
        turbulence_config_for(cfg)).liquid_partition is False
    assert _resolve_microphysics(cfg)[1].liquid_from_closure is False
    # Old configuration KEEPS the in-cloud inhomogeneity thinning (user
    # decision 2026-09-25): two_region at fsd 1.0, and it measurably thins.
    assert cfg.cloud_optics_inhomogeneity == "two_region"
    assert cfg.cloud_fsd == 1.0
    _assert_optics_thinning(cfg, thinned=True)


def test_config_yaml_explicit_cli_flag_overrides_file():
    """Precedence: an explicit CLI flag wins over the --config file default."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    cfg_file = _repo_root() / "config" / "amip" / "amip_production.yaml"
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(str(cfg_file), parser))
    args = _postprocess_args(
        parser.parse_args(_AMIP_DUMMY_PATHS + ["--convection", "bechtold"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.convection == "bechtold"


def test_params_flag_parses():
    parser = build_arg_parser()
    args = parser.parse_args(_AMIP_DUMMY_PATHS + ["--params", "x.yaml"])
    assert args.params == "x.yaml"


def test_params_calibration_applies_to_atm_experimentconfig(tmp_path):
    """A --params calibration entry (registry qualified name) applies to the
    flattened ExperimentConfig scalar via the atm scalar-param map — the same
    path run_amip.main() takes (issue #691)."""
    from legoesm.driver.run_config_yaml import (
        apply_params_to_config,
        build_atm_scalar_param_map,
        load_params_config,
    )
    parser = build_arg_parser()
    cfg = build_config_from_args(
        _postprocess_args(parser.parse_args(_AMIP_DUMMY_PATHS), parser))
    p = tmp_path / "params.yaml"
    p.write_text("atm.clouds.CloudConfig.q_c_diagnostic: 3.0e-4\n")
    out = apply_params_to_config(
        cfg, load_params_config(str(p)), driver="run_amip",
        scalar_param_map=build_atm_scalar_param_map())
    assert out.cloud_q_c_diagnostic == 3.0e-4


def test_aimip_louis_preserves_resolved_surface_scheme():
    """The AIMIP Louis injection must KEEP the run-resolved surface bulk_scheme +
    gustiness (coare3/300) rather than reverting to to_louis_config's default
    constant surface — the clobber that silently made --surface-bulk-scheme a
    no-op on every AIMIP run (anemic evaporation, hfls ~6 vs ~88)."""
    from legoesm.atmosphere.physics.turbulence.config import LouisConfig, SurfaceLayerConfig

    from scripts.run.run_amip import _louis_with_preserved_surface
    # trained Louis carries a DEFAULT (constant) surface, exactly as
    # to_louis_config() builds it from the trained Cd/Ch/z0:
    trained = LouisConfig(surface=SurfaceLayerConfig(Cd_neutral=1.5e-3))
    assert trained.surface.bulk_scheme == "constant"
    # _resolve_turbulence had already applied coare3 + gustiness 300:
    resolved = LouisConfig(surface=SurfaceLayerConfig(
        bulk_scheme="coare3", gustiness_w_zi=300.0,
        z_ref_model_level=False, ocean_q_sfc_saline=True))
    out = _louis_with_preserved_surface(trained, resolved)
    assert out.surface.bulk_scheme == "coare3"        # preserved, not clobbered
    assert out.surface.gustiness_w_zi == 300.0
    assert out.surface.Cd_neutral == 1.5e-3           # trained Cd/Ch/z0 kept
    # the run-resolved surface switches survive too (they were dropped, so
    # --no-surface-z-ref-model-level never reached the trained lane); the
    # values are chosen AGAINST the scheme defaults so a drop is visible
    assert out.surface.z_ref_model_level is False
    assert out.surface.ocean_q_sfc_saline is True
    # the production shape (codex round 2): the trained Louis config pins the
    # height switch False (AIMIP substitutes air T for a missing surface) and
    # the run-resolved surface carries the scheme default True -- the AMIP
    # lane has a real surface temperature, so the resolved value must win
    from legoesm.training.aimip_params import AIMIPClassicalParams
    trained_real = AIMIPClassicalParams.from_defaults().to_louis_config()
    assert trained_real.surface.z_ref_model_level is False
    resolved_default = LouisConfig(surface=SurfaceLayerConfig(bulk_scheme="coare3"))
    assert _louis_with_preserved_surface(trained_real, resolved_default).surface.z_ref_model_level is True


def test_aimip_louis_preserve_surface_noop_without_prev():
    """No prior turbulence config (e.g. turbulence was none) -> trained Louis
    returned unchanged."""
    from legoesm.atmosphere.physics.turbulence.config import LouisConfig, SurfaceLayerConfig

    from scripts.run.run_amip import _louis_with_preserved_surface
    trained = LouisConfig(surface=SurfaceLayerConfig())
    assert _louis_with_preserved_surface(trained, None) is trained


def test_sundqvist_tuning_flags_round_trip_and_override():
    """--sundqvist-{qc-crit,rh-crit,auto-rate} parse and override a
    SundqvistConfig with final precedence; unset knobs stay at the base."""
    from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig

    from scripts.run.run_amip import _apply_sundqvist_overrides
    parser = build_arg_parser()
    d = parser.parse_args(["--dataset", "analytical"])
    assert (d.sundqvist_qc_crit, d.sundqvist_rh_crit, d.sundqvist_auto_rate) == (
        None, None, None)
    args = parser.parse_args(["--dataset", "analytical",
                              "--sundqvist-qc-crit", "1e-4",
                              "--sundqvist-rh-crit", "0.6"])
    base = SundqvistConfig()                       # qc_crit 5e-4, rh_crit 0.8
    out = _apply_sundqvist_overrides(base, args)
    assert out.qc_crit == 1e-4 and out.rh_crit == 0.6
    assert out.auto_rate == base.auto_rate         # untouched knob unchanged


def test_sundqvist_overrides_noop_without_flags():
    """No override flags -> the SAME config object (identity)."""
    from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig

    from scripts.run.run_amip import _apply_sundqvist_overrides
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    base = SundqvistConfig()
    assert _apply_sundqvist_overrides(base, args) is base


def test_sundqvist_overrides_noop_for_non_sundqvist_micro():
    """A sundqvist override is ignored when microphysics != sundqvist."""
    from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig

    from scripts.run.run_amip import _apply_sundqvist_overrides
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical",
                              "--microphysics", "morrison",
                              "--sundqvist-qc-crit", "1e-4"])
    base = SundqvistConfig()
    assert _apply_sundqvist_overrides(base, args) is base


def test_sundqvist_flags_reject_out_of_bounds():
    """Out-of-range tunables fail LOUDLY (argparse accepts any float)."""
    from scripts.run.run_amip import _validate_sundqvist_flags
    parser = build_arg_parser()
    # (positive values only — argparse parses a leading '-' as a flag)
    for flag, bad in [("--sundqvist-qc-crit", "1.0"),      # >> 1.5e-3
                      ("--sundqvist-rh-crit", "0.2"),       # < 0.5
                      ("--sundqvist-auto-rate", "1e-5")]:   # < 1e-4 floor
        args = parser.parse_args(["--dataset", "analytical", flag, bad])
        with pytest.raises(SystemExit):
            _validate_sundqvist_flags(args, parser)


def test_sundqvist_flags_in_bounds_ok():
    from scripts.run.run_amip import _validate_sundqvist_flags
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical",
                              "--sundqvist-qc-crit", "1e-4",
                              "--sundqvist-rh-crit", "0.6",
                              "--sundqvist-auto-rate", "5e-3"])
    _validate_sundqvist_flags(args, parser)   # no raise


def test_sundqvist_flags_rejected_on_mpas_spectral():
    """The overrides are refused on backends that rebuild MicrophysicsConfig at
    run() (MPAS/spectral) and would silently ignore them."""
    from scripts.run.run_amip import _validate_sundqvist_flags
    parser = build_arg_parser()
    mpas = parser.parse_args(["--dataset", "analytical", "--grid-type", "voronoi",
                              "--sundqvist-qc-crit", "1e-4"])
    with pytest.raises(SystemExit):
        _validate_sundqvist_flags(mpas, parser)
    # but no override flags -> no raise even on MPAS
    bare = parser.parse_args(["--dataset", "analytical", "--grid-type", "voronoi"])
    _validate_sundqvist_flags(bare, parser)


def test_evaluate_flags_flow_to_config():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--cmip-output",
        "--evaluate",
        "--evaluation-suite", "Tier1_sanity_checks", "Tier2_atmosphere_monthly",
        "--evaluation-model-id", "legoESM-1-0-test",
        "--evaluation-experiment-id", "amip",
        "--evaluation-variant-id", "r1i1p1f1",
        "--evaluation-data-root-dir", "/tmp/climateeval_data",
        "--evaluation-timerange", "19790101/19791231",
        "--evaluation-fail-missing",
        "--evaluation-download",
        "--evaluation-climateeval-python", "/tmp/climateeval_env/bin/python",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    ev = cfg.output.evaluation
    assert ev.enabled is True
    assert ev.suites == ("Tier1_sanity_checks", "Tier2_atmosphere_monthly")
    assert ev.model_id == "legoESM-1-0-test"
    assert ev.data_root_dir == "/tmp/climateeval_data"
    assert ev.timerange == "19790101/19791231"
    assert ev.fail_on_missing_data is True
    assert ev.download_missing_data is True
    assert ev.climateeval_python == "/tmp/climateeval_env/bin/python"


def test_evaluate_defaults_off():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.output.evaluation.enabled is False


def test_evaluate_default_suites_empty_means_all_tiers():
    """With --evaluate but no --evaluation-suite, suites is empty — the runner
    then discovers + runs ALL bundled suites (every tier)."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--cmip-output", "--evaluate",
        "--evaluation-climateeval-python", sys.executable,
        "--evaluation-data-root-dir", "/tmp",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.output.evaluation.suites == ()


def test_evaluate_requires_cmip_output_rejected():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--evaluate"])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)


def test_evaluate_climateeval_python_unset_rejected():
    """--evaluate with no climateeval_python configured fails LOUDLY at
    validate_strict, before the (multi-hour) run ever starts."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--cmip-output", "--evaluate",
        "--evaluation-data-root-dir", "/tmp",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.output.evaluation.climateeval_python == ""
    with pytest.raises(ValueError, match="climateeval_python"):
        cfg.validate_strict()


def test_evaluate_climateeval_python_nonexistent_rejected():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--cmip-output", "--evaluate",
        "--evaluation-climateeval-python", "/no/such/interpreter",
        "--evaluation-data-root-dir", "/tmp",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="climateeval_python"):
        cfg.validate_strict()


def test_evaluate_data_root_dir_unset_rejected(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--cmip-output", "--evaluate",
        "--evaluation-climateeval-python", sys.executable,
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="data_root_dir"):
        cfg.validate_strict()


def test_evaluate_valid_climateeval_config_passes(tmp_path):
    """A real, executable interpreter + a real directory clears validate_strict."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--cmip-output", "--evaluate",
        "--evaluation-climateeval-python", sys.executable,
        "--evaluation-data-root-dir", str(tmp_path),
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.validate_strict() is None


def test_evaluate_climateeval_python_env_var_default(monkeypatch):
    monkeypatch.setenv("LEGOESM_CLIMATEEVAL_PYTHON", "/env/climateeval/bin/python")
    monkeypatch.setenv("LEGOESM_CLIMATEEVAL_DATA_ROOT", "/env/climateeval_data")
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--cmip-output", "--evaluate"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.output.evaluation.climateeval_python == "/env/climateeval/bin/python"
    assert cfg.output.evaluation.data_root_dir == "/env/climateeval_data"


def test_cloud_sensitivity_flags_round_trip_and_validate():
    """--cloud-p-xr / --cloud-alpha-xr thread into ExperimentConfig; out-of-range
    values fail validate_strict (the xu_randall cloud-fraction sensitivity knobs
    for flattening the moisture-driven overcast runaway)."""
    parser = build_arg_parser()
    d = parser.parse_args(["--dataset", "analytical"])
    assert d.cloud_p_xr is None and d.cloud_alpha_xr is None
    args = _postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--cloud-p-xr", "0.7",
         "--cloud-alpha-xr", "20"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.cloud_p_xr == 0.7 and cfg.cloud_alpha_xr == 20.0
    # unset -> None (byte-identical: CloudConfig default preserved)
    base = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert base.cloud_p_xr is None and base.cloud_alpha_xr is None
    # both bounds enforced by validate_strict
    for flag, val in [("--cloud-p-xr", "5.0"), ("--cloud-alpha-xr", "5000")]:
        bad = build_config_from_args(_postprocess_args(parser.parse_args(
            ["--dataset", "analytical", flag, val]), parser))
        with pytest.raises(Exception):
            bad.validate_strict()


def test_cloud_sensitivity_flags_allowed_on_mpas_spectral():
    """#870 Phase 1 FLIPS the old rejection: --cloud-p-xr/--cloud-alpha-xr now
    REACH the standalone MPAS/spectral radiation (via
    model_driver._standalone_cloud_config reading the same experiment fields),
    so the guard must accept them on every backend — the pre-#870 hard
    rejection blocked a working feature with a false message."""
    from scripts.run.run_amip import _validate_cloud_sensitivity_flags
    parser = build_arg_parser()
    # MPAS + the flags: NO raise (they thread via _standalone_cloud_config).
    mpas = parser.parse_args(["--dataset", "analytical", "--grid-type", "voronoi",
                              "--cloud-p-xr", "0.7"])
    _validate_cloud_sensitivity_flags(mpas, parser)
    # And the values flow into ExperimentConfig on the MPAS path too.
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--grid-type", "voronoi",
         "--cloud-p-xr", "0.7", "--cloud-alpha-xr", "20.0"]), parser))
    assert cfg.cloud_p_xr == 0.7 and cfg.cloud_alpha_xr == 20.0
    # FV path unchanged: no raise with or without flags.
    _validate_cloud_sensitivity_flags(
        parser.parse_args(["--dataset", "analytical", "--cloud-p-xr", "0.7"]),
        parser)


def test_distributed_mode_flag_flows_to_config():
    """--distributed-mode {mpi,spmd} round-trips into ExperimentConfig
    (production cs_spmd is config-file-only without this flag)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.distributed_mode == "mpi"

    cfg_spmd = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--distributed", "--distributed-mode", "spmd",
    ]), parser))
    assert cfg_spmd.distributed is True
    assert cfg_spmd.distributed_mode == "spmd"
    # spmd is a validate_strict-legal combination on the default cube grid;
    # checkpoints + diagnostics are ALSO legal now (cs_spmd steps 5a-5c).
    cfg_spmd._replace(output=cfg_spmd.output._replace(
        diag_days=0, checkpoint_days=0)).validate_strict()

    # Unknown mode is an argparse-level refusal (choices).
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "analytical",
                           "--distributed-mode", "bogus"])


def test_moisture_flux_form_flag_flows_to_dycore_config():
    """#771: --moisture-flux-form must reach the DycoreConfig (which the
    component factory threads into CDGridPrimitiveEquationConfig). Default off;
    --no-moisture-flux-form explicit off."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.dycore.moisture_flux_form is False   # default off

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--moisture-flux-form",
    ]), parser))
    assert cfg_on.dycore.moisture_flux_form is True

    cfg_off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--no-moisture-flux-form",
    ]), parser))
    assert cfg_off.dycore.moisture_flux_form is False


def test_mpas_nu_vert4_t_flag_flows_to_dycore_config():
    """#930: --mpas-nu-vert4-t must reach the DycoreConfig (which the component
    factory threads into MPASPrimitiveEquationConfig.nu_vert4_T — the vertical
    2Δσ-checkerboard cure).  Production default is ON (nonzero); 0 disables."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.dycore.mpas_nu_vert4_T > 0.0   # cure on by default

    cfg_off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--mpas-nu-vert4-t", "0",
    ]), parser))
    assert cfg_off.dycore.mpas_nu_vert4_T == 0.0

    cfg_set = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--mpas-nu-vert4-t", "5e-6",
    ]), parser))
    assert cfg_set.dycore.mpas_nu_vert4_T == pytest.approx(5e-6)


def test_multicontroller_coordinator_flags_parse():
    """Route-B flags round-trip through the parser (they are RUN args consumed
    in main() for the jax.distributed bootstrap, not ExperimentConfig fields)."""
    parser = build_arg_parser()
    a = parser.parse_args(["--enable-latlon-spmd", "--multicontroller",
                           "--coordinator", "localhost:12455"])
    assert a.multicontroller is True
    assert a.coordinator == "localhost:12455"
    # Default: single-controller (both off).
    d = parser.parse_args(["--dataset", "analytical"])
    assert d.multicontroller is False
    assert d.coordinator is None


def test_multicontroller_requires_enable_latlon_spmd(capsys):
    """--multicontroller without --enable-latlon-spmd is refused in main()
    BEFORE any device work (it is only the route-B transport for that lane)."""
    from scripts.run.run_amip import main
    with pytest.raises(SystemExit):
        main(["--grid-type", "latlon", "--dataset", "analytical",
              "--multicontroller"])
    assert "requires --enable-latlon-spmd" in capsys.readouterr().err


def test_top_sponge_flags_flow_to_dycore_config():
    """#836: --sponge-coeff/--sponge-width-m/--sponge-shape/--sponge-scale-height-m
    round-trip into DycoreConfig; default sponge_coeff=0 keeps the sponge OFF."""
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.dycore.sponge_coeff == 0.0          # default OFF

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--sponge-coeff", "1.157e-5",
        "--sponge-width-m", "12000.0",
        "--sponge-shape", "sam_rational",
        "--sponge-scale-height-m", "8000.0",
    ]), parser))
    assert cfg_on.dycore.sponge_coeff == 1.157e-5
    assert cfg_on.dycore.sponge_width_m == 12000.0
    assert cfg_on.dycore.sponge_shape == "sam_rational"
    assert cfg_on.dycore.sponge_scale_height_m == 8000.0


def test_sb81_omega_conversion_flag_flows_to_dycore_config():
    """#1029 ω-side: --sb81-omega-conversion round-trips into DycoreConfig;
    default OFF (the SB81 conversion is opt-in until the #1029(b) lid
    treatment lands)."""
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.dycore.sb81_omega_conversion is False   # default OFF

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--sb81-omega-conversion",
    ]), parser))
    assert cfg_on.dycore.sb81_omega_conversion is True


def test_budget_ledger_flag_flows_to_output_config():
    """--budget-ledger (per-process budget attribution diagnostic)
    round-trips into OutputConfig; default OFF = byte-identical model."""
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.output.budget_ledger is False   # default OFF

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--budget-ledger",
    ]), parser))
    assert cfg_on.output.budget_ledger is True

    # BooleanOptionalAction: a YAML-true value stays CLI-overridable.
    cfg_neg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--no-budget-ledger",
    ]), parser))
    assert cfg_neg.output.budget_ledger is False


def test_convective_precip_efficiency_allows_bechtold():
    """--convective-precip-efficiency now round-trips for bechtold (shared
    split_convective_rain), not just tiedtke; a non-supporting scheme still
    errors."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--convection", "bechtold",
        "--convective-precip-efficiency", "0.6",
    ]), parser))
    assert cfg.convection == "bechtold"
    assert cfg.convective_precip_efficiency == 0.6

    # a scheme without the rain split is rejected at parse time
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical",
            "--convection", "kuo",
            "--convective-precip-efficiency", "0.6",
        ]), parser)


def test_bechtold_conv_top_pa_flows_to_config():
    """--bechtold-conv-top-pa round-trips into ExperimentConfig (the Bechtold
    plume-termination stability cap); default 15000 Pa (150 hPa)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.bechtold_conv_top_pa == 15000.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--bechtold-conv-top-pa", "12000",
    ]), parser))
    assert cfg.bechtold_conv_top_pa == 12000.0


def test_new_convection_knobs_validate_bounds():
    """validate_strict rejects out-of-range convective_precip_efficiency and
    a non-positive bechtold_conv_top_pa (codex audit MEDIUM)."""
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="convective_precip_efficiency"):
        ExperimentConfig(convective_precip_efficiency=1.5).validate_strict()
    with pytest.raises(ValueError, match="bechtold_conv_top_pa"):
        ExperimentConfig(bechtold_conv_top_pa=0.0).validate_strict()
    # in-range passes
    ExperimentConfig(convective_precip_efficiency=0.6,
                     bechtold_conv_top_pa=15000.0).validate_strict()


def test_convective_precip_split_round_trips():
    """--convective-precip-split + autoconv params round-trip into
    ExperimentConfig (the physical Sundqvist autoconversion selector)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.convective_precip_split == "constant"
    assert cfg_default.autoconv_q_c_crit == 5.0e-4
    assert cfg_default.autoconv_pe_max == 0.9

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--convection", "bechtold",
        "--convective-precip-split", "autoconversion",
        "--autoconv-q-c-crit", "8e-4",
        "--autoconv-pe-max", "0.8",
    ]), parser))
    assert cfg.convective_precip_split == "autoconversion"
    assert cfg.autoconv_q_c_crit == 8e-4
    assert cfg.autoconv_pe_max == 0.8

    # an unknown selector is rejected at parse time (argparse choices)
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--dataset", "analytical",
            "--convective-precip-split", "garbage",
        ])


def test_convective_precip_split_validate_bounds():
    """validate_strict rejects an unknown split scheme + out-of-range autoconv
    params; the in-range physical config passes."""
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="convective_precip_split"):
        ExperimentConfig(convective_precip_split="garbage").validate_strict()
    with pytest.raises(ValueError, match="autoconv_q_c_crit"):
        ExperimentConfig(autoconv_q_c_crit=0.0).validate_strict()
    with pytest.raises(ValueError, match="autoconv_pe_max"):
        ExperimentConfig(autoconv_pe_max=1.5).validate_strict()
    ExperimentConfig(convective_precip_split="autoconversion",
                     autoconv_q_c_crit=5.0e-4, autoconv_pe_max=0.9).validate_strict()


def test_bechtold_downdraft_round_trips_and_threads():
    """--bechtold-downdraft-evap/-alpha/-rh-min round-trip into ExperimentConfig
    and thread into the hot-loop BechtoldConfig (the marine humid-BL evaporation
    lever, #847)."""
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-downdraft-evap", "0.3",
        "--bechtold-downdraft-alpha", "0.5",
        "--bechtold-downdraft-rh-min", "0.6",
    ]), parser))
    assert (cfg.bechtold_downdraft_evap, cfg.bechtold_downdraft_alpha,
            cfg.bechtold_downdraft_rh_min) == (0.3, 0.5, 0.6)
    _fn, cc = _resolve_convection(cfg)
    assert (cc.downdraft_evap_efficiency, cc.downdraft_alpha,
            cc.downdraft_RH_min) == (0.3, 0.5, 0.6)
    # default keeps the weak BechtoldConfig defaults (byte-identical)
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    dcc = _resolve_convection(d)[1]
    assert (dcc.downdraft_evap_efficiency, dcc.downdraft_alpha,
            dcc.downdraft_RH_min) == (0.05, 0.3, 0.2)


def test_bechtold_downdraft_validate_bounds():
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="bechtold_downdraft_evap"):
        ExperimentConfig(bechtold_downdraft_evap=0.9).validate_strict()   # > 0.5
    with pytest.raises(ValueError, match="bechtold_downdraft_alpha"):
        ExperimentConfig(bechtold_downdraft_alpha=1.5).validate_strict()  # > 0.9
    with pytest.raises(ValueError, match="bechtold_downdraft_rh_min"):
        ExperimentConfig(bechtold_downdraft_rh_min=1.5).validate_strict()  # > 1.0
    ExperimentConfig(bechtold_downdraft_evap=0.3, bechtold_downdraft_alpha=0.5,
                     bechtold_downdraft_rh_min=0.6).validate_strict()


def test_bechtold_downdraft_transport_round_trips_and_threads():
    """--bechtold-downdraft-transport/-entrain-rate/-detrain-scale round-trip
    into ExperimentConfig and thread into the hot-loop BechtoldConfig (the
    marine-BL ventilation lever); default OFF is byte-identical."""
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-downdraft-transport",
        "--bechtold-downdraft-entrain-rate", "3e-4",
        "--bechtold-downdraft-detrain-scale", "500",
    ]), parser))
    assert cfg.bechtold_downdraft_transport is True
    assert cfg.bechtold_downdraft_entrain_rate == 3e-4
    assert cfg.bechtold_downdraft_detrain_scale_m == 500.0
    cc = _resolve_convection(cfg)[1]
    assert cc.downdraft_transport is True
    assert cc.downdraft_entrain_rate == 3e-4
    assert cc.downdraft_detrain_scale_m == 500.0
    # default OFF => byte-identical BechtoldConfig downdraft defaults
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    dcc = _resolve_convection(d)[1]
    assert dcc.downdraft_transport is False
    assert (dcc.downdraft_entrain_rate,
            dcc.downdraft_detrain_scale_m) == (3.0e-4, 700.0)  # IFS ENTRDD
    # --no- turns OFF a config-file default
    off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--no-bechtold-downdraft-transport"]), parser))
    assert off.bechtold_downdraft_transport is False


def test_bechtold_downdraft_transport_validate_bounds():
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="bechtold_downdraft_entrain_rate"):
        ExperimentConfig(
            bechtold_downdraft_entrain_rate=1e-2).validate_strict()   # > 2e-3
    with pytest.raises(ValueError, match="bechtold_downdraft_detrain_scale_m"):
        ExperimentConfig(
            bechtold_downdraft_detrain_scale_m=5000.0).validate_strict()  # > 3000
    ExperimentConfig(bechtold_downdraft_transport=True,
                     bechtold_downdraft_entrain_rate=3e-4,
                     bechtold_downdraft_detrain_scale_m=500.0).validate_strict()


def test_cloud_inhomogeneity_factor_round_trips():
    """--cloud-inhomogeneity-factor (Cahalan 1994 plane-parallel correction)
    round-trips into ExperimentConfig; default None => CloudConfig default."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--cloud-inhomogeneity-factor", "0.7",
    ]), parser))
    assert cfg.cloud_inhomogeneity_factor == 0.7
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.cloud_inhomogeneity_factor is None


def test_cloud_inhomogeneity_validate_bounds():
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="cloud_inhomogeneity_factor"):
        ExperimentConfig(cloud_inhomogeneity_factor=0.1).validate_strict()  # < 0.3
    with pytest.raises(ValueError, match="cloud_inhomogeneity_factor"):
        ExperimentConfig(cloud_inhomogeneity_factor=1.5).validate_strict()  # > 1.0
    ExperimentConfig(cloud_inhomogeneity_factor=0.7).validate_strict()


def test_cloud_optics_inhomogeneity_round_trips_and_threads():
    """--cloud-optics-inhomogeneity / --cloud-fsd round-trip into
    ExperimentConfig and thread into the hot-loop CloudConfig (the two_region
    sub-grid optic); default is 'constant' + None (byte-identical)."""
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--cloud-optics-inhomogeneity", "two_region", "--cloud-fsd", "0.8",
    ]), parser))
    assert cfg.cloud_optics_inhomogeneity == "two_region"
    assert cfg.cloud_fsd == 0.8
    cc = build_cloud_config(
        cfg.cloud_scheme,
        cloud_optics_inhomogeneity=cfg.cloud_optics_inhomogeneity,
        cloud_fsd=cfg.cloud_fsd)
    assert cc.cloud_optics_inhomogeneity == "two_region"
    assert cc.cloud_fsd == 0.8
    # default: 'constant' scheme + None fsd => CloudConfig defaults
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.cloud_optics_inhomogeneity == "constant"
    assert d.cloud_fsd is None
    dcc = build_cloud_config(d.cloud_scheme,
                             cloud_optics_inhomogeneity=d.cloud_optics_inhomogeneity)
    assert dcc.cloud_optics_inhomogeneity == "constant"
    assert dcc.cloud_fsd == 0.75


def test_cloud_optics_inhomogeneity_validate():
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="cloud_optics_inhomogeneity"):
        ExperimentConfig(cloud_optics_inhomogeneity="bogus").validate_strict()
    with pytest.raises(ValueError, match="cloud_fsd"):
        ExperimentConfig(cloud_fsd=1.5).validate_strict()   # > 1.0
    ExperimentConfig(cloud_optics_inhomogeneity="two_region",
                     cloud_fsd=0.75).validate_strict()


def test_cloud_saturation_scheme_round_trips_and_threads():
    """--cloud-saturation-scheme round-trips into ExperimentConfig and threads
    into the hot-loop CloudConfig (the cloud-fraction RH saturation curve,
    #1521 ice-saturation fix); 'liquid' = legacy byte-identical, no longer
    the default."""
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--cloud-saturation-scheme", "mixed_phase",
    ]), parser))
    assert cfg.cloud_saturation_scheme == "mixed_phase"
    cc = build_cloud_config(
        cfg.cloud_scheme, saturation_scheme=cfg.cloud_saturation_scheme)
    assert cc.saturation_scheme == "mixed_phase"
    # saying nothing => the field default, which is 'mixed_phase' since
    # 2026-09-17 (the 'liquid' curve made cold cloud impossible: at 230 K /
    # 900 hPa ice-saturated air reads RH 0.662 against rh_crit 0.85).
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.cloud_saturation_scheme == "mixed_phase"
    assert build_cloud_config(
        d.cloud_scheme,
        saturation_scheme=d.cloud_saturation_scheme
    ).saturation_scheme == "mixed_phase"


def test_cloud_saturation_scheme_validate():
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="cloud_saturation_scheme"):
        ExperimentConfig(cloud_saturation_scheme="bogus").validate_strict()
    ExperimentConfig(cloud_saturation_scheme="mixed_phase").validate_strict()


def test_louis_cloudtop_entrainment_efficiency_flows_to_config():
    """--cloudtop-entrainment-efficiency round-trips (marine-Sc BL-top
    ventilation: thins excess Sc liquid cloud without an evap trade; the
    structural AMIP albedo fix). Single knob: default 0.0 = off."""
    parser = build_arg_parser()
    cfg0 = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg0.louis_cloudtop_entrainment_efficiency == 0.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--cloudtop-entrainment-efficiency", "0.35",
    ]), parser))
    assert cfg.louis_cloudtop_entrainment_efficiency == 0.35


def test_louis_cloudtop_entrainment_threads_into_louis_config():
    """The efficiency MUST reach LouisConfig via turbulence_config_for (the
    single source of truth for all dycores) — else it is an inert dead field
    (l_mix lesson). Default 0.0 stays byte-identical (no _replace)."""
    from legoesm.driver.physics_pipeline import turbulence_config_for
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "louis",
        "--cloudtop-entrainment-efficiency", "0.4",
    ]), parser))
    assert turbulence_config_for(cfg).louis.cloudtop_entrainment_efficiency == 0.4

    cfg0 = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "louis",
    ]), parser))
    assert turbulence_config_for(cfg0).louis.cloudtop_entrainment_efficiency == 0.0


def test_louis_cloudtop_entrainment_efficiency_validate_strict():
    """validate_strict rejects an efficiency outside [0, 1] (and NaN/Inf)."""
    from legoesm.driver.config import ExperimentConfig
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="louis_cloudtop_entrainment_efficiency"):
            ExperimentConfig(
                louis_cloudtop_entrainment_efficiency=bad).validate_strict()
    for ok in (0.0, 0.2, 1.0):
        ExperimentConfig(louis_cloudtop_entrainment_efficiency=ok).validate_strict()


def test_diagnostic_condensate_scheme_flows_to_config():
    """--diagnostic-condensate-scheme + --adiabatic-lwc-rate round-trip into
    ExperimentConfig (default 'constant' == byte-identical legacy floor)."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.cloud_diagnostic_condensate_scheme == "constant"
    assert cfg_default.cloud_adiabatic_lwc_rate is None

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--clouds", "sundqvist",
        "--diagnostic-condensate-scheme", "adiabatic",
        "--adiabatic-lwc-rate", "2.0e-6",
    ]), parser))
    assert cfg.cloud_diagnostic_condensate_scheme == "adiabatic"
    assert cfg.cloud_adiabatic_lwc_rate == 2.0e-6


def test_diagnostic_condensate_scheme_threads_into_cloud_config():
    """The ExperimentConfig fields thread through the shared build_cloud_config
    into the hot-loop CloudConfig (guards against a dead field)."""
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    cc = build_cloud_config("sundqvist",
                            diagnostic_condensate_scheme="adiabatic",
                            adiabatic_lwc_rate=2.0e-6)
    assert cc.diagnostic_condensate_scheme == "adiabatic"
    assert cc.adiabatic_lwc_rate == 2.0e-6
    # All-None companion call stays byte-identical to the CloudConfig default.
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    assert build_cloud_config("sundqvist") == CloudConfig(scheme="sundqvist")


def test_diagnostic_condensate_scheme_validate_strict():
    """validate_strict rejects an unknown diagnostic-condensate scheme, and
    rejects the adiabatic opt-in on a backend that would silently ignore it."""
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    for ok in ("constant", "adiabatic"):
        # default dycore is cd-grid + rrtmgp cloud-path radiation => OK
        ExperimentConfig(cloud_scheme="sundqvist", radiation="rrtmgp",
                         cloud_diagnostic_condensate_scheme=ok).validate_strict()
    with pytest.raises(ValueError, match="cloud_diagnostic_condensate_scheme"):
        ExperimentConfig(
            cloud_scheme="sundqvist",
            cloud_diagnostic_condensate_scheme="linear").validate_strict()
    # adiabatic on a genuinely-bypassing backend (spectral / MPAS build
    # RadiationConfig directly) is a HARD error — it would silently run the
    # constant floor there (codex dispatch-hardening).  (validate_strict joins
    # all errors, so the match just needs our substring.)
    for bad_disc in ("spectral", "mpas"):
        with pytest.raises(ValueError, match="bypassing the shared cloud pipeline"):
            ExperimentConfig(
                cloud_scheme="sundqvist", radiation="rrtmgp",
                cloud_diagnostic_condensate_scheme="adiabatic",
                dycore=DycoreConfig(discretization=bad_disc)).validate_strict()
    # A cd-grid ALIAS ('centered' / 'finite_volume') DOES thread the floor via
    # the FV pipeline, so the guard must NOT false-positive block it (the review
    # caught this — run_amip defaults --discretization to 'centered').
    for ok_disc in ("centered", "finite_volume"):
        try:
            ExperimentConfig(
                cloud_scheme="sundqvist", radiation="rrtmgp",
                cloud_diagnostic_condensate_scheme="adiabatic",
                dycore=DycoreConfig(discretization=ok_disc)).validate_strict()
        except ValueError as exc:  # unrelated dycore validation may still raise
            assert "bypassing the shared cloud pipeline" not in str(exc), (
                f"adiabatic must not be guard-blocked on cd-grid alias {ok_disc!r}")
    # Cross-field: adiabatic is a silent no-op without a diagnostic-fraction
    # cloud scheme (sundqvist/xu_randall) AND cloud-path radiation (rrtmgp/rrtmg),
    # so those combinations are HARD errors (codex dispatch-hardening).
    with pytest.raises(ValueError, match="would ignore it"):
        ExperimentConfig(
            cloud_scheme="none",
            cloud_diagnostic_condensate_scheme="adiabatic").validate_strict()
    with pytest.raises(ValueError, match="cloud-path radiation"):
        ExperimentConfig(
            cloud_scheme="sundqvist", radiation="gray",
            cloud_diagnostic_condensate_scheme="adiabatic").validate_strict()


def test_use_clubb_cloud_fraction_rejected_on_spectral_only():
    """use_clubb_cloud_fraction is enforced by build_physics_pipeline (cd-grid)
    and combined.make_physics (MPAS); the spectral standalone path builds
    neither, so validate_strict must reject the opt-in there rather than let
    it silently no-op (audit 2026-07-17 dispatch-hardening).  MPAS became a
    legal lane with the CAM6 cloud-fraction port (cloud_scheme='cam6_clubb')."""
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    for bad_disc in ("spectral",):
        with pytest.raises(ValueError, match="silently no-op"):
            ExperimentConfig(
                turbulence="clubb", use_clubb_cloud_fraction=True,
                dycore=DycoreConfig(discretization=bad_disc)).validate_strict()
    # cd-grid aliases DO build the pipeline and MPAS builds make_physics, so
    # the guard must not block them (their own turbulence=='clubb' checks
    # still apply).
    for ok_disc in ("centered", "finite_volume", "mpas"):
        try:
            ExperimentConfig(
                turbulence="clubb", use_clubb_cloud_fraction=True,
                dycore=DycoreConfig(discretization=ok_disc)).validate_strict()
        except ValueError as exc:
            assert "silently no-op" not in str(exc), (
                f"clubb-cf must not be guard-blocked on cd-grid alias {ok_disc!r}")


def test_yaml_settable_bools_have_no_switches():
    """#872 sweep: every store_true flag a shipped YAML can set true is now
    BooleanOptionalAction, so a --config that enables it stays CLI-overridable
    (--no-<flag> => False). The old store_true form made a YAML-true value
    permanently un-overridable (no negative form), breaking one-lever A/B legs
    — hit three times on 2026-07-08 alone (#873 converted the first three)."""
    swept = [
        "aerosol_ccn", "clear_sky_diag", "cmip_output", "diurnal_cycle",
        "land_stomatal_beta", "monthly_means", "orbital_insolation",
        "snow_albedo_feedback", "slab_land_active", "dynamic_albedo",
        # amip_production_latlon24.yaml sets it true (#869) — the filter-off
        # A/B leg needs --no-use-polar-filter (codex: the variant YAML created
        # a fresh instance of exactly this pattern).
        "use_polar_filter",
    ]
    for dest in swept:
        parser = build_arg_parser()
        # Simulate the YAML layer enabling the flag (load_yaml_config applies
        # file values via parser.set_defaults).
        parser.set_defaults(**{dest: True})
        flag = "--no-" + dest.replace("_", "-")
        args = parser.parse_args(["--dataset", "analytical", flag])
        assert getattr(args, dest) is False, (
            f"{flag} must override a YAML-set {dest}=true")
        # And the positive default still holds without the switch.
        args = parser.parse_args(["--dataset", "analytical"])
        assert getattr(args, dest) is True


def test_latlon24_production_variant_pins_polar_filter():
    """#869: the lat-lon production lane variant MUST carry the polar filter
    (the 12-day one-variable A/B convicted filter-off: blowup day 1 vs
    COMPLETED) and the filter-enabled dt=600 (pole clamp lifted, ~10x
    throughput, 30-day soak clean). A silent drop of either re-opens the
    day-9/10 blowup."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    cfg_file = _repo_root() / "config" / "amip" / "amip_production_latlon24.yaml"
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(str(cfg_file), parser))
    args = _postprocess_args(parser.parse_args(_AMIP_DUMMY_PATHS), parser)
    assert args.use_polar_filter is True
    assert args.dt == 600.0
    assert args.grid_type == "latlon" and args.discretization == "latlon_cgrid"
    assert args.resolution == 24 and args.nlev == 20
    # Physics inherited from the production include (one source of truth),
    # except convection: this lane pins `sbm` (#869) because bechtold
    # re-develops a polar-night temperature runaway that blows the run at day
    # ~47 regardless of every numerics lever, while sbm is stable (95-day soak)
    # and lifts hfls 40->70 (#847).  The cube lane keeps bechtold.
    cfg = build_config_from_args(args)
    # Gravity-wave drag is inherited from the production include, which
    # gained the non-orographic component; this assertion still named the
    # orographic-only spelling and so went red with it.
    assert cfg.convection == "sbm"
    assert cfg.gravity_wave_drag == "mcfarlane+e3sm_cam"
    # UNSET (#929 None sentinel; an explicit 0.0 now means "force legacy
    # no-split", not "unset"): the latlon24 YAML clears the inherited bechtold
    # knob to null, and sbm ignores it (sbm_precip_efficiency is its own knob)
    # — see the convective_precip_efficiency note in amip_production_latlon24.yaml.
    assert cfg.convective_precip_efficiency is None


def test_enable_tiled_dycore_flag_flows_to_config():
    """--enable-tiled-dycore round-trips into ExperimentConfig (P4 cube
    sub-face tiling) and validate_strict enforces cube-only."""
    import pytest
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.enable_tiled_dycore is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--enable-tiled-dycore",
    ]), parser))
    assert cfg_on.enable_tiled_dycore is True
    cfg_on.validate_strict()   # default grid = cubed_sphere -> legal

    cfg_bad = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--grid-type", "latlon",
        "--enable-tiled-dycore",
    ]), parser))
    with pytest.raises(ValueError, match="cubed_sphere"):
        cfg_bad.validate_strict()


def test_explicit_zero_sic_scale_and_sst_offset_preserved():
    """An explicit ``--sic-scale 0.0`` / ``--sst-offset 0.0`` must reach the
    config as 0.0 — the builder uses ``is not None``, not ``or``, so a
    legitimate no-sea-ice / no-conversion sensitivity value is not silently
    replaced by the fallback default (1.0 / 0.0)."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--sic-scale", "0.0", "--sst-offset", "0.0",
    ]), parser))
    assert cfg.sic_scale == 0.0
    assert cfg.sst_offset == 0.0
    # The default path still yields the fallbacks.
    cfg_def = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_def.sic_scale == 1.0 and cfg_def.sst_offset == 0.0


def test_preset_dataset_defaults_and_override():
    """A preset dataset defaults sic_scale/sst_offset to the preset's own unit
    conversions (cobe SIC is percent -> 0.01), so a bare ``--dataset cobe`` keeps
    correct units without needing --sic-scale. An explicit ``--sic-scale 0`` (a
    no-sea-ice run) still overrides — the value model_driver forwards into the
    preset config, which used to be dropped by ``_replace(path, T_ice)``."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "cobe", "--forcing-path", "/tmp/cobe.nc",
    ]), parser))
    assert cfg.sic_scale == 0.01          # cobe percent -> fraction (preset default)
    assert cfg.sst_offset == 0.0          # cobe already Kelvin
    cfg0 = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "cobe", "--forcing-path", "/tmp/cobe.nc", "--sic-scale", "0",
    ]), parser))
    assert cfg0.sic_scale == 0.0          # explicit no-ice override honored


def test_external_ozone_aerosol_require_a_file():
    """``--ozone-forcing external`` / ``--aerosol-forcing external`` without a
    file must fail loudly rather than silently substitute the built-in reference
    climatology (parity with the solar/ghg guards)."""
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--ozone-forcing", "external",
        ]), parser)
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--aerosol-forcing", "external",
        ]), parser)


def test_bechtold_use_ifs_cape_closure_round_trips_and_threads():
    """--bechtold-use-ifs-cape-closure round-trips into ExperimentConfig and
    threads into the hot-loop BechtoldConfig (the PR #1095 oracle deep
    closure); default OFF is byte-identical to the legacy closure and --no-
    turns off a config-file default."""
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-use-ifs-cape-closure",
    ]), parser))
    assert cfg.bechtold_use_ifs_cape_closure is True
    cc = _resolve_convection(cfg)[1]
    assert cc.use_ifs_cape_closure is True
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    dcc = _resolve_convection(d)[1]
    assert dcc.use_ifs_cape_closure is True     # default ON since 2026-07-16
    off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--no-bechtold-use-ifs-cape-closure"]), parser))
    assert off.bechtold_use_ifs_cape_closure is False
    occ = _resolve_convection(off)[1]
    assert occ.use_ifs_cape_closure is False    # --no- restores legacy
    assert d.validate_strict() is None


def test_bechtold_use_ifs_cape_closure_survives_amip_round_trip():
    """to_amip_config()/from_amip_config() must carry the closure flag BOTH
    ways (codex: the flat AMIPExperimentConfig filter silently dropped it, so
    a checkpoint restart flipped an explicit selection back to the legacy
    closure)."""
    from legoesm.driver.config import ExperimentConfig
    for flag in (True, False):
        cfg = ExperimentConfig(convection="bechtold",
                               bechtold_use_ifs_cape_closure=flag)
        flat = cfg.to_amip_config()
        assert flat.bechtold_use_ifs_cape_closure is flag
        back = ExperimentConfig.from_amip_config(flat)
        assert back.bechtold_use_ifs_cape_closure is flag


def test_bechtold_use_ifs_cape_closure_legacy_flat_config_gets_default():
    """A legacy flat config object PREDATING the field must resolve to the
    scheme default (True), not silently pin the old closure."""
    from legoesm.driver.config import ExperimentConfig

    flat = ExperimentConfig(convection="bechtold").to_amip_config()

    class _LegacyView:
        """A real flat config with the new field REMOVED (pre-field schema)."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "bechtold_use_ifs_cape_closure":
                raise AttributeError(name)
            return getattr(self._inner, name)

    cfg = ExperimentConfig.from_amip_config(_LegacyView(flat))
    assert cfg.bechtold_use_ifs_cape_closure is True


def test_bechtold_use_ifs_subcloud_evap_round_trips_and_threads():
    """--bechtold-use-ifs-subcloud-evap round-trips into ExperimentConfig,
    survives the flat-AMIP serialization both ways, and threads into the
    hot-loop BechtoldConfig; default OFF (legacy byte-identical)."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-use-ifs-subcloud-evap",
    ]), parser))
    assert cfg.bechtold_use_ifs_subcloud_evap is True
    assert _resolve_convection(cfg)[1].use_ifs_subcloud_evap is True
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    assert d.bechtold_use_ifs_subcloud_evap is True   # default ON 2026-07-16
    off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--no-bechtold-use-ifs-subcloud-evap"]), parser))
    assert off.bechtold_use_ifs_subcloud_evap is False
    assert _resolve_convection(off)[1].use_ifs_subcloud_evap is False
    for flag in (True, False):
        e = ExperimentConfig(convection="bechtold",
                             bechtold_use_ifs_subcloud_evap=flag)
        flat = e.to_amip_config()
        assert flat.bechtold_use_ifs_subcloud_evap is flag
        assert ExperimentConfig.from_amip_config(
            flat).bechtold_use_ifs_subcloud_evap is flag


def test_bechtold_use_ifs_inplume_precip_round_trips_and_threads():
    """--bechtold-use-ifs-inplume-precip round-trips into ExperimentConfig,
    survives the flat-AMIP serialization both ways, threads into the hot-loop
    BechtoldConfig; default OFF (legacy byte-identical)."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-use-ifs-inplume-precip",
    ]), parser))
    assert cfg.bechtold_use_ifs_inplume_precip is True
    assert _resolve_convection(cfg)[1].use_ifs_inplume_precip is True
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    assert d.bechtold_use_ifs_inplume_precip is True   # default ON 2026-07-16
    off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--no-bechtold-use-ifs-inplume-precip"]), parser))
    assert off.bechtold_use_ifs_inplume_precip is False
    assert _resolve_convection(off)[1].use_ifs_inplume_precip is False
    for flag in (True, False):
        e = ExperimentConfig(convection="bechtold",
                             bechtold_use_ifs_inplume_precip=flag)
        flat = e.to_amip_config()
        assert flat.bechtold_use_ifs_inplume_precip is flag
        assert ExperimentConfig.from_amip_config(
            flat).bechtold_use_ifs_inplume_precip is flag


def test_bechtold_dx_m_round_trips_and_threads():
    """--bechtold-dx-m round-trips (CLI -> ExperimentConfig -> flat AMIP ->
    BechtoldConfig); default 0 = legacy."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-dx-m", "417000",
    ]), parser))
    assert cfg.bechtold_dx_m == 417000.0
    assert _resolve_convection(cfg)[1].dx_m == 417000.0
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    assert d.bechtold_dx_m == 0.0
    flat = ExperimentConfig(convection="bechtold",
                            bechtold_dx_m=123456.0).to_amip_config()
    assert flat.bechtold_dx_m == 123456.0
    assert ExperimentConfig.from_amip_config(flat).bechtold_dx_m == 123456.0


def test_bechtold_use_ifs_downdraft_round_trips_and_threads():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-use-ifs-downdraft"]), parser))
    assert cfg.bechtold_use_ifs_downdraft is True
    assert _resolve_convection(cfg)[1].use_ifs_downdraft is True
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    assert d.bechtold_use_ifs_downdraft is True  # flipped 2026-07-17
    for flag in (True, False):
        e = ExperimentConfig(convection="bechtold",
                             bechtold_use_ifs_downdraft=flag)
        flat = e.to_amip_config()
        assert flat.bechtold_use_ifs_downdraft is flag
        assert ExperimentConfig.from_amip_config(
            flat).bechtold_use_ifs_downdraft is flag


def test_bechtold_use_ifs_shallow_closure_round_trips_and_threads():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-use-ifs-shallow-closure"]), parser))
    assert cfg.bechtold_use_ifs_shallow_closure is True
    assert _resolve_convection(cfg)[1].use_ifs_shallow_closure is True
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    assert d.bechtold_use_ifs_shallow_closure is False
    for flag in (True, False):
        e = ExperimentConfig(convection="bechtold",
                             bechtold_use_ifs_shallow_closure=flag)
        flat = e.to_amip_config()
        assert flat.bechtold_use_ifs_shallow_closure is flag
        assert ExperimentConfig.from_amip_config(
            flat).bechtold_use_ifs_shallow_closure is flag


def test_bechtold_capdcycl_and_land_rhebc_round_trip():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-use-ifs-capdcycl", "--bechtold-use-ifs-land-rhebc"]),
        parser))
    cc = _resolve_convection(cfg)[1]
    assert cc.use_ifs_capdcycl is True and cc.use_ifs_land_rhebc is True
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    dcc = _resolve_convection(d)[1]
    assert dcc.use_ifs_capdcycl is True and dcc.use_ifs_land_rhebc is True  # flipped 2026-07-17
    for flag in (True, False):
        e = ExperimentConfig(convection="bechtold",
                             bechtold_use_ifs_capdcycl=flag,
                             bechtold_use_ifs_land_rhebc=flag)
        flat = e.to_amip_config()
        back = ExperimentConfig.from_amip_config(flat)
        assert back.bechtold_use_ifs_capdcycl is flag
        assert back.bechtold_use_ifs_land_rhebc is flag


def test_bechtold_use_ifs_snow_melt_round_trip():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import _resolve_convection
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--bechtold-use-ifs-snow-melt"]), parser))
    assert _resolve_convection(cfg)[1].use_ifs_snow_melt is True
    d = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    assert _resolve_convection(d)[1].use_ifs_snow_melt is True  # flipped 2026-07-17
    for flag in (True, False):
        e = ExperimentConfig(convection="bechtold",
                             bechtold_use_ifs_snow_melt=flag)
        assert ExperimentConfig.from_amip_config(
            e.to_amip_config()).bechtold_use_ifs_snow_melt is flag


def test_ml_parameterization_requires_checkpoint_and_stats():
    """``--physics-parameterization ml`` without checkpoint+stats must fail at
    parse time (the guard used to be dead code inside the --evaluate branch,
    unreachable after its parser.error — audit 2026-07-17)."""
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical",
            "--physics-parameterization", "ml",
            "--convection", "mass_flux", "--turbulence", "louis",
        ]), parser)
    # With both assets the guard passes.
    args = _postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--physics-parameterization", "ml",
        "--convection", "mass_flux", "--turbulence", "louis",
        "--physics-parameterization-checkpoint", "/tmp/ckpt.eqx",
        "--physics-parameterization-stats", "/tmp/stats.npz",
    ]), parser)
    assert args.physics_parameterization == "ml"


def test_truncation_conflicting_grid_or_discretization_errors():
    """--truncation with an explicitly conflicting --grid-type/--discretization
    must be a hard error, not a silent switch to gaussian/spectral (the one
    silent grid fallback in the driver — audit 2026-07-17). The argparse
    defaults (cubed_sphere/centered) are still coerced."""
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--truncation", "42",
            "--grid-type", "latlon",
        ]), parser)
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--truncation", "42",
            "--discretization", "mpas",
        ]), parser)
    # An explicit value EQUAL to the production default also conflicts (the
    # None-sentinel default makes it distinguishable from unset).
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--truncation", "42",
            "--grid-type", "cubed_sphere",
        ]), parser)
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--discretization", "spectral",
            "--grid-type", "latlon",
        ]), parser)
    # Bare --truncation still auto-configures the spectral pair.
    args = _postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--truncation", "42",
    ]), parser)
    assert args.grid_type == "gaussian"
    assert args.discretization == "spectral"
    assert args.resolution == 42
    # Explicit-but-agreeing choices also pass.
    args = _postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--truncation", "42",
        "--grid-type", "gaussian", "--discretization", "spectral",
    ]), parser)
    assert args.grid_type == "gaussian"
    # Bare defaults resolve to the production pair.
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]),
                             parser)
    assert args.grid_type == "cubed_sphere"
    assert args.discretization == "centered"


def test_explicit_zero_p_top_and_stretching_survive():
    """An explicit ``--p-top 0``/``--stretching 0`` must reach GridConfig
    instead of being swallowed by an ``or``-default (so validate_strict can
    reject it loudly); the bare default still yields 200.0/2.0."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--p-top", "0", "--stretching", "0",
    ]), parser))
    assert cfg.grid.p_top_Pa == 0.0
    assert cfg.grid.stretching == 0.0
    cfg_def = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_def.grid.p_top_Pa == 200.0
    assert cfg_def.grid.stretching == 2.0


def test_restart_still_requires_forcing_path_for_real_data():
    """--restart-from does NOT exempt --forcing-path for a real dataset: forcing
    is not in the checkpoint and setup() loads it before the checkpoint, so the
    old exemption only deferred the failure to a confusing deep error
    (audit 2026-07-17)."""
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "custom", "--restart-from", "/tmp/ckpt.zarr",
        ]), parser)
    # analytical needs no forcing file even on restart.
    args = _postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--restart-from", "/tmp/ckpt.zarr",
    ]), parser)
    assert args.restart_from == "/tmp/ckpt.zarr"
    # real dataset WITH a forcing path is fine.
    args = _postprocess_args(parser.parse_args([
        "--dataset", "custom", "--restart-from", "/tmp/ckpt.zarr",
        "--forcing-path", "/tmp/sst.nc",
    ]), parser)
    assert args.forcing_path == "/tmp/sst.nc"


def test_voronoi_default_discretization_is_mpas():
    """2026-07-21 audit: bare --grid-type voronoi must resolve the only
    supported discretization ('mpas'), not the generic 'centered' default that
    dies at the dycore factory on (hydrostatic, centered, mpas)."""
    p = build_arg_parser()
    for grid in ("voronoi", "icosahedral", "mpas_voronoi", "mpas"):
        args = _postprocess_args(p.parse_args(["--grid-type", grid]), p)
        assert args.discretization == "mpas", (
            f"--grid-type {grid} resolved discretization "
            f"{args.discretization!r}, expected 'mpas'")


def test_build_config_resolves_voronoi_discretization_without_postprocess():
    """A caller that builds a config WITHOUT _postprocess_args (codex r2)
    must still get the per-grid default, not the None-sentinel 'centered'."""
    p = build_arg_parser()
    args = p.parse_args(["--grid-type", "voronoi"])
    cfg = build_config_from_args(args)
    assert cfg.dycore.discretization == "mpas"


def test_truncation_only_triggers_spectral_fallback():
    """--truncation N without --discretization spectral must still fall back
    off prognostic schemes (postprocess resolves discretization='spectral'
    BEFORE the fallback runs; the old order made this a silent no-op)."""
    p = build_arg_parser()
    argv = ["--truncation", "21"]
    args = _postprocess_args(p.parse_args(argv), p)
    assert args.discretization == "spectral"
    args = _apply_spectral_scheme_fallback(args, argv, p)
    assert args.convection == "sbm"
    assert args.gravity_wave_drag == "rayleigh"


def test_spectral_fallback_rejects_dependent_option_silent_noop():
    """--truncation 21 --convective-precip-efficiency 0.5: the fallback would
    downgrade convection to sbm, silently ignoring the mass-flux-only option;
    it must error instead of accept-then-ignore (codex r2)."""
    p = build_arg_parser()
    argv = ["--truncation", "21", "--convective-precip-efficiency", "0.5"]
    args = _postprocess_args(p.parse_args(argv), p)
    with pytest.raises(SystemExit):
        _apply_spectral_scheme_fallback(args, argv, p)


def test_hard_sat_override_flags_flow_to_config():
    """--hard-sat-adjust-threshold / --hard-sat-max-heating-k round-trip onto
    the ExperimentConfig flat scalars (day-137 summer-regime drain tuning);
    defaults None keep the per-scheme __param_spec__ values."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.hard_sat_adjust_threshold is None
    assert cfg_default.hard_sat_max_heating_K is None

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--hard-saturation-adjustment",
        "--hard-sat-adjust-threshold", "1.2",
        "--hard-sat-max-heating-k", "10.0",
    ]), parser))
    assert cfg.hard_sat_adjust_threshold == 1.2
    assert cfg.hard_sat_max_heating_K == 10.0
    cfg.validate_strict()


def test_hard_sat_override_without_gate_is_refused():
    """An override without --hard-saturation-adjustment would be silently
    inert (the floats are only read where the boolean gate fires) —
    validate_strict must refuse it."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--hard-sat-max-heating-k", "10.0",
    ]), parser))
    with pytest.raises(ValueError, match="hard_sat_max_heating_K"):
        cfg.validate_strict()


def test_hard_sat_override_bounds_enforced():
    """Out-of-spec-bounds overrides are refused (threshold (1,2) /
    heating (0.5,50) per the warm-rain __param_spec__)."""
    parser = build_arg_parser()
    for flags, match in (
            (["--hard-sat-adjust-threshold", "0.9"], "hard_sat_adjust_threshold"),
            (["--hard-sat-max-heating-k", "100.0"], "hard_sat_max_heating_K")):
        cfg = build_config_from_args(_postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--microphysics", "morrison",
            "--hard-saturation-adjustment", *flags,
        ]), parser))
        with pytest.raises(ValueError, match=match):
            cfg.validate_strict()


def test_hard_saturation_adjustment_requires_a_guarded_scheme():
    """--hard-saturation-adjustment with a scheme that does NOT carry the guard
    is silently inert at runtime (_resolve_microphysics / the MPAS post-step
    drain early-return before the flag is read) — validate_strict must refuse
    it (codex F3).

    The negative set is now sdm / fast_sbm / none, NOT sundqvist: the guard was
    made uniform, so Sundqvist and the ML emulator carry it. sdm and fast_sbm
    are exempt on purpose (they integrate the super-saturation relaxation /
    droplet growth law explicitly — see config.HARD_SAT_GUARD_EXEMPT), and
    'none' has no scheme function at all."""
    from legoesm.atmosphere.physics.microphysics.config import (
        HARD_SAT_GUARD_EXEMPT,
    )
    parser = build_arg_parser()
    assert set(HARD_SAT_GUARD_EXEMPT) == {"sdm", "fast_sbm", "none"}
    for micro in sorted(HARD_SAT_GUARD_EXEMPT):
        cfg = build_config_from_args(_postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--microphysics", micro,
            "--hard-saturation-adjustment",
        ]), parser))
        with pytest.raises(ValueError, match="carrying the guard"):
            cfg.validate_strict()


def test_hard_saturation_adjustment_accepted_by_every_guarded_scheme():
    """The complement, so the test above cannot pass vacuously by
    validate_strict rejecting everything: --hard-saturation-adjustment must be
    ACCEPTED for every guarded scheme run_amip exposes."""
    from legoesm.atmosphere.physics.microphysics.config import (
        HARD_SAT_GUARD_SCHEMES,
    )
    parser = build_arg_parser()
    exposed = [s for s in HARD_SAT_GUARD_SCHEMES
               if s in {"kessler", "sundqvist", "seifert_beheng", "morrison",
                        "thompson", "p3"}]
    assert len(exposed) == 6, exposed
    for micro in exposed:
        cfg = build_config_from_args(_postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--microphysics", micro,
            "--hard-saturation-adjustment",
        ]), parser))
        cfg.validate_strict()   # must NOT raise


def test_morrison_flavor_round_trips():
    """--morrison-flavor sam round-trips into ExperimentConfig; default mg."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--morrison-flavor", "sam",
    ]), parser))
    assert cfg.morrison_flavor == "sam"
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.morrison_flavor == "mg"


def test_morrison_flavor_yaml_route():
    """--config YAML route (keys become parser defaults): morrison_flavor: sam
    must reach ExperimentConfig, and an explicit CLI flag must still win."""
    parser = build_arg_parser()
    parser.set_defaults(morrison_flavor="sam", microphysics="morrison")
    cfg = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg.morrison_flavor == "sam"
    cfg_cli = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--morrison-flavor", "mg",
    ]), parser))
    assert cfg_cli.morrison_flavor == "mg"


def test_bechtold_rprcon_dnoprc_thread_and_validate():
    """bechtold_rprcon / bechtold_dnoprc (IFS in-plume conversion constants,
    the anvil-source levers) thread into the hot-loop BechtoldConfig on BOTH
    resolvers; defaults byte-identical; validate_strict bounds enforced."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import (
        _resolve_convection,
        convection_config_for,
    )
    cfg = ExperimentConfig(convection="bechtold", bechtold_rprcon=5.0e-3,
                           bechtold_dnoprc=1.0e-4)
    leaf = _resolve_convection(cfg)[1]
    assert (leaf.rprcon, leaf.dnoprc) == (5.0e-3, 1.0e-4)
    cc = convection_config_for(cfg)
    assert (cc.bechtold.rprcon, cc.bechtold.dnoprc) == (5.0e-3, 1.0e-4)
    d = _resolve_convection(ExperimentConfig(convection="bechtold"))[1]
    assert (d.rprcon, d.dnoprc) == (1.4e-3, 3.0e-4)
    with pytest.raises(ValueError, match="bechtold_rprcon"):
        ExperimentConfig(bechtold_rprcon=1.0).validate_strict()
    with pytest.raises(ValueError, match="bechtold_dnoprc"):
        ExperimentConfig(bechtold_dnoprc=1.0e-2).validate_strict()


def test_bechtold_epsilon_deep_delta_deep_thread_and_validate():
    """bechtold_epsilon_deep / bechtold_delta_deep (the IFS deep plume-mixing
    rates) thread into the hot-loop BechtoldConfig on BOTH resolvers; defaults
    byte-identical; validate_strict bounds enforced.  Before this wiring the
    deep entrainment rate could not be set from any MIP driver at all, and the
    param spec excluded it on the false ground that the in-scheme (1.3-RH)
    height factor made it "not a constant tunable" -- that factor MULTIPLIES
    the base rate, exactly as it does for delta_deep, which was always tunable.
    """
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import (
        _resolve_convection,
        convection_config_for,
    )
    cfg = ExperimentConfig(convection="bechtold", bechtold_epsilon_deep=4.0e-3,
                           bechtold_delta_deep=2.0e-4)
    leaf = _resolve_convection(cfg)[1]
    assert (leaf.epsilon_deep, leaf.delta_deep) == (4.0e-3, 2.0e-4)
    cc = convection_config_for(cfg)
    assert (cc.bechtold.epsilon_deep, cc.bechtold.delta_deep) == (4.0e-3, 2.0e-4)
    d = _resolve_convection(ExperimentConfig(convection="bechtold"))[1]
    assert (d.epsilon_deep, d.delta_deep) == (1.75e-3, 0.75e-4)
    with pytest.raises(ValueError, match="bechtold_epsilon_deep"):
        ExperimentConfig(bechtold_epsilon_deep=1.0).validate_strict()
    with pytest.raises(ValueError, match="bechtold_delta_deep"):
        ExperimentConfig(bechtold_delta_deep=1.0e-2).validate_strict()


def test_bechtold_mixing_rates_are_settable_from_a_params_file():
    """The --params route, not just the constructor.

    Deleting either scalar-map entry leaves the constructor test above green
    while making the knob unreachable from every driver, which is the exact
    state this wiring was written to end.
    """
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import convection_config_for
    from legoesm.driver.run_config_yaml import (
        apply_params_to_config,
        build_atm_scalar_param_map,
    )
    m = build_atm_scalar_param_map()
    base = ExperimentConfig(convection="bechtold")
    cfg = apply_params_to_config(
        base, {"atm.conv.BechtoldConfig.epsilon_deep": 3.0e-3,
               "atm.conv.BechtoldConfig.delta_deep": 3.0e-5},
        scalar_param_map=m)
    leaf = convection_config_for(cfg).bechtold
    assert (leaf.epsilon_deep, leaf.delta_deep) == (3.0e-3, 3.0e-5)
    with pytest.raises(SystemExit):
        apply_params_to_config(
            base, {"atm.conv.BechtoldConfig.epsilon_deep": 1.0},
            scalar_param_map=m)


def test_amip_round_trip_keeps_the_tuned_convection_scalars():
    """A legacy-format restart must not silently reset a TUNED convection knob.

    The legacy flat schema is a deliberate SUBSET of ExperimentConfig -- some
    140 fields are outside it by design -- so this is not a claim that nothing
    is dropped. It pins the convection scalars the AMIP campaign actually sets
    on its arms, which before 2026-08-13 were dropped: a run restarted from a
    legacy checkpoint quietly continued on defaults, which would have surfaced
    as "the knob does nothing" rather than as an error.

    Codex, reviewing the first version of this test, showed that a
    prefix-based "every convection scalar" assertion was false -- e.g.
    ``sbm_T_min_convect`` is dropped too. Widening the flat schema to satisfy
    that claim is a separate change; this test states only what it checks.
    """
    from legoesm.driver.config import ExperimentConfig
    from legoesm.forcing.amip_config import AMIPExperimentConfig

    tuned = {
        "bechtold_epsilon_deep": 3.0e-3,
        "bechtold_delta_deep": 3.0e-5,
        "bechtold_rprcon": 5.0e-3,
        "bechtold_dnoprc": 1.0e-4,
        "bechtold_downdraft_entrain_rate": 5.0e-4,
        "bechtold_downdraft_detrain_scale_m": 900.0,
        "convective_precip_efficiency": 0.6,
        "autoconv_q_c_crit": 3.0e-4,
    }
    missing = sorted(k for k in tuned
                     if k not in AMIPExperimentConfig._fields)
    assert not missing, f"campaign-tuned knobs absent from the flat schema: {missing}"

    cfg = ExperimentConfig(convection="bechtold", **tuned)
    back = ExperimentConfig.from_amip_config(cfg.to_amip_config())
    for k, v in tuned.items():
        assert getattr(back, k) == v, f"{k} reset by the AMIP round-trip"


def test_inplume_conversion_responds_to_rprcon():
    """Non-vacuity: the in-plume conversion must produce MORE precip at
    higher rprcon and at lower dnoprc on a moist synthetic plume profile —
    the knob the 2026-07-27 anvil campaign tunes.  Guards against the
    silent-constant regression (the fn ignoring its new args)."""
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.convection.bechtold import (
        _ifs_inplume_precip_conversion,
    )

    ncol, nlev = 2, 12
    z = jnp.linspace(12000.0, 200.0, nlev)[None, :].repeat(ncol, 0)
    # Condensate just above the precip-onset threshold: the conversion is
    # RATE-limited there (a saturated profile compresses the rprcon response
    # into the Sundqvist exp plateau and the test goes vacuous).
    q_c_u = jnp.full((ncol, nlev), 5.0e-4)
    T_u = jnp.linspace(210.0, 295.0, nlev)[None, :].repeat(ncol, 0)  # noqa: N806 (T = temperature, domain convention)
    eps = jnp.full((ncol, nlev), 1.0e-4)
    ke = jnp.full((ncol, nlev), 2.0)

    def total_precip(rprcon, dnoprc):
        _, pf = _ifs_inplume_precip_conversion(
            q_c_u, T_u, z, eps, ke, rprcon=rprcon, dnoprc=dnoprc)
        return float(jnp.sum(pf))

    base = total_precip(1.4e-3, 3.0e-4)
    assert total_precip(5.6e-3, 3.0e-4) > base * 1.05
    assert total_precip(1.4e-3, 1.0e-4) > base


def test_mpas_vert_advection_scheme_flag_flows_to_config():
    """--mpas-vert-advection-scheme round-trips into DycoreConfig and is
    rejected outside the MPAS sigma lane rather than running silently inert.

    Default "upwind" = the first-order donor-cell path, bit-identical to
    before; "van_leer" is the monotone 2nd-order TVD option that removes the
    K_sigma = |sigma_dot|*dsigma/2 implicit diffusion measured at +0.822 K/day
    at the tropical UTLS (91.4 hPa, cldF_fsd, N=37)."""
    parser = build_arg_parser()
    mpas = ["--dataset", "analytical", "--grid-type", "voronoi",
            "--discretization", "mpas", "--vertical-coord", "sigma"]

    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(mpas), parser))
    assert cfg_default.dycore.mpas_vert_advection_scheme == "upwind"
    cfg_default.validate_strict()

    cfg_vl = build_config_from_args(_postprocess_args(parser.parse_args(
        mpas + ["--mpas-vert-advection-scheme", "van_leer"]), parser))
    assert cfg_vl.dycore.mpas_vert_advection_scheme == "van_leer"
    cfg_vl.validate_strict()

    # argparse choices reject a typo before anything else runs.
    with pytest.raises(SystemExit):
        parser.parse_args(mpas + ["--mpas-vert-advection-scheme", "vanleer"])

    # hybrid coordinate -> the operator is not wired there -> refuse.
    cfg_hyb = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--grid-type", "voronoi",
         "--discretization", "mpas", "--vertical-coord", "hybrid",
         "--mpas-vert-advection-scheme", "van_leer"]), parser))
    with pytest.raises(ValueError, match="sigma vertical coordinate only"):
        cfg_hyb.validate_strict()


def test_fv3_duo_discretization_flows_to_config():
    """--discretization fv3_duo round-trips into DycoreConfig and passes
    validate_strict (slice 1: dry, physics-off, fp64, nlev in {5, 10}).

    The dry-stack flags mirror what the fv3_duo component-factory branch
    requires; the branch's own refusals (physics on, fp32, bad nlev) are
    covered by tests/atmosphere/hydrostatic/unit/test_fv3_duo_dynamics.py.
    """
    parser = build_arg_parser()
    argv = ["--dataset", "analytical", "--grid-type", "cubed_sphere",
            "--discretization", "fv3_duo", "--resolution", "12",
            "--nlev", "5", "--precision", "fp64",
            "--radiation", "none", "--convection", "none",
            "--microphysics", "none", "--turbulence", "none",
            "--gravity-wave-drag", "none", "--allow-disabled-physics"]
    args = _postprocess_args(parser.parse_args(argv), parser)
    # the disabled-physics gate accepts the stack with the explicit opt-in
    _require_full_physics_for_amip(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.dycore.discretization == "fv3_duo"
    assert cfg.grid.grid_type == "cubed_sphere"
    assert cfg.grid.nlev == 5
    assert cfg.precision == "fp64"
    cfg.validate_strict()

    # The factory's DEFAULT-DENY wall must accept a stock CLI-built duo
    # config — an argparse default drifting off the ExperimentConfig
    # default would otherwise refuse EVERY run_amip fv3_duo launch.
    from legoesm.driver.component_factory import (
        _refuse_fv3_duo_non_default,
    )
    _refuse_fv3_duo_non_default(cfg)

    # argparse rejects a typo before anything else runs.
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "analytical",
                           "--discretization", "fv3duo"])


def test_cmip_resolution_deg_round_trips():
    """--cmip-resolution-deg reaches OutputConfig; default unchanged.

    The CMOR output grid must track the mesh: a refined run written at the
    5-degree default throws the refinement away, and the tropical rain band --
    one to two cells wide -- cannot be scored at all. Before this flag the
    field was reachable from no driver.
    """
    from scripts.run import run_amip
    from legoesm.driver.config import OutputConfig

    p = run_amip.build_arg_parser()
    a = p.parse_args(["--cmip-resolution-deg", "2.5"])
    assert a.cmip_resolution_deg == 2.5
    assert p.parse_args([]).cmip_resolution_deg == OutputConfig().cmip_resolution_deg


def test_capdcycl_land_tau_scale_reaches_the_kernel_and_defaults_to_ifs():
    """The land diurnal-CAPE timescale is a knob, not just an on/off flag.

    The flag disables the diurnal subtraction over BOTH surfaces; this scales
    only the land branch, which is the one whose IFS value assumes a ~10 km
    mesh. 1.0 must reproduce the flag-on behaviour exactly, or every run in the
    campaign so far changes meaning.
    """
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import convection_config_for
    from legoesm.driver.run_config_yaml import (
        apply_params_to_config,
        build_atm_scalar_param_map,
    )
    assert convection_config_for(
        ExperimentConfig(convection="bechtold")).bechtold.capdcycl_land_tau_scale == 1.0
    cfg = apply_params_to_config(
        ExperimentConfig(convection="bechtold"),
        {"atm.conv.BechtoldConfig.capdcycl_land_tau_scale": 0.25},
        scalar_param_map=build_atm_scalar_param_map())
    assert convection_config_for(cfg).bechtold.capdcycl_land_tau_scale == 0.25
    with pytest.raises(ValueError, match="capdcycl_land_tau_scale"):
        ExperimentConfig(bechtold_capdcycl_land_tau_scale=5.0).validate_strict()


def test_sub_daily_diag_days_round_trips_and_never_disables_the_check():
    """A blow-up is reported at the first DIAGNOSTIC SAMPLE, not the first bad
    step, so the cadence bounds how precisely a failure can be located in time.
    An integer-only cadence pinned that bound at one simulated day.  Sub-daily
    values must survive the CLI, reach the config, and — critically — must never
    round down to a zero-step interval, which every guard in the run loop reads
    as "diagnostics disabled" and would silently turn the blow-up check OFF.

    The step arithmetic is exercised through the PRODUCTION helper the run loops
    call, not a copy of it: a local re-implementation would keep passing if the
    real cadence lost its floor.
    """
    from legoesm.driver.diagnostics import diagnostic_interval_steps

    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--diag-days", "0.01",
    ]), parser))
    assert cfg.output.diag_days == pytest.approx(0.01)

    # At the production MPAS timestep: 0.01 d = 11.5 steps -> 11, and a cadence
    # far below one step floors to 1 rather than collapsing to "off".
    assert diagnostic_interval_steps(0.01, 75.0, 999) == 11
    assert diagnostic_interval_steps(1e-6, 75.0, 999) == 1, \
        "a fine cadence silently disabled the blow-up check"
    assert diagnostic_interval_steps(1.0, 75.0, 999) == 1152
    # The "no periodic cadence" sentinel still yields the caller's fallback.
    assert diagnostic_interval_steps(0.0, 75.0, 999) == 999
    # A non-finite cadence is refused, not silently reinterpreted.
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="diag_days"):
            diagnostic_interval_steps(bad, 75.0, 999)

    # The integer default is unchanged.
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.output.diag_days == pytest.approx(5.0)


_FV3_DUO_ARGV = ["--dataset", "analytical", "--grid-type", "cubed_sphere",
                 "--discretization", "fv3_duo", "--resolution", "12",
                 "--nlev", "5", "--precision", "fp64",
                 "--radiation", "none", "--convection", "none",
                 "--microphysics", "none", "--turbulence", "none",
                 "--gravity-wave-drag", "none", "--allow-disabled-physics"]


def _fv3_duo_cfg(extra):
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(_FV3_DUO_ARGV + extra), parser)
    return build_config_from_args(args)


def test_fv3_duo_windows_flags_round_trip_and_validate():
    """--fv3-duo-windows KT + --fv3-duo-window-pad PAD reach DycoreConfig
    and pass validate_strict together; the defaults are None (face
    layout), so an unset pair changes nothing."""
    cfg = _fv3_duo_cfg([])
    assert cfg.dycore.fv3_duo_windows is None
    assert cfg.dycore.fv3_duo_window_pad is None
    cfg.validate_strict()
    cfg = _fv3_duo_cfg(["--fv3-duo-windows", "2", "--fv3-duo-window-pad", "5"])
    assert cfg.dycore.fv3_duo_windows == 2
    assert cfg.dycore.fv3_duo_window_pad == 5
    cfg.validate_strict()


def test_fv3_duo_column_lane_flag_round_trips_and_validates():
    """--fv3-duo-column-lane reaches DycoreConfig (default False) and
    validate_strict refuses it with the window layout."""
    cfg = _fv3_duo_cfg([])
    assert cfg.dycore.fv3_duo_column_lane is False
    cfg = _fv3_duo_cfg(["--fv3-duo-column-lane"])
    assert cfg.dycore.fv3_duo_column_lane is True
    cfg.validate_strict()
    cfg = _fv3_duo_cfg(["--fv3-duo-column-lane", "--fv3-duo-windows", "2",
                        "--fv3-duo-window-pad", "5"])
    with pytest.raises(ValueError, match="rung 7"):
        cfg.validate_strict()
    cfg = _fv3_duo_cfg(["--fv3-duo-column-lane"])
    cfg = cfg._replace(dycore=cfg.dycore._replace(discretization="cdgrid"))
    with pytest.raises(ValueError, match="fv3_duo_column_lane needs"):
        cfg.validate_strict()


def test_fv3_duo_windows_without_pad_is_refused():
    """The pad is a measured per-deck halo width, never defaulted."""
    cfg = _fv3_duo_cfg(["--fv3-duo-windows", "2"])
    with pytest.raises(ValueError, match="fv3_duo_window_pad is REQUIRED"):
        cfg.validate_strict()


def test_fv3_duo_window_pad_without_windows_is_refused():
    cfg = _fv3_duo_cfg(["--fv3-duo-window-pad", "5"])
    with pytest.raises(ValueError, match="without dycore.fv3_duo_windows"):
        cfg.validate_strict()


def test_fv3_duo_windows_below_two_is_refused():
    cfg = _fv3_duo_cfg(["--fv3-duo-windows", "1", "--fv3-duo-window-pad", "5"])
    with pytest.raises(ValueError, match="int >= 2"):
        cfg.validate_strict()


def test_fv3_duo_windows_on_another_discretization_is_refused():
    parser = build_arg_parser()
    argv = [a if a != "fv3_duo" else "cdgrid" for a in _FV3_DUO_ARGV]
    args = _postprocess_args(parser.parse_args(
        argv + ["--fv3-duo-windows", "2", "--fv3-duo-window-pad", "5"]),
        parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="only meaningful with"):
        cfg.validate_strict()


def test_cloud_cover_condensate_q_ref_round_trip_and_bounds():
    """--cloud-cover-condensate-q-ref threads into ExperimentConfig (None by
    default = scheme default, off); out-of-range refused by validate_strict."""
    parser = build_arg_parser()
    base = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert base.cloud_cover_condensate_q_ref is None
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--cloud-cover-condensate-q-ref", "3e-5"]),
        parser))
    assert cfg.cloud_cover_condensate_q_ref == 3.0e-5
    bad = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--cloud-cover-condensate-q-ref", "0.5"]),
        parser))
    with pytest.raises(Exception, match="cloud_cover_condensate_q_ref"):
        bad.validate_strict()


def test_convective_rain_to_surface_round_trips():
    """--convective-rain-to-surface reaches ExperimentConfig AND the
    ConvectionConfig the combined-physics lanes build from it."""
    from legoesm.driver.physics_pipeline import convection_config_for

    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical", "--convection", "bechtold"]),
        parser))
    assert cfg_default.convective_rain_to_surface is True      # default since 2026-09-16
    assert convection_config_for(cfg_default).rain_to_surface is True

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold",
        "--no-convective-rain-to-surface",
    ]), parser))
    assert cfg.convective_rain_to_surface is False
    assert convection_config_for(cfg).rain_to_surface is False


def test_surface_height_and_saline_flags_round_trip():
    """--surface-z-ref-model-level / --surface-ocean-q-sfc-saline reach
    ExperimentConfig AND the surface sub-config turbulence_config_for builds."""
    from legoesm.driver.physics_pipeline import turbulence_config_for

    parser = build_arg_parser()
    default_cfg = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical", "--turbulence", "louis"]), parser))
    # CLI defaults are None = the scheme's own value: the real level height
    # is True at SurfaceLayerConfig (the production default, one place); the
    # sea-water humidity resolves to on where the lane can honour it.
    assert default_cfg.surface_z_ref_model_level is None
    assert default_cfg.surface_ocean_q_sfc_saline is None
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    assert SurfaceLayerConfig().z_ref_model_level is True
    assert turbulence_config_for(default_cfg).louis.surface.z_ref_model_level is True

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "louis",
        "--surface-bulk-scheme", "coare3",
        "--surface-z-ref-model-level", "--surface-ocean-q-sfc-saline",
    ]), parser))
    assert cfg.surface_z_ref_model_level is True
    assert cfg.surface_ocean_q_sfc_saline is True
    cfg = cfg._replace(grid=cfg.grid._replace(grid_type="mpas"))
    surf = turbulence_config_for(cfg).louis.surface
    assert surf.z_ref_model_level is True
    assert surf.ocean_q_sfc_saline is True
    assert surf.bulk_scheme == "coare3"
    # a lane with no ocean/land separation cannot honour the explicit request
    with pytest.raises(ValueError, match="separates ocean from land"):
        turbulence_config_for(cfg._replace(grid=cfg.grid._replace(grid_type="cubed_sphere")))


def test_bechtold_M_b_max_threads_and_validates():
    """bechtold_M_b_max (the cloud-base mass-flux cap) threads into the
    hot-loop BechtoldConfig on BOTH resolvers and through the --params map;
    the default reproduces the production value (0.02, the value the old
    getattr fallback imposed because the field did not exist); validate_strict
    enforces the scheme's declared range (0.02, 0.15).  Before this wiring
    the cap could not be set from any MIP driver: the pipeline read a field
    named ``bechtold_m_b_max`` that no config carried."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import (
        _resolve_convection,
        convection_config_for,
    )
    from legoesm.driver.run_config_yaml import _ATM_SCALAR_PARAM_MAP

    cfg = ExperimentConfig(convection="bechtold", bechtold_M_b_max=0.08)
    assert _resolve_convection(cfg)[1].M_b_max == 0.08
    assert convection_config_for(cfg).bechtold.M_b_max == 0.08
    assert _resolve_convection(ExperimentConfig(convection="bechtold"))[1].M_b_max == 0.05   # default since 2026-09-16
    assert _ATM_SCALAR_PARAM_MAP["atm.conv.BechtoldConfig.M_b_max"] == "bechtold_M_b_max"
    with pytest.raises(ValueError, match="bechtold_M_b_max"):
        ExperimentConfig(bechtold_M_b_max=0.5).validate_strict()


def test_bechtold_enable_cmt_round_trips_and_default_preserves_each_lane():
    """--bechtold-enable-cmt reaches the BechtoldConfig leaf on both resolvers.
    The None default keeps every lane where it was before the MPAS wiring:
    OFF on MPAS (the bridge handed the scheme zero winds), ON elsewhere
    (BechtoldConfig.enable_cmt); codex review of 2bfdf5413 caught the first
    version silently switching the spectral lane's CMT off."""
    from legoesm.driver.physics_pipeline import _resolve_convection, convection_config_for

    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold", "--bechtold-enable-cmt"]), parser))
    assert cfg.bechtold_enable_cmt is True
    assert _resolve_convection(cfg)[1].enable_cmt is True
    assert convection_config_for(cfg).bechtold.enable_cmt is True
    off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold", "--no-bechtold-enable-cmt"]), parser))
    assert convection_config_for(off).bechtold.enable_cmt is False
    dflt = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "bechtold"]), parser))
    assert dflt.bechtold_enable_cmt is None
    assert convection_config_for(dflt._replace(grid=dflt.grid._replace(grid_type="mpas"))).bechtold.enable_cmt is False
    assert convection_config_for(dflt._replace(grid=dflt.grid._replace(grid_type="gaussian"))).bechtold.enable_cmt is True


def test_persistent_dgrid_flag_flows_to_config_1028():
    """--persistent-dgrid reaches DycoreConfig, and OFF is the default.

    #1028: carrying the cube's winds in FV3 D staggering between steps (rather
    than projecting to the corners and back every step) moved a 200-day C36
    Held-Suarez jet from 13.0 to 38.6 m/s.  The knob is only useful if a run
    script can select it, and only safe if the default leaves every existing
    run byte-identical -- both are asserted here.
    """
    parser = build_arg_parser()
    off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
    ]), parser))
    assert off.dycore.persistent_dgrid is False

    on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--persistent-dgrid",
    ]), parser))
    assert on.dycore.persistent_dgrid is True

    # A YAML that turns it on can still be overridden back off from the CLI.
    parser2 = build_arg_parser()
    parser2.set_defaults(persistent_dgrid=True)
    back_off = build_config_from_args(_postprocess_args(parser2.parse_args([
        "--dataset", "analytical", "--no-persistent-dgrid",
    ]), parser2))
    assert back_off.dycore.persistent_dgrid is False


def test_clubb_trop_cloud_top_press_reaches_the_turbulence_kernel():
    """The CLUBB upper domain limit (CAM trop_cloud_top_press) was a scheme
    field no run could set; the flag threads it through the same single source
    the prognostic flag uses, and None keeps the scheme's own 0 (off)."""
    from legoesm.driver.physics_pipeline import turbulence_config_for
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--turbulence", "clubb"]), parser))
    assert cfg_off.clubb_trop_cloud_top_press is None
    tc_off = turbulence_config_for(cfg_off)
    assert tc_off.clubb is None or tc_off.clubb.trop_cloud_top_press == 0.0
    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--turbulence", "clubb",
         "--clubb-trop-cloud-top-press", "15000"]), parser))
    assert turbulence_config_for(cfg_on).clubb.trop_cloud_top_press == 15000.0
    cfg_bad = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--turbulence", "louis",
         "--clubb-trop-cloud-top-press", "15000"]), parser))
    with pytest.raises(ValueError, match="requires turbulence='clubb'"):
        turbulence_config_for(cfg_bad)


def test_clubb_q_flux_scale_reaches_the_turbulence_kernel():
    """The cloud-base mixing probe threads through the same single source as
    the CLUBB top-press knob; None keeps the scheme's own 1.0 and the band
    lands on the kernel config.  The --params class router does NOT reach
    CLUBBConfig on the MPAS lane (the run refused with 'route into the built
    pipeline's config attributes'), which is why this is a driver field."""
    from legoesm.driver.physics_pipeline import turbulence_config_for
    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--turbulence", "clubb"]), parser))
    assert cfg_off.clubb_q_flux_scale is None
    from legoesm.atmosphere.physics.turbulence.integration import materialize_sub_config
    tc_off = materialize_sub_config(turbulence_config_for(cfg_off)).clubb
    assert (tc_off.q_flux_scale, tc_off.q_flux_scale_sigma_lo,
            tc_off.q_flux_scale_sigma_hi) == (1.0, 0.0, 1.0)
    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--turbulence", "clubb",
         "--clubb-q-flux-scale", "2.5",
         "--clubb-q-flux-scale-sigma-band", "0.8", "0.95"]), parser))
    tc_on = turbulence_config_for(cfg_on).clubb
    assert (tc_on.q_flux_scale, tc_on.q_flux_scale_sigma_lo,
            tc_on.q_flux_scale_sigma_hi) == (2.5, 0.8, 0.95)
    cfg_bad = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--turbulence", "louis",
         "--clubb-q-flux-scale", "2.5",
         "--clubb-q-flux-scale-sigma-band", "0.8", "0.95"]), parser))
    with pytest.raises(ValueError, match="requires turbulence='clubb'"):
        turbulence_config_for(cfg_bad)
    # The pipeline refuses on its own (validate_strict may not have run):
    # prognostic closure, a missing band, and an override that does not carry it.
    with pytest.raises(ValueError, match="clubb_prognostic=True does not read it"):
        turbulence_config_for(cfg_on._replace(clubb_prognostic=True))
    with pytest.raises(ValueError, match="requires clubb_q_flux_scale_sigma_band"):
        turbulence_config_for(cfg_on._replace(clubb_q_flux_scale_sigma_band=None))
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    with pytest.raises(ValueError, match="turbulence_override"):
        turbulence_config_for(cfg_on._replace(
            turbulence_override=TurbulenceConfig(scheme="clubb")))
    # ... even when the override carries the same scale (its band may differ).
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    with pytest.raises(ValueError, match="turbulence_override"):
        turbulence_config_for(cfg_on._replace(turbulence_override=TurbulenceConfig(
            scheme="clubb", clubb=CLUBBConfig(q_flux_scale=2.5))))
    # The MPAS lane (model_driver) builds its turbulence config through this
    # same turbulence_config_for, so the analytical-lane assertion covers it.


def test_clubb_q_flux_scale_validate_strict_refuses_bad_band_scheme_and_prognostic():
    """validate_strict refuses the probe without a band, a reversed band, the
    prognostic closure, a band with no scale, an out-of-bounds scale and a
    non-CLUBB scheme; the well-formed pair passes."""
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    parser = build_arg_parser()
    base = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--turbulence", "clubb"]), parser))
    base._replace(clubb_q_flux_scale=2.5,
                  clubb_q_flux_scale_sigma_band=(0.8, 0.95)).validate_strict()
    # A YAML-style list band is accepted by validate_strict (normalised by value).
    base._replace(clubb_q_flux_scale=2.5,
                  clubb_q_flux_scale_sigma_band=[0.8, 0.95]).validate_strict()
    for kw, msg in (
        (dict(clubb_q_flux_scale=2.5), "needs clubb_q_flux_scale_sigma_band"),
        (dict(clubb_q_flux_scale=2.5, clubb_q_flux_scale_sigma_band=(0.9, 0.8)),
         "0 <= lo < hi <= 1"),
        (dict(clubb_q_flux_scale=2.5, clubb_q_flux_scale_sigma_band=(0.8, 0.95),
              clubb_prognostic=True), "prognostic closure does not read it"),
        (dict(clubb_q_flux_scale_sigma_band=(0.8, 0.95)), "does nothing on its own"),
        (dict(clubb_q_flux_scale=50.0, clubb_q_flux_scale_sigma_band=(0.8, 0.95)),
         r"clubb_q_flux_scale=50"),
        (dict(clubb_q_flux_scale=float("nan"), clubb_q_flux_scale_sigma_band=(0.8, 0.95)),
         r"clubb_q_flux_scale=nan"),
        (dict(clubb_q_flux_scale=2.5, clubb_q_flux_scale_sigma_band=(0.8, 0.9, 0.95)),
         "0 <= lo < hi <= 1"),
        (dict(clubb_q_flux_scale=2.5, clubb_q_flux_scale_sigma_band=(0.8, 0.95),
              turbulence_override=TurbulenceConfig(scheme="clubb")),
         "refused with a turbulence_override"),
        (dict(turbulence="louis", clubb_q_flux_scale=2.5,
              clubb_q_flux_scale_sigma_band=(0.8, 0.95)), "is a CLUBB field"),
    ):
        with pytest.raises(ValueError, match=msg):
            base._replace(**kw).validate_strict()


def test_clubb_trop_cloud_top_press_validate_strict_bounds_and_scheme():
    """validate_strict refuses the taper under a non-CLUBB scheme (which the
    pipeline's late check cannot see behind a turbulence_override or the
    FV 'none' early return) and refuses non-physical pressures."""
    parser = build_arg_parser()
    base = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--turbulence", "clubb"]), parser))
    base._replace(clubb_trop_cloud_top_press=15000.0).validate_strict()
    for bad in (-1.0, float("nan"), float("inf"), 1.0e9):
        with pytest.raises(ValueError, match="clubb_trop_cloud_top_press"):
            base._replace(clubb_trop_cloud_top_press=bad).validate_strict()
    with pytest.raises(ValueError, match="CLUBB field"):
        base._replace(turbulence="louis",
                      clubb_trop_cloud_top_press=15000.0).validate_strict()


def test_clubb_liquid_partition_flag_reaches_the_turbulence_kernel():
    """--clubb-liquid-partition must round-trip AND reach the nested config.

    Same two halves as the prognostic flag: a lever that lands on
    ``ExperimentConfig`` but never on the CLUBB sub-config would report success
    while the closure kept throwing its liquid away.
    """
    from legoesm.driver.physics_pipeline import turbulence_config_for
    from legoesm.atmosphere.physics.turbulence.integration import (
        materialize_sub_config,
    )

    parser = build_arg_parser()
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_off.clubb_liquid_partition is False

    cfg_on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "clubb",
        "--clubb-prognostic",
        "--clubb-liquid-partition",
    ]), parser))
    assert cfg_on.clubb_liquid_partition is True
    assert turbulence_config_for(cfg_on).liquid_partition is True

    # ... and the default really is the other value on the same lane, so the
    # assertion above cannot pass by the sub-config defaulting True.
    cfg_plain = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "clubb",
        "--clubb-prognostic",
    ]), parser))
    plain_tc = materialize_sub_config(turbulence_config_for(cfg_plain))
    assert plain_tc.liquid_partition is False


def test_clubb_liquid_partition_requires_clubb_and_the_prognostic_path():
    """The exchanged liquid is the POST-ADVANCE PDF's rcm.

    Only CLUBB diagnoses one, and only the prognostic path advances the moments
    it is computed from, so both are hard requirements rather than hints.
    """
    import pytest
    from legoesm.driver.physics_pipeline import turbulence_config_for

    parser = build_arg_parser()
    wrong_closure = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "louis",
        "--clubb-liquid-partition",
    ]), parser))
    with pytest.raises(ValueError, match="requires turbulence='clubb'"):
        turbulence_config_for(wrong_closure)

    diagnostic = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "clubb",
        "--clubb-liquid-partition",
    ]), parser))
    with pytest.raises(ValueError, match="requires clubb_prognostic=True"):
        turbulence_config_for(diagnostic)


def test_fv3_duo_kessler_reaches_the_config_and_the_wall():
    """``--microphysics kessler`` on the duo argv reaches the config and
    passes the lane's default-deny wall (Kessler alone is the one routed
    scheme); a second active scheme next to it is still refused."""
    from legoesm.driver.component_factory import _refuse_fv3_duo_non_default
    cfg = _fv3_duo_cfg(["--microphysics", "kessler"])
    assert cfg.microphysics == "kessler"
    cfg.validate_strict()
    _refuse_fv3_duo_non_default(cfg)
    cfg = _fv3_duo_cfg(["--microphysics", "kessler", "--turbulence", "louis"])
    _refuse_fv3_duo_non_default(cfg)          # the wall allows the field...
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    with pytest.raises(ValueError, match="silently inert"):   # ...the guard does not
        create_atmosphere_dycore(cfg, create_cubed_sphere(12),
                                 create_sigma_coordinate(5))


def test_land_snow_scheme_and_emissivity_round_trip_and_decks():
    """--land-snow-scheme / --land-snow-emissivity reach ExperimentConfig, and each
    deck states its snowpack explicitly: both bulk (moving production to the
    layered pack is an open user decision)."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--use-multilayer-land"]
    cfg0 = build_config_from_args(_postprocess_args(parser.parse_args(base), parser))
    assert cfg0.land_snow_scheme == "bulk"
    cfg1 = build_config_from_args(_postprocess_args(parser.parse_args(
        base + ["--land-snow-scheme", "layered", "--land-snow-emissivity", "0.975"]),
        parser))
    assert cfg1.land_snow_scheme == "layered"
    assert cfg1.land_snow_emissivity == 0.975
    for deck, want in (("amip_production.yaml", "bulk"),
                       ("amip_sundqvist_l36.yaml", "bulk")):
        p = build_arg_parser()
        rows = load_yaml_config(str(_repo_root() / "config" / "amip" / deck), p)
        assert rows.get("land_snow_scheme") == want, deck
        p.set_defaults(**rows)
        cfg = build_config_from_args(_postprocess_args(
            p.parse_args(_AMIP_DUMMY_PATHS), p))
        assert cfg.land_snow_scheme == want, deck


@pytest.mark.parametrize("argv, match", [
    (["--land-snow-scheme", "layered"], "use_multilayer_land"),
    (["--use-multilayer-land", "--land-snow-emissivity", "0.9"], "land_snow_emissivity"),
])
def test_land_snow_options_refused_when_invalid(argv, match):
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical"] + argv), parser))
    with pytest.raises(ValueError, match=match):
        cfg.validate_strict()


def test_convective_buoyancy_death_memory_reaches_tiedtke_config():
    """--convective-buoyancy-death-memory must reach the Tiedtke scheme config
    the kernel is built from, not stop at ExperimentConfig."""
    from legoesm.driver.physics_pipeline import _resolve_convection

    parser = build_arg_parser()
    base = ["--grid", "gaussian", "--resolution", "21",
            "--convection", "tiedtke"]
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(base), parser))
    cfg_on = build_config_from_args(_postprocess_args(
        parser.parse_args(base + ["--convective-buoyancy-death-memory"]),
        parser))
    assert cfg_on.convective_buoyancy_death_memory is True
    assert _resolve_convection(cfg_off)[1].buoyancy_death_memory is False
    assert _resolve_convection(cfg_on)[1].buoyancy_death_memory is True


@pytest.mark.parametrize("scheme", ["bechtold", "kain_fritsch", "sbm"])
def test_buoyancy_death_memory_refused_off_tiedtke_in_python_config(scheme):
    """A config built in Python (no CLI) must refuse the flag with any scheme
    that does not read it, exactly as the CLI does."""
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig(convection=scheme,
                           convective_buoyancy_death_memory=True)
    with pytest.raises(ValueError, match="convective_buoyancy_death_memory"):
        cfg.validate_strict()
    # control: the same scheme with the flag at its default is not refused for it
    try:
        cfg._replace(convective_buoyancy_death_memory=False).validate_strict()
    except ValueError as e:
        assert "convective_buoyancy_death_memory" not in str(e)
    # and Tiedtke with the flag on is not refused for it
    try:
        cfg._replace(convection="tiedtke").validate_strict()
    except ValueError as e:
        assert "convective_buoyancy_death_memory" not in str(e)


def test_corner_fill_flag_round_trip_and_production_pin(capsys):
    parser = build_arg_parser()
    default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert default.dycore.corner_fill == "avg"
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--corner-fill", "fv3_bgrid_xdir"]), parser))
    assert cfg.dycore.corner_fill == "fv3_bgrid_xdir"
    cfg.validate_strict()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "analytical", "--corner-fill", "fv3_bgrid"])
    assert "invalid choice" in capsys.readouterr().err
    from pathlib import Path
    from legoesm.driver.run_config_yaml import load_yaml_config
    deck = Path(__file__).resolve().parents[2] / "config/amip/amip_production.yaml"
    parser = build_arg_parser()
    keys = load_yaml_config(str(deck), parser)
    assert keys["corner_fill"] == "avg"                          # deck -> loader
    parser.set_defaults(**keys)
    cfg = build_config_from_args(parser.parse_args([]))           # loader -> config
    assert cfg.dycore.corner_fill == "avg"


def test_zm_land_fraction_flows_to_config_and_kernel():
    """--zm-land-fraction threads CLI -> ExperimentConfig -> ZhangMcFarlaneConfig;
    the default requires a land fraction; validate_strict refuses anything else."""
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--convection", "zhang_mcfarlane"]), parser))
    assert cfg.zm_land_fraction == "required"
    cfg_aqua = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--convection", "zhang_mcfarlane",
         "--zm-land-fraction", "none"]), parser))
    assert cfg_aqua.zm_land_fraction == "none"
    with pytest.raises(ValueError, match="zm_land_fraction"):
        cfg._replace(zm_land_fraction="bogus").validate_strict()
    from legoesm.driver.physics_pipeline import _resolve_convection
    assert _resolve_convection(cfg)[1].land_fraction == "required"
    assert _resolve_convection(cfg_aqua)[1].land_fraction == "none"
    # The legacy AMIP export keeps the policy ("none" used to come back
    # "required", so a reloaded aquaplanet config failed setup).
    from legoesm.driver.config import ExperimentConfig
    for c in (cfg, cfg_aqua):
        assert ExperimentConfig.from_amip_config(
            c.to_amip_config()).zm_land_fraction == c.zm_land_fraction


def test_production_deck_pins_the_zm_land_fraction_policy():
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config(
        str(_repo_root() / "config" / "amip" / "amip_production.yaml"),
        build_arg_parser())
    assert keys["convection"] == "zhang_mcfarlane"
    assert keys["zm_land_fraction"] == "required"


def test_mpas_land_params_refresh_flag_flows_to_config():
    """--mpas-land-params-refresh round-trip: default ON (user 2026-09-29),
    --no- form reaches the config, and a YAML false pin is honoured."""
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.mpas_land_params_refresh is True

    cfg_off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--no-mpas-land-params-refresh",
    ]), parser))
    assert cfg_off.mpas_land_params_refresh is False

    parser2 = build_arg_parser()
    parser2.set_defaults(mpas_land_params_refresh=False)   # simulates a YAML pin
    cfg_pin = build_config_from_args(_postprocess_args(parser2.parse_args([
        "--dataset", "analytical"]), parser2))
    assert cfg_pin.mpas_land_params_refresh is False


def test_land_canopy_snow_masking_round_trip_and_refusal():
    """Off by default; the flag reaches ExperimentConfig both ways; the
    production deck does not set it (A/B arm only, user 2026-10-03); on a lane
    where it would be inert, validation refuses it."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--use-multilayer-land",
            "--snow-albedo-feedback"]
    cfg0 = build_config_from_args(_postprocess_args(parser.parse_args(base), parser))
    assert cfg0.land_canopy_snow_masking is False
    cfg1 = build_config_from_args(_postprocess_args(parser.parse_args(
        base + ["--land-canopy-snow-masking"]), parser))
    assert cfg1.land_canopy_snow_masking is True
    # the production deck does not set it, and with the switch on it validates
    p = build_arg_parser()
    rows = load_yaml_config(
        str(_repo_root() / "config" / "amip" / "amip_production.yaml"), p)
    assert "land_canopy_snow_masking" not in rows
    p.set_defaults(**rows)
    prod_on = build_config_from_args(_postprocess_args(
        p.parse_args(_AMIP_DUMMY_PATHS + ["--land-canopy-snow-masking"]), p))
    assert prod_on.land_canopy_snow_masking is True
    prod_on.validate_strict()
    bad = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--land-canopy-snow-masking"]), parser))
    with pytest.raises(ValueError, match="land_canopy_snow_masking"):
        bad.validate_strict()
