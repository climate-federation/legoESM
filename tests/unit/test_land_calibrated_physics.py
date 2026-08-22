"""The land model that was CALIBRATED must be the land model that is DEPLOYED.

Every baked per-PFT land table (albedo, emissivity, snow, soil thermal/hydraulic
scales, and the canopy conductance Vc_max25/g1/LCMA) is fitted offline by
``scripts/run/train_multilayer_land_era5.py`` inside one specific land model.
The coupled AMIP tile used to be assembled from a handful of independent flags,
and it was assembled WRONG: stomata were off, so ``compute_effective_beta`` took
its no-stomata branch and the baked canopy conductance never ran; the soil column
was 6.375 m against the 3 m it was fitted on.

These tests pin the repair: one shared definition
(``legoesm.land.config.calibrated_multilayer_setup``) that BOTH the calibrator
and the coupled driver build from, a gate that refuses a config claiming the
calibrated model while one key disagrees, and the carbon state without which the
same dispatch silently runs a DIFFERENT stomatal model (Jarvis, not Farquhar).
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.land.config import (
    MultiLayerLandConfig,
    apply_calibrated_multilayer,
    calibrated_multilayer_setup,
)
from legoesm.land.soil_grid import make_soil_grid


def test_calibrated_setup_selects_the_farquhar_branch():
    """The setup must pick the stomatal model the tables were fitted under.

    ``stomata_utils.compute_effective_beta`` has three branches: no stomata,
    Jarvis, and coupled Farquhar.  Only ``enabled`` AND ``carbon.scheme ==
    "differland"`` reaches Farquhar; either one alone lands somewhere else.
    """
    cal = calibrated_multilayer_setup()
    assert cal["stomata"].enabled is True
    assert cal["carbon"].scheme == "differland"
    assert cal["bulk_scheme"] == "most"
    # A fresh dict each call: a shared mutable module-level singleton would let
    # one consumer's _replace leak into every other consumer's land model.
    assert calibrated_multilayer_setup() is not cal


def test_calibrated_soil_column_is_not_the_library_default():
    """The calibration column and the library default are different columns.

    Both have eight layers, which is why the restart loader's shape check never
    caught the mismatch — but they span 3 m and 6.4 m and place every soil
    temperature and moisture value at a different depth.
    """
    cal_dz = make_soil_grid(calibrated_multilayer_setup()["soil_grid"]).dz
    default_dz = make_soil_grid(MultiLayerLandConfig().soil_grid).dz
    assert len(cal_dz) == len(default_dz) == 8
    assert float(np.sum(cal_dz)) == pytest.approx(3.0)
    assert not np.allclose(cal_dz, default_dz), (
        "the calibration soil column now equals the library default — if that is "
        "intentional this test is stale, but until then it is the drift this "
        "whole gate exists to catch"
    )


def test_apply_calibrated_multilayer_keeps_everything_else():
    """Applying the setup must change the model, not the per-cell maps."""
    base = MultiLayerLandConfig(albedo_land=0.42, beta_min=0.07,
                                soil_evap_litter_resistance_s_m=123.0)
    out = apply_calibrated_multilayer(base)
    assert out.stomata.enabled and out.carbon.scheme == "differland"
    # untouched fields survive verbatim
    assert out.albedo_land == 0.42
    assert out.beta_min == 0.07
    assert out.soil_evap_litter_resistance_s_m == 123.0


def test_apply_calibrated_multilayer_refuses_a_slab_config():
    """The tables are a multilayer bake; the slab has no equivalent."""
    from legoesm.land.config import LandConfig
    with pytest.raises(TypeError, match="MultiLayerLandConfig"):
        apply_calibrated_multilayer(LandConfig())


def test_offline_calibrator_builds_the_shared_definition():
    """The calibrator's own module state must BE the shared definition.

    This is the anti-drift half: if someone retunes on a different soil column or
    with stomata off and only edits the training script, this goes red instead of
    silently producing tables for a model nothing else runs.
    """
    import scripts.run.train_multilayer_land_era5 as T

    cal = calibrated_multilayer_setup()
    assert T._N_LAYERS == cal["soil_grid"].n_layers
    assert T._SOIL_DEPTH_M == cal["soil_grid"].total_depth
    assert T._SOIL_GROWTH == cal["soil_grid"].growth_factor
    assert T._BULK_SCHEME == cal["bulk_scheme"]
    assert T._STOMATA_ON == cal["stomata"].enabled
    # The whole stomatal / carbon configs, not just the on-off fields: the
    # calibrator derives them from the shared definition, so a future non-default
    # field reaches the fit and the deployment together instead of only one.
    assert T._CALIBRATED["stomata"] == cal["stomata"]
    assert T._CALIBRATED["carbon"] == cal["carbon"]
    assert T._CALIBRATED["surface_scheme"] == cal["surface_scheme"]


def test_amip_flag_flows_to_config_and_gate_rejects_disagreement():
    """The AMIP switch reaches the config, and every overlapping key is CHECKED.

    Checked rather than overridden: a run must never read as the calibrated land
    model while one key quietly runs something else.
    """
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    parser = build_arg_parser()
    cal = calibrated_multilayer_setup()

    off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert off.land_calibrated_physics is False

    # The mesh lane, because it is the only one the switch is allowed on (see
    # test_calibrated_physics_is_refused_off_the_mesh_lane).
    argv = [
        "--dataset", "analytical", "--grid-type", "mpas",
        "--land-mask-file", "lsm.nc",
        "--use-multilayer-land", "--land-calibrated-physics",
        "--land-stomatal-beta", "--land-surface-scheme", "simple_seb",
        "--snow-albedo-feedback", "--mpas-land-beta-soil",
        "--multilayer-n-layers", str(cal["soil_grid"].n_layers),
        "--multilayer-soil-depth", str(cal["soil_grid"].total_depth),
    ]
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(argv), parser))
    assert cfg.land_calibrated_physics is True
    cfg.validate_strict()   # complete and consistent: no error

    # Each overlapping key, disagreeing on its own, must fail.
    with pytest.raises(ValueError, match="land_stomatal_beta"):
        cfg._replace(land_stomatal_beta=False).validate_strict()
    with pytest.raises(ValueError, match="land_surface_scheme"):
        cfg._replace(land_surface_scheme="two_leaf").validate_strict()
    with pytest.raises(ValueError, match="multilayer_soil_depth"):
        cfg._replace(multilayer_soil_depth=6.375).validate_strict()
    with pytest.raises(ValueError, match="multilayer_n_layers"):
        cfg._replace(multilayer_n_layers=10).validate_strict()
    with pytest.raises(ValueError, match="snow_albedo_feedback"):
        cfg._replace(snow_albedo_feedback=False).validate_strict()
    with pytest.raises(ValueError, match="land_gs_max"):
        cfg._replace(land_gs_max=0.5).validate_strict()
    with pytest.raises(ValueError, match="use_multilayer_land"):
        cfg._replace(use_multilayer_land=False).validate_strict()


def test_committed_amip_overlay_is_consistent_with_the_calibrated_model():
    """The shipped overlay must pass its own gate.

    A config file that names the calibrated land model but sets one key against
    it is exactly the defect this work fixes, so the committed file is checked
    rather than trusted.
    """
    import yaml
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    with open("config/amip/amip_land_calibrated.yaml") as fh:
        overlay = yaml.safe_load(fh)
    assert overlay["land_calibrated_physics"] is True

    # Go through the REAL --config path, which also proves every key in the file
    # is a recognised setting (the loader rejects an unknown one outright, so a
    # renamed setting cannot leave this overlay quietly doing nothing).
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(
        "config/amip/amip_land_calibrated.yaml", parser))
    # The overlay is for the MESH (Voronoi/MPAS) lane — the only one that hands
    # the land tile's solved fluxes to the atmosphere — so it is validated there.
    cfg = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical",
                           "--land-mask-file", "lsm.nc",
                           "--grid-type", "mpas"]), parser))
    assert cfg.land_calibrated_physics is True, (
        "the overlay's keys did not reach the config — this test would then pass "
        "without checking anything"
    )
    cfg.validate_strict()

    # Non-vacuous: the same machinery must REJECT an overlay that disagrees.
    with pytest.raises(ValueError, match="land_calibrated_physics"):
        cfg._replace(multilayer_soil_depth=6.375).validate_strict()


def test_lmip_spinup_config_gate():
    """The land IC must be spun up on the same model, and the schema enforces it.

    A soil state is only meaningful for the model it equilibrated under, so a
    spin-up feeding a calibrated coupled run has to run the same land model.
    """
    import copy
    import yaml
    from legoesm.land.lmip_config import validate_config

    with open("config/lmip/amip_mpas4_spinup_calibrated.yaml") as fh:
        raw = yaml.safe_load(fh)
    assert raw["physics"]["calibrated_land_physics"] is True
    validate_config(copy.deepcopy(raw))   # committed config is valid

    for key, bad in (("surface_scheme", "two_leaf_canopy"),
                     ("bulk_scheme", "constant"),
                     ("stomata_enabled", False),
                     ("land_mode", "slab")):
        broken = copy.deepcopy(raw)
        broken["physics"][key] = bad
        with pytest.raises(ValueError, match="calibrated_land_physics|land_mode"):
            validate_config(broken)


def test_every_coupled_land_step_passes_the_leaf_carbon_state():
    """No coupled lane may step the land tile without the leaf-carbon state.

    The Farquhar stomatal model is reached only when the leaf-carbon state is
    handed to the land step; omit it and the SAME configuration silently runs the
    Jarvis model instead, with the baked canopy conductance unused.  There is one
    land step per lane (the shared physics pipeline, the unstructured mesh
    closure, the canopy warm start), and the mesh lane — the one the AMIP
    campaign actually runs — was the one that omitted it.

    Scanning every call site rather than the one that was wrong is the point: the
    next lane someone adds fails here instead of quietly reverting the model.
    Matching is by function NAME, so a call made through a renamed import or a
    wrapper would slip past — the realistic way a new lane appears is a copy of
    an existing call, which uses the real name.  A literal ``carbon_state=None``
    is treated as omitting it, since it selects the same wrong model.
    """
    import ast
    import pathlib

    root = pathlib.Path("packages/coupler/legoesm")
    missing = []
    checked = 0
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name not in ("step_multilayer_land",
                            "step_multilayer_land_with_diagnostics"):
                continue
            checked += 1
            # carbon_state is the 7th positional parameter, so a positional call
            # with >=7 arguments supplies it too.  A literal None is the same as
            # omitting it, so it does not count.
            def _is_none(n):
                return isinstance(n, ast.Constant) and n.value is None

            kw = next((k for k in node.keywords if k.arg == "carbon_state"), None)
            by_keyword = kw is not None and not _is_none(kw.value)
            by_position = len(node.args) >= 7 and not _is_none(node.args[6])
            if not (by_keyword or by_position):
                missing.append(f"{path}:{node.lineno}")
    assert checked >= 3, (
        f"only found {checked} coupled land-step call sites; the scan is not "
        "finding them, so this test proves nothing"
    )
    assert not missing, (
        "coupled land step(s) called without carbon_state, so these lanes run "
        "the Jarvis stomatal model no matter what the configuration asks for: "
        + ", ".join(missing)
    )


def test_mesh_lane_calibrated_run_must_hand_its_fluxes_to_the_atmosphere():
    """On the mesh lane the plant model must actually reach the atmosphere.

    That lane solves the land tile's humidity and fluxes and then publishes them
    only under its own switch; with the switch off the atmosphere keeps a static
    evaporation efficiency and the fitted canopy conductance moves the land tile's
    own temperature and nothing else.  A calibrated run in that state has exactly
    the inert-parameter defect this work removes, so it is refused.
    """
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    parser = build_arg_parser()
    cal = calibrated_multilayer_setup()
    argv = [
        "--dataset", "analytical", "--grid-type", "mpas",
        "--land-mask-file", "lsm.nc",
        "--use-multilayer-land", "--land-calibrated-physics",
        "--land-stomatal-beta", "--land-surface-scheme", "simple_seb",
        "--snow-albedo-feedback", "--mpas-land-beta-soil",
        "--multilayer-n-layers", str(cal["soil_grid"].n_layers),
        "--multilayer-soil-depth", str(cal["soil_grid"].total_depth),
    ]
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(argv), parser))
    assert cfg.mpas_land_beta_soil is True
    cfg.validate_strict()          # complete: no error

    with pytest.raises(ValueError, match="mpas_land_beta_soil"):
        cfg._replace(mpas_land_beta_soil=False).validate_strict()


def test_calibrated_physics_is_refused_off_the_mesh_lane():
    """Only the lane that hands over SOLVED fluxes may deploy the tables.

    Every other lane re-derives the land flux from a scaled saturation humidity.
    That cannot reproduce the calibrated canopy conductance: the land scheme
    applies its throttle to the humidity GRADIENT and bypasses it for snow and
    dew, so the re-derivation can reach the opposite SIGN, and the atmosphere's
    exchange coefficient uses a scalar roughness where the land uses a per-column
    tuned one.  Refusing beats deploying the tables under a coupling that cannot
    express them.
    """
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    parser = build_arg_parser()
    cal = calibrated_multilayer_setup()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-mask-file", "lsm.nc",
        "--use-multilayer-land", "--land-calibrated-physics",
        "--land-stomatal-beta", "--land-surface-scheme", "simple_seb",
        "--snow-albedo-feedback", "--surface-tiled", "--turbulence", "louis",
        "--multilayer-n-layers", str(cal["soil_grid"].n_layers),
        "--multilayer-soil-depth", str(cal["soil_grid"].total_depth),
    ]), parser))
    # Even WITH the tiled surface, a structured grid is refused: the tiled path
    # re-derives the flux rather than receiving it.
    with pytest.raises(ValueError, match="only on the MPAS"):
        cfg.validate_strict()


def test_calibrated_mesh_run_needs_a_turbulence_scheme_that_takes_the_fluxes():
    """A run that could never couple the land fluxes must fail at config time.

    The land tile's solved fluxes are handed to the turbulence kernel, and a
    kernel whose signature has no place for them refuses at RUN time — hours
    into a job.  The same signature scan the runtime guard uses is asked here.
    """
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    from legoesm.atmosphere.physics.turbulence.integration import (
        schemes_accepting_surface_flux,
    )
    accepting = schemes_accepting_surface_flux()
    assert accepting, "no turbulence scheme takes the fluxes — the scan is broken"

    parser = build_arg_parser()
    cal = calibrated_multilayer_setup()
    argv = [
        "--dataset", "analytical", "--grid-type", "mpas",
        "--land-mask-file", "lsm.nc",
        "--use-multilayer-land", "--land-calibrated-physics",
        "--land-stomatal-beta", "--land-surface-scheme", "simple_seb",
        "--snow-albedo-feedback", "--mpas-land-beta-soil",
        "--turbulence", accepting[0],
        "--multilayer-n-layers", str(cal["soil_grid"].n_layers),
        "--multilayer-soil-depth", str(cal["soil_grid"].total_depth),
    ]
    build_config_from_args(
        _postprocess_args(parser.parse_args(argv), parser)).validate_strict()

    # A scheme that cannot take them is refused now, not at run time.
    refusing = [s for s in ("smagorinsky", "louis", "tke", "mynn25",
                            "clubb_lite", "clubb", "holtslag_boville", "ysu",
                            "edmf") if s not in accepting]
    if not refusing:
        pytest.skip("every turbulence scheme takes the surface fluxes")
    bad = argv.copy()
    bad[bad.index("--turbulence") + 1] = refusing[0]
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(bad), parser))
    with pytest.raises(ValueError, match="surface_flux"):
        cfg.validate_strict()


def test_a_mesh_run_with_no_land_anywhere_is_refused():
    """A calibrated run that would build no soil column at all must not start.

    Flat topography with no land-mask file derives a land fraction of zero
    everywhere, so nothing is built and the flux handoff — and the calibrated
    canopy conductance riding it — would be silently inert.  The mesh lane never
    reached the existing version of this check, which was gated on a setting the
    mesh lane does not use.
    """
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    parser = build_arg_parser()
    cal = calibrated_multilayer_setup()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--grid-type", "mpas",
        "--topography", "flat",              # and NO --land-mask-file
        "--use-multilayer-land", "--land-calibrated-physics",
        "--land-stomatal-beta", "--land-surface-scheme", "simple_seb",
        "--snow-albedo-feedback", "--mpas-land-beta-soil",
        "--turbulence", "louis",
        "--multilayer-n-layers", str(cal["soil_grid"].n_layers),
        "--multilayer-soil-depth", str(cal["soil_grid"].total_depth),
    ]), parser))
    with pytest.raises(ValueError, match="silently inert"):
        cfg.validate_strict()


def test_a_wrong_column_land_ic_is_caught_before_any_land_work(tmp_path):
    """The soil-column check must happen where EVERY rank performs it.

    Under a cell-partition the land state is loaded only by ranks that own land,
    so a check there aborts those ranks while the ocean-only ones walk on into
    the next collective and wait forever.  The check therefore runs up front,
    from the config and the restart file's own record — both identical on every
    rank — before any grid, partition or land work.
    """
    import numpy as np
    from legoesm.land.restart import (
        load_land_restart_soil_dz, save_land_restart)
    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
    from legoesm.land.state import MultiLayerLandState
    import jax.numpy as jnp
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    from legoesm.driver.model_driver import ModelDriver

    n_col, n_lay = 4, 8
    state = MultiLayerLandState(
        T_soil=jnp.full((n_col, n_lay), 280.0),
        psi_soil=jnp.full((n_col, n_lay), -1.0),
        theta_soil=jnp.full((n_col, n_lay), 0.3),
        runoff_surface=jnp.zeros(n_col), runoff_subsurface=jnp.zeros(n_col),
        snow_depth=jnp.zeros(n_col), snow_age=jnp.zeros(n_col), TgC=None)
    other = make_soil_grid(SoilGridConfig(n_layers=n_lay, total_depth=6.375,
                                          growth_factor=2.0)).dz
    ic = tmp_path / "wrong_column.npz"
    save_land_restart(ic, state, land_mode="multilayer", t_end_s=0.0,
                      n_steps_completed=0, soil_dz=other)
    assert load_land_restart_soil_dz(ic) is not None, "the stamp was not written"

    cal = calibrated_multilayer_setup()
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--grid-type", "mpas",
        "--land-mask-file", "lsm.nc",
        "--use-multilayer-land", "--land-calibrated-physics",
        "--land-stomatal-beta", "--land-surface-scheme", "simple_seb",
        "--snow-albedo-feedback", "--mpas-land-beta-soil",
        "--turbulence", "louis", "--land-ic", str(ic),
        "--multilayer-n-layers", str(cal["soil_grid"].n_layers),
        "--multilayer-soil-depth", str(cal["soil_grid"].total_depth),
    ]), parser))

    # The preflight alone rejects it — no grid, no partition, no land tile.
    driver = ModelDriver(cfg, output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="different soil column"):
        driver._preflight_land_inputs()

    # And the run refuses to start.
    with pytest.raises(ValueError, match="different soil column"):
        ModelDriver(cfg, output_dir=tmp_path / "out2").setup()

    # A restart on THIS run's column passes, so the check discriminates.
    ok = tmp_path / "right_column.npz"
    save_land_restart(ok, state, land_mode="multilayer", t_end_s=0.0,
                      n_steps_completed=0,
                      soil_dz=make_soil_grid(cal["soil_grid"]).dz)
    ModelDriver(cfg._replace(land_ic_path=str(ok)),
                output_dir=tmp_path / "out3")._preflight_land_inputs()


def test_a_missing_land_ic_is_refused_up_front(tmp_path):
    """A land initial condition that is not there must fail before the run starts.

    Reported only by the loader, this would surface on ranks that own land while
    the rest walked on into the next collective.
    """
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    from legoesm.driver.model_driver import ModelDriver

    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--land-mask-file", "lsm.nc",
        "--use-multilayer-land",
        "--land-ic", str(tmp_path / "does_not_exist.npz"),
    ]), parser))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ModelDriver(cfg, output_dir=tmp_path / "out")._preflight_land_inputs()


def test_one_definition_of_two_soil_columns_being_the_same():
    """The before-the-run check and the loader must agree on "the same column".

    Two separate comparisons with separately written tolerances would eventually
    disagree, and the disagreement would look like a flaky restart.
    """
    import numpy as np
    from legoesm.land.restart import soil_dz_matches

    a = np.array([0.06, 0.09, 0.14, 0.21])
    assert soil_dz_matches(a, a.copy())
    assert soil_dz_matches(a, a * (1.0 + 1e-7))          # float32-vs-64 noise
    assert not soil_dz_matches(a, a * 1.01)              # a real 1% difference
    assert not soil_dz_matches(a, a[:3])                 # different layer count


def test_soil_can_be_started_near_saturation():
    """The soil-moisture fraction must be a REAL control, not an inert one.

    The default seeding maps the initial atmosphere's humidity into the
    plant-available range, which caps the start at FIELD CAPACITY and leaves the
    fraction doing nothing — measured, two coupled arms differing only in it were
    byte-identical.  Starting near saturation is how a run is kept out of the
    dry-soil attractor, so the mode has to be selectable and the fraction has to
    bite.
    """
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--land-mask-file", "lsm.nc",
            "--use-multilayer-land"]

    default = build_config_from_args(_postprocess_args(
        parser.parse_args(base), parser))
    assert default.land_soil_init == "aridity", "the default seeding changed"

    wet = build_config_from_args(_postprocess_args(parser.parse_args(
        base + ["--land-soil-init", "saturation_fraction",
                "--land-soil-moisture-init-frac", "0.9"]), parser))
    assert wet.land_soil_init == "saturation_fraction"
    assert wet.land_soil_moisture_init_frac == pytest.approx(0.9)
    wet.validate_strict()

    with pytest.raises(ValueError, match="land_soil_init"):
        wet._replace(land_soil_init="soaking").validate_strict()


def test_a_near_saturated_seed_stays_inside_the_retention_range():
    """A wet seed must stay strictly below porosity.

    The van-Genuchten retention is singular AT saturation, so seeding exactly at
    porosity would hand the Richards solver an infinite matric head on the first
    step rather than a wet soil.
    """
    import jax.numpy as jnp
    import numpy as np
    from legoesm.driver.model_driver import _THETA_EDGE_GUARD

    theta_sat = np.array([0.43, 0.45, 0.38])
    theta_r = np.array([0.078, 0.067, 0.068])
    seeded = np.asarray(jnp.clip(1.0 * jnp.asarray(theta_sat),
                                 jnp.asarray(theta_r) + _THETA_EDGE_GUARD,
                                 jnp.asarray(theta_sat) - _THETA_EDGE_GUARD))
    assert np.all(seeded < theta_sat), "seeded at or above porosity"
    assert np.all(seeded > theta_r), "seeded at or below the residual"
    # And it really is NEAR saturation, not quietly pulled back to mid-range.
    assert np.all(seeded > 0.95 * theta_sat)


def test_the_default_land_surface_is_the_two_leaf_canopy():
    """The production default must be the canopy, not the simplified scheme.

    Measured on a well-watered column, one day, realistic forcing: the simplified
    scheme evaporates at potential with stomata off (405 W/m2 over forest) and
    throttles BARE GROUND to 2 W/m2 with them on, because its conductance has no
    leaf-area dependence.  It is for academic tests; the canopy is the default.
    """
    from scripts.run.run_amip import (
        build_arg_parser, build_config_from_args, _postprocess_args,
    )
    from legoesm.land.config import LandConfig, MultiLayerLandConfig
    from legoesm.land.surface_scheme import SimpleSEBConfig

    for cfg in (LandConfig(), MultiLayerLandConfig()):
        assert not isinstance(cfg.surface_scheme, SimpleSEBConfig), (
            f"{type(cfg).__name__} defaults to the simplified surface scheme")
        assert cfg.bulk_scheme == "most", (
            f"{type(cfg).__name__} does not default to Monin-Obukhov exchange")

    parser = build_arg_parser()
    multi = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--land-mask-file", "l.nc",
         "--use-multilayer-land"]), parser, []))
    assert multi.land_surface_scheme == "two_leaf"

    # A slab-only run must still START: the canopy runs inside the multilayer
    # tile, so a default it cannot honour must not refuse a run that never asked
    # for a canopy.  The field is left ALONE (rewriting it would make it
    # non-default, which the strict single-purpose lanes reject outright) and is
    # inert on the slab by construction; the warning is what stops that being
    # silent.
    slab = _postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser, [])
    assert slab.land_surface_scheme == "two_leaf"
    assert slab.use_multilayer_land is False

    # But asking for it explicitly without the tile is still refused.
    with pytest.raises(SystemExit):
        _postprocess_args(
            parser.parse_args(["--dataset", "analytical",
                               "--land-surface-scheme", "two_leaf"]),
            parser, ["--land-surface-scheme", "two_leaf"])


def test_a_canopy_gets_per_pft_canopy_parameters(monkeypatch, tmp_path):
    """A canopy scheme on the coupled path must receive CANOPY parameters.

    They already existed and were already per-plant-type — canopy height, C4
    fraction, Vcmax25, roughness ratio, band albedos — built by the same routine
    the offline LMIP simulations use.  The coupled driver simply read a different
    provider, whose only variants are soil ones, so the canopy fell back to
    generic constants and every tuned per-PFT value went inert.
    """
    import sys
    sys.path.insert(0, "tests/unit")
    from test_multilayer_land_driver import _patch_land_loaders, _small_cfg
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.land.canopy.config import CanopyLandParams

    _patch_land_loaders(monkeypatch)
    cfg = _small_cfg()._replace(land_surface_scheme="two_leaf",
                                surfdata_path="data/legoesm_surfdata_c260716.nc")
    import os
    if not os.path.exists(cfg.surfdata_path):
        pytest.skip("harmonized surfdata not staged in this checkout")

    driver = ModelDriver(cfg, output_dir=tmp_path / "canopy")
    driver.setup()
    params = driver.physics.land_ml_params
    assert isinstance(params, CanopyLandParams), (
        f"canopy scheme received {type(params).__name__}, so its tuned per-PFT "
        "values are inert")
    # And they VARY by column — a constant would mean the fallback, dressed up.
    import numpy as np
    for field in ("hc", "rz0m", "Vcmax25_C3_leaf"):
        v = np.asarray(getattr(params, field))
        assert np.ptp(v[np.isfinite(v)]) > 0.0, (
            f"{field} is uniform across columns — that is the generic fallback, "
            "not per-plant-type parameters")
    # HONEST LIMIT, pinned so it is not forgotten (GLM): the original defect
    # named roughness, albedo AND emissivity. Roughness is fixed above. The
    # canopy builder emits a SINGLE emissivity, and the canopy computes its own
    # radiation from soil colour plus leaf optics rather than a bulk vegetation
    # albedo — so the tuned per-PFT emissivity and albedo do NOT reach it. That
    # is a re-fit item, not wiring, and this asserts the current truth so a
    # future change that fixes it fails here and gets noticed.
    assert np.ptp(np.asarray(params.emissivity)) == 0.0, (
        "emissivity now varies by column — if that is intended, this test is "
        "stale and the tuned emissivity may finally be reaching the canopy")
    # And the two datasets agreed on the column count.
    assert np.asarray(params.hc).shape[0] == int(driver.grid.lat.size)


def test_a_canopy_without_its_surfdata_is_refused(monkeypatch, tmp_path):
    """No silent generic-constant canopy: refuse instead."""
    import sys
    sys.path.insert(0, "tests/unit")
    from test_multilayer_land_driver import _patch_land_loaders, _small_cfg
    from legoesm.driver.model_driver import ModelDriver

    _patch_land_loaders(monkeypatch)
    cfg = _small_cfg()._replace(land_surface_scheme="two_leaf", surfdata_path="")
    with pytest.raises(ValueError, match="CANOPY parameters"):
        ModelDriver(cfg, output_dir=tmp_path / "nosd").setup()


def test_a_structured_grid_canopy_survives_radiation(monkeypatch, tmp_path):
    """A canopy run must reach RADIATION, not just setup.

    Radiation asks the land for one broadband albedo.  A bulk surface supplies it
    directly; a canopy parameter set has no such field — it carries the two solar
    band albedos and does its own transfer.  Reading the bulk field threw on
    every structured-grid canopy run, at the first radiation call, after setup had
    already reported success.
    """
    import os
    import sys
    import numpy as np
    sys.path.insert(0, "tests/unit")
    from test_multilayer_land_driver import _patch_land_loaders, _small_cfg
    from legoesm.driver.model_driver import ModelDriver

    sd = "data/legoesm_surfdata_c260716.nc"
    if not os.path.exists(sd):
        pytest.skip("harmonized surfdata not staged in this checkout")
    _patch_land_loaders(monkeypatch)
    driver = ModelDriver(
        _small_cfg()._replace(land_surface_scheme="two_leaf", surfdata_path=sd),
        output_dir=tmp_path / "rad")
    driver.setup()

    # Step it: setup succeeding proves nothing about the radiation call.
    run_seg, carry0, forcing = driver.build_training_segment(2)
    out = run_seg(carry0, 2, forcing)
    _T = getattr(out.T, "data", out.T)
    assert np.all(np.isfinite(np.asarray(_T))), (
        "a structured-grid canopy run produced non-finite temperatures")
    assert np.all(np.isfinite(np.asarray(out.land_ml.T_soil)))
