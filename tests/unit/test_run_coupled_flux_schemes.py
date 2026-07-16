"""CLI + dispatch tests for the per-tile surface-flux schemes wired into
run_coupled: land=MOST, slab ocean=MOST, ocean air-sea=COARE.

Covers (1) the run_coupled argument parser flags + defaults and (2) the slab
ocean "most" dispatch added to SimpleOcean._ocean_turbulent_fluxes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_mod", REPO / "scripts" / "run" / "run_coupled.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_flux_scheme_flag_defaults():
    """Defaults encode the user policy: land=MOST, slab=MOST, ocean air-sea
    scheme defaults 'constant' (set to coare3 at run time)."""
    args = mod.build_parser().parse_args([])
    assert args.land_bulk_scheme == "most"
    assert args.slab_bulk_scheme == "most"
    assert args.surface_bulk_scheme == "constant"


def test_flux_scheme_flags_roundtrip():
    """Each per-tile flag round-trips through the parser."""
    args = mod.build_parser().parse_args([
        "--land-bulk-scheme", "most",
        "--slab-bulk-scheme", "most",
        "--surface-bulk-scheme", "coare3",
    ])
    assert args.land_bulk_scheme == "most"
    assert args.slab_bulk_scheme == "most"
    assert args.surface_bulk_scheme == "coare3"


def test_surface_bulk_scheme_rejects_most():
    """'most' is NOT a valid air-sea scheme -- it must not parse.

    This test previously asserted the opposite ("'most' is now a valid choice
    for the air-sea/atm scheme too") and pinned a phantom. The history:
      2026-06-22 b5c423119  validate_strict starts REJECTING most
                            (_valid_surface_bulk = constant/coare3/large_yeager)
      2026-06-24 57979cdbf  most ADDED to these choices (this test came with it)
                            -> `--surface-bulk-scheme most` parsed, then died at
                            ModelDriver construction
      2026-07-03 e8bb82c53  review adds the flag's NOTE: "'most' is deliberately
                            NOT offered here ... the atmosphere surface layer's
                            compute_surface_fluxes treats 'most' as constant
                            (fixed-roughness LAND scheme), so offering it would
                            silently split the interface" -- but only wrote the
                            comment; the choices list stayed stale.
    The NOTE is the latest word and the code now matches it. Fixing this by
    WIDENING _valid_surface_bulk instead would be worse than the crash:
    turbulence/surface_layer.py gates the MOST path on ("coare3",
    "large_yeager") only, so an accepted "most" silently degrades to the
    constant-coefficient branch -- a loud crash traded for wrong physics.

    The LAND and SLAB tiles legitimately use MOST; that is covered by
    test_flux_scheme_flags_roundtrip via --land-bulk-scheme/--slab-bulk-scheme.
    """
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--surface-bulk-scheme", "most"])


@pytest.mark.parametrize("bad_flag", [
    "--land-bulk-scheme", "--slab-bulk-scheme", "--surface-bulk-scheme",
])
def test_flux_scheme_rejects_unknown(bad_flag):
    """argparse choices reject a typo'd scheme (dispatch hardening at the CLI)."""
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args([bad_flag, "garbage_scheme"])


def test_stomata_flag():
    """Stomatal conductance is DEFAULT ON for coupled runs; --no-stomata opts out."""
    assert mod.build_parser().parse_args([]).stomata is True
    assert mod.build_parser().parse_args(["--stomata"]).stomata is True
    assert mod.build_parser().parse_args(["--no-stomata"]).stomata is False


def test_coupling_diag_days_clamp():
    """diag_days is the atm-ocean coupling interval for the dynamic ocean; it is
    clamped to <=10 (the stale-SST overheating-instability guard), and left
    untouched for slab/two_layer or when already tight."""
    f = mod.clamp_coupling_diag_days
    assert f("dynamic", 20) == 10          # loose -> clamped
    assert f("dynamic", 30) == 10
    assert f("dynamic", 10) == 10          # at the cap -> unchanged
    assert f("dynamic", 5) == 5            # tight -> unchanged
    assert f("slab", 20) == 20             # slab ocean unaffected
    assert f("two_layer", 60) == 60
    assert f("fixed", 30) == 30


def test_slab_ocean_most_dispatch_finite():
    """SimpleOcean._ocean_turbulent_fluxes accepts scheme='most' and returns
    finite SH/LH (the slab=MOST path)."""
    import jax.numpy as jnp
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, _ocean_turbulent_fluxes,
    )
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.core.coupling_fields import AtmToSurface

    shape = (8, 16)
    z = jnp.zeros(shape)
    fc = AtmToSurface(
        sw_down=jnp.full(shape, 200.0), lw_down=jnp.full(shape, 350.0),
        precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, 288.0), q_lowest=jnp.full(shape, 8e-3),
        u_lowest=jnp.full(shape, 4.0), v_lowest=z,
        p_lowest=jnp.full(shape, 1.0e5), p_surface=jnp.full(shape, 1.0e5),
        rho_lowest=jnp.full(shape, 1.15), cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(0.0),
    )
    T_sfc = jnp.full(shape, 295.0)
    q_sfc = saturation_mixing_ratio(T_sfc, fc.p_surface)
    sh, lh = _ocean_turbulent_fluxes(
        T_sfc, q_sfc, fc, SimpleOceanConfig(bulk_scheme="most"))
    assert jnp.all(jnp.isfinite(sh)) and jnp.all(jnp.isfinite(lh))


def test_slab_ocean_unknown_scheme_raises():
    """Dispatch hardening: an unknown slab scheme still raises (not silent)."""
    import jax.numpy as jnp
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, _ocean_turbulent_fluxes,
    )
    from legoesm.core.coupling_fields import AtmToSurface

    shape = (4, 4)
    z = jnp.zeros(shape)
    fc = AtmToSurface(
        sw_down=z, lw_down=z, precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, 288.0), q_lowest=jnp.full(shape, 8e-3),
        u_lowest=jnp.full(shape, 4.0), v_lowest=z,
        p_lowest=jnp.full(shape, 1.0e5), p_surface=jnp.full(shape, 1.0e5),
        rho_lowest=jnp.full(shape, 1.15), cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(0.0),
        has_precipitation=jnp.array(0.0),
    )
    with pytest.raises(ValueError, match="Unknown SimpleOceanConfig.bulk_scheme"):
        _ocean_turbulent_fluxes(
            jnp.full(shape, 295.0), jnp.full(shape, 1e-2), fc,
            SimpleOceanConfig(bulk_scheme="garbage"))


# ===========================================================================
# Land runoff scheme reachability
# ===========================================================================

def test_land_runoff_scheme_reaches_the_slab_land_config():
    """--land-runoff-scheme topmodel must reach LandConfig.runoff_scheme.

    TOPMODEL is implemented in slab_land, param-spec'd as ``land.topmodel``, and
    its dispatch already raises on an unknown name -- but land_scheme_overrides
    built ``LandConfig()`` with NO arguments, pinning runoff_scheme at its
    "bucket" default. The scheme was unselectable from any driver.
    """
    assert mod.build_parser().parse_args([]).land_runoff_scheme == "bucket"
    ov = mod.land_scheme_overrides("slab", "topmodel")
    assert ov["land_config"].runoff_scheme == "topmodel"


def test_land_runoff_default_is_unchanged():
    """A default run must stay byte-identical."""
    assert mod.land_scheme_overrides("slab", "bucket")["land_config"].runoff_scheme == "bucket"


def test_land_runoff_scheme_rejects_multilayer_loudly():
    """MultiLayerLandConfig has NO runoff_scheme field (it resolves runoff
    through its Richards column), so the flag cannot be honoured there. Reject
    rather than silently ignore -- the same failure as `--iwm` being dropped
    under `--vertical-mixing-scheme catke`."""
    with pytest.raises(SystemExit, match="SLAB"):
        mod.land_scheme_overrides("multilayer", "topmodel")


def test_land_runoff_multilayer_ok_at_default():
    """multilayer + the default bucket must NOT raise (nothing was requested)."""
    assert mod.land_scheme_overrides("multilayer", "bucket")["land_mode"] == "multilayer"


def test_land_runoff_cli_rejects_unknown():
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--land-runoff-scheme", "garbage"])


# ===========================================================================
# Sea ice: the whole component had NO cli surface
# ===========================================================================

def test_sea_ice_default_is_none_so_a_default_run_is_unchanged():
    assert mod.build_sea_ice_config(mod.build_parser().parse_args([])) is None


@pytest.mark.parametrize("flag,sub,extra", [
    ("--ice-snow", "snow", []),
    ("--ice-brine", "brine", []),
    # ridging is inert without multi-category ice + a dynamics mode, and the
    # builder now REFUSES that combination -- so ask for a runnable one.
    ("--ice-ponds", "ponds", []),
])
def test_sea_ice_sub_models_are_selectable(flag, sub, extra):
    """Each sub-model is gated by a ``bool = False``.

    Bools are NOT ``:float``-spec-eligible, so ``--params`` can never reach
    them, and run_coupled built ``SeaIceConfig()`` with no arguments and had no
    ``--ice*`` flag at all -- so snow, brine and melt ponds were every one of
    them impossible to switch on from any driver.

    RIDGING is deliberately absent: it needs multi-category ice, which NO
    driver builds, so it gets a rejection test below instead of an offer.
    """
    cfg = mod.build_sea_ice_config(mod.build_parser().parse_args([flag] + extra))
    assert getattr(cfg, sub).enabled is True


def test_enabling_a_sub_model_preserves_its_other_tuned_fields():
    """`_replace(enabled=True)` must not clobber the calibrated defaults."""
    from legoesm.ice.config import SeaIceConfig

    cfg = mod.build_sea_ice_config(mod.build_parser().parse_args(["--ice-snow"]))
    assert cfg.snow._replace(enabled=False) == SeaIceConfig().snow


@pytest.mark.parametrize("flag,field,value,extra", [
    ("--ice-shortwave-scheme", "shortwave_scheme", "delta_eddington", []),
    # each of these is INERT without its companion, and the builder refuses the
    # inert form -- so pass the combination that actually runs.
    ("--ice-bulk-scheme", "bulk_scheme", "most", []),
    ("--ice-stability-scheme", "stability_scheme", "grachev2007_sheba",
     ["--ice-bulk-scheme", "most"]),
])
def test_sea_ice_scheme_fields_are_selectable(flag, field, value, extra):
    """`str` fields -- also unreachable via --params (:float-only)."""
    cfg = mod.build_sea_ice_config(
        mod.build_parser().parse_args([flag, value] + extra))
    assert getattr(cfg, field) == value


@pytest.mark.parametrize("flag", [
    "--ice-shortwave-scheme", "--ice-bulk-scheme",
    "--ice-stability-scheme",
])
def test_sea_ice_scheme_flags_reject_unknown(flag):
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args([flag, "garbage_scheme"])


def test_cli_sea_ice_config_survives_the_params_layer():
    """THE load-bearing pin: assert what main() actually threads.

    apply_coupled_params REBUILDS the bundle and returns its ice config, so a
    CLI-built one had to be seeded INTO it -- otherwise --ice-ridging was
    silently discarded the moment --params was also passed.

    This also shows the gap precisely: ``ice.ridging.e_star`` was ALWAYS
    tunable via --params, while ``ridging.enabled`` is a bool that --params can
    never reach. You could tune ridging's coefficients but never turn ridging
    on.
    """
    import json
    import tempfile
    from pathlib import Path

    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.coupled_config import CoupledConfig

    ice = mod.build_sea_ice_config(mod.build_parser().parse_args(["--ice-snow"]))
    p = Path(tempfile.mkdtemp()) / "params.json"
    p.write_text(json.dumps({"ice.ridging.e_star": 0.5}))
    out = mod.apply_coupled_params(str(p), "analytical", ExperimentConfig(),
                                   CoupledConfig(), None, ice)
    ice_out = out[3]
    assert ice_out.snow.enabled is True, "--ice-snow lost to --params"
    assert ice_out.ridging.e_star == 0.5, "--params did not apply"


def test_main_actually_threads_the_cli_sea_ice_config():
    """main() must BUILD the ice config from args, not pass None.

    Every test above exercises build_sea_ice_config / apply_coupled_params.
    Reverting main()'s `ice_config = build_sea_ice_config(args)` back to
    `ice_config = None` leaves them ALL green -- the flags would parse, the
    helpers would work, and the driver would still get None. That is exactly
    how `--ddm` shipped inert (codex), so pin the call site itself.

    AST rather than a full main() run: main() builds grids and a driver, which
    a unit test cannot afford, but the wiring is a static fact.
    """
    import ast
    from pathlib import Path

    src = Path("scripts/run/run_coupled.py").read_text()
    tree = ast.parse(src)
    main_fn = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = {
        n.func.id for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "build_sea_ice_config" in calls, (
        "main() never calls build_sea_ice_config -- every --ice-* flag would "
        "parse and be silently dropped before reaching CoupledESMDriver"
    )
    # ...and the RESULT must be what the driver receives. Checking only that
    # the kwarg EXISTS is tautological: `ice_config = None` followed by
    # `CoupledESMDriver(..., ice_config=None)` satisfies it while dropping
    # every flag -- exactly the regression this test claims to catch (codex).
    # So follow the dataflow: name assigned from the builder == name passed.
    built = {
        t.id for n in ast.walk(main_fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "build_sea_ice_config"
        for t in n.targets if isinstance(t, ast.Name)
    }
    assert built, "build_sea_ice_config() result is never assigned"

    driver_call = next(
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "CoupledESMDriver"
    )
    passed = {k.arg: k.value for k in driver_call.keywords}
    assert "ice_config" in passed, "CoupledESMDriver is not given ice_config"
    val = passed["ice_config"]
    assert isinstance(val, ast.Name) and val.id in built, (
        f"CoupledESMDriver(ice_config=...) is not the build_sea_ice_config "
        f"result (got {ast.dump(val)}); every --ice-* flag would be dropped"
    )


# --- no --ice-* flag may be silently INERT ---------------------------------
#
# Enabling a sub-model is not the same as reaching it. Codex found --ice-ridging
# and --ice-itd-remap still did NOTHING at the defaults -- i.e. the sea-ice fix
# had reproduced the --ddm mistake it was written to fix. These pin the guards.

def test_ridging_is_rejected_because_no_driver_can_run_it():
    """Ridging needs multi-category ice AND an ice velocity.

    run_coupled builds a scalar-slab SeaIceState (init_surface_state), and NO
    driver anywhere builds a multi-category state -- run_omip_core2 hardcodes
    n_categories=1 too, and step_sea_ice RAISES for n_categories>1 without a
    DynamicSeaIceState. So ridging is oracle-pinned physics that is unreachable
    from every driver: a real unwired-physics gap, tracked separately.

    An earlier draft added --ice-categories/--ice-dynamics here to "reach" it.
    Those were PHANTOMS -- every accepted value crashed on the first step
    (_step_dynamic reads a u_ice the slab state lacks) -- the same shape as
    --surface-bulk-scheme most, which this branch removed. Reject honestly
    instead of offering a flag that cannot run.
    """
    with pytest.raises(SystemExit, match="cannot run"):
        mod.build_sea_ice_config(mod.build_parser().parse_args(["--ice-ridging"]))


def test_the_phantom_flags_are_gone():
    """Regression pin: --ice-categories/--ice-dynamics must not come back.

    They parse cleanly and then die at runtime, which is strictly worse than
    the gap they were meant to close.
    """
    for phantom in ("--ice-categories", "--ice-dynamics"):
        with pytest.raises(SystemExit):
            mod.build_parser().parse_args([phantom, "5"])


def test_there_is_no_itd_remap_flag():
    """itd_remap has exactly two values: 'simple' (the default) and
    'lipscomb2001' (unreachable -- it dispatches only inside the multi-category
    branch no driver builds). A flag whose sole accepted value is the default
    is decoration, so it is deliberately absent rather than offered.
    """
    for phantom in ("--ice-itd-remap",):
        with pytest.raises(SystemExit):
            mod.build_parser().parse_args([phantom, "simple"])


def test_ice_stability_scheme_rejects_the_constant_bulk_branch():
    """The constant branch uses simple_bulk_fluxes and never reads stability."""
    with pytest.raises(SystemExit, match="only read by the MOST bulk branch"):
        mod.build_sea_ice_config(mod.build_parser().parse_args(
            ["--ice-stability-scheme", "grachev2007_sheba"]))
    cfg = mod.build_sea_ice_config(mod.build_parser().parse_args(
        ["--ice-stability-scheme", "grachev2007_sheba",
         "--ice-bulk-scheme", "most"]))
    assert cfg.stability_scheme == "grachev2007_sheba"


@pytest.mark.parametrize("scheme", ["constant", "most", "coare3", "large_yeager"])
def test_ice_bulk_offers_every_scheme_the_dispatch_accepts(scheme):
    """sea_ice._bulk_flux_dispatch accepts all four; the CLI offered only two
    (codex). Note 'most' is GENUINE MOST here (ice roughness), unlike the
    atmosphere where it silently degrades to constant -- which is why
    VALID_SURFACE_BULK omits it but this list keeps it.
    """
    args = ["--ice-bulk-scheme", scheme]
    if scheme != "constant":
        pass  # no cross-field contract on the ice tile
    cfg = mod.build_sea_ice_config(mod.build_parser().parse_args(args))
    if scheme == "constant":
        assert cfg is None      # == the default, nothing requested
    else:
        assert cfg.bulk_scheme == scheme


def test_main_really_passes_the_cli_ice_config_to_the_driver(monkeypatch):
    """EXECUTION-level pin: run main() and capture what the driver receives.

    The AST test above proves the name assigned from build_sea_ice_config is
    the name passed as ice_config=. That is NOT sufficient (codex): main()
    REBINDS ice_config through apply_coupled_params when --params is given, so
    a refactor that reassigns or sanitizes it to None in between still matches
    by name. Only running the thing settles it.

    CoupledESMDriver is imported inside main(), so patch it at its source
    module. The recorder raises immediately -- we want the constructor
    ARGUMENTS, not a driver (which would build grids and cost far more than a
    unit test can afford).
    """
    import legoesm.driver.coupled_esm_driver as ced

    captured = {}

    class _Stop(Exception):
        pass

    def _recorder(*a, **kw):
        captured.update(kw)
        raise _Stop

    monkeypatch.setattr(ced, "CoupledESMDriver", _recorder)
    monkeypatch.setattr(sys, "argv", ["run_coupled", "--ice-snow", "--days", "0"])

    with pytest.raises(_Stop):
        mod.main()

    ice = captured.get("ice_config")
    assert ice is not None, (
        "main() passed ice_config=None despite --ice-snow -- the flag parsed "
        "and was dropped before the driver"
    )
    assert ice.snow.enabled is True, "--ice-snow did not survive to the driver"


def test_main_ice_config_survives_the_params_rebinding(monkeypatch):
    """The specific hole the AST test cannot see (codex): --params REBINDS
    ice_config via apply_coupled_params. A rebinding that dropped the CLI
    config would keep the same variable name and pass the AST check.
    """
    import json
    import tempfile
    from pathlib import Path

    import legoesm.driver.coupled_esm_driver as ced

    captured = {}

    class _Stop(Exception):
        pass

    def _recorder(*a, **kw):
        captured.update(kw)
        raise _Stop

    monkeypatch.setattr(ced, "CoupledESMDriver", _recorder)
    pf = Path(tempfile.mkdtemp()) / "params.json"
    pf.write_text(json.dumps({"ice.ridging.e_star": 0.5}))
    monkeypatch.setattr(sys, "argv", ["run_coupled", "--ice-snow",
                                      "--params", str(pf), "--days", "0"])

    with pytest.raises(_Stop):
        mod.main()

    ice = captured["ice_config"]
    assert ice.snow.enabled is True, "--ice-snow lost to the --params rebinding"
    assert ice.ridging.e_star == 0.5, "--params did not apply"
