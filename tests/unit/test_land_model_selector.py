"""One vocabulary for "which land surface model is this run using".

The coupled driver and ``run_lmip_smoke`` have always named the land surface
through ``CoupledESMConfig.land_mode`` — ``none`` / ``slab`` / ``multilayer`` —
resolved by ``legoesm.land.config.resolve_land_config``.  The AMIP driver
instead exposed two independent booleans, ``slab_land_active`` and
``use_multilayer_land``.  Two booleans have four combinations and the concept
has three states, and the all-false combination does not mean "the simple land
model": it means there is NO land surface model, with land temperature falling
back to the neighbouring prescribed SST minus a lapse rate.  This repo has lost
campaign time to exactly that reading.

These tests pin the bridge between the two spellings.
"""
from __future__ import annotations

import argparse

import pytest

from legoesm.land.config import (
    LAND_MODELS,
    describe_land_model,
    land_model_switches,
)


def test_every_named_model_maps_to_switches_and_back():
    """Round-trip: name -> switches -> name is the identity on all three."""
    assert LAND_MODELS == ("none", "slab", "multilayer")
    for name in LAND_MODELS:
        switches = land_model_switches(name)
        assert describe_land_model(**switches) == name


def test_each_name_is_a_pure_alias_and_sets_nothing_extra():
    """The selector must not invent semantics.

    ``multilayer`` deliberately does NOT also set ``slab_land_active``, even
    though the multilayer soil sits on top of the slab's surface energy balance
    and needs an active tile.  Setting both would make ``--land-model
    multilayer`` a different run from today's ``--use-multilayer-land`` for any
    deck that activates its tile through topography or a land-mask file, which
    is a silent config change — the thing this whole selector exists to prevent.
    Activation stays where it already lives: the mask, the topography, and the
    driver's own validation.
    """
    assert land_model_switches("multilayer") == dict(
        slab_land_active=False, use_multilayer_land=True)
    assert land_model_switches("slab") == dict(
        slab_land_active=True, use_multilayer_land=False)
    assert land_model_switches("none") == dict(
        slab_land_active=False, use_multilayer_land=False)


def test_unknown_name_raises_and_says_what_none_means():
    with pytest.raises(ValueError, match="unknown land_model"):
        land_model_switches("bucket")
    # The message must warn that "none" is the absence of a land model, since
    # reading it as "the simple one" is the documented failure.
    with pytest.raises(ValueError, match="NO land surface model"):
        land_model_switches("typo")


def test_a_land_mask_file_activates_the_tile_on_its_own():
    """A mask path implies activation, so it is part of the resolved answer."""
    assert describe_land_model(slab_land_active=False,
                               use_multilayer_land=False,
                               has_land_mask=True) == "slab"
    assert describe_land_model(slab_land_active=False,
                               use_multilayer_land=False,
                               has_land_mask=False) == "none"


# --------------------------------------------------------------------------
# The driver-side resolution
# --------------------------------------------------------------------------

def _run_amip():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_run_amip_for_test", repo / "scripts" / "run" / "run_amip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**kw):
    base = dict(land_model=None, slab_land_active=False,
                use_multilayer_land=False, land_mask_file="")
    base.update(kw)
    return argparse.Namespace(**base)


def test_named_model_sets_the_switches():
    mod = _run_amip()
    a = _args(land_model="multilayer")
    assert mod._resolve_land_model(a) == "multilayer"
    # Pure alias: it sets the multilayer switch and NOTHING else, so the run is
    # identical to today's --use-multilayer-land.
    assert a.use_multilayer_land is True
    assert a.slab_land_active is False


def test_omitting_the_flag_changes_nothing():
    """NON-VACUITY for "existing decks are unaffected"."""
    mod = _run_amip()
    a = _args(slab_land_active=True)
    assert mod._resolve_land_model(a) == "slab"
    assert a.slab_land_active is True and a.use_multilayer_land is False

    b = _args()
    assert mod._resolve_land_model(b) == "none"
    assert b.slab_land_active is False and b.use_multilayer_land is False


def test_named_model_conflicting_with_a_switch_is_refused():
    """Silently overriding one spelling with the other is the failure mode."""
    mod = _run_amip()
    with pytest.raises(SystemExit, match="conflicts with"):
        mod._resolve_land_model(_args(land_model="none", slab_land_active=True))
    with pytest.raises(SystemExit, match="conflicts with"):
        mod._resolve_land_model(
            _args(land_model="slab", use_multilayer_land=True))


def test_none_with_a_land_mask_file_is_refused():
    """A mask activates the tile by itself, so "none" beside one is a lie."""
    mod = _run_amip()
    with pytest.raises(SystemExit, match="activates the land tile"):
        mod._resolve_land_model(_args(land_model="none",
                                      land_mask_file="/some/mask.nc"))


def test_the_resolver_is_actually_called_from_main_with_real_argv():
    """A resolver nobody calls is a flag that does nothing.

    Every other test here calls ``_resolve_land_model`` directly, so deleting
    its call site in ``main`` would leave them all green while ``--land-model``
    silently stopped working.  This reads the source instead: the call must
    exist, it must sit inside ``main``, and it must be handed the real argv
    rather than the bare ``argv`` parameter — which is ``None`` on the ordinary
    command line, and would drop the conflict check back to truthiness.
    """
    import ast
    import inspect
    from pathlib import Path

    src = Path(inspect.getsourcefile(_run_amip())).read_text()
    tree = ast.parse(src)

    main_fn = next((n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert main_fn is not None, "run_amip.main() not found"

    calls = [n for n in ast.walk(main_fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_resolve_land_model"]
    assert calls, "--land-model is never resolved: main() does not call _resolve_land_model"

    rendered = ast.unparse(calls[0])
    assert "sys.argv" in rendered, (
        f"_resolve_land_model is called as {rendered}; it must receive the "
        "resolved argv (argv if argv is not None else sys.argv[1:]), or an "
        "explicit --no-use-multilayer-land cannot be detected")


def test_an_explicit_negative_switch_is_not_silently_overridden():
    """``--land-model multilayer --no-use-multilayer-land`` must be refused.

    The negative sets False, which is also the default, so a truthiness test
    cannot see it and would let the named model overrule a flag the user
    explicitly wrote.
    """
    mod = _run_amip()
    argv = ["--land-model", "multilayer", "--no-use-multilayer-land"]
    with pytest.raises(SystemExit, match="conflicts with"):
        mod._resolve_land_model(_args(land_model="multilayer"), None, argv)

    # And the positive spelling still resolves normally when nothing conflicts.
    a = _args(land_model="multilayer")
    assert mod._resolve_land_model(a, None, ["--land-model", "multilayer"]) == "multilayer"
    assert a.use_multilayer_land is True


# --------------------------------------------------------------------------
# The AIMIP lat-lon lane, which had no land flags at all
# --------------------------------------------------------------------------

def _run_aimip_latlon():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_run_aimip_latlon_for_test",
        repo / "scripts" / "run" / "run_aimip_latlon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_aimip_latlon_exposes_a_land_model_flag_defaulting_to_today():
    """The lane had NO land flag, so its land model came purely from defaults.

    It builds its ExperimentConfig from a fixed argv carrying no land flags and
    no --topography, so it resolved to "none" — no land surface model at all.
    The flag must default to exactly that, because changing what existing AIMIP
    runs do is a separate decision from making the choice visible.
    """
    mod = _run_aimip_latlon()
    args = mod.build_parser().parse_args([])
    assert args.land_model == "none"
    assert args.land_mask_file == ""
    assert land_model_switches(args.land_model) == dict(
        slab_land_active=False, use_multilayer_land=False)


def test_aimip_latlon_accepts_the_other_two_models():
    mod = _run_aimip_latlon()
    for name in ("slab", "multilayer"):
        assert mod.build_parser().parse_args(
            ["--land-model", name]).land_model == name
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--land-model", "bucket"])


def test_aimip_latlon_applies_the_shared_mapping_not_its_own():
    """Both drivers must resolve the names through the same function.

    Spelling the switches out a second time in this driver is how the two
    definitions drift, which is the defect the selector exists to remove. Read
    the source: build_latlon_config must call land_model_switches.
    """
    import ast
    import inspect
    from pathlib import Path

    src = Path(inspect.getsourcefile(_run_aimip_latlon())).read_text()
    fn = next((n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef)
               and n.name == "build_latlon_config"), None)
    assert fn is not None, "build_latlon_config not found"
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "land_model_switches" in called, (
        "build_latlon_config does not route --land-model through "
        "legoesm.land.config.land_model_switches; a second hand-written "
        "mapping will drift from run_amip's")
