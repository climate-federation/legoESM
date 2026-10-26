"""Three NEMO-faithful dynamics options existed in the model but no OMIP flag
selected them, so a run could not choose what the oracle does.

NEMO ORCA1 runs rn_shlat=2 (no-slip walls), ln_hpg_sco (terrain-following
pressure gradient) and ln_dynvor_een (enstrophy-conserving Coriolis). Our
config supported all three; the driver exposed none of them, and "nemo_sco"
additionally raises unless pgf_quadrature is paired with it, which has no flag
of its own. Every faithfulness run therefore silently used free-slip walls,
the Adcroft/SMC03 pressure gradient and the 4-point-average Coriolis.

These tests pin the CLI surface and the pairing. They deliberately do NOT
assert the numerics: building the real config needs a mesh, and whether our
no_slip reproduces NEMO's shlat=2 partial-slip formulation is an open question
carried from review, not something a unit test can settle.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_omip_core2", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parser():
    return _module()._build_arg_parser()


@pytest.mark.parametrize("flag,dest,good,bad", [
    ("--lateral-side-bc", "lateral_side_bc", "no_slip", "partial_slip"),
    ("--barotropic-coriolis", "barotropic_coriolis", "een", "energy"),
    ("--pgf-scheme", "pgf_scheme", "nemo_sco", "nemo_zco"),
])
def test_flag_accepts_the_nemo_value_and_rejects_a_typo(flag, dest, good, bad):
    p = _parser()
    assert getattr(p.parse_args([flag, good]), dest) == good
    # A typo must not fall through to a default and silently run other physics.
    with pytest.raises(SystemExit):
        p.parse_args([flag, bad])


@pytest.mark.parametrize("dest", [
    "lateral_side_bc", "barotropic_coriolis", "pgf_scheme"])
def test_default_is_none_so_no_existing_run_moved(dest):
    """None means "preserve the config", not "pick a value". Adding these
    flags must not change any run that does not pass them."""
    assert getattr(_parser().parse_args([]), dest) is None


@pytest.mark.parametrize("name", ["lateral_side_bc", "barotropic_coriolis"])
def test_build_tripole_takes_the_flag(name):
    for node in ast.walk(ast.parse(RUNNER.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "build_tripole":
            names = [a.arg for a in node.args.args + node.args.kwonlyargs]
            assert name in names
            return
    pytest.fail("build_tripole not found")


@pytest.mark.parametrize("name", ["lateral_side_bc", "barotropic_coriolis"])
def test_the_flag_reaches_the_config_override(name):
    """A flag parsed but never put in the override dict is inert -- the exact
    defect that made these options unreachable in the first place."""
    assert f'("{name}", {name})' in RUNNER.read_text()


def test_nemo_sco_pairs_the_quadrature_it_requires():
    """Same shape as the msc_stabilize/implicit_K33 pairing: the scheme raises
    without its companion, and the companion has no flag, so selecting
    nemo_sco alone could never work."""
    src = RUNNER.read_text()
    i_sco = src.index('if pgf_scheme == "nemo_sco":')
    i_quad = src.index('_ovr["pgf_quadrature"] = "nemo_trapezoid"')
    assert i_quad > i_sco
    # Conditional, not unconditional: pairing it for every run would change
    # the pressure gradient of every existing adcroft/smc03 run.
    assert "if" in src[i_sco:i_quad].splitlines()[0]


def test_the_model_still_refuses_nemo_sco_without_its_quadrature():
    """The pairing above is only safe while the model keeps rejecting the
    illegal combination; if that guard went away the telescoping terms would
    be silently mismatched instead of raising."""
    pe = (pathlib.Path(__file__).resolve().parents[2] / "packages" / "ocean"
          / "legoesm" / "ocean" / "dynamics"
          / "ocean_pe_latlon_cgrid.py").read_text()
    assert 'pgf_scheme="nemo_sco" requires pgf_quadrature' in pe


def test_nemo_sco_preconditions_are_checked_on_the_merged_config():
    """Both nemo_sco requirements otherwise raise deep in the pressure
    gradient at the first step. GLM review: gate on the RESOLVED config, not
    on the flag, so a base config that already selects the scheme is covered
    and a later override that undoes the pairing is caught."""
    src = RUNNER.read_text()
    i = src.index('if getattr(config, "pgf_scheme", None) == "nemo_sco":')
    blk = src[i:i + 1400]
    assert 'config.pgf_quadrature != "nemo_trapezoid"' in blk
    assert 'getattr(z_coord, "t_depth_ref", None) is None' in blk
    assert "--nemo-vertical" in blk, "the error must name the flag that fixes it"
    # It must run AFTER the config is assembled, or it inspects a stale value.
    assert i > src.index("config = config.replace_flat(**_ovr)")


def test_unknown_flat_override_names_cannot_be_swallowed():
    """If replace_flat ignored unknown names, every flag routed through _ovr
    would be a silent no-op -- the exact failure this whole change is fixing.
    NamedTuple._replace raises, so pin that it still does."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    with pytest.raises(ValueError):
        LatLonCGridOceanConfig().replace_flat(definitely_not_a_field=1)


def test_the_paired_field_is_reachable_from_replace_flat():
    """pgf_quadrature has no CLI flag and is set only by the pairing. If it
    ever moved to another sub-config, replace_flat would raise at run time
    rather than here."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    c = LatLonCGridOceanConfig().replace_flat(pgf_quadrature="nemo_trapezoid")
    assert c.pgf_quadrature == "nemo_trapezoid"


@pytest.mark.parametrize("name,value,read", [
    ("lateral_side_bc", "no_slip", lambda c: c.lateral_side_bc),
    ("barotropic_coriolis", "een", lambda c: c.barotropic.barotropic_coriolis),
])
def test_the_flags_actually_land_in_the_config(name, value, read):
    """The flat name must route to the right sub-config -- barotropic_coriolis
    is nested, lateral_side_bc is top level."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert read(LatLonCGridOceanConfig().replace_flat(**{name: value})) == value


def test_model_side_defaults_are_what_the_gap_claimed():
    """The gap report said we silently ran free-slip and 4-point-average
    Coriolis. If either default ever changes, these flags stop being the thing
    that selects NEMO's choice and this file's premise is stale."""
    from legoesm.ocean.state import BarotropicConfig, LatLonCGridOceanConfig
    assert LatLonCGridOceanConfig().lateral_side_bc == "free_slip"
    assert BarotropicConfig().barotropic_coriolis == "avg"


def test_nemo_vertical_carries_the_depth_ladder_nemo_sco_needs():
    """codex P1: --nemo-vertical loaded only thicknesses, so t_depth_ref
    stayed None and nemo_sco raised even when the flag pair looked right.
    The option was still unreachable -- the guard above just said so politely.

    NEMO's gdept_1d is NOT the running sum of e3t_1d (it comes from NEMO's own
    analytic stretching), so it has to be read, not derived."""
    src = RUNNER.read_text()
    assert "def _load_nemo_gdept_1d(" in src
    i_load = src.index("_nemo_t_depth = _load_nemo_gdept_1d(_vfile)")
    i_use = src.index("t_depth_ref_override=_nemo_t_depth,")
    assert i_use > i_load, "the depths must be loaded before they are passed"
    # and the setup helper must actually hand them to the coordinate builder
    setup = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run"
             / "run_omip.py").read_text()
    assert "dz_ref_override, t_depth_ref_override," in setup


def test_a_mismatched_column_is_refused():
    """Two arrays describing the same column with different level counts is a
    wrong-file error, not something to broadcast past."""
    assert "they describe the same column." in RUNNER.read_text()


def test_missing_gdept_is_tolerated_not_fatal():
    """Vertical files without gdept_1d must keep working for every run that
    does not ask for nemo_sco."""
    src = RUNNER.read_text()
    i = src.index("def _load_nemo_gdept_1d(")
    body = src[i:i + 1200]
    assert 'if "gdept_1d" not in ds:' in body and "return None" in body


def test_kappah_min_flag_exists_and_preserves_the_config_by_default():
    """RETRACTED PREMISE, kept as the record of it.

    This flag was added believing rn_avt0 = 1.2e-5 (namelist_cfg:437) was
    ORCA1's background tracer diffusivity and our 1e-10 was a fidelity gap.
    Wrong: ORCA1 sets ln_zdfiwm = .true. (namelist_cfg:434), and NEMO's
    zdfiwm.F90:378 then overrides the namelist with avtb = 1e-10. Our resolved
    1e-10 MATCHES the oracle. A namelist line is a declared value, not a
    resolved one -- the same lesson already learned about our own configs,
    here missed on the oracle's.

    So the DEFAULT (None) is the NEMO-faithful setting and the flag is a
    departure knob. That is what this pins."""
    p = _parser()
    assert p.parse_args([]).tke_kappah_min is None
    assert p.parse_args(["--tke-kappah-min", "1.2e-5"]).tke_kappah_min == 1.2e-5


def test_the_iwm_override_is_what_sets_the_background_floors():
    """The claim that made this flag look like a fidelity fix, pinned against
    the code that refutes it. Under internal-wave mixing the wave field IS the
    interior background, so BOTH floors come from zdfiwm, not from namzdf_tke.
    ORCA1 runs ln_zdfiwm=.true., so these are the oracle's real values."""
    src = RUNNER.read_text()
    i = src.index("if iwm_enabled:")
    body = src[i:i + 600]
    assert "avmb, avtb = _const.nu_ocean_molecular, 1.0e-10" in body
    assert "zdfiwm_init" in body


def test_negative_diffusivity_is_refused():
    """Codex mutation-tested the first version of this test: it grepped the
    whole driver for "must be >= 0" and PASSED at 97d9cf420~1, before the
    feature existed. So it now CALLS the validator."""
    mod = _module()
    assert mod._checked_kappah_min(1.2e-5) == 1.2e-5
    assert mod._checked_kappah_min(0) == 0.0
    with pytest.raises(ValueError, match="must be >= 0"):
        mod._checked_kappah_min(-1e-6)


def test_the_floor_is_refused_by_every_closure_that_would_drop_it():
    """A flag accepted and silently ignored is the defect this whole series
    keeps finding. Both reviewers flagged it: the check used to live INSIDE
    the TKE branch, so kpp/none took the flag and ran the old floor."""
    mod = _module()
    for closure in ("none", "kpp"):
        with pytest.raises(ValueError, match="requires --tripole-vmix tke"):
            mod.build_tripole_vmix_config(closure, tke_kappah_min=1.2e-5)


def test_the_floor_reaches_all_three_grids():
    """The directive asks for harmonized physics across tripole, MPAS and
    FESOM2. The floor first reached only the tripole builder, so the same
    NEMO card ran with two different background diffusivities depending on
    the grid."""
    src = RUNNER.read_text()
    assert src.count("tke_kappah_min=args.tke_kappah_min") >= 3, (
        "the tripole, MPAS and FESOM lanes must all forward the floor")


def test_the_floor_actually_lands_in_the_tke_config():
    """Not a string match: build the config and read the field back.

    Uses a value that is nobody's default. An earlier version asserted the
    bare card default DIFFERS from 1.2e-5 and went red -- correctly: the
    non-iwm card default already IS 1.2e-5 (nemo_recipe.py:286), which is how
    the refuted premise above survived as long as it did."""
    mod = _module()
    probe = 7.5e-6
    assert mod.build_tripole_vmix_config("tke").tke.kappaH_min != probe
    vm = mod.build_tripole_vmix_config("tke", tke_kappah_min=probe)
    assert vm.scheme == "tke" and vm.tke.kappaH_min == probe


def test_the_momentum_floor_is_left_alone():
    """One variable per arm. kappaM_min is also wrong against NEMO (ours
    1.4e-6 vs rn_avm0 1.2e-4) but changing both at once would make the arm
    unattributable."""
    src = RUNNER.read_text()
    assert "_replace(kappaM_min=" not in src
