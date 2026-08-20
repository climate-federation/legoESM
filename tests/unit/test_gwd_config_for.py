"""``gwd_config_for`` — the third resolver (turbulence / convection / GWD).

Every lane (FV pipeline, MPAS, spectral) previously built
``GravityWaveDragConfig(scheme=...)`` from the scheme STRING alone, so the
tuned ExperimentConfig scalars never reached the kernel on ANY production
AMIP path (only the AIMIP training path consumed them).  This pins the
overlay, the override precedence, and the composite-scheme behaviour.
"""

import inspect
import re

import pytest

from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
)
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.physics_pipeline import gwd_config_for


def _cfg(**kw):
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        **kw,
    )


@pytest.mark.parametrize("scheme", [
    "none", "rayleigh", "lindzen", "mcfarlane", "hines",
    "prognostic_spectral", "e3sm_cam", "ml_emulator", "mcfarlane+hines",
])
def test_defaults_match_bare_config(scheme):
    """Untouched scalars reproduce the bare config EXACTLY — whole tuple, every
    scheme.  Byte-identical defaults are the precondition for wiring the overlay
    into lanes that previously built the bare config."""
    assert (gwd_config_for(_cfg(gravity_wave_drag=scheme))
            == GravityWaveDragConfig(scheme=scheme))


def test_none_scheme_short_circuits():
    got = gwd_config_for(_cfg(gravity_wave_drag="none"))
    assert got.scheme == "none"
    assert got == GravityWaveDragConfig(scheme="none")


def test_mcfarlane_scalars_reach_the_leaf():
    got = gwd_config_for(_cfg(
        gravity_wave_drag="mcfarlane",
        mcfarlane_tau_max=4.0,
        mcfarlane_k_wave=1.0e-4,
        mcfarlane_directional_spread=0.5,
    ))
    assert got.mcfarlane.tau_max == 4.0
    assert got.mcfarlane.k_wave == 1.0e-4
    assert got.mcfarlane.directional_spread == 0.5


def test_hines_scalars_reach_the_leaf():
    got = gwd_config_for(_cfg(
        gravity_wave_drag="hines",
        hines_total_rms_wind=1.2,
        hines_Fmax=0.05,
    ))
    assert got.hines.total_rms_wind == 1.2
    assert got.hines.Fmax == 0.05


def test_composite_scheme_carries_both_tuned_leaves():
    """'mcfarlane+hines' runs BOTH kernels — both overlays must apply."""
    got = gwd_config_for(_cfg(
        gravity_wave_drag="mcfarlane+hines",
        mcfarlane_tau_max=6.0,
        hines_total_rms_wind=1.5,
    ))
    assert got.scheme == "mcfarlane+hines"
    assert got.mcfarlane.tau_max == 6.0
    assert got.hines.total_rms_wind == 1.5


def test_explicit_override_wins_verbatim():
    """An injected full config is never second-guessed by the overlay."""
    inj = GravityWaveDragConfig(scheme="mcfarlane")
    inj = inj._replace(mcfarlane=inj.mcfarlane._replace(tau_max=99.0))
    got = gwd_config_for(_cfg(
        gravity_wave_drag="mcfarlane",
        gravity_wave_drag_override=inj,
        mcfarlane_tau_max=4.0,   # must NOT win over the override
    ))
    assert got is inj
    assert got.mcfarlane.tau_max == 99.0


def test_other_leaves_untouched():
    """Overlaying mcfarlane/hines leaves the other scheme leaves at defaults."""
    got = gwd_config_for(_cfg(gravity_wave_drag="mcfarlane",
                              mcfarlane_tau_max=4.0))
    bare = GravityWaveDragConfig(scheme="mcfarlane")
    assert got.rayleigh == bare.rayleigh
    assert got.lindzen == bare.lindzen
    assert got.e3sm_cam == bare.e3sm_cam
    assert got.prognostic_spectral == bare.prognostic_spectral


def test_resolve_gwd_uses_the_resolver():
    """The FV pipeline path honours the same tuned leaf (no divergence)."""
    from legoesm.driver.physics_pipeline import _resolve_gwd

    fn, kernel_cfg = _resolve_gwd(_cfg(gravity_wave_drag="mcfarlane",
                                       mcfarlane_tau_max=4.0))
    assert fn is not None
    assert kernel_cfg.tau_max == 4.0


def test_cli_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args,
    )
    parser = build_arg_parser()
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.hines_total_rms_wind == 2.0
    assert d.hines_Fmax == 0.1
    assert d.mcfarlane_tau_max == 10.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--hines-total-rms-wind", "1.2", "--hines-fmax", "0.05",
        "--mcfarlane-tau-max", "4.0", "--mcfarlane-k-wave", "1.0e-4",
    ]), parser))
    assert cfg.hines_total_rms_wind == 1.2
    assert cfg.hines_Fmax == 0.05
    assert cfg.mcfarlane_tau_max == 4.0
    assert cfg.mcfarlane_k_wave == 1.0e-4
    assert gwd_config_for(cfg).hines.total_rms_wind == 1.2


def test_every_overlaid_scalar_has_a_run_amip_route():
    """Each scalar gwd_config_for threads must be settable by an operator.
    ``--config`` YAML keys must be argparse dests (load_yaml_config rejects
    anything else), so a flag is the gate for BOTH routes; the --params route
    additionally needs a _ATM_SCALAR_PARAM_MAP entry, which mcfarlane_k_wave
    lacks by design (no __param_spec__ bounds yet)."""
    from legoesm.driver.run_config_yaml import build_atm_scalar_param_map
    from scripts.run.run_amip import build_arg_parser

    dests = {a.dest for a in build_arg_parser()._actions}
    # DERIVED from the resolver source, not hard-coded: a future sixth overlay
    # is then covered automatically (codex round 3 note).  The float filter
    # drops the scheme string / override object the resolver also reads.
    overlaid = sorted(
        n for n in re.findall(r'getattr\(\s*config,\s*"(\w+)"',
                              inspect.getsource(gwd_config_for))
        if isinstance(ExperimentConfig._field_defaults.get(n), float)
    )
    assert len(overlaid) >= 5, f"resolver scan found only {overlaid}"
    # mcfarlane_directional_spread is --params-only (mapped, no flag); every
    # other overlaid scalar must have a flag.
    amap = set(build_atm_scalar_param_map().values())
    for f in overlaid:
        assert f in dests or f in amap, (
            f"{f} is overlaid onto the GWD kernel by gwd_config_for but an "
            "operator cannot set it: no run_amip flag AND no scalar-map entry."
        )


def test_physics_state_seed_honours_a_nested_override():
    """The stateful-carry seed must resolve through the SAME resolver as the
    kernel: init_physics_state sizes gwd_spectrum from
    prognostic_spectral.n_azimuths/.n_wavenumbers/.launch_flux, and an override
    may set all three (codex round 1, finding 2 — the seed at
    model_driver.py:9418 used to build a bare config)."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        PrognosticSpectralConfig,
    )
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig

    inj = GravityWaveDragConfig(
        scheme="prognostic_spectral",
        prognostic_spectral=PrognosticSpectralConfig(
            n_azimuths=8, n_wavenumbers=6, launch_flux=2.5e-3),
    )
    cfg = _cfg(gravity_wave_drag="prognostic_spectral",
               gravity_wave_drag_override=inj)
    ps = init_physics_state(
        3, 8,
        PhysicsConfig(turbulence=TurbulenceConfig(scheme="none"),
                      gravity_wave_drag=gwd_config_for(cfg)),
    )
    assert ps.gwd_spectrum.shape == (3, 8, 6)
    assert float(ps.gwd_spectrum[0, 0, 0]) == 2.5e-3


@pytest.mark.parametrize("field,bad", [
    ("hines_total_rms_wind", -1.0),
    ("hines_total_rms_wind", 0.0),
    ("hines_Fmax", -0.1),
    ("mcfarlane_tau_max", float("nan")),
    ("mcfarlane_k_wave", float("inf")),
    ("mcfarlane_directional_spread", -1.0),
])
def test_validate_strict_rejects_nonpositive_or_nonfinite(field, bad):
    """A negative Fmax makes clip(drag, 0, Fmax) return the NEGATIVE cap at
    every level (constant spurious drag, no error); a non-positive launch rms
    wind silently disables the scheme.  Neither may reach the kernel."""
    with pytest.raises(ValueError, match=field):
        _cfg(gravity_wave_drag="hines", **{field: bad}).validate_strict()


def test_legacy_amip_upconvert_keeps_the_exact_k_wave_default():
    """A legacy checkpoint upconverted through from_amip_config must land on
    the SAME k_wave the kernel default uses (the truncated 6.283185307e-5
    fallback was ~3e-11 off once the overlay went live)."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        McFarlaneConfig,
    )
    from legoesm.forcing.amip_config import AMIPExperimentConfig

    # The legacy flat schema has NO mcfarlane_* fields, so the converter's
    # getattr fallback is what supplies k_wave.
    exp = ExperimentConfig.from_amip_config(AMIPExperimentConfig())
    assert exp.mcfarlane_k_wave == McFarlaneConfig().k_wave
    assert gwd_config_for(
        exp._replace(gravity_wave_drag="mcfarlane"),
    ).mcfarlane == McFarlaneConfig()


@pytest.mark.parametrize("scheme", ["mcfarlane", "hines", "mcfarlane+hines"])
def test_kernel_buildable_for_every_wired_scheme(scheme):
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        get_gwd_fn,
    )
    name, fn, _kcfg = get_gwd_fn(gwd_config_for(_cfg(
        gravity_wave_drag=scheme, hines_total_rms_wind=1.5,
        mcfarlane_tau_max=6.0)))
    assert fn is not None
    assert name
