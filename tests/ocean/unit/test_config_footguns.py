"""Q7 — config-footgun guards (Phase G).

Two footguns flagged in the strategy doc, both resolved by *fail-fast
validation* rather than reordering the 45-field NamedTuple (which would break
positional construction for legacy callers):

1. **EOS dispatch (``eos`` / ``eos_linear``).** ``make_eos_fn`` raises on an
   unknown ``eos`` literal (dispatch discipline). ``LatLonCGridOceanModel._
   validate_config`` now also rejects an unknown ``eos`` at CONSTRUCTION (not
   lazily at the first step), using the single ``VALID_EOS_SCHEMES`` source so
   the valid set is never duplicated.

2. **A_h single source of truth.** On the lat-lon C-grid, horizontal viscosity
   comes ONLY from the dynamics ``config.A_h``. The physics-pathway
   ``LateralMixingConfig`` carries its OWN ``HarmonicConfig.A_h`` — a competing
   source — but is cubed-sphere-only, so ``_validate_config`` rejects any
   non-"none" ``physics.lateral_mixing.scheme`` (including ``"harmonic"``) at
   construction. That guard is what makes ``config.A_h`` the sole A_h source on
   this grid.

   (A purely-static "two field named A_h" AST detector would false-positive:
   A_h legitimately appears in several *different* grid configs — OceanConfig,
   LatLonOceanConfig, HarmonicConfig — which are not a single config's two
   sources. The meaningful single-source check is therefore the runtime guard
   below, per the strategy doc's "curation avoids false positives" note.)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import VALID_EOS_SCHEMES, LinearEOSConfig, make_eos_fn
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    HarmonicConfig,
    LateralMixingConfig,
)
from legoesm.ocean.state import LatLonCGridOceanConfig


# ---------------------------------------------------------------------------
# Footgun 1 — EOS dispatch
# ---------------------------------------------------------------------------


def test_make_eos_fn_accepts_every_valid_scheme():
    """Every literal in the single VALID_EOS_SCHEMES source must dispatch to a
    real callable (catches a set entry with no matching branch)."""
    for scheme in VALID_EOS_SCHEMES:
        fn = make_eos_fn(scheme)
        assert callable(fn), scheme


def test_make_eos_fn_rejects_unknown_scheme():
    """Dispatch discipline: an unknown eos literal must raise ValueError (no
    silent fallback to a default scheme)."""
    with pytest.raises(ValueError, match="Unknown EOS scheme"):
        make_eos_fn("nonlin_typo")


def test_make_eos_fn_linear_defaults_eos_linear():
    """eos='linear' with eos_linear=None uses default LinearEOSConfig (the
    documented coupling); an explicit eos_linear is honoured."""
    fn_default = make_eos_fn("linear", eos_linear=None)
    assert callable(fn_default)
    fn_custom = make_eos_fn("linear", eos_linear=LinearEOSConfig(rho_ref=1025.0))
    assert callable(fn_custom)


def test_validate_config_rejects_unknown_eos_at_construction():
    """Fail-fast: an unknown eos must raise at config validation, not lazily at
    the first step. Uses the same VALID_EOS_SCHEMES source as make_eos_fn."""
    bad = LatLonCGridOceanConfig.from_flat(eos="wrihgt")  # typo
    with pytest.raises(ValueError, match="eos must be one of"):
        LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_accepts_every_valid_eos():
    """Each valid eos literal must pass construction validation — locking the
    validator's set to make_eos_fn's set (no fail-fast/lazy disagreement)."""
    for scheme in VALID_EOS_SCHEMES:
        cfg = LatLonCGridOceanConfig.from_flat(
            eos=scheme,
            eos_depth="geometric" if scheme == "nemo_teos10" else "insitu",
        )
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise


def test_eos_reachable_via_the_yaml_ocean_key():
    """Every VALID_EOS_SCHEMES member must be SELECTABLE through the public YAML
    ``ocean.eos:`` key, i.e. reach the runtime config's ``eos`` field.

    The tests above construct ``LatLonCGridOceanConfig.from_flat(eos=...)``
    directly, which BYPASSES the YAML adapter that production actually uses
    (``OceanExperimentConfig.from_yaml -> to_ocean_config``, run_omip_core2).
    The two shipped templates set ``ocean.eos: wright``, so a rename of the flat
    field or a break in the flat->runtime mapping would silently strand every
    EOS behind the YAML interface while leaving from_flat green. This pins the
    reachability direction (implemented ⇒ selectable), the ocean-EOS analogue of
    the scheme-reachability ratchet.
    """
    from legoesm.ocean.config import OceanExperimentConfig

    for scheme in VALID_EOS_SCHEMES:
        d = {"grid": {"type": "latlon_cgrid", "nlev": 15},
             "ocean": {"eos": scheme}}
        rt = OceanExperimentConfig.from_dict(d).to_ocean_config()
        assert rt.eos == scheme, (
            f"ocean.eos: {scheme!r} did not reach the runtime config "
            f"(got {rt.eos!r}); the YAML->runtime EOS mapping is broken"
        )


def test_shipped_global_ocean_templates_load_with_their_eos():
    """The committed OMIP templates set ``ocean.eos: wright`` and must load
    cleanly through the production YAML path -- a guard against the flat-field
    contract drifting out from under a shipped config (they would otherwise
    raise 'unknown ocean config field' at to_ocean_config)."""
    from pathlib import Path

    from legoesm.ocean.config import OceanExperimentConfig

    for name in ("omip_latlon.yaml", "omip_latlon_kpp.yaml"):
        p = Path("config/templates/global_ocean") / name
        if not p.exists():
            pytest.skip(f"{p} not present")
        rt = OceanExperimentConfig.from_yaml(str(p)).to_ocean_config()
        assert rt.eos in VALID_EOS_SCHEMES, (
            f"{name} resolved eos={rt.eos!r} not in VALID_EOS_SCHEMES"
        )


def test_vertical_mixing_reachable_via_the_yaml_ocean_key():
    """Every ``VALID_VERTICAL_MIXING_SCHEMES`` member must be SELECTABLE through
    the public YAML ``ocean.physics.vertical_mixing.scheme:`` key, i.e. reach the
    runtime config's ``physics.vertical_mixing.scheme`` field.

    The flat lat-lon OMIP driver hands the runtime a fixed KPP override, so the
    ``--config`` YAML adapter is the only general selection path for the mixing
    scheme; a rename of the flat field or a break in the flat->runtime mapping
    would silently strand every non-KPP scheme (e.g. catke) while leaving the
    dispatch green. Pins the reachability direction (implemented ⇒ selectable),
    the vertical-mixing analogue of the EOS pin above.

    This is the config-path HALF of reachability (the YAML value survives into
    the runtime field); the execution half -- that every canonical scheme is
    actually dispatched, not a phantom that hits the unknown-scheme raise -- is
    pinned behaviourally against both dispatchers in
    ``test_vmix_k_profiles_direct.py`` (``TestSchemeReachability``)."""
    from legoesm.ocean.config import OceanExperimentConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        VALID_VERTICAL_MIXING_SCHEMES,
    )

    for scheme in VALID_VERTICAL_MIXING_SCHEMES:
        d = {"grid": {"type": "latlon_cgrid", "nlev": 15},
             "ocean": {"physics": {"vertical_mixing": {"scheme": scheme}}}}
        rt = OceanExperimentConfig.from_dict(d).to_ocean_config()
        got = rt.physics.vertical_mixing.scheme
        assert got == scheme, (
            f"ocean.physics.vertical_mixing.scheme: {scheme!r} did not reach "
            f"the runtime config (got {got!r}); the YAML->runtime mapping is "
            f"broken"
        )


# ---------------------------------------------------------------------------
# Footgun 2 — A_h single source of truth (lat-lon C-grid)
# ---------------------------------------------------------------------------


def test_A_h_single_source_competing_physics_harmonic_rejected():
    """The lat-lon C-grid takes A_h ONLY from config.A_h. A physics-pathway
    LateralMixingConfig with scheme='harmonic' carries a competing
    HarmonicConfig.A_h; that path is cubed-sphere-only, so it must be rejected
    at construction — leaving config.A_h as the sole A_h source."""
    competing = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4,
        physics=OceanPhysicsConfig(
            lateral_mixing=LateralMixingConfig(
                scheme="harmonic", harmonic=HarmonicConfig(A_h=9.9e9),
            ),
        ),
    )
    with pytest.raises(ValueError, match="lat-lon C-grid"):
        LatLonCGridOceanModel._validate_config(competing)


def test_A_h_single_source_clean_config_passes():
    """With physics.lateral_mixing='none' there is no competing A_h source and
    construction validation passes; config.A_h is the single source."""
    clean = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4,
        physics=OceanPhysicsConfig(
            lateral_mixing=LateralMixingConfig(scheme="none"),
        ),
    )
    LatLonCGridOceanModel._validate_config(clean)  # must not raise
    assert clean.lateral_viscosity.A_h == 2.0e4


# ---------------------------------------------------------------------------
# Footgun 3 — momentum_advection dispatch (was a silent fallthrough)
# ---------------------------------------------------------------------------


def test_validate_config_rejects_unknown_momentum_advection():
    """An unknown momentum_advection literal must raise at construction, not
    silently fall through to vector-invariant (the prior behaviour)."""
    bad = LatLonCGridOceanConfig.from_flat(momentum_advection="flux-form")  # typo/not-yet
    with pytest.raises(ValueError, match="momentum_advection must be one of"):
        LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_accepts_valid_momentum_advection():
    """Each currently-valid momentum_advection literal passes validation."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        VALID_MOMENTUM_ADVECTION,
    )
    for scheme in VALID_MOMENTUM_ADVECTION:
        cfg = LatLonCGridOceanConfig.from_flat(momentum_advection=scheme)
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise


def test_validate_config_rejects_unknown_momentum_flux_scheme():
    """An unknown momentum_flux_scheme (used by momentum_advection='flux_form')
    must raise at construction."""
    bad = LatLonCGridOceanConfig.from_flat(
        momentum_advection="flux_form", momentum_flux_scheme="quadratic",
    )
    with pytest.raises(ValueError, match="momentum_flux_scheme must be one of"):
        LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_rejects_unqualified_upwind3():
    """The bare ``"upwind3"`` names no reference model — NEMO's dynadv_up3 and
    Oceananigans' UpwindBiased(3) select the T-point upwind branch by different
    quantities — so a caller must name the arm it claims (``nemo_up3`` /
    ``oceananigans_up3``) or fail construction."""
    bad = LatLonCGridOceanConfig.from_flat(
        momentum_advection="flux_form", momentum_flux_scheme="upwind3",
    )
    with pytest.raises(ValueError, match="momentum_flux_scheme must be one of"):
        LatLonCGridOceanModel._validate_config(bad)


# ---------------------------------------------------------------------------
# Footgun 4 — tracer_advection dispatch (validated only at runtime before)
# ---------------------------------------------------------------------------


def test_validate_config_rejects_unknown_tracer_advection():
    """An unknown tracer_advection literal must raise at construction, not
    survive until the first step's runtime ValueError in
    _compute_advection_flux_div (the prior behaviour — inconsistent with every
    other scheme field, which validate on the static config at construction)."""
    bad = LatLonCGridOceanConfig.from_flat(tracer_advection="van-leer")  # typo
    with pytest.raises(ValueError, match="tracer_advection must be one of"):
        LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_accepts_valid_tracer_advection():
    """Every literal in the single VALID_TRACER_ADVECTION source must pass
    validation — including "som", which is dispatched on a separate step-level
    path and so is absent from _compute_advection_flux_div's if/elif chain."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        VALID_TRACER_ADVECTION,
    )
    assert "som" in VALID_TRACER_ADVECTION  # guard the step-level special case
    for scheme in VALID_TRACER_ADVECTION:
        cfg = LatLonCGridOceanConfig.from_flat(tracer_advection=scheme)
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise


def _dispatch_literals_in(func) -> frozenset:
    """Extract every string literal that ``func`` compares ``tracer_advection``
    against — both ``tracer_advection == "x"`` and ``tracer_advection in (...)``
    — by walking the AST of its source. This introspects the ACTUAL dispatcher
    rather than trusting a hand-maintained list."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    literals: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name)
                and node.left.id == "tracer_advection"):
            continue
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant):
                literals.add(comp.value)
            elif isinstance(op, ast.In) and isinstance(
                comp, (ast.Tuple, ast.List, ast.Set)
            ):
                for elt in comp.elts:
                    if isinstance(elt, ast.Constant):
                        literals.add(elt.value)
    return frozenset(literals)


def test_valid_tracer_advection_matches_dispatch_branches():
    """The single VALID_TRACER_ADVECTION source must list exactly the schemes
    the model can dispatch: the flux-form branches in _compute_advection_flux_div
    (introspected from its AST, not a hand-copied list) plus the step-level "som"
    path. Drift in either direction — a new dispatch branch left out of the
    frozenset (→ false-reject of a working config), or a frozenset entry with no
    handler (→ silent no-op) — fails this gate."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _compute_advection_flux_div,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        VALID_TRACER_ADVECTION,
    )

    dispatch_literals = _dispatch_literals_in(_compute_advection_flux_div)
    # Sanity: the AST walk actually found the chain (non-vacuous guard).
    assert "tvd" in dispatch_literals and "weno5" in dispatch_literals
    # "som" is dispatched on a separate step-level path, not in this function.
    assert "som" not in dispatch_literals
    assert VALID_TRACER_ADVECTION == dispatch_literals | {"som"}


def test_validate_config_accepts_flux_form_with_valid_scheme():
    """flux_form with a valid momentum_flux_scheme passes construction.

    The two UP3 arms are separate scheme names (``nemo_up3`` /
    ``oceananigans_up3``) because their references pick the T-point upwind
    branch differently; the unqualified ``"upwind3"`` names no reference and
    is refused (see test below)."""
    for scheme in ("upwind", "centered", "oceananigans_up3"):
        cfg = LatLonCGridOceanConfig.from_flat(
            momentum_advection="flux_form", momentum_flux_scheme=scheme,
        )
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Footgun 5 — S-42 BBL: in-model hook is rk3_ws-only, not lane-agnostic
# ---------------------------------------------------------------------------


def test_validate_config_rejects_bbl_in_stage_off_the_rk3_ws_lane():
    """bbl_adv_option=2 (NEMO's in-stage Campin-Goosse BBL, nn_bbl_adv=2) is
    honoured ONLY on tracer_time_integrator='rk3_ws' -- the ``elif _tti ==
    "rk3_ws":`` branch is the sole site that builds the BBL context and folds
    bbl_transports/apply_bbl_adv_tendency into the stage-3 tracer RHS. On the
    default euler lane (and ab2/rk3) the field resolves but is never read, so
    a user setting it gets NO boundary layer with no warning
    (docs/ocean/fidelity/nemo_branch_isomorphism_map.md, S-42). Must raise at
    construction instead."""
    bad = LatLonCGridOceanConfig.from_flat(
        tracer_time_integrator="euler", bbl_adv_option=2, bbl_gamma_s=20.0,
    )
    with pytest.raises(ValueError, match="bbl_adv_option=2 is not honoured"):
        LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_accepts_bbl_in_stage_on_the_rk3_ws_lane():
    """The one lane that DOES honour bbl_adv_option=2 (rk3_ws, the certified
    OVERFLOW-zps composition) must still construct cleanly."""
    ok = LatLonCGridOceanConfig.from_flat(
        tracer_time_integrator="rk3_ws", momentum_time_integrator="rk3_ws",
        tracer_advection="fct2", momentum_advection="flux_form",
        momentum_flux_scheme="nemo_up3", vertical_momentum_scheme="nemo_up3",
        bbl_adv_option=2, bbl_gamma_s=20.0,
    )
    LatLonCGridOceanModel._validate_config(ok)  # must not raise


def test_validate_config_guards_diffusive_bbl_selector_and_lane():
    """NEMO ``nn_bbl_ldf=1`` is implemented only in the WS-RK3 stage-3 site."""
    bad_lane = LatLonCGridOceanConfig.from_flat(
        tracer_time_integrator="euler", bbl_diffusive_option=1,
        bbl_aht_m2_s=1000.0,
    )
    with pytest.raises(ValueError, match="bbl_diffusive_option=1 is not honoured"):
        LatLonCGridOceanModel._validate_config(bad_lane)

    bad_coefficient = LatLonCGridOceanConfig.from_flat(
        tracer_time_integrator="rk3_ws", momentum_time_integrator="rk3_ws",
        tracer_advection="fct2", momentum_advection="flux_form",
        momentum_flux_scheme="nemo_up3", vertical_momentum_scheme="nemo_up3",
        bbl_diffusive_option=1,
        bbl_aht_m2_s=0.0,
    )
    with pytest.raises(ValueError, match="requires bbl_aht_m2_s > 0"):
        LatLonCGridOceanModel._validate_config(bad_coefficient)

    ok = LatLonCGridOceanConfig.from_flat(
        tracer_time_integrator="rk3_ws", momentum_time_integrator="rk3_ws",
        tracer_advection="fct2", momentum_advection="flux_form",
        momentum_flux_scheme="nemo_up3", vertical_momentum_scheme="nemo_up3",
        bbl_diffusive_option=1, bbl_aht_m2_s=1000.0,
    )
    LatLonCGridOceanModel._validate_config(ok)


def test_validate_config_accepts_bbl_off_on_any_lane():
    """bbl_adv_option=0 (the default, off) is lane-agnostic -- the new guard
    must not false-reject a config that simply never asked for in-model BBL."""
    for lane in ("euler", "ab2", "rk3"):
        cfg = LatLonCGridOceanConfig.from_flat(
            tracer_time_integrator=lane, bbl_adv_option=0,
        )
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Footgun 6 — S-46/S-47 nemo_up3 requires momentum_time_integrator='rk3_ws'
# ---------------------------------------------------------------------------


def test_validate_config_rejects_nemo_up3_off_the_rk3_ws_lane():
    """momentum_flux_scheme='nemo_up3' selects NEMO's dyn_adv_up3 T-point
    upwind SELECTOR on its own (S-46: keyed by the reference, not the time
    integrator -- 9bbf9f7bd). But NEMO's own face THICKNESS for that same
    routine, e3u(Kmm), is wired only inside the momentum_time_integrator=
    'rk3_ws' stage program (S-47, 60d0c5420); every other integrator falls
    back to the legacy min-of-stretched-face-thickness rule S-47 measured as
    first-order wrong. nemo_up3 paired with any other momentum_time_integrator
    is therefore a combination NEMO itself never runs, and must raise at
    construction."""
    for lane in ("euler", "rk3"):  # momentum_time_integrator has no "ab2" arm
        bad = LatLonCGridOceanConfig.from_flat(
            momentum_advection="flux_form", momentum_flux_scheme="nemo_up3",
            momentum_time_integrator=lane, tracer_time_integrator=lane,
        )
        with pytest.raises(
            ValueError, match="requires momentum_time_integrator='rk3_ws'"
        ):
            LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_accepts_nemo_up3_on_the_rk3_ws_lane():
    """The one lane NEMO actually runs nemo_up3 on -- rk3_ws, which wires the
    S-47 face-thickness fix -- must still construct cleanly."""
    ok = LatLonCGridOceanConfig.from_flat(
        tracer_time_integrator="rk3_ws", momentum_time_integrator="rk3_ws",
        tracer_advection="fct2", momentum_advection="flux_form",
        momentum_flux_scheme="nemo_up3", vertical_momentum_scheme="nemo_up3",
    )
    LatLonCGridOceanModel._validate_config(ok)  # must not raise


def test_validate_config_accepts_oceananigans_up3_on_every_non_rk3_ws_lane():
    """oceananigans_up3 is the OTHER UP3 reference arm (Silvestri/Oceananigans,
    not NEMO) -- the new nemo_up3 x rk3_ws pairing guard must not false-reject
    it on any lane. rk3_ws itself is excluded: it already requires
    momentum_flux_scheme='nemo_up3' specifically (a separate, pre-existing
    guard -- test_validate_config_accepts_nemo_up3_on_the_rk3_ws_lane above),
    so oceananigans_up3 cannot pair with rk3_ws regardless of this guard."""
    for lane in ("euler", "rk3"):  # momentum_time_integrator has no "ab2" arm
        cfg = LatLonCGridOceanConfig.from_flat(
            momentum_advection="flux_form",
            momentum_flux_scheme="oceananigans_up3",
            momentum_time_integrator=lane, tracer_time_integrator=lane,
        )
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise
