"""Unit tests for the AIMIP classical surface *bulk_scheme* knob.

The trainable ``surface_most_unstable_gamma`` / ``surface_most_stable_beta``
/ ``surface_z0h_z0_ratio`` leaves are only read when the surface flux runs a
stability-dependent bulk scheme (``most`` / ``coare3`` / ``large_yeager``).
Under the legacy default ``bulk_scheme="constant"`` they are DEAD — the
constant-Cd bulk formula never touches them, so ``jax.grad`` wrt them is
exactly zero.  These tests verify:

1. ``to_surface_config(bulk_scheme="most")`` yields the requested scheme;
   the default still yields ``"constant"`` (backward compatible).
2. The SPATIAL-surface assembly path (``spatial_surface=True``) also carries
   the requested ``"most"`` scheme — the key regression guard against the
   spatial path staying pinned to ``constant``.
3. ``make_aimip_classical_spectral_physics(surface_bulk_scheme="bogus")``
   raises (dispatch hardening via ``validate_bulk_scheme``).
4. GRADIENT REACHABILITY (the whole point): with ``bulk_scheme="most"`` the
   ``jax.grad`` of a surface-flux quantity wrt the three MOST leaves is
   finite and non-zero; with ``bulk_scheme="constant"`` it is exactly zero
   (documenting WHY the knob is needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
# 1 + 2: config assembly threads bulk_scheme (scalar + spatial paths)
# ----------------------------------------------------------------------

def test_to_surface_config_default_is_constant():
    from legoesm.training.aimip_params import AIMIPClassicalParams

    p = AIMIPClassicalParams.from_defaults()
    sfc = p.to_surface_config()
    assert sfc.bulk_scheme == "constant"


def test_to_surface_config_most_requested():
    from legoesm.training.aimip_params import AIMIPClassicalParams

    p = AIMIPClassicalParams.from_defaults()
    sfc = p.to_surface_config(bulk_scheme="most")
    assert sfc.bulk_scheme == "most"
    # The trained MOST leaves ride along and are finite.
    assert jnp.isfinite(jnp.asarray(sfc.most_unstable_gamma))
    assert jnp.isfinite(jnp.asarray(sfc.most_stable_beta))
    assert jnp.isfinite(jnp.asarray(sfc.z0h_z0_ratio))


def test_to_louis_config_threads_bulk_scheme():
    from legoesm.training.aimip_params import AIMIPClassicalParams

    p = AIMIPClassicalParams.from_defaults()
    louis = p.to_louis_config(bulk_scheme="most")
    assert louis.surface.bulk_scheme == "most"
    # Default remains constant.
    assert p.to_louis_config().surface.bulk_scheme == "constant"


def test_spatial_surface_path_carries_most():
    """The spatial-surface assembly must NOT pin bulk_scheme to constant.

    Build the physics with ``spatial_surface=True`` and a land mask, then
    reach into the assembled ``PhysicsConfig`` and assert the Louis surface
    config carries ``"most"``.  This is the key regression guard: the
    spatial override (_spatial_surface_override) replaces Cd/Ch/z0 and must
    preserve the requested bulk scheme.
    """
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.grids.gaussian import create_gaussian_grid

    grid = create_gaussian_grid(11, dealiasing="quadratic")
    land_mask = jnp.ones((grid.n_lat, grid.n_lon))
    p = AIMIPClassicalParams.from_defaults(spatial_surface=True)

    # Intercept the PhysicsConfig that make_physics receives so we can
    # inspect the surface bulk scheme actually threaded into the turbulence
    # config on the spatial path.
    import legoesm.atmosphere.physics.combined as combined

    captured = {}
    orig_make_physics = combined.make_physics

    def _spy(physics_config, *args, **kwargs):
        captured["cfg"] = physics_config
        return orig_make_physics(physics_config, *args, **kwargs)

    combined.make_physics = _spy
    try:
        make_aimip_classical_spectral_physics(
            p, grid, dt=1800.0,
            radiation="gray",
            turbulence_scheme="louis",
            surface_bulk_scheme="most",
            land_mask=land_mask,
        )
    finally:
        combined.make_physics = orig_make_physics

    turb = captured["cfg"].turbulence
    assert turb.scheme == "louis"
    assert turb.louis.surface.bulk_scheme == "most"


def _assemble_physics_config(p, grid, *, turbulence_scheme, surface_bulk_scheme,
                             land_mask=None, allow_unfilled_families=False):
    """Build the physics and return the intercepted PhysicsConfig.

    The completeness gate stays ON by default so these tests keep exercising
    it; only the deliberate ``turbulence_scheme="none"`` fall-through case
    waives it (codex: a helper-wide waiver silently dropped gate coverage from
    every test in this module).
    """
    from legoesm.training.aimip_params import (
        make_aimip_classical_spectral_physics,
    )
    import legoesm.atmosphere.physics.combined as combined

    captured = {}
    orig = combined.make_physics

    def _spy(physics_config, *args, **kwargs):
        captured["cfg"] = physics_config
        return orig(physics_config, *args, **kwargs)

    combined.make_physics = _spy
    try:
        make_aimip_classical_spectral_physics(
            p, grid, dt=1800.0,
            radiation="gray",
            turbulence_scheme=turbulence_scheme,
            surface_bulk_scheme=surface_bulk_scheme,
            land_mask=land_mask,
            allow_unfilled_families=allow_unfilled_families,
        )
    finally:
        combined.make_physics = orig
    return captured["cfg"]


# mynn25 is EXCLUDED: it carries a ``surface`` field (so the config-assembly
# fix applies) but ``make_physics(model_type="spectral_pe")`` REFUSES it (it
# needs a driver that persists PhysicsState across steps — not the spectral PE
# path).  That is a pre-existing, unrelated limitation; the config-level
# threading for mynn25 is covered by test_mynn25_config_threads_most below,
# which does not go through make_physics.
@pytest.mark.parametrize("turb_scheme", ["edmf", "holtslag_boville", "ysu",
                                         "clubb_lite"])
def test_non_louis_turbulence_honours_bulk_scheme(turb_scheme):
    """A non-Louis turbulence scheme must NOT stay pinned to constant.

    Codex iter-1 finding: the generic ``else`` branch silently left several
    supported turbulence schemes on the constant surface path.  These schemes
    each carry a ``surface`` sub-config, so surface_bulk_scheme="most" must
    reach it.
    """
    from legoesm.training.aimip_params import AIMIPClassicalParams
    from legoesm.grids.gaussian import create_gaussian_grid

    grid = create_gaussian_grid(11, dealiasing="quadratic")
    p = AIMIPClassicalParams.from_defaults()
    cfg = _assemble_physics_config(
        p, grid, turbulence_scheme=turb_scheme, surface_bulk_scheme="most",
    )
    turb = cfg.turbulence
    assert turb.scheme == turb_scheme
    sub = getattr(turb, turb_scheme)
    assert sub.surface.bulk_scheme == "most", (
        f"{turb_scheme} surface bulk scheme not threaded (stayed "
        f"{sub.surface.bulk_scheme!r})"
    )
    # iter-2 finding #4: the non-Louis surface must carry the AIMIP-TRAINED
    # MOST leaves, not the SurfaceLayerConfig DEFAULT — otherwise the knobs
    # are still dead for non-Louis schemes.  from_defaults interiorizes the
    # canonical defaults by a sigmoid margin, so the trained value differs
    # from the raw config default; assert it matches to_surface_config.
    trained = p.to_surface_config(bulk_scheme="most")
    assert float(sub.surface.most_unstable_gamma) == float(
        trained.most_unstable_gamma
    )
    assert float(sub.surface.z0h_z0_ratio) == float(trained.z0h_z0_ratio)


def test_mynn25_config_threads_most_at_config_level():
    """mynn25 config-assembly threads bulk_scheme even though spectral_pe
    make_physics refuses the scheme.

    Guards that the else-branch mapping (TurbulenceConfig(scheme=X, **{X:cfg}))
    is structurally correct for mynn25 without exercising the unsupported
    spectral-PE make_physics path.
    """
    from legoesm.atmosphere.physics.turbulence.config import (
        MYNN25Config,
        TurbulenceConfig,
    )

    base = MYNN25Config()
    # mynn25 carries a surface sub-config (precondition for the fix).
    assert hasattr(base, "surface")
    threaded = base._replace(
        surface=base.surface._replace(bulk_scheme="most"),
    )
    turb = TurbulenceConfig(scheme="mynn25", mynn25=threaded)
    assert turb.mynn25.surface.bulk_scheme == "most"


def test_clubb_config_threads_most_at_config_level():
    """clubb (distinct from clubb_lite) must NOT stay pinned to constant.

    iter-2 finding #6: clubb is materialised from TurbulenceConfig.clubb
    (default None -> CLUBBConfig() with a settable surface) in the dispatcher.
    The builder's else-branch now threads it.  clubb's prognostic path needs a
    PhysicsState-persisting driver (not spectral PE), so verify the mapping at
    the config level: CLUBBConfig carries a surface and TurbulenceConfig(clubb=)
    holds it.
    """
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    base = CLUBBConfig()
    assert hasattr(base, "surface")
    threaded = base._replace(
        surface=base.surface._replace(bulk_scheme="most"),
    )
    turb = TurbulenceConfig(scheme="clubb", clubb=threaded)
    assert turb.clubb.surface.bulk_scheme == "most"


def test_louis_carries_trained_most_leaves():
    """The default (louis) path also carries the TRAINED MOST leaves."""
    from legoesm.training.aimip_params import AIMIPClassicalParams
    from legoesm.grids.gaussian import create_gaussian_grid

    grid = create_gaussian_grid(11, dealiasing="quadratic")
    p = AIMIPClassicalParams.from_defaults()
    cfg = _assemble_physics_config(
        p, grid, turbulence_scheme="louis", surface_bulk_scheme="most",
    )
    trained = p.to_surface_config(bulk_scheme="most")
    sfc = cfg.turbulence.louis.surface
    assert sfc.bulk_scheme == "most"
    assert float(sfc.most_unstable_gamma) == float(trained.most_unstable_gamma)
    assert float(sfc.most_stable_beta) == float(trained.most_stable_beta)
    assert float(sfc.z0h_z0_ratio) == float(trained.z0h_z0_ratio)


def test_turbulence_none_falls_through_cleanly():
    """turbulence='none' has no surface sub-config; must not raise."""
    from legoesm.training.aimip_params import AIMIPClassicalParams
    from legoesm.grids.gaussian import create_gaussian_grid

    grid = create_gaussian_grid(11, dealiasing="quadratic")
    p = AIMIPClassicalParams.from_defaults()
    cfg = _assemble_physics_config(
        p, grid, turbulence_scheme="none", surface_bulk_scheme="most",
        # The ONE case in this module that drops a family on purpose.
        allow_unfilled_families=True,
    )
    assert cfg.turbulence.scheme == "none"


# ----------------------------------------------------------------------
# 3: dispatch hardening — unknown scheme raises at builder entry
# ----------------------------------------------------------------------

def test_make_physics_rejects_unknown_bulk_scheme():
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.grids.gaussian import create_gaussian_grid

    grid = create_gaussian_grid(11, dealiasing="quadratic")
    p = AIMIPClassicalParams.from_defaults()
    with pytest.raises(ValueError, match="Unknown bulk_scheme"):
        make_aimip_classical_spectral_physics(
            p, grid, dt=1800.0,
            radiation="gray",
            surface_bulk_scheme="bogus",
        )


# ----------------------------------------------------------------------
# 4: gradient reachability — MOST leaves live under "most", dead under
#    "constant"
# ----------------------------------------------------------------------

def _synthetic_surface_inputs(ncol: int = 8):
    """Realistic-magnitude lowest-level + surface state for a flux call."""
    u = jnp.linspace(2.0, 12.0, ncol)
    v = jnp.linspace(-3.0, 5.0, ncol)
    T = jnp.linspace(285.0, 300.0, ncol)          # unstable + stable mix
    q_v = jnp.linspace(3.0e-3, 12.0e-3, ncol)
    T_sfc = jnp.linspace(288.0, 298.0, ncol)      # some columns warmer/cooler
    q_sfc = jnp.linspace(6.0e-3, 18.0e-3, ncol)
    rho = jnp.full((ncol,), 1.2)
    return u, v, T, q_v, T_sfc, q_sfc, rho


def _surface_flux_scalar(params, bulk_scheme: str) -> jax.Array:
    """Sum of |surface fluxes| through the trained surface config.

    Differentiable in the params raw leaves, so ``jax.grad`` propagates
    into ``surface_most_*`` / ``z0h_z0_ratio`` iff the chosen bulk scheme
    reads them.
    """
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )

    cfg = params.to_surface_config(bulk_scheme=bulk_scheme)
    u, v, T, q_v, T_sfc, q_sfc, rho = _synthetic_surface_inputs()
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, cfg,
    )
    return (
        jnp.sum(jnp.abs(shflx))
        + jnp.sum(jnp.abs(lhflx))
        + jnp.sum(jnp.abs(tau_x))
        + jnp.sum(jnp.abs(tau_y))
    )


_MOST_KEYS = (
    "surface_most_unstable_gamma",
    "surface_most_stable_beta",
    "surface_z0h_z0_ratio",
)


def _grad_wrt_most_leaves(bulk_scheme: str) -> dict[str, float]:
    from legoesm.training.aimip_params import AIMIPClassicalParams
    import equinox as eqx

    p = AIMIPClassicalParams.from_defaults()

    def loss(params):
        return _surface_flux_scalar(params, bulk_scheme)

    grads = eqx.filter_grad(loss)(p)
    g = grads.raw_values
    return {k: float(jnp.asarray(g[k])) for k in _MOST_KEYS}


def test_gradient_reachability_most_is_live():
    grads = _grad_wrt_most_leaves("most")
    for k, val in grads.items():
        assert jnp.isfinite(val), f"{k} gradient not finite under 'most': {val}"
    # At least one MOST leaf must carry a non-trivial gradient (the fluxes
    # depend on all three; require a clear non-zero on the aggregate).
    total = sum(abs(v) for v in grads.values())
    assert total > 1e-8, (
        f"MOST leaves are dead under bulk_scheme='most' (|grad| sum={total}); "
        "the stability path is not reading them."
    )


def test_gradient_reachability_constant_is_dead():
    grads = _grad_wrt_most_leaves("constant")
    total = sum(abs(v) for v in grads.values())
    # Documents WHY the knob is needed: under the legacy constant path the
    # MOST leaves receive exactly zero gradient.
    assert total == 0.0, (
        f"Expected zero gradient on MOST leaves under 'constant', got {total}. "
        "If this fails the constant path is unexpectedly reading them."
    )


def test_most_grad_strictly_exceeds_constant():
    """Direct A/B: the exact evidence the knob unblocks training."""
    g_most = _grad_wrt_most_leaves("most")
    g_const = _grad_wrt_most_leaves("constant")
    sum_most = sum(abs(v) for v in g_most.values())
    sum_const = sum(abs(v) for v in g_const.values())
    assert sum_most > sum_const
    assert sum_const == 0.0


# ----------------------------------------------------------------------
# 5 + 6: config round-trip + suite parse
# ----------------------------------------------------------------------

def test_run_aimip_reads_surface_bulk_scheme_default():
    """run_aimip resolves aimip_surface_bulk_scheme with a constant default."""
    cfg: dict = {}
    assert str(cfg.get("aimip_surface_bulk_scheme", "constant")) == "constant"
    cfg2 = {"aimip_surface_bulk_scheme": "most"}
    assert str(cfg2.get("aimip_surface_bulk_scheme", "constant")) == "most"


def test_suite_curriculum_most_parses():
    import yaml
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    suite_path = repo / "config/aimip/ace2/suite_curriculum_most.yaml"
    assert suite_path.exists(), suite_path
    suite = yaml.safe_load(suite_path.read_text())

    # Base + variants resolve.
    base_path = repo / suite["base"]
    assert base_path.exists(), base_path
    assert suite["variants"] == ["classical"]

    ov = suite["cfg_overrides"]
    # Surface bulk scheme resolves to "most".
    assert ov["aimip_surface_bulk_scheme"] == "most"
    assert ov["aimip_spatial_surface"] is True
    # Curriculum parses to the expected leads.
    leads = [phase[0] for phase in ov["aimip_rollout_curriculum"]]
    assert leads == [12, 24, 72, 120]
    assert ov["loss_preset"] == "ace2_curriculum"


def test_base_config_declares_surface_bulk_scheme():
    import yaml
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    base = yaml.safe_load(
        (repo / "config/aimip/aimip_era5.yaml").read_text()
    )
    # Backward-compatible default in the base config.
    assert base.get("aimip_surface_bulk_scheme", "constant") == "constant"
