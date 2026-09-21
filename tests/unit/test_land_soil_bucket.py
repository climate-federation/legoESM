"""Unit tests for the prognostic soil-water bucket on the slab-land tile.

Covers the Manabe (1969) bucket added to ``PhysicsPipeline``:
- the soil-moisture evaporation efficiency ``_land_beta`` ramp
  (``beta = beta_min + (1-beta_min)*clip(W/W_max, 0, 1)``);
- ``_bucket_update`` water balance (precip source, beta-limited
  evaporation sink, overflow runoff, non-negativity);
- the coupling to the slab SEB: a drier bucket evaporates less and so
  the land skin equilibrates WARMER (the desert-heating effect);
- the bucket-off path is byte-identical to the legacy saturated surface;
- differentiability of the land skin temperature w.r.t. soil water.
"""
from __future__ import annotations

import jax
import pytest
import jax.numpy as jnp

from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.driver.config import ExperimentConfig
from legoesm.grids.cubed_sphere import create_cubed_sphere

NLEV = 6
N_CS = 4
SHAPE_2D = (6, N_CS, N_CS)
SHAPE_3D = (*SHAPE_2D, NLEV)


def _sigma(nlev=NLEV):
    class _S:
        sigma_full = jnp.linspace(0.1, 0.95, nlev)
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        dsigma = jnp.diff(jnp.linspace(0.05, 1.0, nlev + 1))

        def pressure_at_full(self, p_s):
            return p_s[..., None] * self.sigma_full

        def pressure_at_half(self, p_s):
            return p_s[..., None] * self.sigma_half

        def layer_thickness_dp(self, p_s):
            return p_s[..., None] * self.dsigma
    return _S()


def _bucket_pipeline(active=True, w_max=150.0, beta_min=0.1):
    """All-land gray-radiation pipeline with the soil bucket configured."""
    grid = create_cubed_sphere(N_CS)
    cfg = ExperimentConfig(radiation="gray", microphysics="none",
                           diurnal_cycle=False)
    pipe = build_physics_pipeline(grid, _sigma(), cfg)
    pipe.f_land = jnp.full(SHAPE_2D, 1.0)
    pipe.albedo_land = jnp.full(SHAPE_2D, 0.20)
    pipe.rad_update_steps = 1
    pipe.slab_land_active = True
    pipe.land_soil_bucket = active
    pipe.land_bucket_w_max = w_max
    pipe.land_beta_min = beta_min
    pipe.land_bucket_w_init_frac = 0.5
    return pipe


def _slab_inputs(T_land_val):
    return dict(
        T_land=jnp.full(SHAPE_2D, T_land_val),
        sw_down_sfc=jnp.full(SHAPE_2D, 400.0),
        lw_down_sfc=jnp.full(SHAPE_2D, 340.0),
        T=jnp.full(SHAPE_3D, 285.0),
        p_s=jnp.full(SHAPE_2D, 1.0e5),
        q_v=jnp.full(SHAPE_3D, 0.004),
        u=jnp.full(SHAPE_3D, 4.0),
        v=jnp.zeros(SHAPE_3D),
    )


# ---------------------------------------------------------------------------
# beta ramp
# ---------------------------------------------------------------------------

class TestLandBeta:
    def test_none_when_bucket_off(self):
        pipe = _bucket_pipeline(active=False)
        assert pipe._land_beta(jnp.full(SHAPE_2D, 75.0)) is None

    def test_none_when_w_land_none(self):
        pipe = _bucket_pipeline(active=True)
        assert pipe._land_beta(None) is None

    def test_ramp_endpoints_and_midpoint(self):
        pipe = _bucket_pipeline(active=True, w_max=150.0, beta_min=0.1)
        beta_dry = pipe._land_beta(jnp.zeros(SHAPE_2D))
        beta_wet = pipe._land_beta(jnp.full(SHAPE_2D, 150.0))
        beta_half = pipe._land_beta(jnp.full(SHAPE_2D, 75.0))
        assert jnp.allclose(beta_dry, 0.1)          # beta_min at W=0
        assert jnp.allclose(beta_wet, 1.0)          # saturated at W=W_max
        assert jnp.allclose(beta_half, 0.55)        # linear midpoint
        # clips above capacity (no beta > 1)
        assert jnp.allclose(pipe._land_beta(jnp.full(SHAPE_2D, 300.0)), 1.0)


# ---------------------------------------------------------------------------
# stomatal beta (shared land Jarvis model)
# ---------------------------------------------------------------------------

class TestStomatalBeta:
    """``land_stomatal_beta`` routes the soil availability through the shared
    land Jarvis stomatal model: ``beta = min(beta_soil, beta_canopy)``."""

    _FORCING = dict(
        T_land=jnp.full(SHAPE_2D, 295.0),
        q_air=jnp.full(SHAPE_2D, 0.006),
        p_s=jnp.full(SHAPE_2D, 1.0e5),
    )

    @staticmethod
    def _stomatal_pipe(beta_min=0.1, w_max=150.0):
        from legoesm.land.stomata import StomataConfig
        pipe = _bucket_pipeline(active=True, w_max=w_max, beta_min=beta_min)
        pipe.land_stomatal_beta = True
        pipe.stomata_config = StomataConfig()
        return pipe

    def test_off_returns_soil_beta_even_with_forcing(self):
        # land_stomatal_beta=False ⇒ stomatal forcing ignored, soil ramp kept.
        pipe = _bucket_pipeline(active=True)
        beta = pipe._land_beta(
            jnp.full(SHAPE_2D, 75.0), sw_down_sfc=jnp.full(SHAPE_2D, 500.0),
            **self._FORCING)
        assert jnp.allclose(beta, 0.55)   # soil-only midpoint, unchanged

    def test_falls_back_to_soil_beta_without_forcing(self):
        # Stomatal enabled but no forcing supplied ⇒ graceful soil-only beta.
        pipe = self._stomatal_pipe()
        assert jnp.allclose(pipe._land_beta(jnp.full(SHAPE_2D, 75.0)), 0.55)

    def test_daylight_bounded_and_not_exceeding_soil(self):
        pipe = self._stomatal_pipe()
        beta = pipe._land_beta(
            jnp.full(SHAPE_2D, 150.0),  # wet soil ⇒ beta_soil = 1
            sw_down_sfc=jnp.full(SHAPE_2D, 600.0), **self._FORCING)
        assert jnp.all(beta >= 0.0) and jnp.all(beta <= 1.0)
        assert jnp.all(beta <= 1.0 + 1e-6)   # never exceeds soil availability

    def test_night_closes_stomata(self):
        pipe = self._stomatal_pipe()
        w = jnp.full(SHAPE_2D, 150.0)
        beta_night = pipe._land_beta(
            w, sw_down_sfc=jnp.zeros(SHAPE_2D), **self._FORCING)
        beta_day = pipe._land_beta(
            w, sw_down_sfc=jnp.full(SHAPE_2D, 600.0), **self._FORCING)
        assert jnp.all(beta_night <= beta_day + 1e-6)   # darker ⇒ more closed
        assert float(jnp.max(beta_night)) < 0.05        # ~closed in the dark

    def test_differentiable_wrt_soil_water(self):
        pipe = self._stomatal_pipe()

        def beta_mean(w):
            return jnp.mean(pipe._land_beta(
                w, sw_down_sfc=jnp.full(SHAPE_2D, 600.0), **self._FORCING))

        g = jax.grad(beta_mean)(jnp.full(SHAPE_2D, 60.0))
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# bucket water balance
# ---------------------------------------------------------------------------

def _bucket_args(pipe, w_val, precip, beta_val=None):
    beta = None if beta_val is None else jnp.full(SHAPE_2D, beta_val)
    return dict(
        w_land=jnp.full(SHAPE_2D, w_val),
        precip_total=jnp.full(SHAPE_2D, precip),
        beta_land=beta,
        T_land=jnp.full(SHAPE_2D, 300.0),   # warm => positive q_sat => evap
        T_low=jnp.full(SHAPE_2D, 290.0),
        q_air=jnp.full(SHAPE_2D, 0.002),
        u_low=jnp.full(SHAPE_2D, 4.0),
        v_low=jnp.zeros(SHAPE_2D),
        p_s=jnp.full(SHAPE_2D, 1.0e5),
        dt=600.0,
    )


class TestBucketUpdate:
    def test_noop_when_off(self):
        pipe = _bucket_pipeline(active=False)
        w0 = jnp.full(SHAPE_2D, 75.0)
        w1, runoff = pipe._bucket_update(**_bucket_args(pipe, 75.0, 1e-4, 0.5))
        assert jnp.array_equal(w1, w0)
        assert runoff is None              # inactive -> no runoff diagnostic

    def test_fills_with_heavy_precip(self):
        pipe = _bucket_pipeline(active=True)
        # Moderate precip below the infiltration capacity, dry bucket => water rises.
        w1, _ = pipe._bucket_update(**_bucket_args(pipe, 50.0, 1e-3, 0.3))
        assert jnp.all(w1 > 50.0)

    def test_empties_under_evaporation(self):
        pipe = _bucket_pipeline(active=True)
        # No precip, warm wet surface into dry air => water falls.
        w1, _ = pipe._bucket_update(**_bucket_args(pipe, 100.0, 0.0, 1.0))
        assert jnp.all(w1 < 100.0)

    def test_capped_at_w_max_runoff(self):
        pipe = _bucket_pipeline(active=True, w_max=150.0)
        # Enormous precip over many steps cannot exceed capacity, and the rejected
        # water is reported as runoff (Hortonian + saturation excess), not lost.
        w = jnp.full(SHAPE_2D, 140.0)
        last_runoff = None
        for _ in range(50):
            w, last_runoff = pipe._bucket_update(**{**_bucket_args(pipe, 0.0, 1.0, 1.0),
                                                    "w_land": w})
        assert jnp.all(w <= 150.0 + 1e-6)
        assert jnp.all(last_runoff > 0.0)   # heavy rain on a full bucket -> runoff

    def test_stays_nonnegative(self):
        pipe = _bucket_pipeline(active=True)
        # No precip, strong evaporation, repeatedly => floored at 0.
        w = jnp.full(SHAPE_2D, 5.0)
        for _ in range(200):
            w, _ = pipe._bucket_update(**{**_bucket_args(pipe, 0.0, 0.0, 1.0),
                                          "w_land": w})
        assert jnp.all(w >= 0.0)

    def test_infiltration_excess_runoff_with_bucket_room(self):
        """Intense rain on a bucket with room generates Hortonian runoff and the
        runoff is suppressed when infiltration excess is disabled."""
        pipe = _bucket_pipeline(active=True, w_max=150.0)
        w_on, runoff_on = pipe._bucket_update(**_bucket_args(pipe, 75.0, 5e-2, 1.0))
        assert jnp.all(runoff_on > 0.0)        # rain rate > capacity -> Hortonian
        assert jnp.all(w_on < 150.0)           # bucket still has room
        pipe.land_infiltration_excess = False
        w_off, runoff_off = pipe._bucket_update(**_bucket_args(pipe, 75.0, 5e-2, 1.0))
        assert jnp.all(w_off > w_on)           # more infiltrates without the cap

    def test_drier_evaporates_less(self):
        """Lower beta removes less water for the same atmospheric state."""
        pipe = _bucket_pipeline(active=True)
        w_dry, _ = pipe._bucket_update(**_bucket_args(pipe, 80.0, 0.0, 0.2))
        w_wet, _ = pipe._bucket_update(**_bucket_args(pipe, 80.0, 0.0, 1.0))
        # Both lose water (no precip), but the low-beta surface loses less.
        assert jnp.all(w_dry > w_wet)


# ---------------------------------------------------------------------------
# SEB coupling: dry land equilibrates warmer
# ---------------------------------------------------------------------------

class TestSEBCoupling:
    def test_dry_land_equilibrates_warmer(self):
        pipe = _bucket_pipeline(active=True, beta_min=0.05)
        beta_dry = jnp.full(SHAPE_2D, 0.1)
        beta_wet = jnp.full(SHAPE_2D, 1.0)
        dry = jnp.full(SHAPE_2D, 270.0)
        wet = jnp.full(SHAPE_2D, 270.0)
        inp = _slab_inputs(0.0)
        forcing = {k: inp[k] for k in
                   ("sw_down_sfc", "lw_down_sfc", "T", "p_s", "q_v", "u", "v")}
        for _ in range(500):
            dry = pipe._step_slab_land(dry, **forcing, dt=600.0,
                                       beta_land=beta_dry)
            wet = pipe._step_slab_land(wet, **forcing, dt=600.0,
                                       beta_land=beta_wet)
        # Less evaporative cooling over dry soil => warmer skin.
        assert jnp.all(dry > wet + 1.0)
        assert jnp.all(jnp.isfinite(dry) & jnp.isfinite(wet))

    def test_beta_one_matches_legacy_wet_surface(self):
        """beta_land=None (or 1) reproduces the saturated-surface SEB."""
        pipe = _bucket_pipeline(active=True)
        forcing = {k: v for k, v in _slab_inputs(265.0).items()}
        legacy = pipe._step_slab_land(**forcing, dt=600.0)            # beta None
        beta_one = pipe._step_slab_land(**forcing, dt=600.0,
                                        beta_land=jnp.ones(SHAPE_2D))
        assert jnp.allclose(legacy, beta_one)


# ---------------------------------------------------------------------------
# differentiability
# ---------------------------------------------------------------------------

class TestDifferentiability:
    def test_T_land_differentiable_wrt_soil_water(self):
        pipe = _bucket_pipeline(active=True)
        forcing = {k: v for k, v in _slab_inputs(290.0).items()}

        def _skin_mean(w_land):
            beta = pipe._land_beta(w_land)
            T_new = pipe._step_slab_land(**forcing, dt=600.0, beta_land=beta)
            return jnp.mean(T_new)

        w0 = jnp.full(SHAPE_2D, 60.0)
        g = jax.grad(_skin_mean)(w0)
        assert jnp.all(jnp.isfinite(g))
        # Wetter soil => more evaporation => cooler skin: dT/dW < 0.
        assert jnp.all(g < 0.0)

    def test_bucket_differentiable_wrt_soil_water(self):
        pipe = _bucket_pipeline(active=True)

        def _w_next_mean(w_land):
            beta = pipe._land_beta(w_land)
            args = _bucket_args(pipe, 0.0, 0.0, None)
            args["w_land"] = w_land
            args["beta_land"] = beta
            w_new, _ = pipe._bucket_update(**args)
            return jnp.mean(w_new)

        g = jax.grad(_w_next_mean)(jnp.full(SHAPE_2D, 70.0))
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# full compiled-step integration (build_step_unified threads w_land)
# ---------------------------------------------------------------------------

def _bucket_step_pipeline():
    grid = create_cubed_sphere(N_CS)
    cfg = ExperimentConfig(radiation="gray", microphysics="none",
                           diurnal_cycle=False, turbulence="louis",
                           surface_bulk_scheme="coare3")
    pipe = build_physics_pipeline(grid, _sigma(), cfg)
    pipe.f_land = jnp.full(SHAPE_2D, 0.5)
    pipe.albedo_land = jnp.full(SHAPE_2D, 0.20)
    pipe.rad_update_steps = 1
    pipe.slab_land_active = True
    pipe.surface_tiled = True
    pipe.surface_z0_land = 0.1
    pipe.land_soil_bucket = True
    pipe.land_bucket_w_max = 150.0
    pipe.land_beta_min = 0.1
    pipe.land_bucket_w_init_frac = 0.5
    return pipe


class TestCompiledStepIntegration:
    def test_step_unified_advances_w_land(self):
        """The compiled physics step threads w_land in and returns an
        updated bucket on PhysicsOutput.w_land — and the bucket changes the
        BL moisture tendency vs the saturated-surface path."""
        from legoesm import constants

        def _run(pipe, w_land):
            step_fn = pipe.build_step_unified()
            ad = pipe.adapter
            s2 = ad.shape_2d
            s3 = (*s2, NLEV)
            held3 = jnp.zeros(s3)
            held2 = jnp.zeros(s2)
            o3 = jnp.zeros((ad.ncol, NLEV))
            aer = jnp.zeros((ad.ncol, NLEV))
            return step_fn(
                jnp.bool_(True),
                jnp.full(s3, 285.0), jnp.full(s2, 1.0e5),
                jnp.full(s3, 0.006), jnp.zeros(s3), jnp.zeros(s3),
                jnp.zeros((ad.ncol,)),
                jnp.full(s3, 5.0), jnp.zeros(s3),
                jnp.full(s2, 295.0), jnp.zeros(s2),
                jnp.full(s2, 0.4), jnp.full(s2, 1.0),
                100.0, 43200.0, 600.0,
                jnp.array([]), constants.S_0, o3, aer,
                held3, held2, held2, held2, held2, held2,
                T_land=jnp.full(s2, 300.0),
                w_land=w_land,
            )

        pipe = _bucket_step_pipeline()
        w0 = jnp.full(SHAPE_2D, 30.0)   # fairly dry bucket
        # step_unified returns (physics_out, new_held, T_land, land_ml) since the
        # multilayer-land refactor (#650); land_ml is None here.
        phys_out, _, _, _ = _run(pipe, w0)

        assert phys_out.w_land is not None
        assert phys_out.w_land.shape == w0.shape
        assert jnp.all(jnp.isfinite(phys_out.w_land))
        # The bucket moved (precip source / evaporation sink acted).
        assert not jnp.array_equal(phys_out.w_land, w0)

        # A dry bucket (low beta) yields a different BL moistening than the
        # saturated surface (bucket off): proves beta is wired into the step.
        pipe_off = _bucket_step_pipeline()
        pipe_off.land_soil_bucket = False
        out_off, _, _, _ = _run(pipe_off, None)
        assert out_off.w_land is None
        assert not jnp.allclose(phys_out.dq_v_dt, out_off.dq_v_dt)


def test_land_stomatal_beta_allowed_with_multilayer_land():
    """land_stomatal_beta needs the slab bucket ONLY when the slab land is active.
    use_multilayer_land supplies the root-zone moisture availability from the
    Richards column instead (#715 threads the stomata into MultiLayerLandConfig),
    so validate_strict must WAIVE the bucket requirement — else the SOTA multilayer
    + stomata AMIP config (config/amip/amip_sota.yaml) fails validation."""
    # multilayer + stomata, no slab bucket => the bucket rule must NOT fire
    ml = ExperimentConfig(use_multilayer_land=True, land_stomatal_beta=True,
                          land_soil_bucket=False)
    try:
        ml.validate_strict()
        waived = True
    except ValueError as exc:
        waived = "land_stomatal_beta=True requires land_soil_bucket" not in str(exc)
    assert waived, "multilayer land must waive the slab-bucket requirement for stomata"

    # slab land + stomata + no bucket => the rule STILL fires (guard not weakened)
    slab = ExperimentConfig(use_multilayer_land=False, land_stomatal_beta=True,
                            land_soil_bucket=False)
    try:
        slab.validate_strict()
        fired = False
    except ValueError as exc:
        fired = "land_stomatal_beta=True requires land_soil_bucket" in str(exc)
    assert fired, "slab land must still require the bucket for stomata"


@pytest.mark.parametrize("topography", ["flat", "gaussian"])
def test_surface_tiled_multilayer_requires_named_land(topography):
    """Idealized terrain needs a mask; file-derived land also enables the tile."""
    ml = ExperimentConfig(surface_tiled=True, turbulence="louis",
                          use_multilayer_land=True, slab_land_active=False,
                          land_mask_path="", topography=topography)
    with pytest.raises(ValueError, match=f"topography={topography!r}.*NO land"):
        ml.validate_strict()
    ml._replace(land_mask_path="land_mask.nc").validate_strict()
    ml._replace(topography="elevation.nc").validate_strict()

    # tiled + no land tile at all (no slab, no mask, no multilayer) => STILL fires
    none_tile = ExperimentConfig(surface_tiled=True, turbulence="louis",
                                 use_multilayer_land=False, slab_land_active=False,
                                 land_mask_path="")
    try:
        none_tile.validate_strict()
        fired = False
    except ValueError as exc:
        fired = "surface_tiled=True requires an active land tile" in str(exc)
    assert fired, "tiled surface with NO land tile must still be rejected"


@pytest.mark.parametrize("topography", ["flat", "gaussian"])
def test_slab_land_requires_named_land_on_idealized_terrain(topography):
    """A SLAB land tile on idealized terrain with no mask has no land either.

    The multilayer case above was guarded; the slab one was not, so
    ``slab_land_active=True`` with ``topography='flat'`` validated cleanly and
    then ran a land model over an all-ocean world -- the silently-inert land
    tile that cost the AMIP campaign three tuning waves.  Tiling is NOT part of
    the condition: an untiled slab over zero land is just as inert.
    """
    slab = ExperimentConfig(slab_land_active=True, land_mask_path="",
                            topography=topography)
    with pytest.raises(ValueError, match=f"topography={topography!r}.*NO land"):
        slab.validate_strict()
    slab._replace(land_mask_path="land_mask.nc").validate_strict()
    slab._replace(topography="elevation.nc").validate_strict()

    # An aquaplanet that asks for NO land tile keeps running on idealized
    # terrain -- the guard must not fire on "no land wanted".
    ExperimentConfig(topography=topography).validate_strict()
